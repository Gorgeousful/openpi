from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
    state_sharding: training_utils.TrainState | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restore_items = {
            "train_state": train_state,
            "params": {"params": params},
        }

        restore_args = None
        if state_sharding is not None:
            train_state_sharding, params_sharding = _split_params(state_sharding)
            _warn_if_checkpoint_sharding_differs(
                checkpoint_manager,
                step,
                {
                    "train_state": train_state_sharding,
                    "params": {"params": params_sharding},
                },
            )
            restore_args = ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(
                    item=train_state,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(train_state, train_state_sharding),
                ),
                params=ocp.args.PyTreeRestore(
                    item={"params": params},
                    restore_args=ocp.checkpoint_utils.construct_restore_args(
                        {"params": params}, {"params": params_sharding}
                    ),
                ),
            )

        if restore_args is None:
            restored = checkpoint_manager.restore(step, items=restore_items)
        else:
            restored = checkpoint_manager.restore(step, args=restore_args)
    return _merge_params(restored["train_state"], restored["params"])


def _warn_if_checkpoint_sharding_differs(
    checkpoint_manager: ocp.CheckpointManager,
    step: int | None,
    current_sharding_items: dict[str, object],
) -> None:
    restore_step = checkpoint_manager.latest_step() if step is None else step
    if restore_step is None:
        return

    checkpoint_dir = epath.Path(checkpoint_manager.directory) / str(restore_step)
    for item_name, current_sharding in current_sharding_items.items():
        checkpoint_summary = _checkpoint_sharding_summary(checkpoint_dir / item_name / "_sharding")
        current_summary = _current_sharding_summary(current_sharding)
        if checkpoint_summary and current_summary and checkpoint_summary != current_summary:
            logging.warning(
                "-----\n"
                "Checkpoint sharding metadata differs from current restore sharding.\n"
                "item=%s, step=%s\n"
                "checkpoint_sharding=%s\n"
                "current_restore_sharding=%s\n"
                "Restoring with current sharding.\n"
                "-----",
                item_name,
                restore_step,
                sorted(checkpoint_summary),
                sorted(current_summary),
            )


def _checkpoint_sharding_summary(sharding_path: epath.Path) -> set[tuple]:
    if not sharding_path.exists():
        return set()

    raw_metadata = json.loads(sharding_path.read_text())
    summary = set()
    for raw_sharding in raw_metadata.values():
        metadata = json.loads(raw_sharding)
        summary.add(
            (
                tuple(metadata.get("shape", ())),
                tuple(metadata.get("axis_names", ())),
                _normalize_partition_spec(metadata.get("partition_spec", ())),
            )
        )
    return summary


def _current_sharding_summary(sharding_tree: object) -> set[tuple]:
    leaves = jax.tree_util.tree_leaves(
        sharding_tree,
        is_leaf=lambda x: isinstance(x, jax.sharding.Sharding),
    )
    summary = set()
    for leaf in leaves:
        if isinstance(leaf, jax.sharding.NamedSharding):
            summary.add(
                (
                    tuple(leaf.mesh.shape[name] for name in leaf.mesh.axis_names),
                    tuple(leaf.mesh.axis_names),
                    _normalize_partition_spec(tuple(leaf.spec)),
                )
            )
        elif isinstance(leaf, jax.sharding.Sharding):
            summary.add((type(leaf).__name__, str(leaf)))
    return summary


def _normalize_partition_spec(spec) -> tuple:
    def normalize_axis(axis):
        if axis is None:
            return None
        if isinstance(axis, (tuple, list)):
            return tuple(axis)
        return axis

    return tuple(normalize_axis(axis) for axis in spec)


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
