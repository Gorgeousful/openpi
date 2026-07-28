"""Pi0 variant with an offline DINOv3 Gram distillation loss."""

import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at


class _GramBlock(_gemma.Block):
    """Gemma block that keeps only the requested post-block hidden states in the scan carry."""

    capture_layers: tuple[int, ...] = (11,)

    @nn.compact
    def __call__(
        self, carry, kv_cache, layer_index, positions, attn_mask, adarms_cond, deterministic=True
    ):  # noqa: FBT002
        xs, captured_hidden = carry
        xs, layer_kv_cache = super().__call__(xs, kv_cache, positions, attn_mask, adarms_cond, deterministic)
        capture_mask = jnp.asarray(self.capture_layers) == layer_index
        captured_hidden = jax.tree.map(
            lambda captured, current: jnp.where(
                capture_mask.reshape((-1,) + (1,) * current.ndim),
                current[None, ...],
                captured,
            ),
            captured_hidden,
            xs,
        )
        return (xs, captured_hidden), layer_kv_cache


class _GramGemmaModule(_gemma.Module):
    """Gemma module with the original parameter tree plus selected intermediate states."""

    capture_layers: tuple[int, ...] = (11,)

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = _gemma.Embedder(
            vocab_size=_gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            _GramBlock,
            prevent_cse=False,
            static_argnums=(6,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, 0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            detach_vlm_for_flow=self.detach_vlm_for_flow,
            capture_layers=self.capture_layers,
        )
        self.final_norms = [
            _gemma.RMSNorm(name=_gemma._name("final_norm", i))  # noqa: SLF001
            for i in range(len(self.configs))
        ]

    def __call__(
        self,
        embedded,
        positions,
        mask,
        adarms_cond=None,
        *,
        kv_cache=None,
        deterministic=True,
        output_hidden_states=False,
    ):
        embedded = jax.tree.map(lambda value: value.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        captured_hidden = jax.tree.map(
            lambda value: jnp.zeros((len(self.capture_layers), *value.shape), dtype=value.dtype),
            embedded,
        )
        (embedded, captured_hidden), kv_cache = self.layers(
            (embedded, captured_hidden),
            kv_cache,
            jnp.arange(self.configs[0].depth),
            positions,
            mask,
            adarms_cond,
            deterministic,
        )
        outputs = [
            norm(value, cond)[0] if value is not None else value
            for norm, value, cond in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if output_hidden_states:
            return outputs, kv_cache, captured_hidden
        return outputs, kv_cache


@dataclasses.dataclass(frozen=True)
class Pi0GramConfig(pi0_config.Pi0Config):
    """Pi0 configuration isolated to the Gram distillation experiment."""

    gram_loss_weight: float = 0.1
    gram_layers: list[int] = dataclasses.field(default_factory=lambda: [12])
    gram_remove_negative: bool = True
    gram_use_wrist: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.gram_loss_weight < 0:
            raise ValueError("gram_loss_weight must be non-negative")
        depth = _gemma.get_config(self.paligemma_variant).depth
        if len(set(self.gram_layers)) != len(self.gram_layers):
            raise ValueError("gram_layers must not contain duplicates")
        if any(not 1 <= layer <= depth for layer in self.gram_layers):
            raise ValueError(f"gram_layers must contain values in [1, {depth}] for {self.paligemma_variant}")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Gram":
        return Pi0Gram(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        gram_spec = jax.ShapeDtypeStruct([batch_size, 256, 256], jnp.float16)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                dino_gram={
                    "base_0_rgb": gram_spec,
                    "left_wrist_0_rgb": gram_spec,
                    "right_wrist_0_rgb": gram_spec,
                },
            )
        return observation_spec, action_spec


class Pi0Gram(pi0.Pi0):
    """Pi0 with relation supervision on selected PaliGemma image-token layers."""

    def __init__(self, config: Pi0GramConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.aux_loss_weight = config.aux_loss_weight
        self.aux_ce_chunk_size = config.aux_ce_chunk_size
        self.gram_loss_weight = config.gram_loss_weight
        self.gram_layers = tuple(config.gram_layers)
        self.gram_remove_negative = config.gram_remove_negative
        self.gram_use_wrist = config.gram_use_wrist
        self.use_augmentation = config.use_augmentation

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _GramGemmaModule(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                detach_vlm_for_flow=config.detach_vlm_for_flow,
                capture_layers=tuple(layer - 1 for layer in config.gram_layers),
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width, action_expert_config.width, rngs=rngs
            )
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs
            )
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

    @override
    def _compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        if observation.dino_gram is None:
            raise ValueError("Pi0Gram requires offline DINOv3 targets in observation.dino_gram")

        use_augmentation = train and self.use_augmentation
        if use_augmentation:
            preprocess_rng, rng = jax.random.split(rng)
        else:
            preprocess_rng = None
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=use_augmentation
        )
        noise_rng, time_rng = jax.random.split(rng, 2)
        use_auxiliary = self.aux_loss_weight > 0 and observation.tokenized_auxiliary is not None

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, include_auxiliary=use_auxiliary
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        if prefix_ar_mask.ndim == 2:
            suffix_ar_mask = jnp.broadcast_to(suffix_ar_mask, (prefix_ar_mask.shape[0], suffix_ar_mask.shape[0]))
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        else:
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        if use_auxiliary:
            auxiliary_len = observation.tokenized_auxiliary.shape[1]
            auxiliary_start = prefix_mask.shape[1] - auxiliary_len
            attn_mask = attn_mask.at[:, prefix_mask.shape[1] :, auxiliary_start : prefix_mask.shape[1]].set(False)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _, hidden_states = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            output_hidden_states=True,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # The scan carry retains only the configured post-block states; layer 12 means output of block index 11.
        layer_hidden_states = hidden_states[0].astype(jnp.float32)
        gram_loss_sum = jnp.zeros(actions.shape[0], dtype=jnp.float32)
        gram_term_count = jnp.zeros(actions.shape[0], dtype=jnp.float32)
        for layer_hidden in layer_hidden_states:
            token_offset = 0
            for name in observation.images:
                target_gram = observation.dino_gram.get(name)
                tokens_per_image = 256 if target_gram is None else target_gram.shape[-1]
                image_hidden = layer_hidden[:, token_offset : token_offset + tokens_per_image]
                token_offset += tokens_per_image
                if target_gram is None:
                    continue
                if not self.gram_use_wrist and "wrist" in name:
                    continue
                if image_hidden.shape[1] != target_gram.shape[-1]:
                    raise ValueError(
                        f"Gram token mismatch for {name}: student has {image_hidden.shape[1]}, "
                        f"teacher has {target_gram.shape[-1]}"
                    )

                image_hidden = image_hidden * jax.lax.rsqrt(
                    jnp.sum(jnp.square(image_hidden), axis=-1, keepdims=True) + 1e-6
                )
                student_gram = image_hidden @ jnp.swapaxes(image_hidden, -1, -2)
                target_gram = target_gram.astype(jnp.float32)
                if self.gram_remove_negative:
                    student_gram = jax.nn.relu(student_gram)
                    target_gram = jax.nn.relu(target_gram)
                view_loss = jnp.mean(jnp.square(student_gram - target_gram), axis=(-2, -1))
                view_mask = observation.image_masks[name].astype(jnp.float32)
                gram_loss_sum += view_loss * view_mask
                gram_term_count += view_mask
        gram_loss = gram_loss_sum / jnp.maximum(gram_term_count, 1.0)

        auxiliary_loss = jnp.zeros(action_loss.shape[:-1], dtype=jnp.float32)
        if use_auxiliary:
            auxiliary_loss = self._compute_auxiliary_loss(prefix_out, observation)
        total_loss = (
            action_loss
            + self.aux_loss_weight * auxiliary_loss[..., None]
            + self.gram_loss_weight * gram_loss[..., None]
        )
        return total_loss, {
            "action_loss": jnp.mean(action_loss),
            "auxiliary_loss": jnp.mean(auxiliary_loss),
            "weighted_auxiliary_loss": self.aux_loss_weight * jnp.mean(auxiliary_loss),
            "gram_loss": jnp.mean(gram_loss),
            "weighted_gram_loss": self.gram_loss_weight * jnp.mean(gram_loss),
        }
