"""Visualize cross-patch similarity from selected Gemma visual-token layers."""

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0
import openpi.models.gemma as _gemma
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi05_libero_gram_low_mem_finetune/"
    "pi05_libero_gram_low_mem_finetune-0727/15000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_libero_gram_low_mem_finetune")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=int, default=33611)
    parser.add_argument("--view", choices=("base", "wrist"), default="base")
    parser.add_argument("--anchor", default="12,6", help="16x16 patch coordinate formatted as row,column.")
    parser.add_argument("--layers", type=int, nargs="+", default=(1, 12, 18))
    parser.add_argument("--output-dir", type=Path, default=Path("gram_verify_bowl/model_layers_step15000"))
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--remove-negative", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_anchor(value: str) -> tuple[int, int]:
    try:
        row, column = (int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError(f"Invalid anchor {value!r}; expected row,column") from error
    if not 0 <= row < 16 or not 0 <= column < 16:
        raise ValueError(f"Anchor {(row, column)} is outside the 16x16 patch grid")
    return row, column


def load_observation(config: _config.TrainConfig, index: int) -> _model.Observation:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    if not 0 <= index < len(dataset):
        raise IndexError(f"Dataset index {index} is outside [0, {len(dataset)})")
    item = dataset[index]
    batch = jax.tree.map(lambda value: np.expand_dims(np.asarray(value), axis=0), item)
    observation = _model.Observation.from_dict(batch)
    return jax.tree.map(jnp.asarray, observation)


def extract_view_tokens(model, observation: _model.Observation, view_key: str) -> np.ndarray:
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation, include_auxiliary=False)
    attention_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, _, hidden_states = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=attention_mask,
        positions=positions,
        adarms_cond=[None, None],
        output_hidden_states=True,
    )

    offset = 0
    for name in observation.images:
        if name == view_key:
            tokens = hidden_states[0][:, offset : offset + 256].astype(jnp.float32)
            return np.asarray(jax.device_get(tokens[0]))
        offset += 256
    raise KeyError(f"View {view_key!r} not found in {tuple(observation.images)}")


def token_gram(tokens: np.ndarray, *, remove_negative: bool) -> np.ndarray:
    tokens = tokens.astype(np.float32)
    tokens /= np.sqrt(np.sum(np.square(tokens), axis=-1, keepdims=True) + 1e-6)
    gram = tokens @ tokens.T
    return np.maximum(gram, 0) if remove_negative else gram


def input_image(observation: _model.Observation, view_key: str) -> np.ndarray:
    image = np.asarray(observation.images[view_key][0])
    if image.dtype == np.uint8:
        return image
    return np.clip((image.astype(np.float32) + 1) * 127.5, 0, 255).round().astype(np.uint8)


def make_panel(image_rgb: np.ndarray, similarity: np.ndarray, title: str, anchor: tuple[int, int]) -> np.ndarray:
    similarity = similarity.reshape(16, 16).astype(np.float32)
    if similarity.min() < 0:
        similarity = (similarity + 1) / 2
    heatmap = np.clip(similarity * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    panel = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 0.45, heatmap, 0.55, 0)
    cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    row, column = anchor
    patch_height = image_rgb.shape[0] // 16
    patch_width = image_rgb.shape[1] // 16
    cv2.rectangle(
        panel,
        (column * patch_width, row * patch_height),
        ((column + 1) * patch_width - 1, (row + 1) * patch_height - 1),
        (255, 255, 255),
        2,
    )
    return panel


def top_patches(similarity: np.ndarray, top_k: int) -> list[dict[str, int | float]]:
    order = np.argsort(similarity)[::-1][:top_k]
    return [
        {"row": int(index // 16), "column": int(index % 16), "similarity": float(similarity[index])}
        for index in order
    ]


def main() -> None:
    args = parse_args()
    anchor = parse_anchor(args.anchor)
    anchor_index = anchor[0] * 16 + anchor[1]
    train_config = _config.get_config(args.config_name)
    depth = _gemma.get_config(train_config.model.paligemma_variant).depth
    if any(not 1 <= layer <= depth for layer in args.layers):
        raise ValueError(f"Layers must be within [1, {depth}], got {args.layers}")

    observation = load_observation(train_config, args.index)
    view_key = "base_0_rgb" if args.view == "base" else "left_wrist_0_rgb"
    image_rgb = input_image(observation, view_key)
    target_gram = np.asarray(observation.dino_gram[view_key][0], dtype=np.float32)
    if args.remove_negative:
        target_gram = np.maximum(target_gram, 0)
    target_similarity = target_gram[anchor_index]

    params = _model.restore_params(args.checkpoint_dir / "params", dtype=jnp.bfloat16)
    similarities = {}
    metrics = {}
    for layer in args.layers:
        layer_config = dataclasses.replace(train_config.model, gram_layer=layer)
        model = layer_config.load(params)
        tokens = extract_view_tokens(model, observation, view_key)
        similarity = token_gram(tokens, remove_negative=args.remove_negative)[anchor_index]
        similarities[layer] = similarity
        metrics[str(layer)] = {
            "pearson_with_dino": float(np.corrcoef(similarity, target_similarity)[0, 1]),
            "mse_with_dino": float(np.mean(np.square(similarity - target_similarity))),
            "top_patches": top_patches(similarity, args.top_k),
        }
        del model
        gc.collect()
        jax.clear_caches()

    original = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    row, column = anchor
    patch_height = image_rgb.shape[0] // 16
    patch_width = image_rgb.shape[1] // 16
    cv2.rectangle(
        original,
        (column * patch_width, row * patch_height),
        ((column + 1) * patch_width - 1, (row + 1) * patch_height - 1),
        (0, 0, 255),
        3,
    )
    panels = [original, make_panel(image_rgb, target_similarity, "DINOv3 teacher", anchor)]
    panels.extend(
        make_panel(image_rgb, similarities[layer], f"Gemma layer {layer}", anchor) for layer in args.layers
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"index_{args.index:06d}_{args.view}_anchor_{anchor[0]}_{anchor[1]}.png"
    cv2.imwrite(str(image_path), np.concatenate(panels, axis=1))
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_index": args.index,
        "view": args.view,
        "anchor": {"row": anchor[0], "column": anchor[1]},
        "remove_negative": args.remove_negative,
        "dino_top_patches": top_patches(target_similarity, args.top_k),
        "layers": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved visualization to {image_path}")


if __name__ == "__main__":
    main()
