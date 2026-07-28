"""Verify that offline DINOv3 Gram targets match their LIBERO images."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from precompute_dinov3_gram import DEFAULT_DATASET_ROOT
from precompute_dinov3_gram import DEFAULT_OUTPUT
from precompute_dinov3_gram import DEFAULT_REPO_ID
from precompute_dinov3_gram import DEFAULT_TEACHER
from precompute_dinov3_gram import LeRobotDataset
from precompute_dinov3_gram import extract_gram
from precompute_dinov3_gram import prepare_images
import torch
from transformers import AutoModel

VIEW_FIELDS = {
    "base": ("observation.images.image", "base_0_rgb.npy"),
    "wrist": ("observation.images.wrist_image", "left_wrist_0_rgb.npy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--teacher-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--gram-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/dinov3_gram_verify"))
    parser.add_argument("--indices", type=int, nargs="+", default=(0, 1000, 15085))
    parser.add_argument("--views", choices=tuple(VIEW_FIELDS), nargs="+", default=tuple(VIEW_FIELDS))
    parser.add_argument(
        "--anchors",
        nargs="+",
        default=("4,4", "8,8", "12,12"),
        help="16x16 patch coordinates formatted as row,column.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--remove-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the training-time ReLU before visualizing similarities.",
    )
    return parser.parse_args()


def parse_anchors(values: list[str] | tuple[str, ...]) -> list[tuple[int, int]]:
    anchors = []
    for value in values:
        try:
            row, column = (int(part) for part in value.split(","))
        except ValueError as error:
            raise ValueError(f"Invalid anchor {value!r}; expected row,column") from error
        if not 0 <= row < 16 or not 0 <= column < 16:
            raise ValueError(f"Anchor {(row, column)} is outside the 16x16 patch grid")
        anchors.append((row, column))
    return anchors


def teacher_image(normalized: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), device=normalized.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=normalized.device).view(1, 3, 1, 1)
    image = (normalized * std + mean).clamp(0, 1)[0].permute(1, 2, 0)
    return (image.float().cpu().numpy() * 255).round().astype(np.uint8)


def visualize(image_rgb: np.ndarray, gram: np.ndarray, anchors: list[tuple[int, int]], remove_negative: bool):
    panels = []
    for anchor_num, (row, column) in enumerate(anchors):
        marked = image_rgb.copy()
        patch_size = image_rgb.shape[0] // 16
        cv2.rectangle(
            marked,
            (column * patch_size, row * patch_size),
            ((column + 1) * patch_size - 1, (row + 1) * patch_size - 1),
            (255, 0, 0),
            3,
        )
        if anchor_num == 0:
            panels.append(marked)

        similarity = gram[row * 16 + column].reshape(16, 16).astype(np.float32)
        if remove_negative:
            similarity = np.maximum(similarity, 0)
            similarity_uint8 = np.clip(similarity * 255, 0, 255).astype(np.uint8)
        else:
            similarity_uint8 = np.clip((similarity + 1) * 127.5, 0, 255).astype(np.uint8)
        similarity_uint8 = cv2.resize(
            similarity_uint8, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        heatmap = cv2.applyColorMap(similarity_uint8, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 0.45, heatmap, 0.55, 0)
        cv2.putText(overlay, f"anchor=({row},{column})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        panels.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = parse_args()
    anchors = parse_anchors(args.anchors)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    metadata = json.loads((args.gram_dir / "metadata.json").read_text())
    image_size = int(metadata["input_size"])
    teacher_layer = int(metadata["teacher_layer"])
    if image_size != 256 or int(metadata["num_patch_tokens"]) != 256:
        raise ValueError(f"Expected 256x256 teacher inputs and 256 patches, got metadata={metadata}")

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f"Dataset index {index} is outside [0, {len(dataset)})")

    model = AutoModel.from_pretrained(args.teacher_path, local_files_only=True)
    model.requires_grad_(False).eval().to(device)
    num_layers = int(model.config.num_hidden_layers)
    num_register_tokens = int(model.config.num_register_tokens)
    dtype_map = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
    gram_files = {
        view: np.load(args.gram_dir / VIEW_FIELDS[view][1], mmap_mode="r")
        for view in args.views
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset_index in args.indices:
        sample = dataset[dataset_index]
        global_index = int(np.asarray(sample["index"]).item())
        if global_index != dataset_index:
            raise ValueError(
                f"Dataset position {dataset_index} has global index {global_index}; "
                "the current offline/online indexing assumption is not valid"
            )

        for view in args.views:
            field, _ = VIEW_FIELDS[view]
            normalized = prepare_images(torch.as_tensor(sample[field]).unsqueeze(0), image_size, device)
            recomputed = extract_gram(
                model,
                normalized,
                teacher_layer=teacher_layer,
                num_layers=num_layers,
                num_register_tokens=num_register_tokens,
                expected_patches=256,
                amp_dtype=dtype_map[args.dtype],
            )[0].astype(np.float32)
            stored = np.asarray(gram_files[view][global_index], dtype=np.float32)
            difference = recomputed - stored
            metrics = {
                "dataset_index": dataset_index,
                "global_index": global_index,
                "view": view,
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "max_abs_error": float(np.max(np.abs(difference))),
            }
            results.append(metrics)
            print(json.dumps(metrics))

            panel = visualize(teacher_image(normalized), stored, anchors, args.remove_negative)
            output_path = args.output_dir / f"index_{global_index:06d}_{view}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    summary = {
        "gram_dir": str(args.gram_dir),
        "teacher_path": str(args.teacher_path),
        "results": results,
        "mean_mae": float(np.mean([result["mae"] for result in results])),
        "max_abs_error": float(max(result["max_abs_error"] for result in results)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved verification results to {args.output_dir}")


if __name__ == "__main__":
    main()
