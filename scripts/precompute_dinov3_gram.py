"""Precompute per-view DINOv3 Gram targets for a local LeRobot LIBERO dataset."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_REPO_ID = "/data0/luokang/dataset/luokang/lerobot/libero/libero_all_no_noops_1.0.0_lerobot_10hz"
DEFAULT_DATASET_ROOT = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_all_no_noops_1.0.0_lerobot_10hz"
)
DEFAULT_TEACHER = Path("/data0/luokang/dataset/luokang/ckpts/dinov3-vitl16-pretrain-lvd1689m")
DEFAULT_OUTPUT = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_all_no_noops_1.0.0_lerobot_10hz_dino_gram_vitl16_l24_256"
)


class ImageDataset(Dataset):
    """Read only the fields needed by the frozen teacher."""

    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        return {
            "index": sample["index"],
            "base": sample["observation.images.image"],
            "wrist": sample["observation.images.wrist_image"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--teacher-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--teacher-layer", type=int, default=24, help="One-based transformer block output.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--flush-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_images(images: torch.Tensor, image_size: int, device: torch.device) -> torch.Tensor:
    images = torch.as_tensor(images)
    if images.ndim != 4:
        raise ValueError(f"Expected batched images, got shape {tuple(images.shape)}")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] != 3:
        raise ValueError(f"Expected RGB images, got shape {tuple(images.shape)}")

    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    if images.min() < 0.0:
        images = images / 2.0 + 0.5
    elif images.max() > 1.0:
        images = images / 255.0
    # Match LiberoGramInputs, which horizontally flips both camera views.
    images = torch.flip(images, dims=(-1,))
    images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_gram(
    model,
    images: torch.Tensor,
    *,
    teacher_layer: int,
    num_layers: int,
    num_register_tokens: int,
    expected_patches: int,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    use_hidden_states = teacher_layer != num_layers
    amp_context = (
        torch.autocast(device_type=images.device.type, dtype=amp_dtype)
        if amp_dtype is not None and images.device.type in ("cuda", "cpu")
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        outputs = model(pixel_values=images, output_hidden_states=use_hidden_states)
        hidden = outputs.last_hidden_state if not use_hidden_states else outputs.hidden_states[teacher_layer]

    patch_tokens = hidden[:, 1 + num_register_tokens :, :].float()
    if patch_tokens.shape[1] != expected_patches:
        raise ValueError(
            f"Expected {expected_patches} DINO patch tokens, got {patch_tokens.shape[1]} "
            f"from hidden shape {tuple(hidden.shape)}"
        )
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    gram = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
    return gram.to(torch.float16).cpu().numpy()


def open_outputs(output_dir: Path, num_frames: int, num_patches: int, overwrite: bool):
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = (num_frames, num_patches, num_patches)
    gram_paths = {
        "base": output_dir / "base_0_rgb.npy",
        "wrist": output_dir / "left_wrist_0_rgb.npy",
    }
    if any(path.exists() for path in gram_paths.values()):
        raise FileExistsError(f"Gram outputs already exist in {output_dir}; use --overwrite to restart")
    grams = {
        key: np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        for key, path in gram_paths.items()
    }
    return grams


def main() -> None:
    args = parse_args()
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by the DINOv3 patch size 16")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    try:
        model = AutoModel.from_pretrained(args.teacher_path, local_files_only=True)
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "Failed to load DINOv3 from Transformers. This checkpoint requires a Transformers version "
            "with DINOv3ViTModel support (4.56 or newer)."
        ) from error
    model.requires_grad_(False).eval().to(device)

    num_layers = int(model.config.num_hidden_layers)
    num_register_tokens = int(model.config.num_register_tokens)
    if not 1 <= args.teacher_layer <= num_layers:
        raise ValueError(f"--teacher-layer must be in [1, {num_layers}], got {args.teacher_layer}")
    num_patches = (args.image_size // int(model.config.patch_size)) ** 2
    if num_patches != 256:
        raise ValueError(f"Expected a 16x16 target grid, got {num_patches} patches")

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    grams = open_outputs(args.output_dir, len(dataset), num_patches, args.overwrite)
    metadata = {
        "source_repo_id": args.repo_id,
        "source_dataset": str(args.dataset_root),
        "num_frames": len(dataset),
        "teacher_path": str(args.teacher_path),
        "teacher_layer": args.teacher_layer,
        "input_size": args.image_size,
        "patch_size": int(model.config.patch_size),
        "num_register_tokens": num_register_tokens,
        "num_patch_tokens": num_patches,
        "horizontal_flip": True,
        "feature_normalized": True,
        "negative_similarity_removed": False,
        "dtype": "float16",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    loader = DataLoader(
        ImageDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dtype_map = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = dtype_map[args.dtype]

    for step, batch in enumerate(tqdm(loader, desc="DINOv3 Gram"), start=1):
        indices = np.asarray(batch["index"], dtype=np.int64).reshape(-1)
        for key in ("base", "wrist"):
            images = prepare_images(batch[key], args.image_size, device)
            grams[key][indices] = extract_gram(
                model,
                images,
                teacher_layer=args.teacher_layer,
                num_layers=num_layers,
                num_register_tokens=num_register_tokens,
                expected_patches=num_patches,
                amp_dtype=amp_dtype,
            )
        if step % args.flush_interval == 0:
            for array in grams.values():
                array.flush()

    for array in grams.values():
        array.flush()
    print(f"Saved {len(dataset)} aligned Gram targets to {args.output_dir}")


if __name__ == "__main__":
    main()
