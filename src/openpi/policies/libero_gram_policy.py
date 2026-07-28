import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image[:, ::-1, :].copy()


@dataclasses.dataclass(frozen=True)
class LiberoGramInputs(transforms.DataTransformFn):
    """Convert the minimal LIBERO fields and offline Gram targets to model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "dino_gram" in data:
            base_gram = np.asarray(data["dino_gram"]["base"], dtype=np.float16)
            wrist_gram = np.asarray(data["dino_gram"]["wrist"], dtype=np.float16)
            inputs["dino_gram"] = {
                "base_0_rgb": base_gram,
                "left_wrist_0_rgb": wrist_gram,
                "right_wrist_0_rgb": np.zeros_like(base_gram),
            }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs
