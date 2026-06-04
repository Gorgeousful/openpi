import ast
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_custom_example() -> dict:
    """Creates a random input example for the Libero custom policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image, *, horizontal_flip: bool = False) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if horizontal_flip:
        image = image[:, ::-1, :].copy()
    return image


def _format_grounding_with_loc_tokens(grounding, *, image_height: int, image_width: int) -> str:
    """Formats grounding boxes as PaliGemma location tokens."""
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image dimensions must be positive.")

    if not isinstance(grounding, str):
        grounding = grounding.item() if np.asarray(grounding).ndim == 0 else grounding
    items = ast.literal_eval(grounding) if isinstance(grounding, str) else grounding

    formatted_items = []
    for name, bbox in items:
        name = str(name).strip()
        if not name:
            raise ValueError("Grounding object name must not be empty.")
        if bbox is None:
            formatted_items.append(f"none {name}")
            continue
        if len(bbox) != 4:
            raise ValueError(f"Expected [xmin, ymin, xmax, ymax] bbox for {name!r}, got {bbox!r}.")

        xmin, ymin, xmax, ymax = bbox

        def quantize(value, size):
            return int(np.clip(np.rint(float(value) / size * 1023), 0, 1023))

        loc_ymin = quantize(ymin, image_height)
        loc_xmin = quantize(xmin, image_width)
        loc_ymax = quantize(ymax, image_height)
        loc_xmax = quantize(xmax, image_width)
        formatted_items.append(f"<loc{loc_ymin:04d}><loc{loc_xmin:04d}><loc{loc_ymax:04d}><loc{loc_xmax:04d}> {name}")

    return "; ".join(formatted_items)


def _strip_wrapping_brackets(value) -> str:
    """Removes one pair of wrapping [] or () from a string value."""
    if not isinstance(value, str):
        value = value.item() if np.asarray(value).ndim == 0 else str(value)
    value = str(value).strip()
    if len(value) >= 2 and value[0] in "[(" and value[-1] in "[])":
        matching = {"[": "]", "(": ")"}
        if matching[value[0]] == value[-1]:
            return value[1:-1].strip()
    return value


@dataclasses.dataclass(frozen=True)
class LiberoCustomInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"], horizontal_flip=True)
        wrist_image = _parse_image(data["observation/wrist_image"], horizontal_flip=True)

        if self.model_type in (_model.ModelType.PI0_FAST_THINKING, _model.ModelType.PI0_AR_THINKING):
            inputs = {
                "state": data["observation/state"],
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                },
            }
        else:
            # Create inputs dict. Do not change the keys in the dict below.
            inputs = {
                "state": data["observation/state"],
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                    # Pad any non-existent images with zero-arrays of the appropriate shape.
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                    "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                },
            }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "grounding" in data:
            inputs["grounding"] = _format_grounding_with_loc_tokens(
                data["grounding"],
                image_height=base_image.shape[0],
                image_width=base_image.shape[1],
            )

        if "subtask" in data:
            inputs["subtask"] = data["subtask"]

        if "focus" in data:
            inputs["focus"] = _strip_wrapping_brackets(data["focus"])

        if "phase" in data:
            inputs["phase"] = data["phase"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoCustomOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        outputs = {"actions": np.asarray(data["actions"][:, :7])}
        if "thinking" in data:
            outputs["thinking"] = data["thinking"]
        return outputs
