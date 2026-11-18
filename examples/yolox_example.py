"""Shared model loading and input preparation for the YOLOX examples."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_yolox_model(device, checkpoint_path=None):
    try:
        from yolox.models.build import yolox_m
    except ImportError as error:
        raise ImportError(
            "Install upstream YOLOX separately; see README.md."
        ) from error

    if checkpoint_path is not None and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"YOLOX checkpoint not found: {checkpoint_path}")
    model = yolox_m(pretrained=checkpoint_path is None, device=str(device))
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.eval()


def prepare_yolox_image(image, canvas_size):
    """Return CHW BGR model input and the matching RGB display canvas."""
    from yolox.data.data_augment import preproc

    rgb = np.asarray(image.convert("RGB"))
    bgr_chw, _ = preproc(rgb[..., ::-1], canvas_size)
    display = Image.fromarray(bgr_chw[::-1].transpose(1, 2, 0).astype(np.uint8))
    return torch.from_numpy(bgr_chw), display
