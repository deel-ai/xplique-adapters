"""YOLOX model wrappers for Xplique object detection."""

from typing import Any

from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
    TorchBoxesModelWrapper,
)

from xplique_adapters.object_detection.torch.yolox_formatters import YoloXBoxFormatter


class YoloXBoxesModelWrapper(TorchBoxesModelWrapper):
    """Wrap an evaluation-mode YOLOX model and format its decoded predictions."""

    def __init__(self, model: Any) -> None:
        super().__init__(model, box_formatter=YoloXBoxFormatter())
