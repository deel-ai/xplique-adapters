"""
PyTorch object detection adapters for Xplique.

This module provides PyTorch-specific formatters and wrappers for object detection models.
"""

from xplique_adapters.object_detection.torch.detr_formatters import (
    DetrBoxFormatter,
)
from xplique_adapters.object_detection.torch.detr_wrappers import (
    DetrBoxesModelWrapper,
)
from xplique_adapters.object_detection.torch.torchvision_formatters import (
    TorchvisionBoxFormatter,
)
from xplique_adapters.object_detection.torch.torchvision_wrappers import (
    TorchvisionBoxesModelWrapper,
)
from xplique_adapters.object_detection.torch.yolo_formatters import (
    Yolo11RawBoxFormatter,
    Yolo26RawBoxFormatter,
    YoloOneToManyFormatter,
    YoloOneToOneFormatter,
    YoloResultBoxFormatter,
)
from xplique_adapters.object_detection.torch.yolo_wrappers import (
    Yolo11RawBoxesModelWrapper,
    Yolo26RawBoxesModelWrapper,
    YoloResultBoxesModelWrapper,
)

__all__ = [
    "DetrBoxFormatter",
    "DetrBoxesModelWrapper",
    "TorchvisionBoxFormatter",
    "TorchvisionBoxesModelWrapper",
    "Yolo11RawBoxFormatter",
    "Yolo11RawBoxesModelWrapper",
    "Yolo26RawBoxFormatter",
    "Yolo26RawBoxesModelWrapper",
    "YoloOneToManyFormatter",
    "YoloOneToOneFormatter",
    "YoloResultBoxFormatter",
    "YoloResultBoxesModelWrapper",
]
