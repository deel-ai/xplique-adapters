"""
YOLO model wrappers for Ultralytics YOLO models with Xplique integration.

This module provides wrapper classes for YOLO models to work with Xplique's
object detection explanation methods.
"""

from typing import Any

from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
    TorchBoxesModelWrapper,
)

from xplique_adapters.object_detection.torch.yolo_formatters import (
    YoloOneToManyFormatter,
    YoloOneToOneFormatter,
    YoloResultBoxFormatter,
)


class YoloResultBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for YOLO models that return Results objects.

    This class extends TorchBoxesModelWrapper with YOLO-specific box formatting,
    automatically using YoloResultBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the YOLO result box model wrapper.

        Parameters
        ----------
        model
            PyTorch YOLO object detection model that returns Results objects.
        """
        box_formatter = YoloResultBoxFormatter()
        super().__init__(model, box_formatter=box_formatter)

    def train(self, mode: bool) -> "YoloResultBoxesModelWrapper":
        """
        Override train() to avoid propagating to YOLO model when mode=False.

        When mode=False (typically called from eval()), we just set the training flag
        without propagating to YOLO which would trigger dataset loading.
        """
        if not mode:
            # Just set eval mode without propagating to children
            self.training = False
            return self
        # For train mode, use normal PyTorch behavior
        return super().train(mode)


class Yolo11RawBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for YOLO models that return raw tensor outputs.

    This class extends TorchBoxesModelWrapper with YOLO-specific box formatting,
    automatically using YoloOneToManyFormatter for predictions processing.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the YOLO raw output box model wrapper.

        Parameters
        ----------
        model
            PyTorch YOLO object detection model that returns raw tensors.
        """
        box_formatter = YoloOneToManyFormatter()
        super().__init__(model, box_formatter=box_formatter)

    def train(self, mode: bool) -> "Yolo11RawBoxesModelWrapper":
        """
        Override train() to avoid propagating to YOLO model when mode=False.

        When mode=False (typically called from eval()), we just set the training flag
        without propagating to YOLO which would trigger dataset loading.
        """
        if not mode:
            # Just set eval mode without propagating to children
            self.training = False
            return self
        # For train mode, use normal PyTorch behavior
        return super().train(mode)


class Yolo26RawBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for YOLO models that return raw tensor outputs.

    This class extends TorchBoxesModelWrapper with YOLO-specific box formatting,
    automatically using YoloOneToOneFormatter for predictions processing.
    """

    def __init__(self, model: Any, nb_classes: int) -> None:
        """
        Initialize the YOLO 26 raw output box model wrapper.

        Parameters
        ----------
        model
            PyTorch YOLO object detection model that returns raw tensors.
        nb_classes
            Number of classes in the model output.
        """
        box_formatter = YoloOneToOneFormatter(nb_classes=nb_classes)
        super().__init__(model, box_formatter=box_formatter)

    def train(self, mode: bool) -> "Yolo26RawBoxesModelWrapper":
        """
        Override train() to avoid propagating to YOLO model when mode=False.

        When mode=False (typically called from eval()), we just set the training flag
        without propagating to YOLO which would trigger dataset loading.
        """
        if not mode:
            # Just set eval mode without propagating to children
            self.training = False
            return self
        # For train mode, use normal PyTorch behavior
        return super().train(mode)
