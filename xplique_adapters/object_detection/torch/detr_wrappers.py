"""
DETR model wrappers for Xplique object detection.
"""

from typing import Any

from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
    TorchBoxesModelWrapper,
)

from xplique_adapters.object_detection.torch.detr_formatters import (
    DetrBoxFormatter,
)


class DetrBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for DETR object detection models.

    This class extends TorchBoxesModelWrapper with DETR-specific box formatting,
    automatically using DetrBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any, image_size: tuple) -> None:
        """
        Initialize the DETR box model wrapper.

        Parameters
        ----------
        model
            DETR object detection model.
        image_size
            Size of the input images (height, width).
        """
        box_formatter = DetrBoxFormatter(image_size=image_size)
        super().__init__(model, box_formatter=box_formatter)
