"""
Torchvision model wrappers for Xplique object detection.
"""

from typing import Any

from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
    TorchBoxesModelWrapper,
)

from xplique_adapters.object_detection.torch.torchvision_formatters import (
    TorchvisionBoxFormatter,
)


class TorchvisionBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for Torchvision object detection models.

    This class extends TorchBoxesModelWrapper with Torchvision-specific box formatting,
    automatically using TorchvisionBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any, nb_classes: int) -> None:
        """
        Initialize the Torchvision box model wrapper.

        Parameters
        ----------
        model
            Torchvision object detection model (e.g., FCOS, RetinaNet).
        nb_classes
            Total number of object classes in the detection model.
        """
        box_formatter = TorchvisionBoxFormatter(nb_classes=nb_classes)
        super().__init__(model, box_formatter=box_formatter)
