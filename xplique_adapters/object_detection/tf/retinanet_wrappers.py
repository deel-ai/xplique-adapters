"""
RetinaNet model wrappers for object detection with box formatting.
"""

from typing import Any

from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.box_model_wrapper import (
    TfBoxesModelWrapper,
)

from xplique_adapters.object_detection.tf.retinanet_formatters import (
    RetinaNetProcessedBoxFormatter,
)


class RetinaNetBoxesModelWrapper(TfBoxesModelWrapper):
    """
    Specialized wrapper for RetinaNet object detection models in TensorFlow.

    This class extends TfBoxesModelWrapper with RetinaNet-specific box formatting,
    automatically using RetinaNetProcessedBoxFormatter for predictions processing.
    """

    def __init__(
        self,
        model: Any,
        nb_classes: int,
        input_box_type: BoxType = BoxType(BoxFormat.XYWH, is_normalized=False),  # noqa: B008
        output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),  # noqa: B008
        image_size: tuple = None,  # noqa: RUF013
    ) -> None:
        """
        Initialize the RetinaNet box model wrapper.

        Parameters
        ----------
        model
            TensorFlow RetinaNet object detection model to wrap.
        nb_classes
            Number of object classes in the dataset.
        input_box_type
            Format of boxes from RetinaNet (default XYWH, unnormalized).
        output_box_type
            Desired output format (default XYXY, normalized).
        image_size
            Size of image for coordinate conversion.
        """
        box_formatter = RetinaNetProcessedBoxFormatter(
            nb_classes=nb_classes,
            input_box_type=input_box_type,
            output_box_type=output_box_type,
            image_size=image_size,
        )
        super().__init__(model, box_formatter=box_formatter)
