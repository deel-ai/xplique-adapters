"""
RetinaNet box formatters for converting RetinaNet model outputs to Xplique format.
"""

import tensorflow as tf
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.box_formatter import TfBaseBoxFormatter
from xplique.utils_functions.object_detection.tf.multi_box_tensor import (
    TfMultiBoxTensor,
)


class RetinaNetProcessedBoxFormatter(TfBaseBoxFormatter):
    """
    Box formatter for RetinaNet detection model predictions.

    Handles RetinaNet-specific output format with boxes, confidence scores,
    and class predictions, converting them to unified Xplique format.
    """

    def __init__(
        self,
        nb_classes: int,
        input_box_type: BoxType = BoxType(BoxFormat.XYWH, is_normalized=False),  # noqa: B008
        output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),  # noqa: B008
        image_size: tuple = None,  # noqa: RUF013
    ) -> None:
        """
        Initialize the RetinaNet box formatter.

        Parameters
        ----------
        nb_classes
            Number of object classes in the dataset.
        input_box_type
            Format of boxes from RetinaNet (default XYWH, unnormalized).
        output_box_type
            Desired output format (default XYXY, normalized).
        image_size
            Size of image for coordinate conversion.
        """
        super().__init__(input_box_type, output_box_type)

        self.nb_classes = nb_classes
        self.image_size = image_size

    def forward(self, predictions) -> list[TfMultiBoxTensor]:
        """
        Process RetinaNet predictions handling both single and multi-batch inputs.

        Converts class IDs to one-hot encoding and formats boxes with scores
        and probabilities.

        Parameters
        ----------
        predictions
            Dictionary with 'boxes', 'confidence', and 'classes' keys.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """

        def process_single_batch(batch_idx):
            boxes = predictions["boxes"][batch_idx]
            scores = predictions["confidence"][batch_idx]
            classes = predictions["classes"][batch_idx]

            labels_one_hot = tf.one_hot(
                tf.cast(classes, tf.int32), depth=self.nb_classes
            )
            scores_expanded = scores[:, tf.newaxis]

            pred_dict = {
                "boxes": boxes,
                "scores": scores_expanded,
                "probas": labels_one_hot,
            }
            return self.format_predictions(pred_dict, self.image_size)

        results = []
        for batch_idx in range(predictions["boxes"].shape[0]):
            formatted = process_single_batch(batch_idx)
            results.append(formatted)
        return results
