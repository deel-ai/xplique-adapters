"""
DETR box formatters for converting DETR model outputs to Xplique format.
"""

from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxType,
)
from xplique.utils_functions.object_detection.torch.box_formatter import (
    TorchBaseBoxFormatter,
)
from xplique.utils_functions.object_detection.torch.multi_box_tensor import (
    TorchMultiBoxTensor,
)


class DetrBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for DETR (DEtection TRansformer) object detection models.

    DETR models output predictions in CXCYWH (center-x, center-y, width, height)
    normalized format with separate logits and box coordinates.
    """

    def __init__(self, image_size: tuple) -> None:
        """
        Initialize DETR box formatter with CXCYWH normalized input format.
        """
        super().__init__(
            input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )
        self.image_size = image_size

    def forward(self, predictions) -> list[TorchMultiBoxTensor]:
        """
        Format DETR model predictions.

        Parameters
        ----------
        predictions
            Dictionary with 'pred_logits' and 'pred_boxes' keys containing
            batched predictions from DETR model.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        results = []
        for logits, boxes in zip(predictions["pred_logits"], predictions["pred_boxes"]):
            probas = logits.softmax(-1)[:, :-1]
            scores = probas.max(-1).values.unsqueeze(1)
            pred_dict = {
                "logits": logits,
                "boxes": boxes,
                "scores": scores,
                "probas": probas,
            }
            formatted = self.format_predictions(pred_dict, image_size=self.image_size)
            results.append(formatted)
        return results
