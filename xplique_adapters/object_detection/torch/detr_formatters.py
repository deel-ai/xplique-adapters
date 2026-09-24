"""
DETR box formatters for converting DETR model outputs to Xplique format.
"""

from collections.abc import Mapping

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

_MISSING = object()


def _get_prediction_field(predictions, field: str):
    if isinstance(predictions, Mapping) and field in predictions:
        return predictions[field]
    return getattr(predictions, field, _MISSING)


def _extract_logits_boxes(predictions) -> tuple:
    """Extract logits and boxes from Facebook or HuggingFace DETR outputs."""
    for logits_field in ("pred_logits", "logits"):
        logits = _get_prediction_field(predictions, logits_field)
        boxes = _get_prediction_field(predictions, "pred_boxes")
        if logits is not _MISSING and boxes is not _MISSING:
            if logits is None or boxes is None:
                break
            return logits, boxes

    accepted = (
        "pred_logits and pred_boxes (Facebook DETR), or "
        "logits and pred_boxes (HuggingFace DETR)"
    )
    raise ValueError(f"DETR predictions must contain {accepted}.")


class DetrBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for DETR (DEtection TRansformer) object detection models.

    DETR models output predictions in CXCYWH (center-x, center-y, width, height)
    normalized format with separate logits and box coordinates.
    """

    def __init__(self, image_size: tuple) -> None:
        """
        Initialize DETR box formatter with CXCYWH normalized input format.

        Parameters
        ----------
        image_size
            Input image size as ``(height, width)``. Xplique's translator
            receives the reversed ``(width, height)`` form internally.
        """
        super().__init__(
            input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )
        self.image_size = tuple(image_size)

    def forward(self, predictions) -> list[TorchMultiBoxTensor]:
        """
        Format DETR model predictions.

        Parameters
        ----------
        predictions
            Facebook-style mappings with ``pred_logits`` and ``pred_boxes``,
            HuggingFace-style mappings with ``logits`` and ``pred_boxes``, or
            an output object exposing either accepted logits field as an
            attribute.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        logits_batch, boxes_batch = _extract_logits_boxes(predictions)
        if logits_batch.shape[-1] < 2:
            raise ValueError(
                "DETR logits must contain at least one foreground class and "
                "one final no-object class."
            )

        results = []
        for logits, boxes in zip(logits_batch, boxes_batch):
            probas = logits.softmax(-1)[:, :-1]
            scores = probas.max(-1).values.unsqueeze(1)
            pred_dict = {
                "logits": logits,
                "boxes": boxes,
                "scores": scores,
                "probas": probas,
            }
            formatted = self.format_predictions(
                pred_dict, image_size=(self.image_size[1], self.image_size[0])
            )
            results.append(formatted)
        return results
