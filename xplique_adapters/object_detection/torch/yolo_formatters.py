"""
YOLO box formatters for converting Ultralytics YOLO model outputs to Xplique format.

This module provides formatters for YOLO models that use the ultralytics package.
"""

import torch
import torch.nn.functional as F

try:
    import ultralytics
except ImportError:
    ultralytics = None

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


class YoloResultBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for YOLO (Ultralytics) Results objects.

    Formats predictions from Ultralytics YOLO models that return Results objects
    containing boxes in XYXY format with absolute pixel coordinates, confidence
    scores, and class predictions.
    """

    def __init__(self) -> None:
        """
        Initialize YOLO result formatter with XYXY pixel coordinate input format.
        """
        super().__init__(
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )

    def forward(self, predictions: list) -> list[TorchMultiBoxTensor]:
        """
        Format YOLO Results objects.

        Parameters
        ----------
        predictions
            List of ultralytics.engine.results.Results objects from YOLO model.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        if ultralytics is None:
            raise ImportError(
                "ultralytics package is required for YoloResultBoxFormatter. "
                "Install it with: pip install ultralytics"
            )

        assert isinstance(predictions, list)
        assert isinstance(predictions[0], ultralytics.engine.results.Results)

        formatted_preds = []
        for result in predictions:
            device = result.boxes.cls.device
            num_classes = len(result.names)
            classes_id = result.boxes.cls.long()
            labels_one_hot = F.one_hot(classes_id, num_classes).to(device)
            pred_dict = {
                "boxes": result.boxes.xyxy,
                "scores": result.boxes.conf.unsqueeze(dim=1),
                "probas": labels_one_hot,
            }
            formatted = self.format_predictions(
                pred_dict
            )  # needs boxes, scores, probas
            formatted_preds.append(formatted)
        return formatted_preds


class YoloOneToManyFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for raw YOLO model outputs (tensor format).

    Processes raw YOLO predictions in CXCYWH normalized format before they are
    converted to Results objects. Handles tuple output with detection tensors.
    """

    def __init__(self) -> None:
        """
        Initialize raw YOLO formatter with CXCYWH normalized input format.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True))

    def forward(self, predictions: torch.Tensor) -> list[TorchMultiBoxTensor]:
        """
        Format raw YOLO tensor predictions.

        Parameters
        ----------
        predictions
            Tuple containing (detection_tensor, auxiliary_data) where
            detection_tensor has shape (batch, features, num_boxes) with features
            encoding boxes (4 values) and class probabilities.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        # check the raw structure of the YOLO predictions
        if (
            isinstance(predictions, list)
            and len(predictions) > 0
            and isinstance(predictions[0], ultralytics.engine.results.Results)
        ):
            raise ValueError(
                "YoloOneToManyFormatter received a Results object. "
                "Use YoloResultBoxFormatter for Results objects."
            )
        assert isinstance(predictions, tuple), (
            f"predictions should be a tuple, got {type(predictions)}"
        )
        assert len(predictions) == 2
        assert isinstance(predictions[0], torch.Tensor), (
            f"predictions[0] should be a torch.Tensor, got {type(predictions[0])}"
        )
        is_old_yolo_format = isinstance(predictions[1], list)
        is_new_yolo_format = isinstance(predictions[1], dict)
        assert is_old_yolo_format or is_new_yolo_format, (
            f"predictions[1] should be a list (old format) or dict (new format),"
            f"got {type(predictions[1])}"
        )
        n = len(predictions[1])
        if n != 3:
            hint = (
                f" This looks like a YOLO26/end2end model output (predictions[1] is a dict"
                f" with {n} keys). Use YoloOneToOneFormatter instead of YoloOneToManyFormatter."
                if isinstance(predictions[1], dict)
                else ""
            )
            raise ValueError(
                f"YoloOneToManyFormatter expects predictions[1] to have 3 elements "
                f"(one feature map per detection scale), got {n}.{hint}"
            )
        nb_preds = len(predictions[0])

        formatted_preds = []
        for i in range(nb_preds):
            pred = predictions[0][i]
            boxes = pred.squeeze().permute(1, 0)[:, :4]
            probas = pred.squeeze().permute(1, 0)[:, 4:]  # sigmoid result
            scores = probas.max(-1).values.unsqueeze(1)
            pred_dict = {"boxes": boxes, "scores": scores, "probas": probas}
            formatted = self.format_predictions(
                pred_dict
            )  # needs boxes, scores, probas
            formatted_preds.append(formatted)
        return formatted_preds


class YoloOneToOneFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for raw YOLO model outputs (tensor format).

    Processes raw YOLO predictions in XYXY normalized format before they are
    converted to Results objects. Handles tuple output with detection tensors.
    """

    def __init__(self, nb_classes: int) -> None:
        """
        Initialize raw YOLO formatter with XYXY normalized input format.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True))
        self.nb_classes = nb_classes

    def forward(self, predictions: torch.Tensor) -> list[TorchMultiBoxTensor]:
        """
        Format raw YOLO tensor predictions.

        Parameters
        ----------
        predictions
            Tuple containing (detection_tensor, auxiliary_data) where
            detection_tensor has shape (batch, features, num_boxes) with features
            encoding boxes (4 values) and class probabilities.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        # check the raw structure of the YOLO predictions
        if (
            isinstance(predictions, list)
            and len(predictions) > 0
            and isinstance(predictions[0], ultralytics.engine.results.Results)
        ):
            raise ValueError(
                "YoloOneToOneFormatter received a Results object. "
                "Use YoloResultBoxFormatter for Results objects."
            )
        assert isinstance(predictions, tuple), (
            f"predictions should be a tuple, got {type(predictions)}"
        )
        assert len(predictions) == 2
        assert isinstance(predictions[0], torch.Tensor), (
            f"predictions[0] should be a torch.Tensor, got {type(predictions[0])}"
        )
        is_old_yolo_format = isinstance(predictions[1], list)
        is_new_yolo_format = isinstance(predictions[1], dict)
        assert is_old_yolo_format or is_new_yolo_format, (
            f"predictions[1] should be a list (old format) or dict (new format),"
            f"got {type(predictions[1])}"
        )
        # assert len(predictions[1]) == 3
        nb_preds = len(predictions[0])

        formatted_preds = []
        for i in range(nb_preds):
            pred = predictions[0][i]
            boxes = pred.squeeze()[:, :4]
            scores = pred.squeeze()[:, 4].unsqueeze(1)
            class_id = pred.squeeze()[:, 5]
            # transform the class_id into a one-hot encoding
            probas = torch.zeros((scores.shape[0], self.nb_classes), device=pred.device)
            probas[torch.arange(scores.shape[0]), class_id.long()] = 1

            pred_dict = {"boxes": boxes, "scores": scores, "probas": probas}
            formatted = self.format_predictions(
                pred_dict
            )  # needs boxes, scores, probas
            formatted_preds.append(formatted)
        return formatted_preds


# Backward-compatible aliases (deprecated names).
Yolo11RawBoxFormatter = YoloOneToManyFormatter
Yolo26RawBoxFormatter = YoloOneToOneFormatter
