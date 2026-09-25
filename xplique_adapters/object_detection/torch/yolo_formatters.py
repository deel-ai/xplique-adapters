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
    Box formatter for YOLO one-to-many inference tuples.

    Converts decoded CXCYWH boxes in input-image pixels to XYXY pixels from
    inference tuples rather than Ultralytics Results objects.
    """

    def __init__(self) -> None:
        """
        Initialize YOLO formatter with absolute CXCYWH input and XYXY output.
        """
        super().__init__(
            input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=False),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )

    def forward(self, predictions: torch.Tensor) -> list[TorchMultiBoxTensor]:
        """
        Format raw YOLO tensor predictions.

        Parameters
        ----------
        predictions
            Tuple containing (detection_tensor, auxiliary_data) where
            detection_tensor has shape (batch, 4 + classes, num_boxes) (or
            batch, 1, 4 + classes, num_boxes). The first four values are
            decoded CXCYWH coordinates in input-image pixels, followed by
            class probabilities. The auxiliary dict has exactly the keys
            ``{"boxes", "scores", "feats"}``.

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
        aux = predictions[1]
        if not isinstance(aux, dict):
            raise TypeError(
                "predictions[1] should be an auxiliary dict with keys "
                f"{{'boxes', 'scores', 'feats'}}, got {type(aux)}"
            )
        keys = set(aux)
        if keys == {"one2many", "one2one"}:
            raise ValueError(
                "YoloOneToManyFormatter received an end-to-end auxiliary "
                "dictionary with keys {'one2many', 'one2one'}. "
                "Use YoloOneToOneFormatter instead."
            )
        if keys != {"boxes", "scores", "feats"}:
            raise ValueError(
                "YoloOneToManyFormatter expects predictions[1] to be an auxiliary "
                "dictionary with exactly the keys {'boxes', 'scores', 'feats'}; "
                f"got {sorted(keys)}."
            )
        nb_preds = len(predictions[0])

        formatted_preds = []
        for i in range(nb_preds):
            pred = predictions[0][i]
            if pred.ndim == 3:
                pred = pred.squeeze(0)  # optional singleton axis, not num_boxes
            pred = pred.permute(1, 0)
            centers = pred[:, :2]
            half_sizes = pred[:, 2:4] / 2
            boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)
            probas = pred[:, 4:]  # sigmoid result
            scores = probas.max(-1).values.unsqueeze(1)
            # Xplique's absolute-coordinate translator requires image_size even
            # for this scale-independent CXCYWH-to-XYXY conversion.
            formatted_preds.append(
                TorchMultiBoxTensor(torch.cat((boxes, scores, probas), dim=1))
            )
        return formatted_preds


class YoloOneToOneFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for YOLO end-to-end inference tuples.

    Processes top-k detections with XYXY boxes in input-image pixels from
    inference tuples rather than Ultralytics Results objects.
    """

    def __init__(self, nb_classes: int) -> None:
        """
        Initialize YOLO formatter with absolute XYXY input and output.
        """
        super().__init__(
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )
        self.nb_classes = nb_classes

    def forward(self, predictions: torch.Tensor) -> list[TorchMultiBoxTensor]:
        """
        Format raw YOLO tensor predictions.

        Parameters
        ----------
        predictions
            Tuple containing (detection_tensor, auxiliary_data) where
            detection_tensor has shape (batch, num_boxes, 6), with absolute
            XYXY coordinates, score, and class ID per detection. The auxiliary
            dict has exactly the keys ``{"one2many", "one2one"}``.

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
        aux = predictions[1]
        if not isinstance(aux, dict):
            raise TypeError(
                "predictions[1] should be an auxiliary dict with keys "
                f"{{'one2many', 'one2one'}}, got {type(aux)}"
            )
        if set(aux) != {"one2many", "one2one"}:
            raise ValueError(
                "YoloOneToOneFormatter expects predictions[1] to be an auxiliary "
                "dictionary with exactly the keys {'one2many', 'one2one'}; "
                f"got {sorted(aux)}."
            )
        nb_preds = len(predictions[0])

        formatted_preds = []
        for i in range(nb_preds):
            pred = predictions[0][i]
            boxes = pred[:, :4]
            scores = pred[:, 4].unsqueeze(1)
            class_id = pred[:, 5]
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
