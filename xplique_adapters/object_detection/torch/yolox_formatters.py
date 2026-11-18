"""Format native YOLOX predictions as Xplique object-detection outputs."""

import torch
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


class YoloXBoxFormatter(TorchBaseBoxFormatter):
    """Format decoded YOLOX output ``(B, anchors, 5 + classes)``.

    YOLOX outputs absolute CXCYWH coordinates, sigmoid objectness, and sigmoid
    class probabilities. Column 4 is the joint detection confidence
    (objectness times the highest class probability) used for filtering and
    attribution. The class probability vector is kept unchanged.
    """

    def __init__(self) -> None:
        super().__init__(
            input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=False),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )

    def forward(self, predictions: torch.Tensor) -> list[TorchMultiBoxTensor]:
        """Convert each batch item to absolute XYXY boxes and class scores."""
        if not isinstance(predictions, torch.Tensor):
            raise TypeError(
                f"YOLOX predictions must be a torch.Tensor, got {type(predictions)}."
            )
        if predictions.ndim != 3 or predictions.shape[-1] < 6:
            raise ValueError(
                "YOLOX predictions must have shape (batch, anchors, 5 + classes), "
                f"got {tuple(predictions.shape)}."
            )

        results = []
        for prediction in predictions:
            centers = prediction[:, :2]
            half_sizes = prediction[:, 2:4] / 2
            boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=-1)
            probas = prediction[:, 5:]
            scores = prediction[:, 4:5] * probas.max(dim=-1, keepdim=True).values
            results.append(
                TorchMultiBoxTensor(torch.cat((boxes, scores, probas), dim=-1))
            )
        return results
