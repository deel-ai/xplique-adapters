"""Convert Torchvision detection outputs to Xplique object-detection format.

Expected input is a list of per-image dictionaries containing ``boxes``
``(num_detections, 4)`` in absolute XYXY format, ``scores``
``(num_detections,)``, and integer ``labels`` ``(num_detections,)``.
Boxes remain in absolute XYXY format; labels are converted to one-hot class
probabilities. The formatter returns one formatted prediction per image.
"""

import torch.nn.functional as F
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


class TorchvisionBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for Torchvision object detection models (FCOS, RetinaNet, SSD, etc.).

    Torchvision models output predictions in XYXY format with absolute pixel coordinates
    and discrete class labels that need to be converted to one-hot encoded probabilities.
    """

    def __init__(self, nb_classes: int) -> None:
        """
        Initialize Torchvision box formatter.

        Parameters
        ----------
        nb_classes
            Total number of object classes in the detection model.
        """
        super().__init__(
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )
        self.nb_classes = nb_classes

    def forward(self, predictions) -> list[TorchMultiBoxTensor]:
        """
        Format Torchvision model predictions.

        Parameters
        ----------
        predictions
            List of dictionaries, one per image, each containing 'boxes',
            'scores', and 'labels' keys.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        results = []
        for prediction in predictions:
            prediction["scores"] = prediction["scores"].unsqueeze(dim=1)
            labels_one_hot = F.one_hot(
                prediction["labels"], num_classes=self.nb_classes
            ).to(prediction["scores"].device)
            prediction["probas"] = labels_one_hot
            formatted = self.format_predictions(prediction)
            results.append(formatted)
        return results
