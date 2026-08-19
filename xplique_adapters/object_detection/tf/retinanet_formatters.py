"""Format decoded KerasCV RetinaNet predictions for Xplique."""

from collections.abc import Mapping

import tensorflow as tf
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.box_formatter import TfBaseBoxFormatter
from xplique.utils_functions.object_detection.tf.box_manager import (
    TfBoxCoordinatesTranslator,
)
from xplique.utils_functions.object_detection.tf.multi_box_tensor import (
    TfMultiBoxTensor,
)


class RetinaNetProcessedBoxFormatter(TfBaseBoxFormatter):
    """Format a batch of decoded KerasCV RetinaNet predictions.

    ``predictions`` must be a mapping whose values have a batch dimension:
    ``boxes`` has shape ``(B, N, 4)`` in absolute ``xywh`` format,
    ``confidence`` and ``classes`` have shape ``(B, N)``, and ``num_detections``
    may provide the valid count for each image. KerasCV's optional ``scores``
    field is accepted as per-class probabilities with shape ``(B, N, C)``.

    The formatter returns one :class:`TfMultiBoxTensor` per image. The public
    image-size convention is ``(height, width)``; Xplique receives ``(width,
    height)`` internally.
    """

    def __init__(
        self,
        nb_classes: int,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the formatter for absolute ``xywh`` decoded boxes."""
        if nb_classes < 1:
            raise ValueError(f"nb_classes must be positive, got {nb_classes}.")
        if image_size is not None and len(image_size) != 2:
            raise ValueError(
                f"image_size must be a (height, width) pair, got {image_size!r}."
            )

        super().__init__(
            BoxType(BoxFormat.XYWH, is_normalized=False),
            BoxType(BoxFormat.XYXY, is_normalized=True),
        )

        self.nb_classes = nb_classes
        self.image_size = image_size

    @classmethod
    def _from_box_types(
        cls,
        nb_classes: int,
        input_box_type: BoxType,
        output_box_type: BoxType,
        image_size: tuple[int, int] | None = None,
    ) -> "RetinaNetProcessedBoxFormatter":
        """Create the internal normalized formatter used by latent decoding."""
        formatter = cls(nb_classes=nb_classes, image_size=image_size)
        formatter.input_box_type = input_box_type
        formatter.output_box_type = output_box_type
        formatter.box_translator = TfBoxCoordinatesTranslator(
            input_box_type, output_box_type
        )
        return formatter

    def _xplique_image_size(self) -> tuple[int, int] | None:
        if self.image_size is None:
            return None
        height, width = self.image_size
        return width, height

    def forward(self, predictions) -> list[TfMultiBoxTensor]:
        """Format one decoded result per image in a batched prediction mapping."""
        if not isinstance(predictions, Mapping):
            raise ValueError(
                "RetinaNet predictions must be a mapping containing "
                "'boxes', 'confidence', and 'classes'."
            )

        required = {"boxes", "confidence", "classes"}
        missing = sorted(required.difference(predictions))
        if missing:
            raise ValueError(
                "RetinaNet predictions are missing required fields "
                f"{missing}; expected 'boxes', 'confidence', and 'classes'."
            )

        boxes = tf.convert_to_tensor(predictions["boxes"])
        confidence = tf.convert_to_tensor(predictions["confidence"])
        classes = tf.convert_to_tensor(predictions["classes"])

        if boxes.shape.rank != 3 or (
            boxes.shape[-1] is not None and boxes.shape[-1] != 4
        ):
            raise ValueError(
                "RetinaNet 'boxes' must have shape (batch, detections, 4), "
                f"got {boxes.shape}."
            )
        if confidence.shape.rank != 2:
            raise ValueError(
                "RetinaNet 'confidence' must have shape (batch, detections), "
                f"got {confidence.shape}."
            )
        if classes.shape.rank != 2:
            raise ValueError(
                "RetinaNet 'classes' must have shape (batch, detections), "
                f"got {classes.shape}."
            )
        if (
            boxes.shape[0] is not None
            and confidence.shape[0] is not None
            and boxes.shape[0] != confidence.shape[0]
        ):
            raise ValueError("RetinaNet fields must have the same batch dimension.")
        if (
            boxes.shape[1] is not None
            and confidence.shape[1] is not None
            and boxes.shape[1] != confidence.shape[1]
        ):
            raise ValueError(
                "RetinaNet 'confidence' must have the same detection dimension "
                "as 'boxes'."
            )
        if (
            boxes.shape[0] is not None
            and classes.shape[0] is not None
            and boxes.shape[0] != classes.shape[0]
        ) or (
            boxes.shape[1] is not None
            and classes.shape[1] is not None
            and boxes.shape[1] != classes.shape[1]
        ):
            raise ValueError(
                "RetinaNet 'classes' must have the same batch and detection "
                "dimensions as 'boxes'."
            )

        class_scores = predictions.get("scores")
        if class_scores is not None:
            class_scores = tf.convert_to_tensor(class_scores)
            if class_scores.shape.rank != 3 or (
                class_scores.shape[-1] is not None
                and class_scores.shape[-1] != self.nb_classes
            ):
                raise ValueError(
                    "RetinaNet 'scores' must contain per-class probabilities with "
                    f"shape (batch, detections, {self.nb_classes}), "
                    f"got {class_scores.shape}."
                )

        num_detections = predictions.get("num_detections")
        if num_detections is not None:
            num_detections = tf.convert_to_tensor(num_detections)
            if num_detections.shape.rank != 1:
                raise ValueError(
                    "RetinaNet 'num_detections' must have shape (batch,), "
                    f"got {num_detections.shape}."
                )
            if (
                boxes.shape[0] is not None
                and num_detections.shape[0] is not None
                and boxes.shape[0] != num_detections.shape[0]
            ):
                raise ValueError(
                    "RetinaNet 'num_detections' must have one value per image."
                )

        batch_size = boxes.shape[0]
        if batch_size is None:
            raise ValueError(
                "RetinaNet formatter requires a statically known batch dimension "
                "to return one per-image result."
            )

        results = []
        image_size = self._xplique_image_size()
        for batch_idx in range(batch_size):
            item_boxes = boxes[batch_idx]
            item_confidence = confidence[batch_idx]
            item_classes = classes[batch_idx]
            item_scores = class_scores[batch_idx] if class_scores is not None else None

            if num_detections is not None:
                count = tf.cast(num_detections[batch_idx], tf.int32)
                indices = tf.range(count)
                item_boxes = tf.gather(item_boxes, indices)
                item_confidence = tf.gather(item_confidence, indices)
                item_classes = tf.gather(item_classes, indices)
                if item_scores is not None:
                    item_scores = tf.gather(item_scores, indices)

            item_classes_int = tf.cast(item_classes, tf.int32)
            if tf.executing_eagerly() and bool(
                tf.reduce_any(
                    (item_classes_int < 0) | (item_classes_int >= self.nb_classes)
                )
            ):
                raise ValueError(
                    "RetinaNet 'classes' contains an ID outside the valid class "
                    f"range [0, {self.nb_classes})."
                )

            if item_scores is None:
                probas = tf.one_hot(
                    item_classes_int,
                    depth=self.nb_classes,
                    dtype=item_confidence.dtype,
                )
            else:
                probas = tf.cast(item_scores, item_confidence.dtype)

            results.append(
                self.format_predictions(
                    {
                        "boxes": item_boxes,
                        "scores": item_confidence[:, tf.newaxis],
                        "probas": probas,
                    },
                    image_size=image_size,
                )
            )
        return results
