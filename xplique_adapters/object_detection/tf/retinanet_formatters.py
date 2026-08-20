"""Format decoded KerasCV RetinaNet predictions for Xplique."""

from collections.abc import Mapping

import tensorflow as tf
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.box_formatter import TfBaseBoxFormatter
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
    height)`` internally. When ``num_detections`` is present, rows after the
    valid count are treated as padding and are not returned. Rows with class
    ``-1`` are likewise padding and are never passed to ``tf.one_hot``.
    """

    def __init__(
        self,
        nb_classes: int,
        image_size: tuple[int, int] | None = None,
        *,
        _input_box_type: BoxType | None = None,
        _output_box_type: BoxType | None = None,
    ) -> None:
        """Initialize the formatter for decoded absolute ``xywh`` boxes.

        Parameters
        ----------
        nb_classes
            Number of foreground classes represented by the model.
        image_size
            Input image dimensions as ``(height, width)``. Required when
            converting the formatter's absolute input boxes to normalized
            output boxes.

        Raises
        ------
        ValueError
            If ``nb_classes`` is not positive or ``image_size`` is not a pair.
        """
        if nb_classes < 1:
            raise ValueError(f"nb_classes must be positive, got {nb_classes}.")
        if image_size is not None and len(image_size) != 2:
            raise ValueError(
                f"image_size must be a (height, width) pair, got {image_size!r}."
            )

        input_box_type = _input_box_type or BoxType(BoxFormat.XYWH, is_normalized=False)
        output_box_type = _output_box_type or BoxType(
            BoxFormat.XYXY, is_normalized=True
        )
        super().__init__(input_box_type, output_box_type)

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
        """Create a formatter with an explicit coordinate contract.

        This factory is used by the latent decoder, whose output is already
        normalized ``xyxy``. Keeping the coordinate types in the constructor
        avoids mutating the translator after the base formatter is initialized.

        Parameters
        ----------
        nb_classes
            Number of foreground classes represented by the model.
        input_box_type
            Format and normalization of the decoded input boxes.
        output_box_type
            Format and normalization expected by Xplique.
        image_size
            Input image dimensions as ``(height, width)`` when conversion
            between absolute and normalized coordinates is required.

        Returns
        -------
        formatter
            A configured RetinaNet formatter.
        """
        return cls(
            nb_classes=nb_classes,
            image_size=image_size,
            _input_box_type=input_box_type,
            _output_box_type=output_box_type,
        )

    def _xplique_image_size(self) -> tuple[int, int] | None:
        """Return the public image size in Xplique's ``(width, height)`` order."""
        if self.image_size is None:
            return None
        height, width = self.image_size
        return width, height

    def _validate_image_size(self) -> None:
        """Validate dimensions before translating absolute coordinates."""
        if self.image_size is None and (
            not self.input_box_type.is_normalized
            or not self.output_box_type.is_normalized
        ):
            raise ValueError(
                "image_size is required to convert between absolute and "
                "normalized RetinaNet boxes."
            )

    @staticmethod
    def _validate_prediction_shapes(
        boxes: tf.Tensor,
        confidence: tf.Tensor,
        classes: tf.Tensor,
        class_scores: tf.Tensor | None,
        num_detections: tf.Tensor | None,
        nb_classes: int,
    ) -> None:
        """Validate ranks and statically known dimensions of prediction fields."""

        expected = {
            "boxes": (boxes, 3, 4),
            "confidence": (confidence, 2, None),
            "classes": (classes, 2, None),
        }
        if class_scores is not None:
            expected["scores"] = (class_scores, 3, nb_classes)
        if num_detections is not None:
            expected["num_detections"] = (num_detections, 1, None)

        for name, (tensor, rank, final_dim) in expected.items():
            if tensor.shape.rank != rank or (
                final_dim is not None
                and tensor.shape[-1] is not None
                and tensor.shape[-1] != final_dim
            ):
                shape = (
                    f"(batch, detections, {final_dim})"
                    if rank == 3 and final_dim is not None
                    else f"rank {rank}"
                )
                if name == "scores":
                    shape = f"per-class probabilities with shape {shape}"
                raise ValueError(
                    f"RetinaNet '{name}' must have {shape}; got {tensor.shape}."
                )

        for name, (tensor, _, _) in expected.items():
            if name == "boxes" or name == "num_detections":
                continue
            for axis, axis_name in ((0, "batch"), (1, "detection")):
                if (
                    boxes.shape[axis] is not None
                    and tensor.shape[axis] is not None
                    and boxes.shape[axis] != tensor.shape[axis]
                ):
                    raise ValueError(
                        f"RetinaNet '{name}' must have the same {axis_name} "
                        "dimension as 'boxes'."
                    )

        if (
            num_detections is not None
            and boxes.shape[0] is not None
            and num_detections.shape[0] is not None
            and boxes.shape[0] != num_detections.shape[0]
        ):
            raise ValueError(
                "RetinaNet 'num_detections' must have one value per image."
            )

    def forward(self, predictions) -> list[TfMultiBoxTensor]:
        """Format one decoded result per image in a batched prediction mapping.

        Parameters
        ----------
        predictions
            Mapping containing ``boxes`` with shape ``(B, N, 4)``,
            ``confidence`` and ``classes`` with shape ``(B, N)``, and optional
            ``scores`` with shape ``(B, N, nb_classes)``. ``num_detections``
            may provide one valid row count per image.

        Returns
        -------
        formatted_predictions
            List of :class:`TfMultiBoxTensor` objects, one per image. Each
            tensor has width ``nb_classes + 5``.

        Raises
        ------
        ValueError
            If required fields are missing, fields have incompatible shapes,
            class IDs are invalid, or image dimensions are required but absent.
        """
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

        class_scores = predictions.get("scores")
        if class_scores is not None:
            class_scores = tf.convert_to_tensor(class_scores)

        num_detections = predictions.get("num_detections")
        if num_detections is not None:
            num_detections = tf.convert_to_tensor(num_detections)

        self._validate_prediction_shapes(
            boxes,
            confidence,
            classes,
            class_scores,
            num_detections,
            self.nb_classes,
        )
        self._validate_image_size()

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
                if tf.executing_eagerly():
                    count_value = int(count.numpy())
                    if count_value < 0 or (
                        item_boxes.shape[0] is not None
                        and count_value > item_boxes.shape[0]
                    ):
                        raise ValueError(
                            "RetinaNet 'num_detections' must be between zero and "
                            f"the number of boxes ({item_boxes.shape[0]}), got "
                            f"{count_value}."
                        )
                indices = tf.range(count)
                item_boxes = tf.gather(item_boxes, indices)
                item_confidence = tf.gather(item_confidence, indices)
                item_classes = tf.gather(item_classes, indices)
                if item_scores is not None:
                    item_scores = tf.gather(item_scores, indices)

            item_classes_int = tf.cast(item_classes, tf.int32)
            if tf.executing_eagerly() and bool(tf.reduce_any(item_classes_int < -1)):
                raise ValueError(
                    "RetinaNet 'classes' contains an ID below the padding value "
                    f"-1 or above the valid class range [0, {self.nb_classes})."
                )
            if tf.executing_eagerly() and bool(
                tf.reduce_any(item_classes_int >= self.nb_classes)
            ):
                raise ValueError(
                    "RetinaNet 'classes' contains an ID outside the valid class "
                    f"range [0, {self.nb_classes})."
                )

            # KerasCV uses -1 for padded class IDs. Filter those rows before
            # one-hot encoding while retaining every valid row when no count is given.
            valid_rows = item_classes_int >= 0
            item_boxes = tf.boolean_mask(item_boxes, valid_rows)
            item_confidence = tf.boolean_mask(item_confidence, valid_rows)
            item_classes_int = tf.boolean_mask(item_classes_int, valid_rows)
            if item_scores is not None:
                item_scores = tf.boolean_mask(item_scores, valid_rows)

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
