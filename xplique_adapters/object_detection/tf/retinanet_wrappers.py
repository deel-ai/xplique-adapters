"""
RetinaNet model wrappers for object detection with box formatting.
"""

from typing import Any, Literal

import tensorflow as tf
from xplique.utils_functions.object_detection.tf.box_model_wrapper import (
    TfBoxesModelWrapper,
    _pad_and_stack_box_predictions,
)

from xplique_adapters.object_detection.tf.retinanet_formatters import (
    RetinaNetProcessedBoxFormatter,
)


_SUPPORTED_BOX_FORMATS = frozenset(
    {"xyxy", "xywh", "center_xywh", "rel_xyxy", "rel_xywh"}
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
        image_size: tuple[int, int] | None = None,
        prediction_mode: Literal["raw", "decoded"] = "raw",
    ) -> None:
        """Initialize a wrapper for raw or already-decoded RetinaNet output."""
        if prediction_mode not in {"raw", "decoded"}:
            raise ValueError(
                "prediction_mode must be either 'raw' or 'decoded', "
                f"got {prediction_mode!r}."
            )
        box_formatter = RetinaNetProcessedBoxFormatter(
            nb_classes=nb_classes, image_size=image_size
        )
        super().__init__(model, box_formatter=box_formatter)
        self.prediction_mode = prediction_mode

    def _box_format(self) -> str:
        box_format = getattr(self.model, "bounding_box_format", None)
        if box_format is None:
            decoder = getattr(self.model, "prediction_decoder", None)
            box_format = getattr(decoder, "bounding_box_format", None)
        if box_format is None:
            # Decoded callables without KerasCV metadata use the formatter contract.
            box_format = "xywh"
        box_format = str(box_format).lower()
        if box_format not in _SUPPORTED_BOX_FORMATS:
            raise ValueError(
                "Unsupported RetinaNet bounding_box_format "
                f"{box_format!r}; supported formats are "
                f"{sorted(_SUPPORTED_BOX_FORMATS)}."
            )
        return box_format

    def _canonicalize_predictions(self, predictions, x):
        box_format = self._box_format()
        if box_format == "xywh":
            return predictions

        try:
            from keras_cv.src import bounding_box
        except ImportError as error:  # pragma: no cover - only reached for bad installs
            raise ImportError(
                "KerasCV is required to convert RetinaNet bounding boxes."
            ) from error

        image_shape = tf.shape(x)[1:4]
        canonical = dict(predictions)
        canonical["boxes"] = bounding_box.convert_format(
            predictions["boxes"],
            source=box_format,
            target="xywh",
            image_shape=image_shape,
        )
        return canonical

    def call(self, x, training=None, **kwargs):
        """Call, decode when requested, and format the wrapped RetinaNet once."""
        if self.prediction_mode == "raw":
            call_kwargs = dict(kwargs)
            call_kwargs["training"] = False
            raw_predictions = self.model(x, **call_kwargs)
            decode_predictions = getattr(self.model, "decode_predictions", None)
            if not callable(decode_predictions):
                raise AttributeError(
                    "RetinaNetBoxesModelWrapper prediction_mode='raw' requires "
                    "the wrapped model to provide decode_predictions()."
                )
            predictions = decode_predictions(raw_predictions, x)
        else:
            call_kwargs = dict(kwargs)
            if training is not None:
                call_kwargs["training"] = training
            predictions = self.model(x, **call_kwargs)

        predictions = self._canonicalize_predictions(predictions, x)
        list_of_predictions = self.box_formatter(predictions)
        if self.output_as_list:
            return list_of_predictions
        return _pad_and_stack_box_predictions(list_of_predictions)
