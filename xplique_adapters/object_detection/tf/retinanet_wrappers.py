"""
RetinaNet model wrappers for object detection with box formatting.
"""

from collections.abc import Mapping
from typing import Any, Literal

import tensorflow as tf
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
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
_BOX_TYPE_FORMATS = {
    BoxFormat.XYXY: {False: "xyxy", True: "rel_xyxy"},
    BoxFormat.XYWH: {False: "xywh", True: "rel_xywh"},
    BoxFormat.CXCYWH: {False: "center_xywh"},
}


class RetinaNetBoxesModelWrapper(TfBoxesModelWrapper):
    """Wrap a KerasCV RetinaNet and format its decoded detections.

    In ``"raw"`` mode, the wrapped model is called once with
    ``training=False`` and its ``decode_predictions()`` method is called before
    formatting. In ``"decoded"`` mode, the wrapped callable is expected to
    return the decoded prediction mapping directly. The wrapper preserves
    :class:`TfBoxesModelWrapper`'s list-versus-padded-tensor behavior.

    Decoded box formats are resolved from ``model.bounding_box_format``, then
    ``model.prediction_decoder.bounding_box_format``, and finally the explicit
    ``input_box_type`` argument. Supported formats are ``xyxy``, ``xywh``,
    ``center_xywh``, ``rel_xyxy``, and ``rel_xywh``. Axis orders such as
    ``yxyx`` are rejected rather than guessed.
    """

    def __init__(
        self,
        model: Any,
        nb_classes: int,
        image_size: tuple[int, int] | None = None,
        prediction_mode: Literal["raw", "decoded"] = "raw",
        input_box_type: BoxType | str | None = None,
    ) -> None:
        """Initialize a wrapper for raw or already-decoded RetinaNet output.

        Parameters
        ----------
        model
            KerasCV RetinaNet model, or a callable returning decoded
            predictions when ``prediction_mode="decoded"``.
        nb_classes
            Number of foreground classes represented by the model.
        image_size
            Input image dimensions as ``(height, width)``. Required when the
            decoded boxes are absolute coordinates and must be normalized.
        prediction_mode
            ``"raw"`` to invoke ``decode_predictions`` after the model call,
            or ``"decoded"`` when the callable already returns decoded fields.
        input_box_type
            Explicit fallback format for decoded boxes when neither the model
            nor its prediction decoder exposes ``bounding_box_format``. A
            :class:`BoxType` preserves the absolute/normalized distinction;
            a KerasCV format string is also accepted.

        Raises
        ------
        ValueError
            If ``prediction_mode`` is not ``"raw"`` or ``"decoded"``.
        """
        if prediction_mode not in {"raw", "decoded"}:
            raise ValueError(
                "prediction_mode must be either 'raw' or 'decoded', "
                f"got {prediction_mode!r}."
            )
        self.input_box_type = input_box_type
        box_format = self._resolve_box_format(model, input_box_type)
        formatter_input_box_type = (
            self._canonical_box_type(box_format) if box_format is not None else None
        )
        box_formatter = RetinaNetProcessedBoxFormatter(
            nb_classes=nb_classes,
            image_size=image_size,
            _input_box_type=formatter_input_box_type,
        )
        super().__init__(model, box_formatter=box_formatter)
        self.prediction_mode = prediction_mode

    @staticmethod
    def _resolve_box_format(
        model: Any, input_box_type: BoxType | str | None
    ) -> str | None:
        """Resolve a decoded box format without requiring a model instance."""
        box_format = getattr(model, "bounding_box_format", None)
        if box_format is None:
            decoder = getattr(model, "prediction_decoder", None)
            box_format = getattr(decoder, "bounding_box_format", None)
        if box_format is None:
            if input_box_type is None:
                return None
            return RetinaNetBoxesModelWrapper._format_from_box_type(input_box_type)
        if isinstance(box_format, BoxType):
            return RetinaNetBoxesModelWrapper._format_from_box_type(box_format)
        return str(box_format).lower()

    @staticmethod
    def _canonical_box_type(box_format: str) -> BoxType:
        """Return the formatter type for the canonical ``xywh`` representation."""
        return BoxType(
            BoxFormat.XYWH,
            is_normalized=box_format.startswith("rel_"),
        )

    @staticmethod
    def _format_from_box_type(input_box_type: BoxType | str) -> str:
        """Convert an explicit Xplique or KerasCV box type to a KerasCV name."""
        if isinstance(input_box_type, BoxType):
            try:
                return _BOX_TYPE_FORMATS[input_box_type.format][
                    input_box_type.is_normalized
                ]
            except KeyError as error:
                raise ValueError(
                    "Unsupported RetinaNet box type: "
                    f"{input_box_type.format!r}, normalized="
                    f"{input_box_type.is_normalized}."
                ) from error
        if isinstance(input_box_type, str):
            return input_box_type.lower()
        raise ValueError(
            "input_box_type must be a BoxType or KerasCV format string, got "
            f"{type(input_box_type).__name__}."
        )

    def _box_format(self) -> str:
        """Resolve and validate the KerasCV format of decoded boxes.

        Returns
        -------
        box_format
            Lowercase KerasCV format name.

        Raises
        ------
        ValueError
            If the model does not expose a supported box format.
        """
        box_format = self._resolve_box_format(self.model, self.input_box_type)
        if box_format is None:
            raise ValueError(
                "RetinaNet decoded predictions do not declare a supported "
                "bounding_box_format. Set input_box_type explicitly."
            )
        if box_format not in _SUPPORTED_BOX_FORMATS:
            raise ValueError(
                "Unsupported RetinaNet bounding_box_format "
                f"{box_format!r}; supported formats are "
                f"{sorted(_SUPPORTED_BOX_FORMATS)}."
            )
        return box_format

    def _canonicalize_predictions(self, predictions, x):
        """Convert decoded boxes to the formatter's ``xywh`` contract.

        Parameters
        ----------
        predictions
            Decoded RetinaNet prediction mapping.
        x
            Input image batch. Its shape supplies dimensions for relative box
            formats.

        Returns
        -------
        canonical_predictions
            Prediction mapping with boxes in absolute ``xywh`` or normalized
            ``rel_xywh`` format, depending on the source normalization.

        Raises
        ------
        ValueError
            If the box format is unsupported or required prediction fields are
            absent.
        ImportError
            If KerasCV is unavailable while a conversion is required.
        """
        if not isinstance(predictions, Mapping):
            raise ValueError(
                "RetinaNet decoded predictions must be a mapping containing "
                "'boxes', 'confidence', and 'classes'."
            )
        if "boxes" not in predictions:
            raise ValueError("RetinaNet decoded predictions are missing 'boxes'.")

        box_format = self._box_format()
        target_format = "rel_xywh" if box_format.startswith("rel_") else "xywh"
        if box_format == target_format:
            return predictions

        try:
            from keras_cv import bounding_box
        except ImportError as error:  # pragma: no cover - only reached for bad installs
            raise ImportError(
                "KerasCV is required to convert RetinaNet bounding boxes."
            ) from error

        image_shape = (
            tf.shape(x)[1:4]
            if box_format.startswith("rel_") != target_format.startswith("rel_")
            else None
        )
        canonical = dict(predictions)
        canonical["boxes"] = bounding_box.convert_format(
            predictions["boxes"],
            source=box_format,
            target=target_format,
            image_shape=image_shape,
        )
        return canonical

    def call(self, x, training=None, **kwargs):
        """Call, decode when requested, and format the wrapped RetinaNet once.

        Parameters
        ----------
        x
            Input image batch with shape ``(B, height, width, channels)``.
        training
            Training flag used in decoded mode. Raw mode always forces
            ``training=False`` to avoid post-processing side effects.
        **kwargs
            Additional keyword arguments forwarded to the wrapped callable.

        Returns
        -------
        predictions
            A list of per-image :class:`TfMultiBoxTensor` objects, or a padded
            tensor when ``output_as_list`` is false.

        Raises
        ------
        AttributeError
            If raw mode is selected but the wrapped model has no callable
            ``decode_predictions`` method.
        """
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
