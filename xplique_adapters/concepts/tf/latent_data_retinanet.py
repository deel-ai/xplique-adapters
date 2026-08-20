"""
TensorFlow RetinaNet latent data extraction for concept-based explanations.

This module provides components for extracting and manipulating latent representations
from TensorFlow/Keras RetinaNet object detection models for use with Xplique's
concept-based explanation methods.

See https://github.com/keras-team/keras-cv/blob/master/keras_cv/src/models/object_detection/retinanet/retinanet.py

Copyright 2023 The KerasCV Authors. Portions of this file are modified from the
upstream source and are licensed under the Apache License, Version 2.0. See
https://github.com/keras-team/keras-cv/blob/master/LICENSE.
"""

import types

import numpy as np
import tensorflow as tf
from keras_cv.src import bounding_box
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.converters import _decode_deltas_to_boxes
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.concepts.latent_extractor import LatentData, LatentExtractorBuilder
from xplique.concepts.tf.latent_extractor import TfLatentExtractor

from xplique_adapters.object_detection.tf import (
    RetinaNetProcessedBoxFormatter,
)


def _decode_retinanet_predictions(
    model, cls_pred, box_pred, image_shape, anchor_image_shape=None
):
    """Decode RetinaNet head outputs into normalized ``rel_xyxy`` predictions.

    Parameters
    ----------
    model
        KerasCV RetinaNet model exposing ``bounding_box_format``,
        ``anchor_generator``, and the generator's ``bounding_box_format``.
    cls_pred
        Classification logits with shape ``(batch, detections, classes)``.
    box_pred
        Box deltas with shape ``(batch, detections, 4)``.
    image_shape
        Tensor or sequence containing ``(height, width, channels)``. This
        shape is used for delta decoding and output normalization.
    anchor_image_shape
        Optional Python-compatible image shape for anchor generation. When it
        is omitted, ``image_shape`` is used and a statically known TensorFlow
        shape is converted to a tuple in graph mode.

    Returns
    -------
    predictions
        Mapping containing decoded ``boxes`` in ``rel_xyxy`` format, per-class
        ``scores``, maximum ``confidence``, and integer ``classes``.

    Raises
    ------
    ValueError
        If the model does not declare ``bounding_box_format``.
    """
    model_box_format = getattr(model, "bounding_box_format", None)
    if model_box_format is None:
        raise ValueError(
            "RetinaNet model must define bounding_box_format to decode latent "
            "predictions."
        )

    box_variance = [0.1, 0.1, 0.2, 0.2]
    if anchor_image_shape is None:
        anchor_image_shape = image_shape
        if not tf.executing_eagerly():
            static_shape = tf.get_static_value(image_shape)
            if static_shape is not None:
                anchor_image_shape = tuple(int(value) for value in static_shape)
    anchors = model.anchor_generator(image_shape=anchor_image_shape)
    anchors = ops.concatenate(list(anchors.values()), axis=0)
    boxes = _decode_deltas_to_boxes(
        anchors=anchors,
        boxes_delta=box_pred,
        anchor_format=model.anchor_generator.bounding_box_format,
        box_format=model_box_format,
        variance=box_variance,
        image_shape=image_shape,
    )
    boxes = bounding_box.convert_format(
        boxes,
        source=model_box_format,
        target="rel_xyxy",
        image_shape=image_shape,
    )

    probas = ops.sigmoid(cls_pred)
    return {
        "boxes": boxes,
        "scores": probas,
        "confidence": ops.max(probas, axis=-1),
        "classes": ops.argmax(probas, axis=-1),
    }


class TfLatentDataRetinanet(LatentData):
    """Store RetinaNet's multi-scale backbone features and image shape.

    This class encapsulates the multi-scale feature pyramid outputs from the ResNet
    backbone of a RetinaNet model. Features are typically stored as a dictionary with
    keys like 'P3', 'P4', 'P5' representing different pyramid levels.

    When an explainer replaces the selected activation with a perturbation
    batch, all companion feature levels are materially repeated from their
    first item. The selected activation's leading dimension is authoritative.

    Attributes
    ----------
    resnet_features
        Dictionary of feature maps from ResNet backbone. Keys are pyramid levels
        (e.g., 'P3', 'P4', 'P5') and values are tensors with shapes like:
        - P3: (batch, 80, 80, 512)
        - P4: (batch, 40, 40, 1024)
        - P5: (batch, 20, 20, 2048)
    image_shape
        Tuple representing the shape of the input image (height, width, channels).
    index_activations
        Index specifying which feature map to use as activations. Default is -1 (last feature).
    anchor_image_shape
        Optional static ``(height, width, channels)`` shape supplied to the
        anchor generator. KerasCV anchor generation may require Python values
        when the decoder runs inside a TensorFlow graph.
    """

    index_activations = -1
    resnet_features: dict
    # typically a dict of resnet features with keys like 'P3', 'P4', 'P5'
    # Feature P3: np.shape (1, 80, 80, 512)
    # Feature P4: shape (1, 40, 40, 1024)
    # Feature P5: shape (1, 20, 20, 2048)
    image_shape: tuple

    def __init__(
        self,
        resnet_features: dict,
        image_shape: tuple,
        index_activations: int = -1,
        anchor_image_shape: tuple | None = None,
    ) -> None:
        """
        Initialize RetinaNet latent data with feature maps and image shape.

        Parameters
        ----------
        resnet_features
            Dictionary of feature maps from ResNet backbone.
        image_shape
            Tuple representing the input image shape.
        index_activations
            Index specifying which feature map to use. Default is -1.
        anchor_image_shape
            Optional static image shape for anchor generation, usually captured
            from the input shape during the extractor's feature pass.
        """
        self.resnet_features = resnet_features
        self.image_shape = image_shape
        self.index_activations = index_activations
        self.anchor_image_shape = anchor_image_shape

    def __len__(self) -> int:
        """
        Return the batch size from the feature maps.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        key = list(self.resnet_features.keys())[self.index_activations]
        return self.resnet_features[key].shape[0]

    def print_infos(self) -> None:
        """
        Print information about stored feature maps and image shape.

        Displays the keys and shapes of all feature maps in the ResNet features
        dictionary, as well as the input image shape.
        """
        print("LatentDataRetinanet:")
        for key, feature in self.resnet_features.items():
            print(f"\tFeature {key}: shape {feature.shape}")
        print(f"Image shape: {self.image_shape}")

    def get_activations(
        self, as_numpy: bool = True, keep_gradients: bool = False
    ) -> np.ndarray | tf.Tensor:
        """
        Extract the feature map at the specified index.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information (not used in this implementation).

        Returns
        -------
        activations
            Feature map as numpy array or tensor, depending on as_numpy parameter.
        """
        key = list(self.resnet_features.keys())[self.index_activations]
        activations = self.resnet_features[key]

        if as_numpy and isinstance(activations, tf.Tensor):
            activations = activations.numpy()

        return activations

    def check_features_positive(self) -> None:
        """
        Verify that all feature maps contain only non-negative values.

        Raises
        ------
        ValueError
            If any feature map contains negative values.
        """
        for name, feature in self.resnet_features.items():
            feature = tf.convert_to_tensor(feature)
            if bool(tf.reduce_any(feature < 0)):
                raise ValueError(
                    f"Feature {name!r} contains negative values, which is unexpected."
                )

    def set_activations(self, values: tf.Tensor | np.ndarray) -> None:
        """
        Update the selected feature and re-batch companion feature levels.

        The replacement activation's leading dimension determines the number
        of perturbations. Every other feature level is repeated from its first
        item because all outputs represent perturbations of one source image.

        Parameters
        ----------
        values
            New feature map values as tf.Tensor or np.ndarray.

        Raises
        ------
        TypeError
            If values is not a tf.Tensor or np.ndarray.
        """
        if not isinstance(values, (tf.Tensor, np.ndarray)):
            raise TypeError(
                f"Unsupported type: {type(values)}. Expected tf.Tensor or np.ndarray"
            )

        values = tf.convert_to_tensor(values)
        # When a perturbation-based explainer batches multiple perturbed
        # coefficients, all feature maps must match the new batch size.
        # All items are perturbations of the same single image, so repetition is safe.
        if values.shape[0] == 0:
            raise ValueError("Replacement activations cannot have an empty batch.")
        key = list(self.resnet_features.keys())[self.index_activations]
        for name, feature in self.resnet_features.items():
            if name != key and feature.shape[0] == 0:
                raise ValueError(f"Feature {name!r} has an empty source batch.")

        new_batch_size = tf.shape(values)[0]
        self.resnet_features[key] = values
        for k, feature in self.resnet_features.items():
            if k != key:
                feature = tf.convert_to_tensor(feature)
                self.resnet_features[k] = tf.repeat(feature[:1], new_batch_size, axis=0)


class RetinaNetExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for RetinaNet models.

    This class provides methods to construct a TfLatentExtractor specifically
    configured for RetinaNet object detection models. It defines the forward
    pass split into feature extraction (g) and prediction decoding (h).
    """

    @classmethod
    def build(
        cls, model, nb_classes, index_activations=-1, batch_size: int = 1
    ) -> "TfLatentExtractor":
        """
        Build a LatentExtractor for a RetinaNet model.

        This method creates custom g and h functions that split the model's forward
        pass: g extracts backbone features, and h processes them through the detection
        head and decodes predictions.

        Parameters
        ----------
        model
            TensorFlow RetinaNet model instance with feature_extractor, feature_pyramid,
            classification_head, and box_head attributes.
        nb_classes
            Number of object classes the model detects.
        index_activations
            Index specifying which feature pyramid level to use as activations. Default is -1.
        batch_size
            Batch size for processing. Default is 1.

        Notes
        -----
        ``g`` snapshots the statically known input shape for anchor generation.
        ``h`` still uses the TensorFlow ``image_shape`` tensor for decoding and
        normalization, so the same ``(height, width, channels)`` convention is
        preserved throughout the latent path.
        Returns
        -------
        latent_extractor
            Configured TfLatentExtractor instance for the RetinaNet model.
        """

        def g(self, samples: tf.Tensor) -> TfLatentDataRetinanet:
            backbone_outputs = self.feature_extractor(samples, training=False)
            return TfLatentDataRetinanet(
                backbone_outputs,
                tf.shape(samples)[1:4],
                index_activations=index_activations,
                anchor_image_shape=tuple(samples.shape[1:4]),
            )

        def h(self, latent_data: TfLatentDataRetinanet) -> dict:
            backbone_outputs, image_shape = (
                latent_data.resnet_features,
                latent_data.image_shape,
            )
            features = self.feature_pyramid(backbone_outputs, training=False)
            cls_outputs = []
            box_outputs = []
            for feature in features:
                cls_outputs.append(self.classification_head(feature))
                box_outputs.append(self.box_head(feature))

            # 4. Concatenate outputs
            cls_outputs = tf.concat(
                [
                    tf.reshape(c, [tf.shape(c)[0], -1, self.num_classes])
                    for c in cls_outputs
                ],
                axis=1,
            )
            box_outputs = tf.concat(
                [tf.reshape(b, [tf.shape(b)[0], -1, 4]) for b in box_outputs], axis=1
            )

            return _decode_retinanet_predictions(
                model,
                cls_outputs,
                box_outputs,
                image_shape,
                anchor_image_shape=latent_data.anchor_image_shape,
            )

        model.h = types.MethodType(h, model)
        model.g = types.MethodType(g, model)

        processed_formatter = RetinaNetProcessedBoxFormatter._from_box_types(
            nb_classes,
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        )
        latent_extractor = TfLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=TfLatentDataRetinanet,
            output_formatter=processed_formatter,
            batch_size=batch_size,
        )
        return latent_extractor
