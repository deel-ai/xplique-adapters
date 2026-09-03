"""
DETR (Detection Transformer) latent data extraction for concept-based explanations.

This module provides components for extracting and manipulating latent representations
from DETR object detection models for use with Xplique's concept-based explanation methods.

See https://github.com/facebookresearch/detr/blob/main/util/misc.py

Copyright 2020-present Facebook, Inc. Portions of this file are modified from
the upstream source and are licensed under the Apache License, Version 2.0.
See https://github.com/facebookresearch/detr/blob/main/LICENSE.
"""

import types
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
import torchvision
from torch import Tensor
from xplique.concepts.latent_extractor import LatentData, LatentExtractorBuilder
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor

from ...object_detection.torch import DetrBoxFormatter

_FACEBOOK_ATTRIBUTES = (
    "backbone",
    "transformer",
    "input_proj",
    "query_embed",
    "class_embed",
    "bbox_embed",
)
_HUGGINGFACE_BASE_ATTRIBUTES = (
    "backbone",
    "position_embedding",
    "input_projection",
    "encoder",
    "decoder",
    "query_position_embeddings",
)
_HUGGINGFACE_HEAD_ATTRIBUTES = ("class_labels_classifier", "bbox_predictor")


class NestedTensor:
    """
    Container for tensors with associated masks for handling variable-sized inputs.

    This class wraps tensors and their corresponding masks, enabling operations
    on batched inputs with different sizes (common in DETR models).

    Attributes
    ----------
    tensors
        The main tensor data.
    mask
        Optional boolean mask indicating valid regions (False) and padding (True).
    """

    def __init__(self, tensors: torch.Tensor, mask: Tensor | None):
        """
        Initialize a NestedTensor with tensors and an optional mask.

        Parameters
        ----------
        tensors
            The main tensor data.
        mask
            Optional boolean mask for valid regions.
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> "NestedTensor":
        """
        Move tensors and mask to the specified device.

        Parameters
        ----------
        device
            Target device (e.g., 'cuda', 'cpu').

        Returns
        -------
        nested_tensor
            New NestedTensor instance on the target device.
        """
        # type: (Device) -> NestedTensor
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> tuple:
        """
        Decompose the NestedTensor into separate tensors and mask.

        Returns
        -------
        tensors
            The main tensor data.
        mask
            The boolean mask.
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        """
        Return string representation of the tensors.

        Returns
        -------
        representation
            String representation of the tensor data.
        """
        return str(self.tensors)


class LatentDataDetr(LatentData):
    """
    Stores latent representations (features and positional encodings) from DETR's backbone.

    This class retains every feature level and positional encoding returned by a DETR
    backbone. Features are stored as NestedTensor objects whose masks use ``True`` for
    padding. Only the final backbone level is exposed as concept activations and passed
    to the encoder-decoder, matching both Facebook and HuggingFace DETR forward paths.
    Decoding arbitrary earlier levels is intentionally unsupported because the input
    projection is constructed for the final level's channel count.

    Attributes
    ----------
    features
        List of NestedTensor objects containing backbone feature maps at different scales.
    pos
        List of positional encoding tensors corresponding to each feature scale.
    """

    BACKBONE_LEVEL_INDEX = -1

    def __init__(
        self,
        features: list,
        pos: list[torch.Tensor],
    ):
        """
        Initialize DETR latent data with features and positional encodings.

        Parameters
        ----------
        features
            List of NestedTensor objects from the backbone.
        pos
            List of positional encoding tensors.
        """
        self.features = features
        self.pos = pos

    def selected_feature(self) -> NestedTensor:
        """Return the final backbone feature consumed by DETR."""
        return self.features[self.BACKBONE_LEVEL_INDEX]

    def selected_position(self) -> torch.Tensor:
        """Return the positional encoding for the final backbone feature."""
        return self.pos[self.BACKBONE_LEVEL_INDEX]

    def selected_padding_mask(self) -> torch.Tensor:
        """Return the final feature's padding mask, or an empty padding mask."""
        feature = self.selected_feature()
        if feature.mask is not None:
            return feature.mask
        return torch.zeros(
            (feature.tensors.shape[0], *feature.tensors.shape[-2:]),
            dtype=torch.bool,
            device=feature.tensors.device,
        )

    def __len__(self) -> int:
        """
        Return the batch size from the feature tensors.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        return len(self.selected_feature().tensors)

    def detach(self) -> None:
        """
        Detach all tensors from the computation graph.

        This method detaches features and positional encodings, preventing
        gradient computation through these tensors.
        """
        for feature in self.features:
            feature.tensors = feature.tensors.detach()
            if feature.mask is not None:
                feature.mask = feature.mask.detach()
        self.pos = [position.detach() for position in self.pos]

    def to(self, device: torch.device) -> "LatentDataDetr":
        """
        Move all data to the specified device.

        Parameters
        ----------
        device
            Target device (e.g., torch.device('cuda') or torch.device('cpu')).

        Returns
        -------
        latent_data
            New LatentDataDetr instance with data on the target device.
        """
        new_features = [feature.to(device) for feature in self.features]
        new_pos = [position.to(device) for position in self.pos]
        return LatentDataDetr(new_features, new_pos)

    def get_activations(
        self, as_numpy: bool = True, keep_gradients: bool = False
    ) -> np.ndarray | torch.Tensor:
        """
        Extract the feature tensors as activations.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Feature tensors as numpy array or PyTorch tensor. If 4D (N, C, H, W),
            converted to (N, H, W, C) format for compatibility.
        """
        activations = self.selected_feature().tensors

        if not keep_gradients:
            activations = activations.detach()

        is_4d = len(activations.shape) == 4
        if is_4d:
            activations = activations.permute(0, 2, 3, 1)

        if as_numpy:
            activations = activations.cpu().numpy()

        return activations

    def set_activations(self, values: torch.Tensor) -> None:
        """
        Update the feature tensors with new activation values.

        Parameters
        ----------
        values
            New feature tensor values. Expected format is (N, H, W, C), which will
            be converted to PyTorch's (N, C, H, W) format.
        """
        # tensorflow/numpy -> torch
        # activations: (N, H, W, C) -> (N, C, H, W)
        is_4d = len(values.shape) == 4
        if is_4d:
            values = values.permute(0, 3, 1, 2)

        index = self.BACKBONE_LEVEL_INDEX
        selected_feature = self.selected_feature()
        selected_feature.tensors = values

        # When a perturbation-based explainer batches multiple perturbed
        # coefficients, mask and pos must match the new batch size.
        # All items are perturbations of the same single image, so repeat its
        # batch companions from the first item.
        new_batch_size = values.shape[0]
        if (
            selected_feature.mask is not None
            and selected_feature.mask.shape[0] != new_batch_size
        ):
            selected_feature.mask = selected_feature.mask[:1].repeat(
                (new_batch_size,) + (1,) * (selected_feature.mask.ndim - 1)
            )
        position = self.selected_position()
        if position.shape[0] != new_batch_size:
            self.pos[index] = position[:1].repeat(
                (new_batch_size,) + (1,) * (position.ndim - 1)
            )


def _max_by_axis(shapes: list[list[int]]) -> list[int]:
    maxes = shapes[0].copy()
    for shape in shapes[1:]:
        for index, item in enumerate(shape):
            maxes[index] = max(maxes[index], item)
    return maxes


def _pad_images(
    tensor_list: list[Tensor] | Tensor,
    max_size: list[int] | tuple,
) -> NestedTensor:
    padded_images = []
    padded_masks = []
    for image in tensor_list:
        padding = [maximum - current for maximum, current in zip(max_size, image.shape)]
        padded_images.append(
            torch.nn.functional.pad(
                image, (0, padding[2], 0, padding[1], 0, padding[0])
            )
        )
        valid_region = torch.zeros_like(image[0], dtype=torch.int, device=image.device)
        padded_masks.append(
            torch.nn.functional.pad(
                valid_region, (0, padding[2], 0, padding[1]), "constant", 1
            ).to(torch.bool)
        )
    return NestedTensor(torch.stack(padded_images), torch.stack(padded_masks))


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(
    tensor_list: list[Tensor] | Tensor,
) -> NestedTensor:
    max_size = []
    for dimension in range(tensor_list[0].dim()):
        maximum = torch.max(
            torch.stack([image.shape[dimension] for image in tensor_list]).to(
                torch.float32
            )
        ).to(torch.int64)
        max_size.append(maximum)
    return _pad_images(tensor_list, tuple(max_size))


def _nested_tensor_from_tensor_list(
    tensor_list: list[Tensor] | Tensor,
) -> NestedTensor:
    if len(tensor_list) == 0:
        raise ValueError("DETR input must contain at least one image.")
    if any(image.ndim != 3 for image in tensor_list):
        raise ValueError("DETR expects images with shape (C, H, W) before batching.")
    channels = tensor_list[0].shape[0]
    if any(image.shape[0] != channels for image in tensor_list):
        raise ValueError("All DETR images must have the same number of channels.")
    if torchvision._is_tracing():
        return _onnx_nested_tensor_from_tensor_list(tensor_list)
    max_size = _max_by_axis([list(image.shape) for image in tensor_list])
    return _pad_images(tensor_list, max_size)


def _facebook_g(self, samples) -> LatentDataDetr:
    if isinstance(samples, (list, torch.Tensor)):
        samples = _nested_tensor_from_tensor_list(samples)
    features, positions = self.backbone(samples)
    return LatentDataDetr(features, positions)


def _facebook_h(self, latent_data: LatentDataDetr):
    source = latent_data.selected_feature().tensors
    hidden_states = self.transformer(
        self.input_proj(source),
        latent_data.selected_padding_mask(),
        self.query_embed.weight,
        latent_data.selected_position(),
    )[0]
    outputs_class = self.class_embed(hidden_states)
    outputs_coord = self.bbox_embed(hidden_states).sigmoid()
    output = {
        "pred_logits": outputs_class[-1],
        "pred_boxes": outputs_coord[-1],
    }
    if self.aux_loss:
        output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
    return output


def _huggingface_g(self, samples) -> LatentDataDetr:
    nested_samples = _nested_tensor_from_tensor_list(samples)
    pixel_values = nested_samples.tensors
    valid_mask = nested_samples.mask.logical_not()
    backbone_features = self.model.backbone(pixel_values, valid_mask)
    features = [
        NestedTensor(feature_map, feature_mask.logical_not())
        for feature_map, feature_mask in backbone_features
    ]
    positions = [
        self.model.position_embedding(
            shape=feature.tensors.shape,
            device=feature.tensors.device,
            dtype=pixel_values.dtype,
            mask=feature.mask.logical_not(),
        )
        for feature in features
    ]
    return LatentDataDetr(features, positions)


def _huggingface_h(self, latent_data: LatentDataDetr):
    feature_map = latent_data.selected_feature().tensors
    valid_mask = latent_data.selected_padding_mask().logical_not()
    flattened_features = (
        self.model.input_projection(feature_map).flatten(2).transpose(1, 2)
    )
    spatial_positions = latent_data.selected_position().flatten(2).transpose(1, 2)
    flattened_mask = valid_mask.flatten(1)
    encoder_outputs = self.model.encoder(
        inputs_embeds=flattened_features,
        attention_mask=flattened_mask,
        spatial_position_embeddings=spatial_positions,
    )

    query_positions = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(
        feature_map.shape[0], 1, 1
    )
    decoder_outputs = self.model.decoder(
        inputs_embeds=torch.zeros_like(query_positions),
        attention_mask=None,
        spatial_position_embeddings=spatial_positions,
        object_queries_position_embeddings=query_positions,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=flattened_mask,
    )
    sequence_output = decoder_outputs.last_hidden_state
    return {
        "logits": self.class_labels_classifier(sequence_output),
        "pred_boxes": self.bbox_predictor(sequence_output).sigmoid(),
    }


def _missing_attributes(instance, attributes: tuple[str, ...]) -> list[str]:
    return [name for name in attributes if getattr(instance, name, None) is None]


def _is_facebook_detr(model) -> bool:
    return not _missing_attributes(model, _FACEBOOK_ATTRIBUTES)


def _is_huggingface_detr(model) -> bool:
    return not _missing_huggingface_attributes(model)


def _missing_huggingface_attributes(model) -> list[str]:
    base_model = getattr(model, "model", None)
    if base_model is None:
        return ["model"]
    missing = [
        f"model.{name}"
        for name in _missing_attributes(base_model, _HUGGINGFACE_BASE_ATTRIBUTES)
    ]
    missing.extend(_missing_attributes(model, _HUGGINGFACE_HEAD_ATTRIBUTES))
    if getattr(getattr(model, "config", None), "model_type", None) != "detr":
        missing.append("config.model_type='detr'")
    return missing


def _is_transformers_v4_detr(model) -> bool:
    base_model = getattr(model, "model", None)
    backbone = getattr(base_model, "backbone", None)
    return (
        base_model is not None
        and getattr(backbone, "conv_encoder", None) is not None
        and not _missing_attributes(model, _HUGGINGFACE_HEAD_ATTRIBUTES)
        and getattr(getattr(model, "config", None), "model_type", None) == "detr"
    )


def _transformers_version() -> str:
    try:
        return version("transformers")
    except PackageNotFoundError:
        return "unknown"


class DetrExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for DETR models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for DETR (Detection Transformer) object detection models. It defines
    the forward pass split into backbone feature extraction (g) and transformer-based
    prediction (h). Caller-supplied Facebook reference DETR models and HuggingFace
    Transformers 5 ``DetrForObjectDetection`` models are supported.
    """

    @classmethod
    def build(
        cls,
        model: Callable,
        device: str = "cuda",
        batch_size: int = 1,
        image_size: tuple = (800, 800),
    ) -> "TorchLatentExtractor":
        """
        Build a LatentExtractor for a DETR model.

        This method creates custom g and h functions that split the model's forward pass:
        g extracts backbone features and positional encodings, and h processes them through
        the transformer and prediction heads.

        Parameters
        ----------
        model
            Caller-supplied Facebook reference DETR model or HuggingFace
            Transformers 5 ``DetrForObjectDetection`` model.
        device
            Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
        batch_size
            Batch size for processing. Default is 1.
        image_size
            Tuple specifying the input image size (height, width) for the model.
            Default is (800, 800).
        Returns
        -------
        latent_extractor
            Configured TorchLatentExtractor instance for the DETR model.
        """
        is_facebook = _is_facebook_detr(model)
        is_huggingface = _is_huggingface_detr(model)
        if not is_facebook and _is_transformers_v4_detr(model):
            detected_version = _transformers_version()
            raise TypeError(
                "DetrExtractorBuilder requires transformers>=5,<6 for HuggingFace "
                f"DETR models; detected transformers {detected_version}."
            )
        if not is_facebook and not is_huggingface:
            facebook_missing = ", ".join(
                _missing_attributes(model, _FACEBOOK_ATTRIBUTES)
            )
            huggingface_missing = ", ".join(_missing_huggingface_attributes(model))
            raise TypeError(
                "DetrExtractorBuilder requires a Facebook reference DETR model "
                f"(missing: {facebook_missing}) or a HuggingFace Transformers 5 "
                "DetrForObjectDetection model "
                f"(missing base-model attributes: {huggingface_missing})."
            )
        if (
            is_facebook
            and getattr(model, "aux_loss", False)
            and not callable(getattr(model, "_set_aux_loss", None))
        ):
            raise TypeError(
                "DetrExtractorBuilder requires callable model._set_aux_loss "
                "when model.aux_loss is enabled."
            )

        if is_facebook:
            g = _facebook_g
            h = _facebook_h
        else:
            g = _huggingface_g
            h = _huggingface_h

        model.g = types.MethodType(g, model)
        model.h = types.MethodType(h, model)

        processed_formatter = DetrBoxFormatter(image_size=image_size)
        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=LatentDataDetr,
            output_formatter=processed_formatter,
            batch_size=batch_size,
            device=device,
        )
        return latent_extractor
