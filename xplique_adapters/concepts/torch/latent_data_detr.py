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

import numpy as np
import torch
import torchvision
from torch import Tensor
from xplique.concepts.latent_extractor import LatentData, LatentExtractorBuilder
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor

from ...object_detection.torch import DetrBoxFormatter


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

    This class encapsulates the multi-scale features and positional encodings extracted
    from the DETR (Detection Transformer) backbone network. Features are stored as
    NestedTensor objects to handle variable-sized inputs.

    Attributes
    ----------
    features
        List of NestedTensor objects containing backbone feature maps at different scales.
    pos
        List of positional encoding tensors corresponding to each feature scale.
    selected_index
        Index of the feature level exposed as activations and consumed by the
        DETR decoder. The final level is selected by default.
    """

    def __init__(
        self,
        features: list,
        pos: list[torch.Tensor],
        selected_index: int = -1,
    ):
        """
        Initialize DETR latent data with features and positional encodings.

        Parameters
        ----------
        features
            List of NestedTensor objects from the backbone.
        pos
            List of positional encoding tensors.
        selected_index
            Index of the selected feature level. Defaults to the final level.
        """
        self.features = features
        self.pos = pos
        self.selected_index = selected_index

    def __len__(self) -> int:
        """
        Return the batch size from the feature tensors.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        return len(self.features[self.selected_index].tensors)

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
        return LatentDataDetr(new_features, new_pos, self.selected_index)

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
        activations = self.features[self.selected_index].tensors

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

        selected_feature = self.features[self.selected_index]
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
        position = self.pos[self.selected_index]
        if position.shape[0] != new_batch_size:
            self.pos[self.selected_index] = position[:1].repeat(
                (new_batch_size,) + (1,) * (position.ndim - 1)
            )


class DetrExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for DETR models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for DETR (Detection Transformer) object detection models. It defines
    the forward pass split into backbone feature extraction (g) and transformer-based
    prediction (h).
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
            PyTorch DETR model instance with backbone, transformer, input_proj,
            query_embed, class_embed, and bbox_embed attributes.
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
        selected_index = -1

        def nested_tensor_from_tensor_list(tensor_list: list[Tensor]):
            # TODO make this more general
            if tensor_list[0].ndim == 3:
                if torchvision._is_tracing():
                    # nested_tensor_from_tensor_list() does not export well to ONNX
                    # call _onnx_nested_tensor_from_tensor_list() instead
                    return _onnx_nested_tensor_from_tensor_list(tensor_list)

                # TODO make it support different-sized images
                max_size = _max_by_axis([list(img.shape) for img in tensor_list])
                # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
                batch_shape = [len(tensor_list)] + max_size
                b, _, h, w = batch_shape
                dtype = tensor_list[0].dtype
                device_tensor = tensor_list[0].device
                tensor = torch.zeros(batch_shape, dtype=dtype, device=device_tensor)
                mask = torch.ones((b, h, w), dtype=torch.bool, device=device_tensor)
                # work around for
                # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                # m[: img.shape[1], :img.shape[2]] = False
                # which is not yet supported in onnx
                padded_imgs = []
                padded_masks = []
                for img in tensor_list:
                    padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
                    padded_img = torch.nn.functional.pad(
                        img, (0, padding[2], 0, padding[1], 0, padding[0])
                    )
                    padded_imgs.append(padded_img)

                    m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
                    padded_mask = torch.nn.functional.pad(
                        m, (0, padding[2], 0, padding[1]), "constant", 1
                    )
                    padded_masks.append(padded_mask.to(torch.bool))

                tensor = torch.stack(padded_imgs)
                mask = torch.stack(padded_masks)
            else:
                raise ValueError("not supported")
            return NestedTensor(tensor, mask)

        def _max_by_axis(the_list):
            # type: (List[List[int]]) -> List[int]
            maxes = the_list[0]
            for sublist in the_list[1:]:
                for index, item in enumerate(sublist):
                    maxes[index] = max(maxes[index], item)
            return maxes

        @torch.jit.unused
        def _onnx_nested_tensor_from_tensor_list(
            tensor_list: list[Tensor],
        ) -> NestedTensor:
            print("_onnx_nested_tensor_from_tensor_list")
            max_size = []
            for i in range(tensor_list[0].dim()):
                max_size_i = torch.max(
                    torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
                ).to(torch.int64)
                max_size.append(max_size_i)
            max_size = tuple(max_size)

            # work around for
            # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # m[: img.shape[1], :img.shape[2]] = False
            # which is not yet supported in onnx
            padded_imgs = []
            padded_masks = []
            for img in tensor_list:
                padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
                padded_img = torch.nn.functional.pad(
                    img, (0, padding[2], 0, padding[1], 0, padding[0])
                )
                padded_imgs.append(padded_img)

                m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
                padded_mask = torch.nn.functional.pad(
                    m, (0, padding[2], 0, padding[1]), "constant", 1
                )
                padded_masks.append(padded_mask.to(torch.bool))

            tensor = torch.stack(padded_imgs)
            mask = torch.stack(padded_masks)

            return NestedTensor(tensor, mask=mask)

        def g(self, samples) -> LatentDataDetr:
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)

            return LatentDataDetr(features, pos, selected_index=selected_index)

        def h(self, latent_data: LatentDataDetr):
            features, pos = latent_data.features, latent_data.pos

            src, mask = features[selected_index].decompose()
            if mask is None:
                mask = torch.zeros(
                    (src.shape[0], src.shape[-2], src.shape[-1]),
                    dtype=torch.bool,
                    device=src.device,
                )
            hs = self.transformer(
                self.input_proj(src), mask, self.query_embed.weight, pos[selected_index]
            )[0]

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
            return out

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
