"""
YOLO latent data extraction for concept-based explanations.

This module provides components for extracting and manipulating latent representations
from YOLO object detection models for use with Xplique's concept-based explanation methods.

See https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py

This file modifies code from Ultralytics contributors and is licensed under the
GNU Affero General Public License v3.0. See
https://github.com/ultralytics/ultralytics/blob/main/LICENSE.
"""

import copy
import types
from enum import Enum

import numpy as np
import torch
import ultralytics
from torch import Tensor
from xplique.concepts.latent_extractor import LatentData, LatentExtractorBuilder
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor

from ...object_detection.torch import YoloOneToManyFormatter, YoloOneToOneFormatter


class LatentDataYolo(LatentData):
    """
    Stores latent representations (activations and intermediate outputs) from YOLO models.

    This class encapsulates intermediate activations and layer outputs during YOLO's
    forward pass. The x attribute contains the main activation tensor at the split point,
    while y contains a list of outputs from earlier layers needed for skip connections.

    Attributes
    ----------
    x
        Main activation tensor at the model split point.
    y
        List of intermediate layer outputs, used for skip connections in later layers.
    """

    def __init__(self, x: torch.Tensor, y: list[torch.Tensor | None]):
        """
        Initialize YOLO latent data with activation tensor and intermediate outputs.

        Parameters
        ----------
        x
            Main activation tensor at the split point.
        y
            List of intermediate layer outputs.
        """
        self.x = x
        self.y = y

    def get_activations(
        self, as_numpy: bool = True, keep_gradients: bool = False
    ) -> np.ndarray | torch.Tensor:
        """
        Extract the main activation tensor.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Activation tensor as numpy array or PyTorch tensor. If 4D (N, C, H, W),
            converted to (N, H, W, C) format for compatibility.
        """
        activations = self.x

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
        Update the main activation tensor with new values.

        Parameters
        ----------
        values
            New activation tensor values. Expected format is (N, H, W, C), which will
            be converted to PyTorch's (N, C, H, W) format.
        """
        is_4d = len(values.shape) == 4
        if is_4d:
            values = values.permute(0, 3, 1, 2)
        self.x = values

        # When a perturbation-based explainer batches multiple perturbed
        # coefficients, skip-connection tensors in y must match the new batch
        # size. All items are perturbations of the same single image so expand
        # is safe.
        new_batch_size = values.shape[0]
        self.y = [
            yi[:1].expand(new_batch_size, *[-1] * (yi.dim() - 1))
            if yi is not None
            else None
            for yi in self.y
        ]

    def __str__(self) -> str:
        """
        Return string representation of the latent data.

        Returns
        -------
        representation
            String describing the shapes of x and length of y.
        """
        return f"LatentDataYolo(x shape: {self.x.shape}, y length: {len(self.y)})"


def make_head_end2end_differentiable(
    model: "ultralytics.nn.tasks.DetectionModel",
) -> "ultralytics.nn.tasks.DetectionModel":
    """
    Create a differentiable version of a YOLO26 model's end2end detection head.

    In ``Detect.forward`` the one2one branch is intentionally computed from detached
    features (``x_detach = [xi.detach() for xi in x]``) so that the bipartite-matching
    training loss cannot back-propagate into the shared backbone.  For attribution
    methods this breaks the gradient graph.

    This function creates a deep copy of the model and replaces ``Detect.forward`` with
    an identical version that simply omits the ``.detach()`` call, so gradients flow
    through the one2one branch end-to-end.  All downstream operations (``_inference``,
    ``postprocess`` via ``topk`` + ``gather``) are already differentiable.

    Parameters
    ----------
    model
        Original YOLO DetectionModel instance (must have ``end2end=True`` head).

    Returns
    -------
    differentiable_model
        Deep copy of the model with a patched ``Detect.forward`` that preserves
        gradients through the one2one branch.
    """

    # Root cause in upstream ultralytics.nn.modules.head.Detect.forward:
    #
    #   if self.end2end:
    #       x_detach = [xi.detach() for xi in x]
    #       one2one = self.forward_head(x_detach, **self.one2one)
    #
    # That detach severs the computation graph before the one2one branch. The
    # decoded end2end detections then come from tensors with no grad_fn, so
    # attribution methods fail at backward() with "tensor does not require grad".
    #
    # Our patch keeps the original control flow and only replaces the detached
    # features with the live tensors from x.

    def _differentiable_forward(self, x):
        """Detect.forward without .detach() on the one2one branch."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            # This is the critical fix: upstream uses x_detach here.
            one2one = self.forward_head(x, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    differentiable_model = copy.deepcopy(model)

    head_detect = next(iter(differentiable_model.children()))[-1]
    if head_detect.training:
        print("head_detect was in training mode, switching to eval mode")
        head_detect.eval()
        head_detect.training = False

    head_detect.forward = types.MethodType(_differentiable_forward, head_detect)

    return differentiable_model


class YoloExtractorMode(Enum):
    """Select which YOLO detection path the latent extractor should expose."""

    ONE_TO_MANY = "one_to_many"
    ONE_TO_ONE = "one_to_one"


class YoloExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for YOLO models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for YOLO (You Only Look Once) object detection models. It allows
    splitting the model at any layer to extract intermediate features. It also
    bypasses postprocessing/NMS-style filtering by using raw head outputs, to
    preserve gradients for attribution methods.

    YOLO Architecture Overview
    ----------
    Backbone (sequential 0->9):
        Input -> Layer 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

    Neck: three Resolution Paths, concatenation nodes are in parentheses:
        High-resolution path (P3 - small objects, 80x80):
        10 -> (11) -> 12 -> 13 -> (14) -> 15
        Mid-resolution path (P4 - medium objects, 40x40):
        10 -> (11) -> 12 -> 13 -> (14) -> 15 -> 16 -> (17) -> 18
        Low-resolution path (P5 - large objects, 20x20):
        10 -> (11) -> 12 -> 13 -> (14) -> 15 -> 16 -> (17) -> 18 -> 19 -> (20) -> 21

    Head:
        one-to-many (default): uses the original dense YOLO-style head.
        one-to-one: for end2end-capable heads (e.g., YOLO26), this branch is available in
        addition. It is patched to be fully differentiable by removing the detach() in
        the forward pass.
    """

    @classmethod
    def build(
        cls,
        model: "ultralytics.nn.tasks.DetectionModel",
        extraction_layer: int,
        batch_size: int = 1,
        nb_classes: int | None = None,
        mode: YoloExtractorMode = YoloExtractorMode.ONE_TO_MANY,
        device: str = "cuda",
    ) -> "TorchLatentExtractor":
        """
        Build a LatentExtractor for a YOLO model.

        This method creates custom g and h functions that split the model's forward pass
        at a specified layer: g runs layers up to extraction_layer, and h runs remaining layers.
        The model is modified to preserve gradients through the detection head.

        Parameters
        ----------
        model
            Ultralytics YOLO DetectionModel instance.
        extraction_layer
            Index of the layer where the model should be split. Must not be a layer
            that only takes input from the previous layer (f != -1).
        batch_size
            Batch size for processing. Default is 1.
        device
            Device to run the model on ('cpu' or 'cuda'). Default is 'cuda'.
        nb_classes
            Number of classes for ``YoloOneToOneFormatter`` when using
            ``ONE_TO_ONE`` mode without providing a custom formatter.
        mode
            Detection path to expose through the extractor. ``ONE_TO_MANY`` keeps
            the dense YOLO11-style path. ``ONE_TO_ONE`` uses the YOLO26 end2end
            path with the differentiable one2one forward patch.

        Returns
        -------
        latent_extractor
            Configured TorchLatentExtractor instance for the YOLO model.

        Raises
        ------
        ValueError
            If extraction_layer is out of valid range or points to an invalid layer type.
        """

        def g(self, x) -> LatentDataYolo:
            y = []  # outputs
            for m in self.model[:extraction_layer]:
                if m.f != -1:  # if not from previous layer
                    x = (
                        y[m.f]
                        if isinstance(m.f, int)
                        else [x if j == -1 else y[j] for j in m.f]
                    )  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return LatentDataYolo(x, y)

        def h(self, latent_data: LatentDataYolo) -> Tensor:

            x, y = latent_data.x, latent_data.y
            # to avoid in-place modifications
            # x = x.clone()
            y = [yi.clone() if yi is not None else None for yi in y]

            for m in self.model[extraction_layer:]:
                if m.f != -1:  # if not from previous layer
                    x = (
                        y[m.f]
                        if isinstance(m.f, int)
                        else [x if j == -1 else y[j] for j in m.f]
                    )  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.save else None)
            return x

        # check on extraction_layer value
        if extraction_layer < 0 or extraction_layer >= len(model.model):
            raise ValueError(
                f"extraction_layer must be between 0 and {len(model.model) - 1}"
            )
        if model.model[extraction_layer].f != -1:
            raise ValueError(
                f"extraction_layer must not be a layer that takes input only from the previous layer (but here f={model.model[extraction_layer].f})"
            )

        if model.training:
            print("model was in training mode, switching to eval mode")
            model.eval()
            model.training = False

        if mode == YoloExtractorMode.ONE_TO_MANY:
            differentiable_model = model
            formatter = YoloOneToManyFormatter()
        elif mode == YoloExtractorMode.ONE_TO_ONE:
            head_detect = model.model[-1]
            if not getattr(head_detect, "end2end", False):
                raise ValueError(
                    "ONE_TO_ONE mode requires a YOLO end2end detection head with one2one branches."
                )
            differentiable_model = make_head_end2end_differentiable(model)
            if nb_classes is None:
                raise ValueError(
                    "ONE_TO_ONE mode requires nb_classes when no custom formatter is provided."
                )
            formatter = YoloOneToOneFormatter(nb_classes=nb_classes)
        else:
            raise ValueError(f"Unsupported YoloExtractorMode: {mode}")

        differentiable_model.g = types.MethodType(g, differentiable_model)
        differentiable_model.h = types.MethodType(h, differentiable_model)

        latent_extractor = TorchLatentExtractor(
            differentiable_model,
            differentiable_model.g,
            differentiable_model.h,
            latent_data_class=LatentDataYolo,
            output_formatter=formatter,
            batch_size=batch_size,
            device=device,
        )
        return latent_extractor
