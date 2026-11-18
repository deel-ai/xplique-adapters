"""YOLOX latent extraction for concept-based explanations."""

import copy
import types

import numpy as np
import torch
from xplique.concepts.latent_extractor import LatentData, LatentExtractorBuilder
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor

from ...object_detection.torch import YoloXBoxFormatter


class LatentDataYoloX(LatentData):
    """Store YOLOX FPN features while exposing one level as activations."""

    def __init__(
        self,
        fpn_outs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        extraction_layer: int,
    ) -> None:
        self.fpn_outs = list(fpn_outs)
        self.extraction_layer = extraction_layer

    def __len__(self) -> int:
        return self.fpn_outs[self.extraction_layer].shape[0]

    def detach(self) -> None:
        self.fpn_outs = [output.detach() for output in self.fpn_outs]

    def get_activations(
        self, as_numpy: bool = True, keep_gradients: bool = False
    ) -> np.ndarray | torch.Tensor:
        activations = self.fpn_outs[self.extraction_layer]
        if not keep_gradients:
            activations = activations.detach()
        if activations.ndim == 4:
            activations = activations.permute(0, 2, 3, 1)
        if as_numpy:
            return activations.cpu().numpy()
        return activations

    def set_activations(self, values: torch.Tensor) -> None:
        if values.ndim == 4:
            values = values.permute(0, 3, 1, 2)
        self.fpn_outs[self.extraction_layer] = values

        batch_size = values.shape[0]
        self.fpn_outs = [
            output[:1].expand(batch_size, *output.shape[1:])
            if output.shape[0] != batch_size
            else output
            for output in self.fpn_outs
        ]

    def to(self, device: torch.device) -> "LatentDataYoloX":
        return LatentDataYoloX(
            [output.to(device) for output in self.fpn_outs], self.extraction_layer
        )


class YoloXExtractorBuilder(LatentExtractorBuilder):
    """Build a CRAFT-compatible extractor by splitting YOLOX at its FPN."""

    @classmethod
    def build(
        cls,
        model: torch.nn.Module,
        device: str = "cuda",
        extraction_layer: int = -1,
        batch_size: int = 1,
    ) -> TorchLatentExtractor:
        """Build an extractor for one of the backbone's FPN output levels."""
        if not hasattr(model, "backbone") or not hasattr(model, "head"):
            raise TypeError("YOLOX model must expose 'backbone' and 'head' modules.")

        fpn_count = 3
        if not -fpn_count <= extraction_layer < fpn_count:
            raise ValueError(
                f"extraction_layer must be between {-fpn_count} and {fpn_count - 1}, "
                f"got {extraction_layer}."
            )
        layer_index = extraction_layer % fpn_count

        def g(self, inputs):
            return LatentDataYoloX(self.backbone(inputs), layer_index)

        def h(self, latent_data: LatentDataYoloX):
            return self.head(latent_data.fpn_outs)

        extractor_model = copy.deepcopy(model).eval()
        extractor_model.g = types.MethodType(g, extractor_model)
        extractor_model.h = types.MethodType(h, extractor_model)

        return TorchLatentExtractor(
            extractor_model,
            extractor_model.g,
            extractor_model.h,
            latent_data_class=LatentDataYoloX,
            output_formatter=YoloXBoxFormatter(),
            batch_size=batch_size,
            device=device,
        )
