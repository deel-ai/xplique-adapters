from types import SimpleNamespace

import pytest
import torch

from xplique_adapters.concepts.torch.latent_data_detr import (
    DetrExtractorBuilder,
    LatentDataDetr,
    NestedTensor,
)


class _Backbone(torch.nn.Module):
    def forward(self, samples):
        tensors = samples.tensors
        batch_size, _, height, width = tensors.shape
        mask = samples.mask
        if mask is not None:
            mask = mask.clone()
        features = [
            NestedTensor(tensors[:, :1], mask),
            NestedTensor(tensors[:, :2], None if mask is None else mask.clone()),
        ]
        positions = [
            torch.zeros(batch_size, 2, height, width, device=tensors.device),
            torch.zeros(batch_size, 2, height, width, device=tensors.device),
        ]
        return features, positions


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_mask = None

    def forward(self, source, mask, query_embed, positional_encoding):
        del positional_encoding
        self.last_mask = mask
        pooled = source.flatten(2).mean(-1)
        hidden_states = pooled[:, None, :] + query_embed[None, :, :]
        return hidden_states.unsqueeze(0), None


class _FacebookDetr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.transformer = _Transformer()
        self.input_proj = torch.nn.Conv2d(2, 2, kernel_size=1)
        self.query_embed = torch.nn.Embedding(2, 2)
        self.class_embed = torch.nn.Linear(2, 3)
        self.bbox_embed = torch.nn.Linear(2, 4)
        self.aux_loss = False


def _build_extractor():
    model = _FacebookDetr()
    extractor = DetrExtractorBuilder.build(
        model, device="cpu", batch_size=1, image_size=(4, 4)
    )
    return model, extractor


def test_selected_index_is_consistently_the_last_backbone_level():
    _model, extractor = _build_extractor()
    samples = torch.randn(1, 3, 4, 4)

    latent_data = extractor.input_to_latent(samples)
    first_feature = latent_data.features[0].tensors.clone()
    first_position = latent_data.pos[0].clone()

    assert isinstance(latent_data, LatentDataDetr)
    assert latent_data.selected_index == -1
    assert len(latent_data) == 1
    assert latent_data.get_activations(as_numpy=False).shape == (1, 4, 4, 2)

    latent_data.set_activations(torch.ones(3, 4, 4, 2))

    assert len(latent_data) == 3
    assert latent_data.features[-1].tensors.shape == (3, 2, 4, 4)
    assert latent_data.features[0].tensors.shape == (1, 1, 4, 4)
    assert torch.equal(latent_data.features[0].tensors, first_feature)
    assert torch.equal(latent_data.pos[0], first_position)


def test_rebatching_is_materialized_and_supports_two_directions():
    _model, extractor = _build_extractor()
    latent_data = extractor.input_to_latent(torch.randn(1, 3, 4, 4))
    original_mask = latent_data.features[-1].mask
    original_position = latent_data.pos[-1]

    latent_data.set_activations(torch.randn(3, 4, 4, 2))

    assert latent_data.features[-1].mask.shape == (3, 4, 4)
    assert latent_data.pos[-1].shape == (3, 2, 4, 4)
    assert latent_data.features[-1].mask.data_ptr() != original_mask.data_ptr()
    assert latent_data.pos[-1].data_ptr() != original_position.data_ptr()

    latent_data.set_activations(torch.randn(2, 4, 4, 2))

    assert len(latent_data) == 2
    assert latent_data.features[-1].mask.shape[0] == 2
    assert latent_data.pos[-1].shape[0] == 2


def test_none_mask_survives_detach_and_to():
    _model, extractor = _build_extractor()
    latent_data = extractor.input_to_latent(torch.randn(1, 3, 4, 4))
    latent_data.features[-1].mask = None

    latent_data.detach()
    moved = latent_data.to("cpu")

    assert latent_data.features[-1].mask is None
    assert moved.features[-1].mask is None
    assert len(moved.features) == 2
    assert len(moved.pos) == 2


def test_h_fabricates_all_valid_mask_for_none_selected_mask():
    model, extractor = _build_extractor()
    latent_data = extractor.input_to_latent(torch.randn(2, 3, 4, 4))
    latent_data.features[-1].mask = None

    predictions = model.h(latent_data)

    assert predictions["pred_logits"].shape == (2, 2, 3)
    assert model.transformer.last_mask.shape == (2, 4, 4)
    assert model.transformer.last_mask.dtype is torch.bool
    assert not model.transformer.last_mask.any()


def test_builder_requires_facebook_attributes():
    model = SimpleNamespace(
        backbone=object(),
        transformer=object(),
        input_proj=object(),
        query_embed=object(),
        class_embed=object(),
    )

    with pytest.raises(TypeError, match="bbox_embed.*Facebook-reference-compatible"):
        DetrExtractorBuilder.build(model, device="cpu")


def test_builder_requires_auxiliary_loss_callback():
    model = SimpleNamespace(
        backbone=object(),
        transformer=object(),
        input_proj=object(),
        query_embed=object(),
        class_embed=object(),
        bbox_embed=object(),
        aux_loss=True,
        _set_aux_loss=None,
    )

    with pytest.raises(TypeError, match="callable.*_set_aux_loss"):
        DetrExtractorBuilder.build(model, device="cpu")


def test_split_detr_path_preserves_input_gradients():
    model, extractor = _build_extractor()
    samples = torch.randn(1, 3, 4, 4, requires_grad=True)

    latent_data = extractor.input_to_latent(samples)
    activations = latent_data.get_activations(as_numpy=False, keep_gradients=True)
    latent_data.set_activations(activations)
    predictions = model.h(latent_data)
    formatted = extractor.output_formatter(predictions)[0]
    formatted.sum().backward()

    assert samples.grad is not None
    assert torch.any(samples.grad != 0)
