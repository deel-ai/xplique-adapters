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
        self.last_samples = samples
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


class _Output(SimpleNamespace):
    def __getitem__(self, index):
        return (self.last_hidden_state,)[index]


class _HuggingFaceBackbone(torch.nn.Module):
    def forward(self, pixel_values, pixel_mask):
        self.last_pixel_values = pixel_values
        self.last_pixel_mask = pixel_mask
        features = [
            (pixel_values[:, :1], pixel_mask),
            (pixel_values[:, :2], pixel_mask.clone()),
        ]
        return features


class _HuggingFacePositionEmbedding(torch.nn.Module):
    def forward(self, *, shape, device, dtype, mask):
        self.last_mask = mask
        return torch.zeros(
            shape[0],
            2,
            shape[-2],
            shape[-1],
            device=device,
            dtype=dtype,
        )


class _HuggingFaceEncoder(torch.nn.Module):
    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        spatial_position_embeddings,
    ):
        self.last_attention_mask = attention_mask
        return _Output(last_hidden_state=inputs_embeds + spatial_position_embeddings)


class _HuggingFaceDecoder(torch.nn.Module):
    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        spatial_position_embeddings,
        object_queries_position_embeddings,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        del attention_mask, spatial_position_embeddings
        self.last_encoder_attention_mask = encoder_attention_mask
        pooled = encoder_hidden_states.mean(dim=1, keepdim=True)
        hidden = inputs_embeds + object_queries_position_embeddings + pooled
        return _Output(last_hidden_state=hidden)


class _HuggingFaceDetrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _HuggingFaceBackbone()
        self.position_embedding = _HuggingFacePositionEmbedding()
        self.input_projection = torch.nn.Conv2d(2, 2, kernel_size=1)
        self.encoder = _HuggingFaceEncoder()
        self.decoder = _HuggingFaceDecoder()
        self.query_position_embeddings = torch.nn.Embedding(2, 2)


class _HuggingFaceDetrForObjectDetection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _HuggingFaceDetrModel()
        self.class_labels_classifier = torch.nn.Linear(2, 3)
        self.bbox_predictor = torch.nn.Linear(2, 4)
        self.config = SimpleNamespace(model_type="detr", auxiliary_loss=False)


def _build_extractor():
    model = _FacebookDetr()
    extractor = DetrExtractorBuilder.build(
        model, device="cpu", batch_size=1, image_size=(4, 4)
    )
    return model, extractor


def _build_huggingface_extractor():
    model = _HuggingFaceDetrForObjectDetection()
    extractor = DetrExtractorBuilder.build(
        model, device="cpu", batch_size=1, image_size=(4, 4)
    )
    return model, extractor


def test_final_backbone_level_is_the_detr_latent_representation():
    model, extractor = _build_extractor()
    samples = torch.randn(1, 3, 4, 4)

    latent_data = extractor.input_to_latent(samples)
    first_feature = latent_data.features[0].tensors.clone()
    first_position = latent_data.pos[0].clone()

    assert isinstance(latent_data, LatentDataDetr)
    assert len(latent_data) == 1
    assert latent_data.get_activations(as_numpy=False).shape == (1, 4, 4, 2)

    latent_data.set_activations(torch.ones(3, 4, 4, 2))

    assert len(latent_data) == 3
    assert latent_data.features[-1].tensors.shape == (3, 2, 4, 4)
    assert latent_data.features[0].tensors.shape == (1, 1, 4, 4)
    assert torch.equal(latent_data.features[0].tensors, first_feature)
    assert torch.equal(latent_data.pos[0], first_position)
    predictions = model.h(latent_data)
    assert predictions["pred_logits"].shape == (3, 2, 3)


def test_detr_nested_tensor_supports_different_spatial_sizes():
    model, _extractor = _build_extractor()
    image_a = torch.randn(3, 5, 7)
    image_b = torch.randn(3, 3, 4)

    latent_data = model.g([image_a, image_b])
    samples = model.backbone.last_samples

    assert samples.tensors.shape == (2, 3, 5, 7)
    assert not samples.mask[0].any()
    assert not samples.mask[1, :3, :4].any()
    assert samples.mask[1, 3:, :].all()
    assert samples.mask[1, :, 4:].all()
    assert len(latent_data) == 2


@pytest.mark.parametrize("samples", [[], [torch.randn(3, 4)]])
def test_detr_nested_tensor_rejects_invalid_image_batches(samples):
    model, _extractor = _build_extractor()

    with pytest.raises(ValueError, match="at least one image|shape.*C, H, W"):
        model.g(samples)


def test_detr_nested_tensor_rejects_different_channel_counts():
    model, _extractor = _build_extractor()

    with pytest.raises(ValueError, match="same number of channels"):
        model.g([torch.randn(3, 4, 4), torch.randn(1, 4, 4)])


def test_huggingface_detr_split_uses_final_backbone_level():
    model, extractor = _build_huggingface_extractor()
    samples = torch.randn(1, 3, 4, 4, requires_grad=True)

    latent_data = extractor.input_to_latent(samples)
    first_feature = latent_data.features[0].tensors.clone()
    activations = latent_data.get_activations(as_numpy=False, keep_gradients=True)
    latent_data.set_activations(activations.repeat(3, 1, 1, 1))
    predictions = model.h(latent_data)

    assert activations.shape == (1, 4, 4, 2)
    assert torch.equal(latent_data.features[0].tensors, first_feature)
    assert predictions["logits"].shape == (3, 2, 3)
    assert predictions["pred_boxes"].shape == (3, 2, 4)
    assert torch.all(
        (predictions["pred_boxes"] >= 0) & (predictions["pred_boxes"] <= 1)
    )
    predictions["logits"].sum().backward()
    assert samples.grad is not None


def test_huggingface_detr_converts_padding_mask_to_valid_pixel_mask():
    model, _extractor = _build_huggingface_extractor()
    image_a = torch.randn(3, 5, 7)
    image_b = torch.randn(3, 3, 4)

    latent_data = model.g([image_a, image_b])
    pixel_mask = model.model.backbone.last_pixel_mask
    model.h(latent_data)

    assert pixel_mask.dtype is torch.bool
    assert pixel_mask[0].all()
    assert pixel_mask[1, :3, :4].all()
    assert not pixel_mask[1, 3:, :].any()
    assert not pixel_mask[1, :, 4:].any()
    assert torch.equal(
        model.model.encoder.last_attention_mask,
        latent_data.selected_padding_mask().logical_not().flatten(1),
    )
    assert model.model.position_embedding.last_mask.dtype is torch.bool


def test_real_huggingface_detr_split_preserves_outputs_and_gradients():
    transformers = pytest.importorskip("transformers")
    backbone_config = transformers.ResNetConfig(
        num_channels=3,
        embedding_size=8,
        hidden_sizes=[8, 16, 24, 32],
        depths=[1, 1, 1, 1],
        out_features=["stage4"],
    )
    config = transformers.DetrConfig(
        backbone_config=backbone_config,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=3,
        num_labels=2,
    )
    model = transformers.DetrForObjectDetection(config).eval()
    samples = torch.rand(1, 3, 32, 40, requires_grad=True)
    expected = model(samples)
    extractor = DetrExtractorBuilder.build(
        model, device="cpu", batch_size=1, image_size=(32, 40)
    )

    latent_data = extractor.input_to_latent(samples)
    actual = model.h(latent_data)

    assert torch.allclose(actual["logits"], expected.logits)
    assert torch.allclose(actual["pred_boxes"], expected.pred_boxes)
    actual["logits"].sum().backward()
    assert samples.grad is not None


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

    with pytest.raises(TypeError, match="Facebook reference.*HuggingFace"):
        DetrExtractorBuilder.build(model, device="cpu")


def test_builder_rejects_transformers_v4_detr():
    model = _HuggingFaceDetrForObjectDetection()
    del model.model.position_embedding
    model.model.backbone.conv_encoder = object()

    with pytest.raises(TypeError, match=r"transformers>=5,<6.*detected"):
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
