import pytest
import torch

from xplique_adapters.object_detection.torch import (
    DetrBoxesModelWrapper,
    DetrBoxFormatter,
)


def _predictions(batch_size=1, num_queries=2, num_foreground_classes=2):
    return {
        "pred_logits": torch.randn(batch_size, num_queries, num_foreground_classes + 1),
        "pred_boxes": torch.rand(batch_size, num_queries, 4),
    }


def test_detr_formatter_init():
    formatter = DetrBoxFormatter((800, 600))

    assert formatter.image_size == (800, 600)


def test_detr_formatter_supports_facebook_schema():
    formatter = DetrBoxFormatter((800, 600))

    results = formatter(_predictions(num_foreground_classes=3))

    assert len(results) == 1
    assert results[0].shape == torch.Size([2, 8])
    assert results[0].probas().shape[-1] == 3


def test_detr_formatter_supports_huggingface_schema():
    formatter = DetrBoxFormatter((800, 600))
    predictions = _predictions(num_foreground_classes=2)
    predictions["logits"] = predictions.pop("pred_logits")

    results = formatter(predictions)

    assert results[0].shape == torch.Size([2, 7])


def test_detr_formatter_supports_huggingface_model_output():
    ModelOutput = pytest.importorskip("transformers").utils.ModelOutput
    formatter = DetrBoxFormatter((800, 600))
    predictions = _predictions(num_foreground_classes=2)
    output = ModelOutput(
        logits=predictions["pred_logits"], pred_boxes=predictions["pred_boxes"]
    )

    results = formatter(output)

    assert results[0].shape == torch.Size([2, 7])


def test_detr_formatter_missing_fields_error_lists_accepted_schemas():
    formatter = DetrBoxFormatter((800, 600))

    with pytest.raises(ValueError, match="pred_logits.*pred_boxes.*logits"):
        formatter({"scores": torch.ones(1)})


def test_detr_formatter_requires_foreground_and_no_object_classes():
    formatter = DetrBoxFormatter((800, 600))
    predictions = _predictions(num_foreground_classes=0)

    with pytest.raises(ValueError, match="foreground.*no-object"):
        formatter(predictions)


def test_detr_formatter_excludes_no_object_probability():
    formatter = DetrBoxFormatter((600, 800))
    logits = torch.tensor([[[1.0, 2.0, 5.0]]])
    predictions = {"pred_logits": logits, "pred_boxes": torch.zeros(1, 1, 4)}

    result = formatter(predictions)[0]
    expected_probas = logits.softmax(-1)[..., :-1]

    assert result.shape == torch.Size([1, 7])
    assert torch.allclose(result.probas(), expected_probas[0])
    assert torch.allclose(result.scores(), expected_probas.max(-1).values[0])


def test_detr_formatter_converts_rectangular_image_size_exactly():
    formatter = DetrBoxFormatter((600, 800))
    predictions = {
        "pred_logits": torch.zeros(1, 1, 2),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
    }

    boxes = formatter(predictions)[0].boxes()

    assert torch.allclose(boxes, torch.tensor([[0.0, 0.0, 800.0, 600.0]]))


def test_detr_formatter_preserves_gradients():
    formatter = DetrBoxFormatter((600, 800))
    logits = torch.randn(1, 2, 3, requires_grad=True)
    boxes = torch.rand(1, 2, 4, requires_grad=True)

    result = formatter({"pred_logits": logits, "pred_boxes": boxes})[0]
    result.sum().backward()

    assert logits.grad is not None
    assert boxes.grad is not None


def test_detr_wrapper_accepts_positional_image_size_and_forwards_pixel_mask():
    class SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received_x = None
            self.received_pixel_mask = None

        def forward(self, x, *, pixel_mask=None):
            self.received_x = x
            self.received_pixel_mask = pixel_mask
            return _predictions(batch_size=x.shape[0], num_foreground_classes=2)

    model = SpyModel()
    wrapper = DetrBoxesModelWrapper(model, (600, 800))
    x = torch.randn(2, 3, 16, 20)
    pixel_mask = torch.zeros(2, 16, 20, dtype=torch.bool)

    results = wrapper(x, pixel_mask=pixel_mask)

    assert len(results) == 2
    assert model.received_x is x
    assert model.received_pixel_mask is pixel_mask


def test_detr_wrapper_forwards_facebook_outputs():
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return _predictions(batch_size=x.shape[0], num_foreground_classes=2)

    wrapper = DetrBoxesModelWrapper(MockModel(), (600, 800))

    results = wrapper(torch.randn(2, 3, 16, 20))

    assert isinstance(results, list)
    assert len(results) == 2
