import torch

from xplique_adapters.object_detection.torch import (
    DetrBoxesModelWrapper,
    DetrBoxFormatter,
)


def test_detr_formatter_init():
    """Test that DetrBoxFormatter can be initialized with image_size."""
    image_size = (800, 800)
    formatter = DetrBoxFormatter(image_size=image_size)
    assert formatter is not None
    assert formatter.image_size == image_size


def test_detr_formatter_forward():
    """Test DetrBoxFormatter with mock predictions outputs absolute pixel coordinates."""
    image_size = (800, 800)
    formatter = DetrBoxFormatter(image_size=image_size)

    # Mock DETR predictions (normalized CXCYWH format)
    batch_size = 2
    num_queries = 100
    num_classes = 91

    predictions = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes + 1),
        "pred_boxes": torch.rand(batch_size, num_queries, 4),
    }

    results = formatter.forward(predictions)

    assert isinstance(results, list)
    assert len(results) == batch_size
    assert results[0].shape == torch.Size([num_queries, 96])

    # Verify boxes are in absolute pixel coordinates (not normalized)
    # Boxes should be in range [0, image_size], not [0, 1]
    boxes = results[0].boxes()
    assert boxes.shape == torch.Size([num_queries, 4])

    # Check that at least some boxes exceed 1.0 (indicating pixel coordinates)
    # Since input boxes are random in [0,1], output should be scaled to image_size
    assert boxes.max() > 1.0, "Boxes should be in pixel format, not normalized"
    # Note: DETR boxes in CXCYWH format can extend outside image bounds after conversion
    # This is expected behavior, so we don't enforce strict bounds checking here


def test_detr_wrapper_init():
    """Test that DetrBoxesModelWrapper can be initialized."""

    # Create a simple mock model
    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return {
                "pred_logits": torch.randn(batch_size, 100, 92),
                "pred_boxes": torch.rand(batch_size, 100, 4),
            }

    model = MockModel()
    image_size = (800, 800)
    wrapper = DetrBoxesModelWrapper(model, image_size=image_size)
    assert wrapper is not None


def test_detr_wrapper_forward():
    """Test DetrBoxesModelWrapper forward pass."""

    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return {
                "pred_logits": torch.randn(batch_size, 100, 92),
                "pred_boxes": torch.rand(batch_size, 100, 4),
            }

    model = MockModel()
    image_size = (800, 800)
    wrapper = DetrBoxesModelWrapper(model, image_size=image_size)

    # Test with dummy input
    x = torch.randn(2, 3, 224, 224)
    results = wrapper(x)

    assert isinstance(results, list)
    assert len(results) == 2
