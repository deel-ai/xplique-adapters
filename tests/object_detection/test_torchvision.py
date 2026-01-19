import torch

from xplique_adapters.object_detection.torch import (
    TorchvisionBoxesModelWrapper,
    TorchvisionBoxFormatter,
)


def test_torchvision_formatter_init():
    """Test that TorchvisionBoxFormatter can be initialized."""
    nb_classes = 20
    formatter = TorchvisionBoxFormatter(nb_classes)
    assert formatter is not None
    assert formatter.nb_classes == nb_classes


def test_torchvision_formatter_forward():
    """Test TorchvisionBoxFormatter with mock predictions."""
    nb_classes = 20
    formatter = TorchvisionBoxFormatter(nb_classes)

    # Mock Torchvision predictions (list of dicts, one per image)
    batch_size = 2
    num_boxes_per_image = [5, 3]

    predictions = []
    for num_boxes in num_boxes_per_image:
        predictions.append(
            {
                "boxes": torch.rand(num_boxes, 4) * 100,  # XYXY format in pixels
                "scores": torch.rand(num_boxes),
                "labels": torch.randint(0, nb_classes, (num_boxes,)),
            }
        )

    results = formatter.forward(predictions)

    assert isinstance(results, list)
    assert len(results) == batch_size
    assert results[0].shape[0] == num_boxes_per_image[0]
    assert results[1].shape[0] == num_boxes_per_image[1]


def test_torchvision_wrapper_init():
    """Test that TorchvisionBoxesModelWrapper can be initialized."""

    # Create a simple mock model
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return [
                {
                    "boxes": torch.rand(5, 4) * 100,
                    "scores": torch.rand(5),
                    "labels": torch.randint(0, 20, (5,)),
                }
                for _ in range(x.shape[0])
            ]

    model = MockModel()
    nb_classes = 20
    wrapper = TorchvisionBoxesModelWrapper(model, nb_classes)
    assert wrapper is not None


def test_torchvision_wrapper_forward():
    """Test TorchvisionBoxesModelWrapper forward pass."""

    class MockModel(torch.nn.Module):
        def forward(self, x):
            return [
                {
                    "boxes": torch.rand(5, 4) * 100,
                    "scores": torch.rand(5),
                    "labels": torch.randint(0, 20, (5,)),
                }
                for _ in range(x.shape[0])
            ]

    model = MockModel()
    nb_classes = 20
    wrapper = TorchvisionBoxesModelWrapper(model, nb_classes)

    # Test with dummy input
    x = torch.randn(2, 3, 224, 224)
    results = wrapper(x)

    assert isinstance(results, list)
    assert len(results) == 2
