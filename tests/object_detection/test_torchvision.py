from contextlib import redirect_stdout
from collections import OrderedDict
from io import StringIO

import torch
import torchvision
import xplique_adapters.concepts.torch.latent_data_retinanet as retinanet_module

from xplique_adapters.concepts.torch.latent_data_retinanet import (
    LatentDataRetinanet,
    RetinaNetExtractorBuilder,
)
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


def test_retinanet_latent_data_rebatches_features_and_metadata():
    tensors = torch.zeros((1, 3, 8, 8))
    images = torchvision.models.detection.image_list.ImageList(tensors, [(8, 8)])
    features = OrderedDict(
        {
            "0": torch.ones((1, 2, 2, 2)),
            "1": torch.full((1, 1, 1, 2), 2.0),
        }
    )
    latent_data = LatentDataRetinanet(images, [(8, 8)], features)
    image_tensor = latent_data.images.tensors

    latent_data.set_activations(torch.full((3, 1, 1, 2), 4.0))

    assert len(latent_data) == 3
    assert latent_data.features["1"].shape == (3, 2, 1, 1)
    assert latent_data.features["0"].shape == (3, 2, 2, 2)
    assert torch.all(latent_data.features["0"] == 1.0)
    assert len(latent_data.images.image_sizes) == 3
    assert len(latent_data.original_image_sizes) == 3
    assert latent_data.images.tensors is image_tensor

    latent_data.set_activations(torch.full((2, 1, 1, 2), 5.0))

    assert len(latent_data) == 2
    assert len(latent_data.images.image_sizes) == 2
    assert len(latent_data.original_image_sizes) == 2
    assert latent_data.images.tensors is image_tensor


def test_retinanet_builder_uses_canonical_name():
    assert RetinaNetExtractorBuilder.__name__ == "RetinaNetExtractorBuilder"


def test_retinanet_builder_does_not_write_to_stdout(monkeypatch):
    """
    Test that the builder does no longer print to stdout, and
    uses logging instead
    """
    class Model:
        pass

    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(retinanet_module, "TorchLatentExtractor", FakeExtractor)
    output = StringIO()
    with redirect_stdout(output):
        RetinaNetExtractorBuilder.build(Model(), device="cpu")

    assert output.getvalue() == ""
