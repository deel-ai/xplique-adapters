import os

import numpy as np
import pytest
import torch
from PIL import Image
from xplique.concepts import HolisticCraftTorch as Craft
from xplique.utils_functions.object_detection.torch.multi_box_tensor import (
    TorchMultiBoxTensor,
)

from xplique_adapters.concepts.torch.latent_data_yolox import (
    LatentDataYoloX,
    YoloXExtractorBuilder,
)
from xplique_adapters.object_detection.torch import YoloXBoxFormatter

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture(scope="function", params=["cpu", "cuda"])
def device_param(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(scope="function")
def image_data(device_param):
    raw_image = Image.open(os.path.join(test_dir, "img.jpg")).convert("RGB")
    resized_image = raw_image.resize((64, 64))
    input_tensor = (
        torch.from_numpy(np.array(resized_image, copy=True))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
    )
    return raw_image, input_tensor.to(device_param)


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    from yolox.exp.default.yolox_m import Exp

    _image, input_tensor = image_data
    model = Exp().get_model().to(device_param).eval()
    with torch.no_grad():
        predictions = model(input_tensor)
    return model, predictions


@pytest.mark.yolox
def test_model_outputs(image_data, model_data):
    _image, input_tensor = image_data
    _model, predictions = model_data

    assert predictions.shape == (input_tensor.shape[0], 84, 85)
    assert torch.isfinite(predictions).all()


@pytest.fixture(scope="function")
def dataset_classes():
    from yolox.data.datasets.coco_classes import COCO_CLASSES

    class_names = list(COCO_CLASSES)
    label_to_color = {"person": "r", "car": "g", "giraffe": "y"}
    return class_names, len(class_names), label_to_color


@pytest.fixture(scope="function")
def latent_extractor_data(dataset_classes, model_data, device_param):
    _class_names, _num_classes, _label_to_color = dataset_classes
    model, _predictions = model_data
    return YoloXExtractorBuilder.build(
        model,
        extraction_layer=-1,
        batch_size=1,
        device=str(device_param),
    )


@pytest.mark.yolox
def test_latent_extractor(
    image_data, dataset_classes, latent_extractor_data, model_data
):
    _image, input_tensor = image_data
    class_names, num_classes, _label_to_color = dataset_classes

    results = latent_extractor_data(input_tensor)

    assert num_classes == 80
    assert class_names[0] == "person"
    assert class_names[23] == "giraffe"
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], TorchMultiBoxTensor)
    assert results[0].shape == (84, 5 + num_classes)
    assert torch.isfinite(results[0]).all()
    torch.testing.assert_close(results[0], YoloXBoxFormatter()(model_data[1])[0])


@pytest.mark.yolox
def test_latent_extractor_gradients(image_data, latent_extractor_data):
    _image, input_tensor = image_data
    differentiable_input = input_tensor.detach().requires_grad_(True)

    results = latent_extractor_data(differentiable_input)
    results[0][:, 4].sum().backward()

    assert differentiable_input.grad is not None
    assert torch.isfinite(differentiable_input.grad).all()
    assert differentiable_input.grad.abs().sum() > 0


@pytest.fixture(scope="function")
def craft_data(image_data, latent_extractor_data, device_param):
    from overcomplete.optimization import SemiNMF
    from xplique.concepts.torch.factorizer import OvercompleteFactorizer

    _image, input_tensor = image_data
    factorizer = OvercompleteFactorizer(
        optimizer_class=SemiNMF,
        nb_concepts=3,
        device=device_param,
    )
    craft = Craft(
        latent_extractor=latent_extractor_data,
        number_of_concepts=3,
        device=str(device_param),
        factorizer=factorizer,
    )
    craft.fit(input_tensor)
    return craft


@pytest.mark.yolox
def test_craft_fit_encode_decode(image_data, craft_data):
    _image, input_tensor = image_data
    craft = craft_data

    encoded = craft.encode(input_tensor)
    assert len(encoded) == 1
    latent_data, coefficients = encoded[0]
    assert coefficients.shape == (1, 2, 2, 3)
    assert np.isfinite(coefficients).all()

    decoded = craft.decode(latent_data, coefficients)
    assert isinstance(decoded, TorchMultiBoxTensor)
    assert decoded.shape == (84, 85)
    assert torch.isfinite(decoded).all()


class MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, inputs):
        return [
            self.conv(inputs),
            self.conv(torch.nn.functional.avg_pool2d(inputs, 2)),
            self.conv(torch.nn.functional.avg_pool2d(inputs, 4)),
        ]


class MockHead(torch.nn.Module):
    def forward(self, fpn_outs):
        predictions = []
        for feature in fpn_outs:
            value = feature.mean(dim=(1, 2, 3)).unsqueeze(1)
            boxes = torch.cat((value, value, value.abs() + 1, value.abs() + 1), dim=1)
            scores = value.sigmoid()
            probas = torch.cat((scores, 1 - scores), dim=1)
            predictions.append(torch.cat((boxes, scores, probas), dim=1))
        return torch.stack(predictions, dim=1)


class MockYoloX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()
        self.head = MockHead()


def test_latent_data_round_trip_and_expand_other_levels():
    features = [
        torch.randn(1, 4, 8 // (2**index), 8 // (2**index)) for index in range(3)
    ]
    latent_data = LatentDataYoloX(features, extraction_layer=1)
    activations = latent_data.get_activations(as_numpy=False)
    assert activations.shape == (1, 4, 4, 4)

    latent_data.set_activations(activations.expand(3, -1, -1, -1))

    assert latent_data.fpn_outs[1].shape == (3, 4, 4, 4)
    for index in (0, 2):
        torch.testing.assert_close(
            latent_data.fpn_outs[index], features[index].expand(3, -1, -1, -1)
        )
    assert len(latent_data) == 3
    assert latent_data.to(torch.device("cpu")).extraction_layer == 1


def test_extractor_rejects_invalid_layer_and_model():
    with pytest.raises(ValueError, match="extraction_layer"):
        YoloXExtractorBuilder.build(MockYoloX(), extraction_layer=3, device="cpu")
    with pytest.raises(TypeError, match="backbone.*head"):
        YoloXExtractorBuilder.build(torch.nn.Identity(), device="cpu")
