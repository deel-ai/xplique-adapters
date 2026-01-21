import os
import pprint

import numpy as np
import pytest
import torch
import torchvision.transforms as T
import xplique
from PIL import Image
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.concepts import HolisticCraftTorch as Craft
from xplique.concepts import PartialExplainer
from xplique.plots import plot_attributions, plot_image_detections
from xplique.utils_functions.common.torch.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.torch.multi_box_tensor import (
    TorchMultiBoxTensor as MultiBoxTensor,
)
from xplique.wrappers import TorchWrapper

from xplique_adapters.concepts.torch.latent_data_detr import DetrExtractorBuilder

pp = pprint.PrettyPrinter(indent=4)
print(torch.__version__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="function", params=["cpu", "cuda"])
def device_param(request):
    device_str = request.param
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device


@pytest.fixture(scope="function")
def image_data(device_param):
    # Load and preprocess the image
    raw_image = Image.open(os.path.join(test_dir, "img.jpg"))

    # standard PyTorch mean-std input image normalization
    transform = T.Compose(
        [
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # preprocess and batch the image
    input_tensor = transform(raw_image).unsqueeze(0).to(device_param)

    return raw_image, input_tensor


def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    _, input_tensor = image_data
    model = torch.hub.load(
        "facebookresearch/detr", "detr_resnet50", pretrained=True
    ).to(device_param)
    model.eval()
    processed_results = model(input_tensor)
    return model, processed_results


def test_model_outputs(model_data):
    _, processed_results = model_data
    assert list(processed_results.keys()) == ["pred_logits", "pred_boxes"]


def test_gradients_model_original(image_data, model_data):
    _image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert not check, (
        "Model gradients should be not computed successfully when calling the default Detr model."
    )


@pytest.fixture(scope="function")
def dataset_classes():
    # COCO classes
    CLASSES = [
        "N/A",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    nb_classes = len(CLASSES)
    label_to_color = {
        "person": "r",
        "bicycle": "b",
        "car": "g",
        "motorcycle": "y",
        "truck": "orange",
    }
    return CLASSES, nb_classes, label_to_color


@pytest.fixture(scope="function")
def latent_extractor_data(dataset_classes, model_data, device_param):
    _classes_names, _nb_classes, _label_to_color = dataset_classes
    model, _ = model_data

    latent_extractor = DetrExtractorBuilder.build(
        model, device=str(device_param), batch_size=2
    )
    return latent_extractor


def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    results = latent_extractor(input_tensor)
    print("Latent Data Detr:", results)
    assert isinstance(results, list), (
        "Results should be a list of MultiBoxTensor objects"
    )
    assert len(results) == 1, "Should have one result per batch item"
    assert results[0].shape == torch.Size([100, 96]), (
        "MultiBoxTensor shape should be [100, 96]."
    )

    filtered_results = results[0].filter(confidence=0.85)
    plot_image_detections(image, filtered_results, classes_names, label_to_color)


def test_latent_extractor_gradients(image_data, latent_extractor_data):
    _image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."


def test_latent_extractor_saliency(
    image_data, dataset_classes, latent_extractor_data, device_param
):
    image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    targets = latent_extractor(input_tensor)
    filtered_targets = targets[0].filter(
        confidence=0.9, class_id=classes_names.index("car")
    )
    assert filtered_targets.shape == torch.Size([2, 96])
    box_to_explain = np.expand_dims(filtered_targets.detach().cpu().numpy(), axis=0)

    latent_extractor.output_as_tensor = True
    input_tensor_tf_dim = input_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

    torch_wrapped_model = TorchWrapper(
        latent_extractor, device=device_param, is_channel_first=True
    )

    explainer = Saliency(
        torch_wrapped_model, operator=xplique.Tasks.OBJECT_DETECTION, batch_size=1
    )
    explanation = explainer.explain(input_tensor_tf_dim, targets=box_to_explain)

    plot_attributions(
        explanation,
        [np.array(image)],
        img_size=6.0,
        cmap="jet",
        alpha=0.3,
        absolute_value=False,
        clip_percentile=0.5,
    )


@pytest.fixture(scope="function")
def craft_data(image_data, latent_extractor_data, device_param):
    _image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    craft = Craft(
        latent_extractor=latent_extractor,
        number_of_concepts=10,
        device=str(device_param),
    )
    craft.fit(input_tensor)
    return craft


def test_craft_reencode(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    craft = craft_data

    # The encode method now returns a list of tuples [(latent_data, coeffs_u), ...]
    latent_data_coeffs_u_list = craft.encode(input_tensor)

    # Should have one tuple per image in the batch
    assert len(latent_data_coeffs_u_list) == input_tensor.shape[0], (
        f"Expected {input_tensor.shape[0]} tuples, got {len(latent_data_coeffs_u_list)}"
    )

    # Get the first tuple (since we're only testing with one image)
    latent_data, coeffs_u = latent_data_coeffs_u_list[0]

    print(coeffs_u.shape)
    assert coeffs_u.shape == (1, 25, 25, 10), (
        "Latent data shape should be (1, 25, 25, 10)."
    )

    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, MultiBoxTensor), (
        "Decoded result should be an MultiBoxTensor directly"
    )
    print(result.shape)
    assert result.shape == torch.Size([100, 96]), (
        "Decoded MultiBoxTensor shape should be [100, 96]."
    )

    filtered_result = result.filter(confidence=0.85)
    plot_image_detections(image, filtered_result, classes_names, label_to_color)


def test_craft_decoder_modes(image_data, dataset_classes, craft_data, device_param):
    _image, input_tensor = image_data
    _classes_names, _nb_classes, _label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    latent_data, coeffs_u = encoded_data[0]
    decoder = craft.make_concept_decoder(latent_data)

    # Decoder should always return tensor (unified behavior with TensorFlow)
    output_tensor = decoder(coeffs_u)
    assert hasattr(output_tensor, "shape"), "Decoder should always return a tensor"
    assert output_tensor.shape == torch.Size([1, 100, 96]), (
        f"Expected shape [1, 100, 96], got {output_tensor.shape}"
    )

    # For filtering, use decode directly to get MultiBoxTensor
    nbc_tensor = craft.decode(latent_data, coeffs_u)
    assert hasattr(nbc_tensor, "filter"), (
        "decode should return MultiBoxTensor with filter method"
    )

    # Simulate what a perturbation-based explainer (e.g. Sobol) does:
    # it stacks batch_size=2 perturbed copies of coeffs_u and calls the decoder once.
    coeffs_u_batch2 = np.concatenate(
        [coeffs_u, coeffs_u], axis=0
    )  # shape [2, 25, 25, 10]
    output_batch2 = decoder(coeffs_u_batch2)
    assert output_batch2.shape == torch.Size([2, 100, 96]), (
        f"Expected shape [2, 100, 96], got {output_batch2.shape}"
    )


def test_craft_concepts(image_data, craft_data):
    _image, input_tensor = image_data
    craft = craft_data
    craft.display_images_per_concept(
        input_tensor, order=None, filter_percentile=80, clip_percentile=5
    )


def test_craft_gradient_input(image_data, dataset_classes, craft_data):
    _image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
    craft = craft_data

    # Test compute_gradient_input
    operator = xplique.Tasks.OBJECT_DETECTION
    class_id = classes_names.index("person")
    partial_explainer = PartialExplainer(
        GradientInput,
        operator=operator,
        reducer=None,
    )
    explanation = craft.compute_explanation_per_concept(
        input_tensor,
        class_id=class_id,
        confidence=0.6,
        partial_explainer=partial_explainer,
    )

    # Verify explanation shape
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[1:3] == (25, 25), (
        "Should match coeffs_u spatial dimensions"
    )
    assert explanation.shape[3] == 10, "Should match number of concepts"

    importances_gi = craft.estimate_importance(
        input_tensor, operator, class_id, confidence=0.6, method="gradient_input"
    )

    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for 'person': {order}")
    assert np.all(order == np.array([2, 6, 5, 4, 7, 9, 8, 0, 3, 1])), (
        "Concepts order is not as expected"
    )
    craft.display_images_per_concept(
        input_tensor, order=order, filter_percentile=80, clip_percentile=5
    )


def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode(differentiable=True) preserves gradients through the encoding pipeline."""
    _image, input_tensor = image_data
    craft = craft_data

    # Ensure input has gradients enabled
    input_with_grad = input_tensor.clone().detach().requires_grad_(True)

    # Test differentiable encoding
    encoded_data = craft.encode(input_with_grad, differentiable=True)
    _latent_data, coeffs_u = encoded_data[0]

    # Verify coeffs_u is a tensor with gradients
    assert isinstance(coeffs_u, torch.Tensor), (
        "coeffs_u should be a torch.Tensor in differentiable mode"
    )
    assert coeffs_u.requires_grad, "coeffs_u should have gradients enabled"

    # Create a simple loss and check gradient flow
    loss = coeffs_u.sum()
    loss.backward()
    gradients = input_with_grad.grad

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert gradients.abs().sum() > 0, "Gradients should be non-zero"
