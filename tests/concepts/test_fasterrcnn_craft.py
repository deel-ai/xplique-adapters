import os
import pprint

import numpy as np
import pytest
import torch
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

from xplique_adapters.concepts.torch.latent_data_faster_rcnn import (
    FasterRcnnExtractorBuilder,
)

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

    # Load model weights for transforms
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    preprocess = weights.transforms()

    # Resize and preprocess the image
    image_size = (800, 800)
    if image_size is not None:
        raw_image_resized = raw_image.resize(image_size)
    else:
        raw_image_resized = raw_image

    # Apply the weights transforms
    input_tensor = preprocess(raw_image_resized).unsqueeze(0).to(device_param)

    return raw_image, input_tensor


def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    _, input_tensor = image_data
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model = model.to(device_param)
    processed_results = model(input_tensor)
    return model, processed_results


def test_model_outputs(model_data):
    _, processed_results = model_data
    # FasterRCNN returns a list of dicts with detection results
    assert isinstance(processed_results, list), "FasterRCNN should return a list"
    assert len(processed_results) > 0, "Results list should not be empty"
    assert isinstance(processed_results[0], dict), "Each result should be a dict"
    assert "boxes" in processed_results[0], "Results should contain 'boxes'"
    assert "labels" in processed_results[0], "Results should contain 'labels'"
    assert "scores" in processed_results[0], "Results should contain 'scores'"


def test_gradients_model_original(image_data, model_data):
    _image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check, (
        "Model gradients should be computed successfully when calling the default FasterRCNN model."
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
    _classes_names, nb_classes, _label_to_color = dataset_classes
    model, _ = model_data

    latent_extractor = FasterRcnnExtractorBuilder.build(
        model,
        device=str(device_param),
        nb_classes=nb_classes,
        extraction_location="resnet",
        extraction_layer=-1,
    )
    return latent_extractor


def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    results = latent_extractor(input_tensor)
    print("Latent Data FasterRCNN:", results)
    assert isinstance(results, list), (
        "Results should be a list of MultiBoxTensor objects"
    )
    assert len(results) == 1, "Should have one result per batch item"
    # FasterRCNN output shape varies based on the feature pyramid level
    assert isinstance(results[0], MultiBoxTensor), "Result should be a MultiBoxTensor"

    filtered_results = results[0].filter(confidence=0.5)
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
    _classes_names, _nb_classes, _label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    targets = latent_extractor(input_tensor)

    # Use a lower confidence threshold to ensure we get detections
    confidence_threshold = 0.3

    filtered_targets = targets[0].filter(confidence=confidence_threshold)

    # Ensure we have detections to explain
    assert len(filtered_targets) > 0, (
        f"No detections found at confidence threshold {confidence_threshold}"
    )

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
    # FasterRCNN spatial dimensions depend on extraction location and layer
    assert len(coeffs_u.shape) == 4, "Coeffs should be 4D tensor"
    assert coeffs_u.shape[0] == 1, "Batch dimension should be 1"
    assert coeffs_u.shape[-1] == 10, "Should have 10 concepts"

    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, MultiBoxTensor), (
        "Decoded result should be an MultiBoxTensor directly"
    )
    print(result.shape)
    assert isinstance(result, MultiBoxTensor), "Result should be MultiBoxTensor"

    filtered_result = result.filter(confidence=0.5)
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
    # FasterRCNN output shape varies, just check it's not empty
    assert output_tensor.shape[0] > 0, "Decoder output should not be empty"

    # For filtering, use decode directly to get MultiBoxTensor
    nbc_tensor = craft.decode(latent_data, coeffs_u)
    assert hasattr(nbc_tensor, "filter"), (
        "decode should return MultiBoxTensor with filter method"
    )

    # Simulate what a perturbation-based explainer (e.g. Sobol) does:
    # it stacks batch_size=2 perturbed copies of coeffs_u and calls the decoder once.
    coeffs_u_batch2 = np.concatenate([coeffs_u, coeffs_u], axis=0)
    output_batch2 = decoder(coeffs_u_batch2)
    assert output_batch2.shape[0] == 2, (
        f"Expected batch dimension 2, got {output_batch2.shape[0]}"
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
    confidence = 0.5
    partial_explainer = PartialExplainer(
        GradientInput,
        operator=operator,
        reducer=None,
    )
    explanation = craft.compute_explanation_per_concept(
        input_tensor,
        class_id=class_id,
        confidence=confidence,
        partial_explainer=partial_explainer,
    )

    # Verify explanation has correct dimensions
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[-1] == 10, "Should match number of concepts"

    importances_gi = craft.estimate_importance(
        input_tensor,
        operator,
        class_id,
        confidence=confidence,
        method="gradient_input",
    )

    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for 'person': {order}")
    assert len(order) == 10, "Should have 10 importance scores"
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
