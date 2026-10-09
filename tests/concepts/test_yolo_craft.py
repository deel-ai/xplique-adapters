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
    TorchMultiBoxTensor,
)
from xplique.wrappers import TorchWrapper

from xplique_adapters.concepts.torch.latent_data_yolo import (
    YoloExtractorBuilder,
    YoloExtractorMode,
)
from xplique_adapters.object_detection.torch import (
    YoloOneToManyFormatter,
    YoloOneToOneFormatter,
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

    # YOLO expects images to be resized to 640x640
    image_size = (640, 640)
    raw_image_resized = raw_image.resize(image_size)

    # Convert PIL image to tensor
    input_tensor = (
        torch.from_numpy(np.array(raw_image_resized)).permute(2, 0, 1).float() / 255.0
    )
    input_tensor = input_tensor.unsqueeze(0).to(device_param)

    return raw_image, input_tensor


def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="function", params=["yolo11n.pt", "yolo26n.pt"])
def yolo_model_version(request):
    return request.param


@pytest.fixture(scope="function")
def model_data(image_data, device_param, yolo_model_version):
    _, input_tensor = image_data
    from ultralytics import YOLO

    model = YOLO(yolo_model_version).to(device_param)
    detection_model = model.model
    detection_model.eval()
    detection_model = detection_model.to(device_param)
    processed_results = detection_model(input_tensor)
    return model, processed_results, yolo_model_version


def test_model_outputs(model_data):
    _, processed_results, _ = model_data
    # YOLO detection model returns tensors
    assert isinstance(processed_results, tuple), "YOLO should return tuple"
    assert len(processed_results) > 0, "Results should not be empty"


def test_gradients_model_original(image_data, model_data):
    _image, input_tensor = image_data
    model, _, _ = model_data
    detection_model = model.model
    check = check_model_gradients(detection_model, input_tensor)
    assert check, (
        "Model gradients should be computed successfully when calling the default YOLO model."
    )


@pytest.fixture(scope="function")
def dataset_classes(model_data):
    # Ultralytics COCO heads use contiguous class IDs; derive names from the
    # loaded model instead of the sparse torchvision-style table.
    model, _, _ = model_data
    names = model.names
    nb_classes = len(names)
    head = model.model.model[-1]
    assert head.nc == nb_classes
    CLASSES = [names[class_id] for class_id in range(nb_classes)]
    label_to_color = {
        "person": "r",
        "bicycle": "b",
        "car": "g",
        "motorcycle": "y",
        "truck": "orange",
    }
    return CLASSES, nb_classes, label_to_color


def _has_one_to_one_branch(detection_model):
    return getattr(detection_model.model[-1], "one2one_cv2", None) is not None


@pytest.fixture(scope="function")
def latent_extractor_data(dataset_classes, model_data, device_param):
    _classes_names, nb_classes, _label_to_color = dataset_classes
    model, _, _yolo_model_version = model_data
    detection_model = model.model

    # Select the branch from the head architecture, not from its end2end flag:
    # the flag's default changed across Ultralytics 8.4.x releases.
    if _has_one_to_one_branch(detection_model):
        mode = YoloExtractorMode.ONE_TO_ONE
    else:
        mode = YoloExtractorMode.ONE_TO_MANY

    latent_extractor = YoloExtractorBuilder.build(
        detection_model,
        extraction_layer=10,
        batch_size=1,
        nb_classes=nb_classes,
        mode=mode,
        device=str(device_param),
    )
    return latent_extractor


def test_latent_extractor(
    image_data, dataset_classes, latent_extractor_data, model_data
):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    _, _, yolo_model_version = model_data
    latent_extractor = latent_extractor_data

    assert nb_classes == 80
    assert classes_names[0] == "person"
    assert classes_names[23] == "giraffe"
    assert _has_one_to_one_branch(model_data[0].model) == (
        yolo_model_version == "yolo26n.pt"
    )

    results = latent_extractor(input_tensor)
    print("Latent Data YOLO:", results)
    assert isinstance(results, list), (
        "Results should be a list of MultiBoxTensor objects"
    )
    assert len(results) == 1, "Should have one result per batch item"
    # YOLO output shape varies based on the extraction layer
    assert isinstance(results[0], TorchMultiBoxTensor), (
        "Result should be a MultiBoxTensor"
    )
    assert results[0].shape[-1] == 5 + nb_classes
    assert torch.isfinite(results[0]).all()

    if yolo_model_version == "yolo26n.pt":
        probas = results[0][:, 5:]
        assert probas.shape[-1] == nb_classes
        torch.testing.assert_close(
            probas.sum(dim=-1),
            torch.ones(probas.shape[0], device=probas.device),
        )

    filtered_results = results[0].filter(confidence=0.5)
    plot_image_detections(image, filtered_results, classes_names, label_to_color)


def test_one_to_one_requires_one_to_one_branch():
    """ONE_TO_ONE is rejected for heads without one-to-one layers (e.g. YOLO11)."""
    from ultralytics import YOLO

    detection_model = YOLO("yolo11n.yaml").model
    head = detection_model.model[-1]
    assert getattr(head, "one2one_cv2", None) is None
    with pytest.raises(ValueError, match="one-to-one branches"):
        YoloExtractorBuilder.build(
            detection_model,
            extraction_layer=10,
            nb_classes=head.nc,
            mode=YoloExtractorMode.ONE_TO_ONE,
            device="cpu",
        )


def _yolo26_reference(end2end, image):
    """Upstream evaluation output of a fresh YOLO26 model with ``end2end`` set."""
    from ultralytics import YOLO

    torch.manual_seed(0)
    model = YOLO("yolo26n.yaml").model.eval()
    model.end2end = end2end
    with torch.no_grad():
        upstream = model(image)
    return model, upstream


def _assert_input_gradients(extractor, image, nb_columns):
    differentiable_image = image.detach().requires_grad_(True)
    actual = extractor(differentiable_image)[0]
    actual[:, :nb_columns].sum().backward()
    assert differentiable_image.grad is not None
    assert torch.isfinite(differentiable_image.grad).all()
    assert differentiable_image.grad.abs().sum() > 0
    return actual


@pytest.mark.parametrize("end2end", [True, False])
def test_yolo26_one_to_one_matches_upstream_and_keeps_gradients(end2end):
    """ONE_TO_ONE matches upstream end-to-end inference whatever the caller's flag."""
    image = torch.rand(1, 3, 64, 64)
    model, _ = _yolo26_reference(end2end, image)
    head = model.model[-1]
    formatter = YoloOneToOneFormatter(nb_classes=head.nc)

    extractor = YoloExtractorBuilder.build(
        model,
        extraction_layer=10,
        nb_classes=head.nc,
        mode=YoloExtractorMode.ONE_TO_ONE,
        device="cpu",
    )
    # The caller's model is untouched: no bindings, same end2end flag.
    assert extractor.model is not model
    assert not hasattr(model, "g") and not hasattr(model, "h")
    assert bool(head.end2end) == end2end
    copied_head = extractor.model.model[-1]
    assert copied_head.forward.__func__ is type(copied_head).forward

    _, upstream = _yolo26_reference(True, image)
    assert upstream[0].shape[-1] == 6
    expected = formatter(upstream)[0]

    actual = _assert_input_gradients(extractor, image, nb_columns=5)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("end2end", [True, False])
def test_yolo26_one_to_many_matches_upstream_and_keeps_gradients(end2end):
    """ONE_TO_MANY on a YOLO26 head matches upstream dense inference."""
    image = torch.rand(1, 3, 64, 64)
    model, _ = _yolo26_reference(end2end, image)
    head = model.model[-1]

    extractor = YoloExtractorBuilder.build(
        model,
        extraction_layer=10,
        mode=YoloExtractorMode.ONE_TO_MANY,
        device="cpu",
    )
    assert bool(head.end2end) == end2end

    _, upstream = _yolo26_reference(False, image)
    assert upstream[0].shape[1] == 4 + head.nc
    expected = YoloOneToManyFormatter()(upstream)[0]

    actual = _assert_input_gradients(extractor, image, nb_columns=5)
    torch.testing.assert_close(actual, expected)


def test_yolo26_example_one_to_one_cpu_path():
    """The attribution example builds a CPU ONE_TO_ONE extractor for YOLO26."""
    from examples import run_object_detection_attributions as example

    config = {**example.MODEL_CONFIGS["yolo26"], "model_path": "yolo26n.yaml"}
    model, _ = example.load_model("yolo26", config, torch.device("cpu"))
    head = model.model.model[-1]
    assert _has_one_to_one_branch(model.model)

    extractor = example.create_wrapper(
        "yolo26", model, config, torch.device("cpu"), use_raw_wrapper=True
    )
    assert extractor.device == torch.device("cpu")
    output = extractor(torch.rand(1, 3, 64, 64))[0]
    assert output.shape[-1] == 5 + head.nc
    assert torch.isfinite(output).all()
    probas = output[:, 5:]
    assert probas.shape[-1] == head.nc
    torch.testing.assert_close(probas.sum(dim=-1), torch.ones(len(probas)))


@pytest.mark.parametrize("end2end", [True, False])
def test_yolo26_fused_head_preserves_outputs_and_validates_branches(end2end):
    image = torch.rand(1, 3, 64, 64)
    model, _ = _yolo26_reference(end2end, image)
    model.fuse(verbose=False)
    head = model.model[-1]
    mode = YoloExtractorMode.ONE_TO_ONE if end2end else YoloExtractorMode.ONE_TO_MANY
    formatter = (
        YoloOneToOneFormatter(nb_classes=head.nc)
        if end2end
        else YoloOneToManyFormatter()
    )
    with torch.no_grad():
        expected = formatter(model(image))[0]
    extractor = YoloExtractorBuilder.build(
        model,
        extraction_layer=10,
        nb_classes=head.nc,
        mode=mode,
        device="cpu",
    )
    actual = _assert_input_gradients(extractor, image, nb_columns=5)
    torch.testing.assert_close(actual, expected)

    if head.cv2 is None:
        with pytest.raises(ValueError, match="removed.*fused"):
            YoloExtractorBuilder.build(model, extraction_layer=10, device="cpu")
    if getattr(head, "one2one_cv2", None) is None:
        with pytest.raises(ValueError, match="one-to-one branches"):
            YoloExtractorBuilder.build(
                model,
                extraction_layer=10,
                nb_classes=head.nc,
                mode=YoloExtractorMode.ONE_TO_ONE,
                device="cpu",
            )


def test_yolo26_extractor_preserves_caller_training_state():
    image = torch.rand(1, 3, 64, 64)
    model, _ = _yolo26_reference(False, image)
    model.train()
    # Mixed modes must also survive construction and attribution.
    model.model[0].eval()
    original_modes = [module.training for module in model.modules()]
    extractor = YoloExtractorBuilder.build(
        model,
        extraction_layer=10,
        nb_classes=model.model[-1].nc,
        mode=YoloExtractorMode.ONE_TO_ONE,
        device="cpu",
    )
    _assert_input_gradients(extractor, image, nb_columns=5)
    assert [module.training for module in model.modules()] == original_modes
    assert not hasattr(model, "g") and not hasattr(model, "h")
    assert model.model[-1].end2end is False


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
    confidence_threshold = 0.2

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

    # YOLO outputs can have negative values in feature maps, so we use oc_semi_nmf
    from overcomplete.optimization import SemiNMF
    from xplique.concepts.torch.factorizer import OvercompleteFactorizer

    factorizer = OvercompleteFactorizer(
        optimizer_class=SemiNMF, nb_concepts=10, device=device_param
    )

    craft = Craft(
        latent_extractor=latent_extractor,
        number_of_concepts=10,
        device=str(device_param),
        factorizer=factorizer,
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
    # YOLO spatial dimensions depend on extraction layer
    assert len(coeffs_u.shape) == 4, "Coeffs should be 4D tensor"
    assert coeffs_u.shape[0] == 1, "Batch dimension should be 1"
    assert coeffs_u.shape[-1] == 10, "Should have 10 concepts"

    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, TorchMultiBoxTensor), (
        "Decoded result should be an MultiBoxTensor directly"
    )
    print(result.shape)
    assert isinstance(result, TorchMultiBoxTensor), "Result should be MultiBoxTensor"

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
    # YOLO output shape varies, just check it's not empty
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
