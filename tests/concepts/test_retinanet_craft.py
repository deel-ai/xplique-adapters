import os
import pprint

import numpy as np
import pytest
import tensorflow as tf
import xplique
from keras_cv.models import RetinaNet
from PIL import Image
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.concepts import HolisticCraftTf as Craft
from xplique.concepts import PartialExplainer
from xplique.plots import plot_attributions, plot_image_detections
from xplique.utils_functions.common.tf.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.tf.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.multi_box_tensor import (
    TfMultiBoxTensor as MultiBoxTensor,
)

from xplique_adapters.concepts.tf.latent_data_retinanet import RetinaNetExtractorBuilder
from xplique_adapters.object_detection.tf import RetinaNetBoxesModelWrapper

pp = pprint.PrettyPrinter(indent=4)
print(tf.__version__)

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=["cpu", "gpu"])
def device_param(request):
    if request.param == "gpu" and not tf.config.list_physical_devices("GPU"):
        pytest.skip("GPU not available")
    elif request.param == "cpu":
        tf.config.set_visible_devices([], "GPU")
        print("[test] GPU disabled: running on CPU")
    return request.param


@pytest.fixture(scope="function")
def image_data(device_param):
    device_name = f"/{device_param.upper()}:0"
    with tf.device(device_name):
        # Load and preprocess the image
        raw_image = Image.open(os.path.join(test_dir, "img.jpg"))

        def resize_image(image, target_height=640):
            width, height = image.size
            target_width = int(width * target_height / height)
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        resized_image = resize_image(raw_image)
        image = resized_image.crop((100, 0, 740, 640))

        img_np = np.array(image, dtype=np.float32)
        input_tensor = tf.expand_dims(img_np, axis=0)
        return image, input_tensor


def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 640)
    assert image.size == expected_size


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    device_name = f"/{device_param.upper()}:0"
    with tf.device(device_name):
        _, input_tensor = image_data
        model = RetinaNet.from_preset("retinanet_resnet50_pascalvoc")
        processed_results = model.predict(input_tensor)
        return model, processed_results


def test_model_outputs(model_data):
    _, processed_results = model_data
    assert list(processed_results.keys()) == [
        "boxes",
        "confidence",
        "classes",
        "num_detections",
    ]


def test_retinanet_wrapper_real_raw_and_decoded_paths(image_data, model_data):
    _image, input_tensor = image_data
    model, _processed_results = model_data
    image_size = tuple(input_tensor.shape[1:3])

    raw_wrapper = RetinaNetBoxesModelWrapper(model, nb_classes=20, image_size=image_size)
    raw_results = raw_wrapper(input_tensor)

    class DecodedModel:
        bounding_box_format = model.bounding_box_format
        prediction_decoder = model.prediction_decoder

        def __call__(self, images, **kwargs):
            return model.decode_predictions(model(images, training=False), images)

    decoded_wrapper = RetinaNetBoxesModelWrapper(
        DecodedModel(),
        nb_classes=20,
        image_size=image_size,
        prediction_mode="decoded",
    )
    decoded_results = decoded_wrapper(input_tensor)

    assert len(raw_results) == len(decoded_results) == 1
    assert raw_results[0].shape[-1] == decoded_results[0].shape[-1] == 25


def test_gradients_predict(image_data, model_data):
    _image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model.predict, input_tensor)
    assert check is False, (
        "Model gradients should not be computed successfully in normal / predict mode."
    )


def test_gradients_model_call(image_data, model_data):
    _image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check, (
        "Model gradients should be computed successfully when calling the model."
    )


@pytest.fixture(scope="module")
def dataset_classes():
    CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
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
    device_name = f"/{device_param.upper()}:0"
    with tf.device(device_name):
        _classes_names, nb_classes, _label_to_color = dataset_classes
        model, _ = model_data

        latent_extractor = RetinaNetExtractorBuilder.build(model, nb_classes)
        return latent_extractor


def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    # Test in tensor mode
    latent_extractor.output_as_tensor = True
    results = latent_extractor(input_tensor)
    print("Latent Data Retinanet (tensor mode):", results.shape)
    assert results.shape == (1, 76725, 25), (
        "Latent data shape should be (1, 76725, 25)."
    )

    # Test in list mode
    latent_extractor.output_as_list = True
    results_list = latent_extractor(input_tensor)
    print("Latent Data Retinanet (list mode):", type(results_list))
    assert isinstance(results_list, list), (
        "Should return a list of MultiBoxTensor objects in list mode"
    )
    assert len(results_list) == 1, "Should have one result per batch item"
    assert isinstance(results_list[0], MultiBoxTensor), (
        "First element should be a MultiBoxTensor"
    )
    assert results_list[0].shape == (76725, 25), (
        "MultiBoxTensor shape should be (76725, 25)."
    )

    filtered_results_list = results_list[0].filter(confidence=0.85)
    box_type = BoxType(BoxFormat.XYXY, is_normalized=True)
    plot_image_detections(
        image, filtered_results_list, classes_names, label_to_color, box_type=box_type
    )


def test_latent_extractor_gradients(image_data, latent_extractor_data):
    _image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Set to tensor mode before checking gradients
    latent_extractor.output_as_tensor = True
    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."


def test_latent_extractor_saliency(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    # Set to tensor mode for operator compatibility
    latent_extractor.output_as_tensor = True
    operator = xplique.Tasks.OBJECT_DETECTION_BOX_PROBA

    # Get targets in list mode first, then filter
    latent_extractor.output_as_list = True
    targets = latent_extractor(input_tensor)
    box_to_explain = targets[0].filter(
        confidence=0.9, class_id=classes_names.index("car")
    )

    # Set back to tensor mode for the explainer
    latent_extractor.output_as_tensor = True
    box_to_explain = box_to_explain.to_batched_tensor()

    explainer = Saliency(latent_extractor, operator=operator, batch_size=None)
    explanation = explainer.explain(input_tensor, targets=box_to_explain)

    # Check explanation shape (should match input image dimensions)
    assert explanation.shape == (1, 640, 640, 1), (
        f"Expected shape (1, 640, 640, 1), got {explanation.shape}"
    )

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
    device_name = f"/{device_param.upper()}:0"
    with tf.device(device_name):
        _image, input_tensor = image_data
        latent_extractor = latent_extractor_data

        # Ensure latent extractor is in list mode for CRAFT
        latent_extractor.output_as_list = True
        craft = Craft(latent_extractor=latent_extractor, number_of_concepts=10)
        craft.fit(input_tensor)
        return craft


def test_craft_reencode(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    print("nb encoded data:", len(encoded_data))
    latent_data, coeffs_u = encoded_data[0]
    print(coeffs_u.shape)
    assert coeffs_u.shape == (1, 20, 20, 10), (
        "Latent data shape should be (1, 20, 20, 10)."
    )

    decoded_data = craft.decode(latent_data, coeffs_u)
    print(f"decoded_data type = {type(decoded_data)}")
    assert isinstance(decoded_data, MultiBoxTensor), (
        "Should return an MultiBoxTensor directly"
    )
    assert decoded_data.shape == (76725, 25), (
        "MultiBoxTensor shape should be (76725, 25)."
    )

    filtered_decoded_data = decoded_data.filter(confidence=0.85)
    box_type = BoxType(BoxFormat.XYXY, is_normalized=True)
    plot_image_detections(
        image, filtered_decoded_data, classes_names, label_to_color, box_type=box_type
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
    assert explanation.shape[1:3] == (20, 20), (
        "Should match coeffs_u spatial dimensions"
    )
    assert explanation.shape[3] == 10, "Should match number of concepts"

    # Test estimate_importance_gradient_input
    importances_gi = craft.estimate_importance(
        input_tensor, operator, class_id, confidence=0.6, method="gradient_input"
    )
    assert importances_gi.shape == (10,), (
        "Should return importance scores for each concept"
    )

    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for 'person': {order}")
    assert np.all(order == np.array([3, 1, 7, 4, 2, 8, 0, 5, 9, 6])), (
        "Concepts order is not as expected"
    )
    craft.display_images_per_concept(
        input_tensor, order=order, filter_percentile=80, clip_percentile=5
    )


def test_craft_decoder_modes(image_data, dataset_classes, craft_data):
    """Test that the decoder works in both tensor and list modes"""
    _image, input_tensor = image_data
    _classes_names, _nb_classes, _label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    latent_data, coeffs_u = encoded_data[0]
    decoder = craft.make_concept_decoder(latent_data)

    # Decoder should always return tensor (unified behavior with PyTorch)
    output_tensor = decoder(coeffs_u)
    assert hasattr(output_tensor, "shape"), "Decoder should always return a tensor"
    assert output_tensor.shape == (1, 76725, 25), "Should have correct tensor shape"

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


def test_multibox_tensor_filter(image_data, dataset_classes, latent_extractor_data):
    """Test MultiBoxTensor filtering functionality"""
    _image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    # Get results in list mode to access MultiBoxTensor
    latent_extractor.output_as_list = True
    targets = latent_extractor(input_tensor)
    nbc_tensor = targets[0]  # targets is a list, get the first MultiBoxTensor

    # Test filtering by class_id and confidence
    filtered = nbc_tensor.filter(class_id=classes_names.index("person"), confidence=0.5)
    assert hasattr(filtered, "shape"), "Filtered result should have shape attribute"
    assert len(filtered.shape) == 2, "Filtered result should be 2D (boxes, features)"
    assert filtered.shape[1] == 25, "Should preserve feature dimension"

    print(f"Original: {nbc_tensor.shape}, Filtered: {filtered.shape}")

    # Test filtering with high confidence (should return fewer boxes)
    filtered_high = nbc_tensor.filter(
        class_id=classes_names.index("person"), confidence=0.9
    )
    assert filtered_high.shape[0] <= filtered.shape[0], (
        "Higher confidence should return fewer or equal boxes"
    )


def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode(differentiable=True) preserves gradients through the encoding pipeline."""
    _image, input_tensor = image_data
    craft = craft_data

    # Test differentiable encoding with gradient tape
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)

        # Encode with differentiable mode
        encoded_data = craft.encode(input_tensor, differentiable=True)
        _latent_data, coeffs_u = encoded_data[0]

        # Verify coeffs_u is a tensor
        assert isinstance(coeffs_u, tf.Tensor), (
            "coeffs_u should be a tf.Tensor in differentiable mode"
        )

        # Create a simple loss
        loss = tf.reduce_sum(coeffs_u)

    # Check gradient flow
    gradients = tape.gradient(loss, input_tensor)

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert tf.reduce_sum(tf.abs(gradients)) > 0, "Gradients should be non-zero"
