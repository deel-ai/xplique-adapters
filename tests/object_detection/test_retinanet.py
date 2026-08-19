import numpy as np
import pytest
import tensorflow as tf

from xplique_adapters.concepts.tf.latent_data_retinanet import (
    TfLatentDataRetinanet,
    _decode_retinanet_predictions,
)
from xplique_adapters.object_detection.tf import (
    RetinaNetBoxesModelWrapper,
    RetinaNetProcessedBoxFormatter,
)


def _decoded_predictions(batch_size=1, num_boxes=2):
    return {
        "boxes": tf.constant(
            [[[10.0, 20.0, 100.0, 40.0], [0.0, 0.0, 5.0, 5.0]]] * batch_size
        ),
        "confidence": tf.constant([[0.8, 0.2]] * batch_size),
        "classes": tf.constant([[1, 0]] * batch_size, dtype=tf.int32),
        "num_detections": tf.constant([num_boxes] * batch_size, dtype=tf.int32),
    }


def test_retinanet_formatter_requires_decoded_fields():
    formatter = RetinaNetProcessedBoxFormatter(2, image_size=(100, 200))

    with pytest.raises(ValueError, match="missing required fields"):
        formatter({"boxes": tf.zeros((1, 1, 4))})


def test_retinanet_formatter_removes_padding_and_converts_rectangular_boxes():
    predictions = _decoded_predictions(num_boxes=1)
    predictions["classes"] = tf.constant([[1, -1]], dtype=tf.int32)
    formatter = RetinaNetProcessedBoxFormatter(2, image_size=(100, 200))

    results = formatter(predictions)

    assert len(results) == 1
    assert results[0].shape == (1, 7)
    np.testing.assert_allclose(
        results[0].boxes().numpy(), [[0.05, 0.2, 0.55, 0.6]], rtol=1e-6
    )
    np.testing.assert_allclose(results[0].probas().numpy(), [[0.0, 1.0]])


def test_retinanet_formatter_accepts_per_class_scores():
    predictions = _decoded_predictions(num_boxes=2)
    predictions["scores"] = tf.constant([[[0.1, 0.9], [0.7, 0.3]]])
    formatter = RetinaNetProcessedBoxFormatter(2, image_size=(100, 200))

    result = formatter(predictions)[0]

    np.testing.assert_allclose(result.probas().numpy(), [[0.1, 0.9], [0.7, 0.3]])


def test_retinanet_formatter_rejects_invalid_score_shape():
    predictions = _decoded_predictions(num_boxes=1)
    predictions["scores"] = tf.zeros((1, 2, 3))
    formatter = RetinaNetProcessedBoxFormatter(2, image_size=(100, 200))

    with pytest.raises(ValueError, match="per-class probabilities"):
        formatter(predictions)


def test_retinanet_formatter_requires_dimensions_for_pixel_conversion():
    with pytest.raises(ValueError, match="image size"):
        RetinaNetProcessedBoxFormatter(2)(_decoded_predictions(num_boxes=1))


def test_retinanet_wrapper_decoded_mode_forwards_kwargs():
    calls = []

    def model(x, **kwargs):
        calls.append(kwargs)
        return _decoded_predictions(batch_size=x.shape[0], num_boxes=1)

    wrapper = RetinaNetBoxesModelWrapper(
        model, 2, image_size=(100, 200), prediction_mode="decoded"
    )
    results = wrapper(tf.zeros((2, 100, 200, 3)), training=True)

    assert len(results) == 2
    assert calls == [{"training": True}]

    wrapper.output_as_list = False
    padded_results = wrapper(tf.zeros((2, 100, 200, 3)), training=True)
    assert isinstance(padded_results, tf.Tensor)
    assert padded_results.shape == (2, 1, 7)
    assert calls == [{"training": True}, {"training": True}]


def test_retinanet_wrapper_raw_mode_decodes_once_and_preserves_gradients():
    class RawModel:
        bounding_box_format = "xyxy"

        def __init__(self):
            self.calls = []
            self.decode_calls = 0

        def __call__(self, x, **kwargs):
            self.calls.append(kwargs)
            return {"raw": x}

        def decode_predictions(self, predictions, images):
            self.decode_calls += 1
            base = tf.reshape(predictions["raw"][:, 0, 0, 0], (-1, 1, 1))
            boxes = tf.concat(
                [
                    tf.zeros_like(base),
                    tf.zeros_like(base),
                    base + 100.0,
                    base + 40.0,
                ],
                axis=-1,
            )
            return {
                "boxes": boxes,
                "confidence": tf.ones((tf.shape(images)[0], 1)),
                "classes": tf.zeros((tf.shape(images)[0], 1), dtype=tf.int32),
                "num_detections": tf.ones((tf.shape(images)[0],), dtype=tf.int32),
            }

    model = RawModel()
    wrapper = RetinaNetBoxesModelWrapper(model, 1, image_size=(100, 200))
    inputs = tf.Variable(tf.ones((2, 1, 1, 1)))

    with tf.GradientTape() as tape:
        result = wrapper(inputs)[0]
        loss = tf.reduce_sum(result.tensor)
    gradient = tape.gradient(loss, inputs)

    assert model.calls == [{"training": False}]
    assert model.decode_calls == 1
    assert gradient is not None


def test_retinanet_wrapper_raw_mode_requires_decoder():
    wrapper = RetinaNetBoxesModelWrapper(
        lambda x, **kwargs: _decoded_predictions(),
        2,
        image_size=(100, 200),
    )

    with pytest.raises(AttributeError, match="decode_predictions"):
        wrapper(tf.zeros((1, 100, 200, 3)))


def test_tf_retinanet_latent_data_rebatches_and_checks_all_features():
    latent_data = TfLatentDataRetinanet(
        {
            "P3": tf.ones((1, 2, 2, 1)),
            "P4": np.ones((1, 1, 1, 1), dtype=np.float32),
        },
        tf.constant([100, 200, 3]),
        index_activations=0,
    )
    replacement = tf.ones((3, 2, 2, 1)) * 2

    latent_data.set_activations(replacement)

    assert latent_data.resnet_features["P3"].shape[0] == 3
    assert latent_data.resnet_features["P4"].shape[0] == 3
    np.testing.assert_allclose(latent_data.get_activations(as_numpy=False), 2.0)

    @tf.function(input_signature=[tf.TensorSpec((None, 2, 2, 1), tf.float32)])
    def set_dynamic(values):
        latent_data.set_activations(values)
        return tf.shape(latent_data.resnet_features["P4"])

    assert tuple(set_dynamic(tf.ones((2, 2, 2, 1))).numpy()) == (2, 1, 1, 1)

    invalid_data = TfLatentDataRetinanet(
        {"P4": tf.constant([[[[-1.0]]]])}, tf.constant([100, 200, 3])
    )
    with pytest.raises(ValueError, match="P4"):
        invalid_data.check_features_positive()


def test_tf_retinanet_latent_decoder_uses_model_format_without_decoder():
    class AnchorGenerator:
        bounding_box_format = "xyxy"

        def __call__(self, image_shape):
            assert tuple(image_shape) == (100, 200, 3)
            return {"P3": tf.constant([[0.0, 0.0, 100.0, 50.0]])}

    class Model:
        bounding_box_format = "xywh"
        anchor_generator = AnchorGenerator()

    predictions = _decode_retinanet_predictions(
        Model(),
        tf.zeros((1, 1, 2)),
        tf.zeros((1, 1, 4)),
        tf.constant([100, 200, 3]),
    )

    assert "prediction_decoder" not in Model.__dict__
    np.testing.assert_allclose(predictions["boxes"].numpy(), [[[0.0, 0.0, 0.5, 0.5]]])
