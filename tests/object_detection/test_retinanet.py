import tensorflow as tf

from xplique_adapters.object_detection.tf import (
    RetinaNetBoxesModelWrapper,
    RetinaNetProcessedBoxFormatter,
)


def test_retinanet_formatter_init():
    """Test that RetinaNetProcessedBoxFormatter can be initialized."""
    nb_classes = 20
    formatter = RetinaNetProcessedBoxFormatter(nb_classes)
    assert formatter is not None
    assert formatter.nb_classes == nb_classes


def test_retinanet_formatter_forward():
    """Test RetinaNetProcessedBoxFormatter with mock predictions."""
    nb_classes = 20
    image_size = (224, 224)
    formatter = RetinaNetProcessedBoxFormatter(nb_classes, image_size=image_size)

    # Mock RetinaNet predictions
    batch_size = 2
    num_boxes = 100

    predictions = {
        "boxes": tf.random.uniform((batch_size, num_boxes, 4), 0, 100),
        "confidence": tf.random.uniform((batch_size, num_boxes), 0, 1),
        "classes": tf.random.uniform(
            (batch_size, num_boxes), 0, nb_classes, dtype=tf.int32
        ),
    }

    results = formatter.forward(predictions)

    assert isinstance(results, list)
    assert len(results) == batch_size
    assert float(tf.reduce_max(results[0][:, :4])) > 1.0


def test_retinanet_wrapper_init():
    """Test that RetinaNetBoxesModelWrapper can be initialized."""

    # Create a simple mock callable
    def mock_model(x, **kwargs):
        batch_size = 2
        return {
            "boxes": tf.random.uniform((batch_size, 100, 4), 0, 100),
            "confidence": tf.random.uniform((batch_size, 100), 0, 1),
            "classes": tf.random.uniform((batch_size, 100), 0, 20, dtype=tf.int32),
            "num_detections": tf.constant([100, 100]),
        }

    nb_classes = 20
    image_size = (224, 224)
    wrapper = RetinaNetBoxesModelWrapper(mock_model, nb_classes, image_size=image_size)
    assert wrapper is not None


def test_retinanet_wrapper_forward():
    """Test RetinaNetBoxesModelWrapper forward pass."""

    def mock_model(x, **kwargs):
        batch_size = 2
        return {
            "boxes": tf.random.uniform((batch_size, 100, 4), 0, 100),
            "confidence": tf.random.uniform((batch_size, 100), 0, 1),
            "classes": tf.random.uniform((batch_size, 100), 0, 20, dtype=tf.int32),
            "num_detections": tf.constant([100, 100]),
        }

    nb_classes = 20
    image_size = (224, 224)
    wrapper = RetinaNetBoxesModelWrapper(mock_model, nb_classes, image_size=image_size)

    # Test with dummy input
    x = tf.random.normal((2, 224, 224, 3))
    results = wrapper(x)

    assert isinstance(results, list)
    assert len(results) == 2
