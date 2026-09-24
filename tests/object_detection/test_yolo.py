import pytest
import torch
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat

from xplique_adapters.object_detection.torch import (
    Yolo11RawBoxesModelWrapper,
    Yolo26RawBoxesModelWrapper,
    YoloOneToManyFormatter,
    YoloOneToOneFormatter,
    YoloResultBoxesModelWrapper,
    YoloResultBoxFormatter,
)


def test_yolo_result_formatter_init():
    """Test that YoloResultBoxFormatter can be initialized."""
    formatter = YoloResultBoxFormatter()
    assert formatter is not None


def test_yolo_raw_formatter_init():
    """Test that YoloOneToManyFormatter can be initialized."""
    formatter = YoloOneToManyFormatter()
    assert formatter is not None


def test_yolo_raw_formatter_forward():
    """Test YoloOneToManyFormatter with mock predictions."""
    formatter = YoloOneToManyFormatter()

    # Mock YOLO raw predictions (tuple format)
    batch_size = 2
    num_boxes = 10
    num_classes = 80

    # YOLO inference output: (batch, 1, features, num_boxes)
    # features = 4 (box coords) + num_classes (class probas)
    raw_output = torch.rand(batch_size, 1, 4 + num_classes, num_boxes)
    predictions = (raw_output, [[None], [None], [None]])  # type: ignore

    results = formatter.forward(predictions)  # type: ignore

    assert isinstance(results, list)
    assert len(results) == batch_size


@pytest.mark.parametrize("extra_axis", [False, True], ids=["native", "singleton_axis"])
def test_yolo_raw_formatter_single_detection_absolute_pixels(extra_axis):
    formatter = YoloOneToManyFormatter()
    # Decoded CXCYWH pixels with three class probabilities.
    detection = torch.tensor([100.0, 200.0, 40.0, 80.0, 0.1, 0.8, 0.3])
    raw_output = detection.reshape(1, 7, 1)
    if extra_axis:
        raw_output = raw_output.unsqueeze(1)

    results = formatter((raw_output, {"boxes": None, "scores": None, "feats": None}))

    assert len(results) == 1
    assert results[0].shape == (1, 8)
    torch.testing.assert_close(
        results[0], torch.tensor([[80.0, 160.0, 120.0, 240.0, 0.8, 0.1, 0.8, 0.3]])
    )
    assert formatter.input_box_type.format is BoxFormat.CXCYWH
    assert not formatter.input_box_type.is_normalized
    assert formatter.output_box_type.format is BoxFormat.XYXY
    assert not formatter.output_box_type.is_normalized


def test_yolo_raw_wrapper_init():
    """Test that Yolo11RawBoxesModelWrapper can be initialized."""

    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            raw_output = torch.rand(batch_size, 84, 8400)
            return (raw_output, [[None], [None], [None]])

    model = MockModel()
    wrapper = Yolo11RawBoxesModelWrapper(model)
    assert wrapper is not None


def test_yolo_result_wrapper_init():
    """Test that YoloResultBoxesModelWrapper can be initialized."""

    class MockModel(torch.nn.Module):
        def forward(self, x):
            return []  # Would return Results objects

    model = MockModel()
    wrapper = YoloResultBoxesModelWrapper(model)
    assert wrapper is not None


def test_yolo26_raw_formatter_init():
    """Test that YoloOneToOneFormatter can be initialized with nb_classes."""
    formatter = YoloOneToOneFormatter(nb_classes=80)
    assert formatter is not None


def test_yolo26_raw_formatter_forward():
    """Test YoloOneToOneFormatter with mock predictions."""
    num_classes = 80
    formatter = YoloOneToOneFormatter(nb_classes=num_classes)

    batch_size = 2
    num_boxes = 10
    # YOLO26 inference output: (batch, num_boxes, 6) — boxes, score, class_id
    raw_output = torch.cat(
        [
            torch.rand(batch_size, num_boxes, 4) * 640,  # boxes (XYXY pixels)
            torch.rand(batch_size, num_boxes, 1),  # score
            torch.randint(
                0, num_classes, (batch_size, num_boxes, 1)
            ).float(),  # class_id
        ],
        dim=-1,
    )
    predictions = (raw_output, [[None], [None], [None]])

    results = formatter.forward(predictions)  # type: ignore

    assert isinstance(results, list)
    assert len(results) == batch_size


def test_yolo26_raw_formatter_single_detection_absolute_pixels():
    formatter = YoloOneToOneFormatter(nb_classes=3)
    # Decoded XYXY pixels, score, class ID.
    raw_output = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 0.75, 2.0]]])

    results = formatter((raw_output, {"one2many": None, "one2one": None}))

    assert len(results) == 1
    assert results[0].shape == (1, 8)
    torch.testing.assert_close(
        results[0], torch.tensor([[10.0, 20.0, 30.0, 40.0, 0.75, 0.0, 0.0, 1.0]])
    )
    assert formatter.input_box_type.format is BoxFormat.XYXY
    assert not formatter.input_box_type.is_normalized
    assert formatter.output_box_type.format is BoxFormat.XYXY
    assert not formatter.output_box_type.is_normalized


def test_yolo26_raw_wrapper_init():
    """Test that Yolo26RawBoxesModelWrapper can be initialized with nb_classes."""

    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            num_boxes = 10
            raw_output = torch.rand(batch_size, num_boxes, 6)
            return (raw_output, [[None], [None], [None]])

    model = MockModel()
    wrapper = Yolo26RawBoxesModelWrapper(model, nb_classes=80)
    assert wrapper is not None


def test_yolo_one_to_many_formatter_rejects_end2end_dict():
    """YoloOneToManyFormatter must raise ValueError with a helpful hint when
    predictions[1] is a dict with 2 keys (YOLO26/end2end model output)."""
    formatter = YoloOneToManyFormatter()

    batch_size = 2
    num_boxes = 10
    num_classes = 80
    raw_output = torch.rand(batch_size, 1, 4 + num_classes, num_boxes)
    # Simulate YOLO26 end2end output: dict with 2 keys instead of list of 3
    predictions = (raw_output, {"one2many": None, "one2one": None})  # type: ignore

    with pytest.raises(ValueError, match="YoloOneToOneFormatter"):
        formatter.forward(predictions)  # type: ignore


def test_yolo_one_to_many_formatter_rejects_wrong_list_length():
    """YoloOneToManyFormatter must raise ValueError when predictions[1] is a list
    with length != 3."""
    formatter = YoloOneToManyFormatter()

    batch_size = 2
    num_boxes = 10
    num_classes = 80
    raw_output = torch.rand(batch_size, 1, 4 + num_classes, num_boxes)
    predictions = (raw_output, [[None], [None]])  # type: ignore — only 2 scales

    with pytest.raises(ValueError, match="3 elements"):
        formatter.forward(predictions)  # type: ignore
