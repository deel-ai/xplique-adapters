import pytest
import torch

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

    # Raw YOLO output: (batch, 1, features, num_boxes)
    # features = 4 (box coords) + num_classes (class probas)
    raw_output = torch.rand(batch_size, 1, 4 + num_classes, num_boxes)
    predictions = (raw_output, [[None], [None], [None]])  # type: ignore

    results = formatter.forward(predictions)  # type: ignore

    assert isinstance(results, list)
    assert len(results) == batch_size


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
    # Raw YOLO26 output: (batch, num_boxes, 4 + 1 + 1) — boxes, score, class_id
    raw_output = torch.cat(
        [
            torch.rand(batch_size, num_boxes, 4),  # boxes (XYXY normalized)
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
