import pytest
import torch
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat

from xplique_adapters.object_detection.torch import (
    YoloXBoxesModelWrapper,
    YoloXBoxFormatter,
)


def test_yolox_formatter_converts_boxes_and_preserves_objectness():
    formatter = YoloXBoxFormatter()
    predictions = torch.tensor([[[100.0, 200.0, 40.0, 80.0, 0.7, 0.2, 0.9]]])

    results = formatter(predictions)

    assert len(results) == 1
    torch.testing.assert_close(
        results[0], torch.tensor([[80.0, 160.0, 120.0, 240.0, 0.63, 0.2, 0.9]])
    )
    assert results[0].filter(class_id=1, confidence=0.6).shape[0] == 1
    assert results[0].filter(class_id=1, confidence=0.8).shape[0] == 0
    assert formatter.input_box_type.format is BoxFormat.CXCYWH
    assert formatter.output_box_type.format is BoxFormat.XYXY


@pytest.mark.parametrize(
    ("predictions", "exception", "message"),
    [
        ([torch.zeros(1, 1, 6)], TypeError, "torch.Tensor"),
        (torch.zeros(1, 6), ValueError, "shape"),
        (torch.zeros(1, 1, 5), ValueError, r"5 \+ classes"),
    ],
)
def test_yolox_formatter_rejects_invalid_predictions(predictions, exception, message):
    with pytest.raises(exception, match=message):
        YoloXBoxFormatter()(predictions)


def test_yolox_wrapper_formats_model_output():
    class MockYoloX(torch.nn.Module):
        def forward(self, inputs):
            return torch.tensor(
                [[[10.0, 20.0, 4.0, 8.0, 0.6, 0.8]]], device=inputs.device
            ).expand(inputs.shape[0], -1, -1)

    wrapper = YoloXBoxesModelWrapper(MockYoloX())
    outputs = wrapper(torch.zeros(2, 3, 32, 32))

    assert len(outputs) == 2
    assert outputs[0].shape == (1, 6)