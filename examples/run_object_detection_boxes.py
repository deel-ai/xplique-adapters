"""Run supported object detectors on one image and save detections."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from xplique.plots import plot_image_detections
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType


COCO_CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
COCO_CLASSES_YOLO = [name for name in COCO_CLASSES if name != "N/A"]
DEFAULT_COLORS = ["r", "orange", "b", "g", "y", "purple", "cyan", "magenta"]
CLASS_COLORS = {
    "person": "r",
    "car": "#03FFF2D1",
    "motorcycle": "y",
    "bicycle": "b",
}


MODELS = (
    "detr", "fasterrcnn", "retinanet", "keras_retinanet", "fcos", "ssd",
    "yolo11", "yolo26",
)


def get_label_to_color(class_names):
    return {
        class_name: CLASS_COLORS.get(
            class_name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        )
        for idx, class_name in enumerate(class_names)
    }


def load_model(name, device):
    if name == "detr":
        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)
        return model.to(device).eval(), None, (800, 800)
    if name == "fasterrcnn":
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        return fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval(), weights, None
    if name == "retinanet":
        from torchvision.models.detection import (
            RetinaNet_ResNet50_FPN_V2_Weights,
            retinanet_resnet50_fpn_v2,
        )
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        return retinanet_resnet50_fpn_v2(weights=weights).to(device).eval(), weights, None
    if name == "keras_retinanet":
        from keras_cv.models import RetinaNet

        return RetinaNet.from_preset("retinanet_resnet50_pascalvoc"), None, (640, 640)
    if name == "fcos":
        from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        return fcos_resnet50_fpn(weights=weights).to(device).eval(), weights, None
    if name == "ssd":
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        return ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval(), weights, (320, 320)
    if name in ("yolo11", "yolo26"):
        from ultralytics import YOLO
        model = YOLO(f"{name}n.pt", verbose=False)
        if name == "yolo26":
            model.model.model[-1].end2end = False
        return model, None, (640, 640)
    raise ValueError(f"Unsupported model: {name}")


def run_model(name, model, image, weights, image_size, device):
    display_image = image.resize(image_size) if image_size else image
    if name == "keras_retinanet":
        import tensorflow as tf
        from xplique_adapters.object_detection.tf import RetinaNetProcessedBoxFormatter

        tensor = tf.convert_to_tensor(
            np.asarray(display_image, dtype=np.float32)[None, ...],
            dtype=tf.float32,
        )
        predictions = model.predict(tensor, verbose=0)
        predictions = {
            key: tf.convert_to_tensor(value) for key, value in predictions.items()
        }
        result = RetinaNetProcessedBoxFormatter(
            nb_classes=20, image_size=image_size
        )(predictions)[0]
        return display_image, result, [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ], False

    tensor = T.ToTensor()(display_image).unsqueeze(0)
    if weights is not None:
        tensor = weights.transforms()(tensor)
    tensor = tensor.to(device)

    if name in ("yolo11", "yolo26"):
        from xplique_adapters.object_detection.torch.yolo_wrappers import YoloResultBoxesModelWrapper
        result = YoloResultBoxesModelWrapper(model)(tensor)[0]
        class_names = COCO_CLASSES_YOLO
    elif name == "detr":
        from xplique_adapters.object_detection.torch.detr_wrappers import DetrBoxesModelWrapper
        result = DetrBoxesModelWrapper(model, image_size=display_image.size[::-1])(tensor)[0]
        class_names = COCO_CLASSES
    else:
        from xplique_adapters.object_detection.torch.torchvision_wrappers import TorchvisionBoxesModelWrapper
        result = TorchvisionBoxesModelWrapper(model, nb_classes=91)(tensor)[0]
        class_names = COCO_CLASSES
    return display_image, result, class_names, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="img.jpg", help="Input image path")
    parser.add_argument("--output-dir", default="object_detection_boxes")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB")
    figures = []

    for name in args.models:
        print(f"[{name}] loading model")
        model, weights, image_size = load_model(name, device)
        display_image, detections, class_names, normalized = run_model(
            name, model, image, weights, image_size, device
        )
        filtered = detections.filter(confidence=args.confidence)
        print(f"[{name}] {filtered.shape[0]} detections above {args.confidence}")
        box_type = BoxType(BoxFormat.XYXY, is_normalized=normalized)
        fig = plot_image_detections(
            display_image,
            filtered,
            class_names,
            get_label_to_color(class_names),
            box_type=box_type,
        )
        fig.savefig(os.path.join(args.output_dir, f"{name}.png"), bbox_inches="tight")
        figures.append((name, display_image, filtered, class_names))
        plt.close(fig)
        del model
    print(f"Saved detections to {args.output_dir}")


if __name__ == "__main__":
    main()
