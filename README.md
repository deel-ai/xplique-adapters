# xplique-adapters

Model-specific adapters for [Xplique](https://github.com/fredericboisnard/xplique), keeping third-party and proprietary model code out of the core library.

---

## Why xplique-adapters?

[Xplique](https://github.com/fredericboisnard/xplique) is a framework-agnostic explainability library. Its core should only depend on standard scientific Python (NumPy, TensorFlow, PyTorch) — not on third-party model ecosystems like Ultralytics YOLO, HuggingFace Transformers, or keras-cv.

`xplique-adapters` is the thin integration layer that bridges those ecosystems. It contains:

- **Formatters** — convert model-specific output formats to the common Xplique box representation.
- **Wrappers** — wrap models in a Xplique-compatible interface, handling model-specific quirks (e.g. YOLO's training mode side-effects).
- **Latent extractors** — split a model at an intermediate layer to expose feature maps for concept-based explanation methods (CRAFT).

Because these components depend on optional, often proprietary or large packages (`ultralytics`, `transformers`, `keras-cv`…), keeping them here avoids polluting the core Xplique install.

---

## Installation

```bash
pip install xplique_adapters
```

Install only the extras you need:

| Extra | Installs | Use case |
|---|---|---|
| `yolo` | `ultralytics` | YOLO 11 / YOLO 2.6 models |
| `detr` | `transformers` | DETR (HuggingFace) |
| `torchvision` | `torchvision` | FCOS, RetinaNet, Faster R-CNN, SSD |
| `retinanet` | `keras-cv` | TensorFlow RetinaNet |
| `all` | all of the above | everything |

```bash
pip install "xplique-adapters[yolo,torchvision] @ git+https://..."
```

---

## Architecture overview

```
xplique_adapters/
├── object_detection/
│   ├── torch/          # PyTorch formatters & wrappers
│   └── tf/             # TensorFlow formatters & wrappers
└── concepts/
    ├── torch/          # PyTorch latent data containers & extractor builders
    └── tf/             # TensorFlow latent data containers & extractor builders
```

---

## Object detection adapters

### Concepts

| Concept | Description |
|---|---|
| **Formatter** | A callable that converts a model's raw prediction into a list of `MultiBoxTensor` objects (one per image). Handles box coordinate conversion, score extraction, and class probability encoding. |
| **Wrapper** | A `nn.Module` (or TF equivalent) that wraps a detection model and applies the right formatter internally, so Xplique attribution methods can treat it as a black box. |

---

### PyTorch — `xplique_adapters.object_detection.torch`

#### YOLO (Ultralytics)

Three formatters cover the two main YOLO output modes:

| Class | Input format | When to use |
|---|---|---|
| `YoloResultBoxFormatter` | Ultralytics `Results` objects (XYXY, absolute px) | Default YOLO inference — `model(images)` returns `Results` |
| `YoloOneToManyFormatter` | Raw tuple `(tensor, aux)`, CXCYWH normalized | YOLO 11 internal raw output (before post-processing) |
| `YoloOneToOneFormatter` | Raw tuple `(tensor, aux)`, XYXY normalized | YOLO 26 internal raw output (before post-processing) |

Corresponding wrappers:

| Class | Pairs with |
|---|---|
| `YoloResultBoxesModelWrapper` | `YoloResultBoxFormatter` |
| `Yolo11RawBoxesModelWrapper` | `YoloOneToManyFormatter` |
| `Yolo26RawBoxesModelWrapper` | `YoloOneToOneFormatter` |

> All YOLO wrappers override `train(mode=False)` to prevent accidentally triggering Ultralytics dataset-loading side effects when Xplique calls `.eval()`.

**Usage:**

```python
from ultralytics import YOLO
from xplique_adapters.object_detection.torch import YoloResultBoxesModelWrapper

model = YOLO("yolo11n.pt").model
wrapper = YoloResultBoxesModelWrapper(model)
```

---

#### DETR (HuggingFace Transformers)

| Class | Role | Details |
|---|---|---|
| `DetrBoxFormatter` | Formatter | DETR outputs CXCYWH normalized boxes + logits. The formatter softmaxes logits, extracts scores and converts boxes to XYXY absolute coordinates. Requires `image_size`. |
| `DetrBoxesModelWrapper` | Wrapper | Wraps a HuggingFace DETR model with `DetrBoxFormatter`. |

**Usage:**

```python
from transformers import DetrForObjectDetection
from xplique_adapters.object_detection.torch import DetrBoxesModelWrapper

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
wrapper = DetrBoxesModelWrapper(model, image_size=(640, 640))
```

---

#### Torchvision models (FCOS, RetinaNet, SSD, Faster R-CNN, …)

All `torchvision.models.detection` models share the same output format (list of dicts with `boxes`, `scores`, `labels`).

| Class | Role | Details |
|---|---|---|
| `TorchvisionBoxFormatter` | Formatter | Converts discrete `labels` to one-hot probabilities. Requires `nb_classes`. |
| `TorchvisionBoxesModelWrapper` | Wrapper | Wraps any torchvision detection model with `TorchvisionBoxFormatter`. |

**Usage:**

```python
import torchvision
from xplique_adapters.object_detection.torch import TorchvisionBoxesModelWrapper

model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
wrapper = TorchvisionBoxesModelWrapper(model, nb_classes=91)  # COCO
```

---

### TensorFlow — `xplique_adapters.object_detection.tf`

#### RetinaNet (keras-cv)

| Class | Role | Details |
|---|---|---|
| `RetinaNetProcessedBoxFormatter` | Formatter | Formats decoded batched predictions (`boxes[B,N,4]`, `confidence[B,N]`, `classes[B,N]`, optional `num_detections[B]` and per-class `scores[B,N,C]`). Uses absolute `xywh` input and normalized `xyxy` output. |
| `RetinaNetBoxesModelWrapper` | Wrapper | Wraps a KerasCV RetinaNet in `prediction_mode="raw"` (default, calls `decode_predictions`) or explicit `prediction_mode="decoded"`. |

RetinaNet public image sizes use `(height, width)`. Padding rows are removed
according to `num_detections`; without that field all valid rows are retained
and class `-1` rows are treated as padding. The wrapper infers KerasCV's
`bounding_box_format` from the model and supports
`xyxy`, `xywh`, `center_xywh`, `rel_xyxy`, and `rel_xywh` formats.
Decoded callables without either format attribute must pass an explicit
`input_box_type`; normalized `rel_` formats do not require `image_size`, while
absolute formats do.

**Usage:**

```python
import keras_cv
from xplique_adapters.object_detection.tf import RetinaNetBoxesModelWrapper

model = keras_cv.models.RetinaNet(...)
wrapper = RetinaNetBoxesModelWrapper(
    model, nb_classes=80, image_size=(640, 640), prediction_mode="raw"
)
```

---

## Concepts / latent extractors

These components are used with Xplique's **CRAFT** (Concept Recursive Activation Factorization for Trees) method, which requires splitting a model into an *encoder* (up to a chosen layer) and a *decoder* (the rest).

Each latent extractor pair consists of:
- A **`LatentData`** container — holds the intermediate activations and any additional state (image sizes, skip-connection tensors, …) needed by the second half of the model.
- A **`LatentExtractorBuilder`** — monkey-patches or hooks the model to capture the right tensors at inference time and return a `LatentData` instance.

---

### PyTorch — `xplique_adapters.concepts.torch`

| LatentData class | Builder class | Model family | Notes |
|---|---|---|---|
| `LatentDataYolo` | `YoloExtractorBuilder` | Ultralytics YOLO | Stores main activation `x` + list of skip-connection tensors `y` |
| `LatentDataDetr` | `DetrExtractorBuilder` | HuggingFace DETR | Includes `NestedTensor` helper for variable-size inputs and masking |
| `LatentDataRetinanet` | `RetinaNetExtractorBuilder` | torchvision RetinaNet | Multi-scale FPN features as `OrderedDict`; perturbation rebatching repeats companion features and image metadata |
| `LatentDataFasterRcnn` | `FasterRcnnExtractorBuilder` | torchvision Faster R-CNN | Multi-scale ResNet/FPN features; `extraction_layer` selects which scale |
| `LatentDataFcos` | `FcosExtractorBuilder` | torchvision FCOS | Multi-scale FPN features; `extraction_layer` selects which scale |
| `LatentDataSSD` | `SSDExtractorBuilder` | torchvision SSD (MobileNetV3) | Backbone features as dict; `extraction_layer` selects which feature map |

All `LatentData` classes expose:
- `get_activations(as_numpy, keep_gradients)` — returns activations in `(N, H, W, C)` format.
- `set_activations(values)` — replaces activations (used by CRAFT when perturbing the latent space).

---

### TensorFlow — `xplique_adapters.concepts.tf`

| LatentData class | Builder class | Model family | Notes |
|---|---|---|---|
| `TfLatentDataRetinanet` | `RetinaNetExtractorBuilder` | keras-cv RetinaNet | ResNet backbone pyramid features `P3/P4/P5`; `index_activations` selects scale |

---

## Adding a new adapter

1. Implement a **formatter** subclassing `TorchBaseBoxFormatter` or `TfBaseBoxFormatter`.
2. Implement a **wrapper** subclassing `TorchBoxesModelWrapper` or `TfBoxesModelWrapper`, instantiating your formatter in `__init__`.
3. Export both from the relevant `__init__.py`.
4. If CRAFT support is needed, implement a `LatentData` subclass and a `LatentExtractorBuilder`.
5. Add an optional dependency in `setup.py` under `extras_require` for the new model's package.
6. Add tests under `tests/`.

---

## License

This project is distributed under multiple licenses.

Original code is licensed under the MIT License. Some model-specific latent
adapters contain modified third-party code and are subject to their respective
upstream licenses:

- DETR and TensorFlow RetinaNet adapters: Apache License 2.0
- Torchvision RetinaNet, FCOS, SSD, and Faster R-CNN adapters: BSD 3-Clause
    License
- YOLO adapter: GNU Affero General Public License v3.0

See [`LICENSE`](LICENSE) for the project-level statement and the complete
third-party license texts in [`LICENSES/`](LICENSES/).
