# xplique-adapters

Model-specific adapters for [Xplique](https://github.com/fredericboisnard/xplique), keeping third-party and proprietary model code out of the core library.

---

## Why xplique-adapters?

[Xplique](https://github.com/fredericboisnard/xplique) is a framework-agnostic explainability library. Its core should only depend on standard scientific Python (NumPy, TensorFlow, PyTorch) — not on third-party model ecosystems like Ultralytics YOLO or keras-cv.

`xplique-adapters` is the thin integration layer that bridges those ecosystems. It contains:

- **Formatters** — convert model-specific output formats to the common Xplique box representation.
- **Wrappers** — wrap models in a Xplique-compatible interface, handling model-specific quirks (e.g. YOLO's training mode side-effects).
- **Latent extractors** — split a model at an intermediate layer to expose feature maps for concept-based explanation methods (CRAFT).

Because these components depend on optional, often proprietary or large packages (`ultralytics`, `keras-cv`…), keeping them here avoids polluting the core Xplique install.

---

## Installation

```bash
pip install xplique_adapters
```

Install only the extras you need:

| Extra | Installs | Use case |
|---|---|---|
| `yolo` | `ultralytics` | YOLO 11 / YOLO 2.6 models |
| `detr` | `transformers>=5,<6` | DETR (HuggingFace) |
| `torchvision` | `torchvision` | FCOS, RetinaNet, Faster R-CNN, SSD |
| `retinanet` | `keras-cv` | TensorFlow RetinaNet |
| `all` | all of the above | everything |

The Facebook reference DETR implementation is not installed as a package extra. Install
the `torchvision` extra for its PyTorch dependencies, then load the upstream model through
Torch Hub as shown below. The model source and checkpoint are downloaded into Torch's
cache by the caller.

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

Each wrapper is paired with a formatter for a specific model output stage. A
wrapper is not automatically a raw-model or post-processing wrapper: this is
determined by the formatter it uses.

| Model family | Wrapper input | Post-processing status | Why this wrapper exists |
|---|---|---|---|
| Ultralytics YOLO (`YoloResultBoxesModelWrapper`) | `Results` objects | Post-processed, typically after decoding and NMS | Standard `model(images)` inference output |
| Ultralytics YOLO (`Yolo11RawBoxesModelWrapper`, `Yolo26RawBoxesModelWrapper`) | Raw internal tensors | Before the usual result/NMS post-processing | Needed when explaining internal YOLO branches or raw model outputs |
| DETR | Model output object | Raw model predictions; formatter decodes scores and boxes | DETR has no anchor/NMS pipeline like YOLO |
| Torchvision detectors | List of dictionaries | Post-processed detections, typically after thresholding/NMS | Standard torchvision inference output (`boxes`, `scores`, `labels`) |
| KerasCV RetinaNet | Decoded prediction dictionary | Decoded predictions; NMS depends on the producing code | The adapter receives `boxes`, `confidence`, and `classes` tensors |

YOLO is the only family exposing both public post-processed results and raw
internal model outputs. The wrapper must therefore match the selected YOLO
path; the raw wrappers must not be used with Ultralytics `Results` objects.

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

#### DETR (Facebook reference and HuggingFace Transformers)

| Class | Role | Details |
|---|---|---|
| `DetrBoxFormatter` | Formatter | DETR outputs CXCYWH normalized boxes + logits. The formatter softmaxes logits, extracts scores and converts boxes to XYXY absolute coordinates. Requires `image_size`. |
| `DetrBoxesModelWrapper` | Wrapper | Wraps Facebook reference or HuggingFace DETR models with `DetrBoxFormatter`. |
| `DetrExtractorBuilder` | Latent extractor | Splits a caller-supplied Facebook reference DETR or HuggingFace Transformers 5 DETR model for CRAFT. |

##### Facebook reference DETR

Install the PyTorch and Torchvision dependencies, then load the reference model through
Torch Hub. `DetrExtractorBuilder` attaches the latent `g()` and `h()` split to the loaded
model; it does not download or manage the model itself.

```bash
pip install "xplique-adapters[torchvision]"
```

```python
import torch

from xplique_adapters.concepts.torch.latent_data_detr import DetrExtractorBuilder
from xplique_adapters.object_detection.torch import DetrBoxesModelWrapper

model = torch.hub.load(
    "facebookresearch/detr",
    "detr_resnet50",
    pretrained=True,
).eval()

# Use this wrapper for standard Xplique object-detection predictions.
wrapper = DetrBoxesModelWrapper(model, image_size=(800, 800))

# Use this extractor when running CRAFT or another latent-space method.
latent_extractor = DetrExtractorBuilder.build(
    model,
    device="cuda",
    image_size=(800, 800),
)
```

The complete CRAFT example is available in
[`examples/torch_detr_craft.py`](examples/torch_detr_craft.py).

##### HuggingFace Transformers DETR

Install the `detr` extra, which provides `transformers>=5,<6`:

```bash
pip install "xplique-adapters[detr]"
```

```python
from transformers import DetrForObjectDetection

from xplique_adapters.concepts.torch.latent_data_detr import DetrExtractorBuilder
from xplique_adapters.object_detection.torch import DetrBoxesModelWrapper

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
wrapper = DetrBoxesModelWrapper(model, image_size=(640, 640))
latent_extractor = DetrExtractorBuilder.build(
    model,
    device="cuda",
    image_size=(640, 640),
)
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
according to `num_detections`; without that field all rows are retained. The
wrapper infers KerasCV's `bounding_box_format` from the model and supports
`xyxy`, `xywh`, `center_xywh`, `rel_xyxy`, and `rel_xywh` formats.

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
| `LatentDataDetr` | `DetrExtractorBuilder` | Facebook reference and HuggingFace Transformers 5 DETR | Extracts the final backbone level and supports variable-size inputs and masking. Facebook models are caller-supplied, typically through Torch Hub. |
| `LatentDataRetinanet` | `RetinanetExtractorBuilder` | torchvision RetinaNet | Multi-scale FPN features as `OrderedDict`; `extraction_layer` selects which scale; perturbation rebatching repeats companion features and image metadata |
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
