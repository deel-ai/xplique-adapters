"""
Attribution Methods on YOLO with PyTorch

This script demonstrates multiple attribution methods on a YOLO object detection model.

For each image, it computes attributions on the first and last detected boxes.

The images are loaded from a pre-processed .npz file (giraffes.npz).

Features:
- Configurable attribution methods via ATTRIBUTION_METHODS dictionary
- Smart caching: skips already-computed results to save time
- Combined visualization showing all results in a grid layout

Output:
- Individual detection and attribution images for each enabled method
- Combined visualization (always regenerated) showing all results in a grid
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

import tensorflow as tf
import time

# Configure TensorFlow to grow memory allocation as needed
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Warning: Could not set GPU memory growth: {e}")

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
import xplique
from xplique.attributions import (
    Saliency,
    GradientInput,
    SquareGrad,
    VarGrad,
    SmoothGrad,
    Occlusion,
    HsicAttributionMethod,
    SobolAttributionMethod,
    Rise,
)
from xplique.plots import plot_image_detections
from xplique.plots.image import generate_heatmap

# ============================================================================
# Configuration Parameters
# ============================================================================

# Detection parameters
class_name = "giraffe"  # Target class for detection
confidence_threshold = 0.3  # Confidence threshold for filtering boxes
num_images = 6  # Number of images to process (first N images from dataset)
num_boxes_per_image = 2  # Number of boxes to explain per image (first and last box)

# Attribution methods to compute (set to True/False to enable/disable)
ATTRIBUTION_METHODS = {
    "saliency": True,  # Gradient-based attribution
    "gradientinput": True,  # Gradient × Input
    "squaregrad": True,  # Squared gradients
    "vargrad": True,  # Variance of gradients
    "smoothgrad": True,  # Smooth gradients (averaged over noisy samples)
    "occlusion": True,  # Perturbation-based attribution
    "hsic": True,  # HSIC attribution
    "sobol": True,  # Sobol attribution
    "rise": True,  # RISE attribution
}

# Dataset file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "assets", "giraffes.npz")

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "torch_yolo_attributions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Load Multiple Images from NPZ File
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Giraffe Images from NPZ Dataset")
print("=" * 80)

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_FILE}\n"
        f"Please run 'python create_giraffe_npz.py' first to create the dataset."
    )

print(f"Loading dataset from {DATASET_FILE}...")
data = np.load(DATASET_FILE)
all_images = data["arr_0"]  # Shape: (N, 800, 800, 3), dtype: uint8
print(f"Loaded {len(all_images)} images from npz file")

# Select first N images
selected_images = all_images[:num_images]
print(f"Processing first {num_images} images")
print(f"Image shape: {selected_images[0].shape}, dtype: {selected_images[0].dtype}")

# YOLO uses 640x640 input
attribution_size = (640, 640)
print(f"Resizing to {attribution_size} for YOLO")

# Convert to PIL Images and resize
raw_images = []
for i, img_array in enumerate(selected_images):
    img = Image.fromarray(img_array, mode="RGB")
    img = img.resize(attribution_size)
    raw_images.append(img)

print(f"Successfully prepared {len(raw_images)} images")

# ============================================================================
# 2. Load YOLO Model
# ============================================================================
print("\n" + "=" * 80)
print("2. Loading YOLO Model")
print("=" * 80)

print("Loading YOLO model...")
MODEL_PATH = "yolo11n.pt"
yolo_model = YOLO(MODEL_PATH, verbose=False)
yolo_model.eval()
print("YOLO model loaded successfully")

# Get COCO class names
CLASSES = list(yolo_model.names.values())
nb_classes = len(CLASSES)
print(f"Loaded {nb_classes} COCO class names from YOLO model")
class_id = CLASSES.index(class_name)

# YOLO preprocessing
transform = T.Compose([T.Resize(attribution_size), T.ToTensor()])
processed_inputs = [transform(img).unsqueeze(0).to(device) for img in raw_images]

print(f"\nProcessed {len(processed_inputs)} images")
print(f"Input shape per image: {processed_inputs[0].shape}")

# ============================================================================
# 3. Build Raw Wrapper and Get Predictions
# ============================================================================
print("\n" + "=" * 80)
print("3. Building Raw Wrapper and Running Detection")
print("=" * 80)
print("NOTE: YOLO requires the raw wrapper for all attribution methods")

# Build raw wrapper
from xplique_adapters.concepts.torch.latent_data_yolo import YoloExtractorBuilder

print("Building YOLO raw wrapper...")
detection_model = yolo_model.model  # Extract inner PyTorch model
raw_wrapper = YoloExtractorBuilder.build(
    detection_model, extraction_layer=10, batch_size=1
)
raw_wrapper.output_as_tensor = True
# raw_wrapper.training = False
print("Raw wrapper built successfully!")

# Get predictions with raw wrapper
print("\nRunning detection with raw wrapper...")
all_filtered_boxes = []
for idx, processed_input in enumerate(processed_inputs):
    with torch.no_grad():
        predictions = raw_wrapper(processed_input)

    # Filter by class and confidence
    filtered_boxes = predictions[0].filter(
        class_id=class_id, confidence=confidence_threshold
    )
    all_filtered_boxes.append(filtered_boxes)
    print(f"  Image {idx}: {filtered_boxes.shape[0]} boxes detected for '{class_name}'")

# ============================================================================
# 4. Save Detection Visualizations (ALL boxes per image)
# ============================================================================
print("\n" + "=" * 80)
print("4. Saving Detection Visualizations")
print("=" * 80)

label_to_color = {f"{class_name}": "r"}

for img_idx, (raw_image, filtered_boxes) in enumerate(
    zip(raw_images, all_filtered_boxes)
):
    if filtered_boxes.shape[0] == 0:
        print(f"Image {img_idx}: No boxes - skipping")
        continue

    # Save ONE image per image showing ALL detected boxes together
    output_path = os.path.join(OUTPUT_DIR, f"01_detection_img{img_idx:02d}.png")

    # Show first and last box on the image (for visual diversity)
    num_detected = filtered_boxes.shape[0]
    if num_detected == 1:
        box_indices = [0]
    elif num_detected >= 2:
        box_indices = [0, num_detected - 1]  # First and last
    else:
        box_indices = []

    if not box_indices:
        print(f"Image {img_idx}: No boxes - skipping")
        continue

    selected_boxes = filtered_boxes[box_indices].cpu().detach()
    print(f"Image {img_idx}: Showing {len(box_indices)} boxes (first and last)")

    fig = plot_image_detections(raw_image, selected_boxes, CLASSES, label_to_color)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# 5. Initialize Attribution Methods with Raw Wrapper
# ============================================================================
print("\n" + "=" * 80)
print("5. Initializing Attribution Methods")
print("=" * 80)

# Wrap raw wrapper with TorchWrapper for attributions
from xplique.wrappers import TorchWrapper as XpliqueTorchWrapper

torch_wrapped_raw = XpliqueTorchWrapper(
    raw_wrapper, device=device, is_channel_first=True, requires_grad=True
)

# Initialize all explainers
explainers = {}

if ATTRIBUTION_METHODS["saliency"]:
    explainers["saliency"] = Saliency(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
    )
    print("  - Saliency explainer initialized")

if ATTRIBUTION_METHODS["gradientinput"]:
    explainers["gradientinput"] = GradientInput(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
    )
    print("  - GradientInput explainer initialized")

if ATTRIBUTION_METHODS["squaregrad"]:
    explainers["squaregrad"] = SquareGrad(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
    )
    print("  - SquareGrad explainer initialized")

if ATTRIBUTION_METHODS["vargrad"]:
    explainers["vargrad"] = VarGrad(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
        nb_samples=50,
        noise=0.15,
    )
    print("  - VarGrad explainer initialized")

if ATTRIBUTION_METHODS["smoothgrad"]:
    explainers["smoothgrad"] = SmoothGrad(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
        nb_samples=50,
        noise=0.15,
    )
    print("  - SmoothGrad explainer initialized")

if ATTRIBUTION_METHODS["occlusion"]:
    explainers["occlusion"] = Occlusion(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
        patch_size=(100, 100),
        patch_stride=(50, 50),
    )
    print("  - Occlusion explainer initialized")

if ATTRIBUTION_METHODS["hsic"]:
    explainers["hsic"] = HsicAttributionMethod(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
    )
    print("  - Hsic explainer initialized")

if ATTRIBUTION_METHODS["sobol"]:
    explainers["sobol"] = SobolAttributionMethod(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
        grid_size=8,
        nb_design=8,
    )
    print("  - Sobol explainer initialized")

if ATTRIBUTION_METHODS["rise"]:
    explainers["rise"] = Rise(
        torch_wrapped_raw,
        operator=xplique.Tasks.OBJECT_DETECTION,
        batch_size=1,
        nb_samples=4000,
    )
    print("  - Rise explainer initialized")

# Method configurations (file prefix and display name)
method_configs = {
    "saliency": {"prefix": "02", "name": "Saliency"},
    "gradientinput": {"prefix": "03", "name": "GradientInput"},
    "squaregrad": {"prefix": "04", "name": "SquareGrad"},
    "vargrad": {"prefix": "05", "name": "VarGrad"},
    "smoothgrad": {"prefix": "06", "name": "SmoothGrad"},
    "occlusion": {"prefix": "07", "name": "Occlusion"},
    "hsic": {"prefix": "08", "name": "Hsic"},
    "sobol": {"prefix": "09", "name": "Sobol"},
    "rise": {"prefix": "10", "name": "Rise"},
}

# Initialize timing storage
method_timings = {method: [] for method in method_configs.keys()}

# ============================================================================
# 6. Generate Attributions
# ============================================================================
print("\n" + "=" * 80)
print("6. Generating Attributions")
print("=" * 80)

# Process each image and each box
for img_idx, (raw_image, processed_input, filtered_boxes) in enumerate(
    zip(raw_images, processed_inputs, all_filtered_boxes)
):
    print(f"\n{'-' * 80}")
    print(f"Image {img_idx}: Processing Attributions")
    print(f"{'-' * 80}")

    num_boxes_available = filtered_boxes.shape[0]
    if num_boxes_available == 0:
        print(f"  No boxes detected - skipping attribution for image {img_idx}")
        continue

    # Select first and last box for visual diversity
    if num_boxes_available == 1:
        box_indices = [0]
    else:
        box_indices = [0, num_boxes_available - 1]  # First and last

    print(
        f"  Processing {len(box_indices)} boxes: first and last (out of {num_boxes_available} detected)"
    )

    # Prepare inputs in TensorFlow format (channel-last)
    tf_inputs = tf.convert_to_tensor(processed_input.cpu().numpy())
    tf_inputs = tf.transpose(tf_inputs, [0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)

    # Process each selected box
    for enumerate_idx, actual_box_idx in enumerate(box_indices):
        print(
            f"\n  Box {enumerate_idx + 1}/{len(box_indices)} (index {actual_box_idx}):"
        )

        single_box = filtered_boxes[actual_box_idx : actual_box_idx + 1].cpu().detach()
        print(f"    Coords: {single_box[0, :4].tolist()}")
        single_box_ext = tf.expand_dims(single_box, axis=0)

        # Process each enabled method
        for method_name, explainer in explainers.items():
            config = method_configs[method_name]
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{config['prefix']}_{method_name}_img{img_idx:02d}_box{enumerate_idx:02d}.png",
            )

            print(f"    - Computing {config['name']}...")

            # Time the attribution computation
            start_time = time.time()

            # Compute explanation
            explanations = explainer.explain(tf_inputs, single_box_ext)
            heatmap = generate_heatmap(
                explanations[0], size=tf_inputs[0].shape[:2], clip_percentile=0.5
            )

            # Visualize and save
            fig = plot_image_detections(
                raw_image, single_box, CLASSES, label_to_color, heatmap=heatmap
            )
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            elapsed_time = time.time() - start_time
            method_timings[method_name].append(elapsed_time)
            print(f"      Saved: {output_path} ({elapsed_time:.2f}s)")

print("\n" + "=" * 80)
print("7. Creating Combined Visualization")
print("=" * 80)

# Auto-detect all existing attribution files to determine the grid structure
import glob

# Find all attribution files (they have img*_box* pattern)
attribution_files = []
for method_name, config in method_configs.items():
    method_files = glob.glob(
        os.path.join(OUTPUT_DIR, f"{config['prefix']}_{method_name}_img*_box*.png")
    )
    attribution_files.extend(method_files)

if not attribution_files:
    print("No attribution files found - skipping combined visualization")
else:
    # Parse attribution files to get all image-box pairs
    detected_images = set()
    image_box_pairs = set()
    for attr_file in attribution_files:
        basename = os.path.basename(attr_file)
        parts = basename.replace(".png", "").split("_")
        # Format: {prefix}_{method}_img{XX}_box{YY}.png
        img_num = int(parts[2].replace("img", ""))
        box_num = int(parts[3].replace("box", ""))
        detected_images.add(img_num)
        image_box_pairs.add((img_num, box_num))

    image_box_pairs = sorted(image_box_pairs)
    total_columns = len(image_box_pairs)

    print(f"Detected images: {sorted(detected_images)}")
    print(f"Total columns (image-box combinations): {total_columns}")

    # Determine which methods have files
    methods_with_files = []
    for method_name, config in method_configs.items():
        method_files = glob.glob(
            os.path.join(OUTPUT_DIR, f"{config['prefix']}_{method_name}_img*_box*.png")
        )
        if method_files:
            methods_with_files.append(method_name)
            print(f"  Found {len(method_files)} files for {config['name']}")

    num_rows = 1 + len(methods_with_files)

    print(f"Creating grid with {num_rows} rows × {total_columns} columns")

    # Create combined visualization
    fig, axes = plt.subplots(
        num_rows,
        total_columns,
        figsize=(3 * total_columns, 3 * num_rows),
        subplot_kw={"xticks": [], "yticks": [], "frame_on": False},
    )

    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if total_columns == 1:
        axes = axes.reshape(-1, 1)

    # Row 0: Detections (show ALL boxes per image)
    # Map image index to column index
    img_to_cols = {}
    for col_idx, (img_idx, box_idx) in enumerate(image_box_pairs):
        if img_idx not in img_to_cols:
            img_to_cols[img_idx] = []
        img_to_cols[img_idx].append(col_idx)

    for col_idx, (img_idx, box_idx) in enumerate(image_box_pairs):
        # For detection row, use the image file (all boxes together)
        # But only show it in the first column for this image
        if img_to_cols[img_idx][0] == col_idx:
            # This is the first box column for this image - show detection
            detection_path = os.path.join(
                OUTPUT_DIR, f"01_detection_img{img_idx:02d}.png"
            )
            if os.path.exists(detection_path):
                detection_img = plt.imread(detection_path)
                axes[0, col_idx].imshow(detection_img)
            else:
                axes[0, col_idx].text(
                    0.5,
                    0.5,
                    f"Img{img_idx}\nDetection\nNot Found",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        else:
            # This is not the first box column for this image - leave empty or repeat
            detection_path = os.path.join(
                OUTPUT_DIR, f"01_detection_img{img_idx:02d}.png"
            )
            if os.path.exists(detection_path):
                detection_img = plt.imread(detection_path)
                axes[0, col_idx].imshow(detection_img)

        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        for spine in axes[0, col_idx].spines.values():
            spine.set_visible(False)

if total_columns > 0:
    axes[0, 0].set_ylabel(
        "Detections",
        fontsize=13,
        fontweight="bold",
        rotation=0,
        ha="right",
        va="center",
    )

# Attribution rows
row_idx = 1
for method_name in methods_with_files:
    config = method_configs[method_name]

    for col_idx, (img_idx, box_idx) in enumerate(image_box_pairs):
        attribution_path = os.path.join(
            OUTPUT_DIR,
            f"{config['prefix']}_{method_name}_img{img_idx:02d}_box{box_idx:02d}.png",
        )
        if os.path.exists(attribution_path):
            attribution_img = plt.imread(attribution_path)
            axes[row_idx, col_idx].imshow(attribution_img)
        else:
            axes[row_idx, col_idx].text(
                0.5,
                0.5,
                f"{config['name']}\nNot Computed",
                ha="center",
                va="center",
                fontsize=9,
                color="red",
            )

        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        for spine in axes[row_idx, col_idx].spines.values():
            spine.set_visible(False)

        if col_idx == 0:
            # Include timing in ylabel if available
            if method_timings[method_name]:
                mean_time = np.mean(method_timings[method_name])
                print(
                    f"  {config['name']} average time: {mean_time:.2f}s over {len(method_timings[method_name])} runs : {[f'{t:.2f}s' for t in method_timings[method_name]]}"
                )
                label_text = f"{config['name']}\n({mean_time:.2f}s)"
            else:
                label_text = config["name"]

            axes[row_idx, col_idx].set_ylabel(
                label_text,
                fontsize=13,
                fontweight="bold",
                rotation=0,
                ha="right",
                va="center",
            )

    row_idx += 1

plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
combined_path = os.path.join(OUTPUT_DIR, "00_combined_all_results.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
print(f"Saved combined visualization: {combined_path}")
plt.close()

print("\n" + "=" * 80)
print("All outputs saved to:", OUTPUT_DIR)
print("=" * 80)
