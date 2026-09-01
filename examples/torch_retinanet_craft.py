"""
CRAFT on RetinaNet with PyTorch - Multiple Images

This script demonstrates CRAFT (Concept-based Recursive Activation Factorization)
on a RetinaNet object detection model with multiple giraffe images.

The images are loaded from a pre-processed .npz file (giraffes.npz) which contains
50 giraffe images already resized to 800x800 pixels.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
import xplique
from xplique.concepts import HolisticCraftTorch as Craft
from xplique.plots import plot_image_detections
from xplique_adapters.concepts.torch.latent_data_retinanet import (
    RetinaNetExtractorBuilder,
)

# Configuration parameters
class_name = "giraffe"  # Target class for detection

# Dataset file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "assets", "giraffes.npz")

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "torch_retinanet_craft")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Load Images from NPZ File
# ============================================================================
print("\n" + "=" * 80)
print("1. Loading Giraffe Images from NPZ Dataset")
print("=" * 80)

# Check if dataset file exists
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_FILE}\n"
        f"Please run 'python create_giraffe_npz.py' first to create the dataset."
    )

# Load the .npz file
print(f"Loading dataset from {DATASET_FILE}...")
data = np.load(DATASET_FILE)
all_images = data["arr_0"]  # Shape: (N, 800, 800, 3), dtype: uint8
print(f"Loaded {len(all_images)} images from npz file")
print(f"Image shape: {all_images.shape[1:]}, dtype: {all_images.dtype}")

# Convert numpy arrays to PIL Images for compatibility with the rest of the code
raw_images = []
for img_array in all_images:
    img = Image.fromarray(img_array, mode="RGB")
    raw_images.append(img)

print(f"Successfully prepared {len(raw_images)} images")

# ============================================================================
# 2. Load RetinaNet Model
# ============================================================================
print("\n" + "=" * 80)
print("3. Loading RetinaNet Model")
print("=" * 80)

# Load RetinaNet Model
print("Loading RetinaNet model...")


weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
model = retinanet_resnet50_fpn_v2(weights=weights)
model.eval()
model = model.to(device)
print("RetinaNet model loaded successfully")

# Get COCO class names from model weights (official list)
CLASSES = weights.meta["categories"]
nb_classes = len(CLASSES)
print(f"Loaded {nb_classes} COCO class names from model weights")
class_id = CLASSES.index(class_name)

# RetinaNet preprocessing test
images_size = (800, 800)
preprocess = weights.transforms()
input_tensors = [img.resize(images_size) for img in raw_images]
input_tensors = [preprocess(img) for img in input_tensors]
input_tensor = torch.stack(input_tensors, dim=0).to(device)  # Shape: (N, 3, 800, 800)

print(f"\nBatched input tensor shape: {input_tensor.shape}")
print(f"Number of images: {len(raw_images)}")

# ============================================================================
# 3. Display Sample Images with Predicted Bounding Boxes
# ============================================================================
print("\n" + "=" * 80)
print("2. Displaying Sample Images with Predicted Bounding Boxes")
print("=" * 80)

# Define color mapping for different classes
label_to_color = {
    f"{class_name}": "r",
    "person": "b",
    "zebra": "g",
    "elephant": "orange",
    "bird": "purple",
}

# Build latent extractor (needed for predictions)
latent_extractor = RetinaNetExtractorBuilder.build(
    model, device=str(device), nb_classes=nb_classes, extraction_layer=-1
)

# Get predictions for first 20 images in batches
num_display_images = 20
batch_size = 4  # Process images in smaller batches to fit GPU memory
results = []

for batch_start in range(0, num_display_images, batch_size):
    batch_end = min(batch_start + batch_size, num_display_images)
    batch_tensor = input_tensor[batch_start:batch_end]
    with torch.no_grad():
        batch_results = latent_extractor(batch_tensor)
    results.extend(batch_results)

# Create grid of images with bounding boxes (5 columns)
nb_cols = 5
nb_rows = (num_display_images + nb_cols - 1) // nb_cols
fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(20, 16))

for idx in range(num_display_images):
    # Filter boxes for the target class with confidence threshold
    boxes = results[idx].filter(class_id=class_id, confidence=0.8)

    # Get the raw image
    img_to_display = raw_images[idx]

    # Display image with boxes on the specific axis
    row = idx // nb_cols  # Integer division to get row index
    col = idx % nb_cols  # Modulo to get column index
    ax = axes[row, col]
    plot_image_detections(img_to_display, boxes, CLASSES, label_to_color, ax=ax)
    ax.axis("off")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "01_sample_images_with_boxes.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
print(f"Displayed {num_display_images} images with predicted bounding boxes")
plt.close()

# ============================================================================
# 4. Display Detailed View of One Image
# ============================================================================
print("\n" + "=" * 80)
print("4. Displaying Detailed View of One Image")
print("=" * 80)

# Select which image to display in detail
display_image_idx = 4
print(f"Showing detailed view of image at index {display_image_idx}")

# Get detections for the selected image
display_image_tensor = input_tensor[
    display_image_idx : display_image_idx + 1
]  # Shape: (1, 3, 800, 800)
single_result = latent_extractor(display_image_tensor)
print(f"Number of boxes detected: {len(single_result)}")

# Filter boxes for the target class
box_to_display = single_result[0].filter(class_id=class_id, confidence=0.3)
print(f"Number of boxes for '{class_name}': {box_to_display.shape[0]}")

# Display the raw image with boxes overlaid
display_image = raw_images[display_image_idx].resize(images_size)
fig = plot_image_detections(display_image, box_to_display, CLASSES, label_to_color)

output_path = os.path.join(OUTPUT_DIR, "03_detailed_single_image_boxes.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# ============================================================================
# 5. Initialize and Run CRAFT on All Images
# ============================================================================
print("\n" + "=" * 80)
print("5. Initializing and Running CRAFT on All Images")
print("=" * 80)

craft = Craft(
    latent_extractor=latent_extractor, number_of_concepts=10, device=str(device)
)
craft.fit(input_tensor)
print("CRAFT fitting completed")

# ============================================================================
# 6. Display Concepts
# ============================================================================
print("\n" + "=" * 80)
print("6. Displaying Concepts")
print("=" * 80)

fig = craft.display_images_per_concept(
    input_tensor[:10], order=None, filter_percentile=80, clip_percentile=5
)

output_path = os.path.join(OUTPUT_DIR, "04_concepts_unordered.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# ============================================================================
# 7. Rank Concepts by Importance for Target Class (gradient-input method)
# ============================================================================
print("\n" + "=" * 80)
print(
    f"7. Ranking Concepts by Importance for '{class_name}' class, with gradient-input method"
)
print("=" * 80)

# Rank concepts for the target class
operator = xplique.Tasks.OBJECT_DETECTION
importances_gi = craft.estimate_importance(
    input_tensor, operator, class_id, confidence=0.3, method="gradient_input"
)
order = importances_gi.argsort()[::-1]

print(f"Concept importances for '{class_name}': {importances_gi}")

fig = craft.display_images_per_concept(
    input_tensor[:5], order=order, filter_percentile=80, clip_percentile=5
)

output_path = os.path.join(OUTPUT_DIR, "05_concepts_ordered_gi.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# Display bar plot of concept importances for target class
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(importances_gi))), importances_gi)
plt.xlabel("Concept ID")
plt.ylabel("Importance (Gradient-Input)")
plt.title(f'Concept Importances for "{class_name}" class (Gradient-Input)')

output_path = os.path.join(OUTPUT_DIR, "06_importance_bar_gi.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# ============================================================================
# 8. Display Top Images per Concept (gradient-input method)
# ============================================================================
print("\n" + "=" * 80)
print("8. Displaying Top Images per Concept (gradient-input method)")
print("=" * 80)

fig = craft.display_top_images_per_concept(
    input_tensor,
    topk=5,
    filter_percentile=80,
    clip_percentile=5,
    order=order,  # Use the importance order from selected class
)

output_path = os.path.join(OUTPUT_DIR, "07_top_images_per_concept_gi.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# ============================================================================
# 9. Rank Concepts by Importance using Sobol Method
# ============================================================================
print("\n" + "=" * 80)
print(f"9. Ranking Concepts by Importance for '{class_name}' class (Sobol)")
print("=" * 80)

# Estimate importance using Sobol method (use fewer images as it's more expensive)
num_images_sobol = 5
print(f"Running Sobol attribution with {num_images_sobol} images...")

importances_sobol = craft.estimate_importance(
    input_tensor[:num_images_sobol],
    operator,
    class_id,
    confidence=0.3,
    method="sobol",
    grid_size=4,
    nb_design=8,
)
order_sobol = importances_sobol.argsort()[::-1]

print(f"Concept importances (Sobol) for '{class_name}': {importances_sobol}")

fig = craft.display_images_per_concept(
    input_tensor[:5], order=order_sobol, filter_percentile=80, clip_percentile=5
)

output_path = os.path.join(OUTPUT_DIR, "08_concepts_ordered_sobol.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# Display bar plot of concept importances for Sobol
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(importances_sobol))), importances_sobol)
plt.xlabel("Concept ID")
plt.ylabel("Importance (Sobol)")
plt.title(f'Concept Importances for "{class_name}" class (Sobol)')

output_path = os.path.join(OUTPUT_DIR, "09_importance_bar_sobol.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# ============================================================================
# 10. Display Top Images per Concept (Sobol Order)
# ============================================================================
print("\n" + "=" * 80)
print("10. Displaying Top Images per Concept (Sobol Order)")
print("=" * 80)

fig = craft.display_top_images_per_concept(
    input_tensor,
    topk=5,
    filter_percentile=80,
    clip_percentile=5,
    order=order_sobol,  # Use the Sobol importance order
)

output_path = os.path.join(OUTPUT_DIR, "10_top_images_per_concept_sobol.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

print("\n" + "=" * 80)
print("All outputs saved to:", OUTPUT_DIR)
print("=" * 80)


# ============================================================================
# 11. Alternative: Using PartialExplainer API with SquareGrad
# ============================================================================
print("\n" + "=" * 80)
print("11. Demonstrating PartialExplainer API with SquareGrad")
print("=" * 80)

# Clear GPU cache to free memory from previous operations
import gc

torch.cuda.empty_cache()
gc.collect()
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

from xplique.concepts import PartialExplainer
from xplique.attributions import SquareGrad

# Create a SquareGrad explainer wrapper
print("Creating PartialExplainer with SquareGrad method...")
squaregrad_explainer = PartialExplainer(
    explainer_class=SquareGrad,
    operator=operator,
    reducer=None,  # Important: Don't reduce over concept dimension!
    nb_samples=20,  # Reduced from 50 to save GPU memory
    noise=0.15,
)

explanation_squaregrad = craft.compute_explanation_per_concept(
    partial_explainer=squaregrad_explainer,
    images=input_tensor,
    class_id=class_id,
    confidence=0.3,
)
importances_squaregrad = craft.reduce_to_importance(
    explanation=explanation_squaregrad,
    spatial_reducer="max",
    abs_before_reduce=True,
    aggregation_reducer="mean",
)

print(f"\nSquareGrad importance scores: {importances_squaregrad}")
order_squaregrad = importances_squaregrad.argsort()[::-1]
print(f"SquareGrad concept ranking: {order_squaregrad}")

# Display concepts ordered by SquareGrad importance
fig = craft.display_images_per_concept(
    input_tensor[:5], order=order_squaregrad, filter_percentile=80, clip_percentile=5
)

output_path = os.path.join(OUTPUT_DIR, "11_concepts_ordered_squaregrad.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()

# Display bar plot for SquareGrad
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(importances_squaregrad))), importances_squaregrad)
plt.xlabel("Concept ID")
plt.ylabel("Importance (SquareGrad)")
plt.title(
    f'Concept Importances for "{class_name}" class (SquareGrad via PartialExplainer)'
)

output_path = os.path.join(OUTPUT_DIR, "11_importance_bar_squaregrad.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.close()


# ============================================================================
# 12. Display Top Images per Concept (SquareGrad method)
# ============================================================================
print("\n" + "=" * 80)
print("12. Displaying Top Images per Concept (SquareGrad method)")
print("=" * 80)

fig = craft.display_top_images_per_concept(
    input_tensor,
    topk=5,
    filter_percentile=80,
    clip_percentile=5,
    order=order_squaregrad,  # Use the importance order from SquareGrad
)

output_path = os.path.join(OUTPUT_DIR, "12_top_images_per_concept_squaregrad.png")



# ============================================================================
# 13. Test Importance, Prevalence (with GradientInput)
# ============================================================================

# Create a GradientInput explainer wrapper
from xplique.concepts import PartialExplainer
from xplique.attributions import GradientInput

print("Creating PartialExplainer with GradientInput method...")
gradient_input_explainer = PartialExplainer(
    explainer_class=GradientInput,
    operator=operator,
    reducer=None,
)
explanation = craft.compute_explanation_per_concept(
    images=input_tensor,
    partial_explainer=gradient_input_explainer,
    class_id=class_id,
    confidence=0.3
)

# importance
importances_gradient_input = craft.reduce_to_importance(
    explanation, spatial_reducer="max", aggregation_reducer="mean")
print(f"\nGradientInput importance scores: {importances_gradient_input}")

order_gradient_input = importances_gradient_input.argsort()[::-1]
print(f"GradientInput concept ranking: {order_gradient_input}")

# prevalence
prevalence = craft.reduce_to_prevalence(explanation)
print(f"\nPrevalence scores: {prevalence}")

prevalence_order = prevalence.argsort()[::-1]
print(f"Prevalence concept ranking: {prevalence_order}")
