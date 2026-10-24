
from pathlib import Path
import json
import re

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)

DISEASE_MODEL_PATH = MODELS / "disease_model_10class.keras"
DISEASE_CLASSES_PATH = MODELS / "disease_classes_10class.json"
SEVERITY_MODEL_PATH = MODELS / "best_severity_model_clean.keras"


# =========================================================
# LOAD MODELS
# =========================================================

for required_path in (
    DISEASE_MODEL_PATH,
    DISEASE_CLASSES_PATH,
    SEVERITY_MODEL_PATH,
):
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required model or class file not found: {required_path}"
        )

disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

with DISEASE_CLASSES_PATH.open("r", encoding="utf-8") as file:
    disease_classes = json.load(file)

if disease_model.output_shape[-1] != len(disease_classes):
    raise ValueError(
        "The disease model output count does not match the saved class list."
    )

severity_model = tf.keras.models.load_model(SEVERITY_MODEL_PATH)

severity_classes = [
    "Healthy (0%)",
    "Low (0.1%-10%)",
    "Mild (11%-25%)",
    "Moderate (26%-50%)",
    "Severe (51%+)",
]

if severity_model.output_shape[-1] != len(severity_classes):
    raise ValueError(
        "The severity model output count does not match severity_classes."
    )


# =========================================================
# PROTOTYPE HEALTH SCORES
# =========================================================

# These are heuristic values, not validated plant-health measurements.
HEALTH_SCORES = [100, 95, 82, 62, 25]


# =========================================================
# DISEASE-SPECIFIC RECOMMENDATIONS
# =========================================================

RECOMMENDATIONS = {
    "healthy": [
        "Continue regular inspection of leaves and stems.",
        "Water at the base of the plant and avoid prolonged leaf wetness.",
        "Maintain good airflow and remove plant debris.",
        "Recheck the plant if new spots, curling, or yellowing develop.",
    ],

    "bacterial_spot": [
        "Inspect for small dark, water-soaked spots and surrounding yellowing.",
        "Avoid handling plants when foliage is wet.",
        "Clean tools and follow local guidance for removing affected material.",
        "Seek local agricultural advice before applying any control product.",
    ],

    "early_blight": [
        "Inspect older leaves for expanding brown spots, often with concentric rings.",
        "Remove badly affected leaves when appropriate and dispose of them safely.",
        "Avoid overhead watering and rotate tomato crops where possible.",
        "Confirm the diagnosis before choosing disease-control measures.",
    ],

    "late_blight": [
        "Inspect for rapidly spreading dark lesions and possible pale growth under leaves.",
        "Limit leaf wetness and avoid handling wet plants.",
        "Seek prompt local agricultural advice because late blight can spread rapidly.",
        "Do not rely on the model alone to decide whether to destroy plants or apply treatment.",
    ],

    "leaf_mold": [
        "Inspect the undersides of leaves for olive-green or yellowish mold growth.",
        "Improve ventilation and avoid prolonged high humidity.",
        "Remove severely affected foliage when appropriate.",
        "Use locally approved control measures only after confirming the disease.",
    ],

    "septoria_leaf_spot": [
        "Inspect lower leaves for numerous small spots with pale centers and dark margins.",
        "Avoid splashing soil or water onto foliage.",
        "Remove heavily affected leaves when appropriate and dispose of them safely.",
        "Maintain crop hygiene and consult local guidance if symptoms spread.",
    ],

    "spider_mites": [
        "Inspect leaf undersides for tiny mites, stippling, or fine webbing.",
        "Check several leaves because symptoms may be unevenly distributed.",
        "Avoid unnecessary broad-spectrum pesticide use that could harm beneficial insects.",
        "Confirm the presence of mites before selecting a locally approved control method.",
    ],

    "target_spot": [
        "Inspect for brown lesions that may develop concentric rings.",
        "Reduce prolonged leaf wetness and improve airflow around plants.",
        "Remove affected plant debris and follow good crop hygiene.",
        "Seek confirmation because target spot can resemble other tomato leaf diseases.",
    ],

    "tomato_mosaic_virus": [
        "Look for mottled leaf coloration, distortion, or unusual growth.",
        "Clean hands and tools between plants and avoid handling healthy plants after suspect plants.",
        "Do not propagate plants from a suspect plant until the diagnosis is clarified.",
        "Seek local agricultural guidance on virus confirmation and plant management.",
    ],

    "yellow_leaf_curl_virus": [
        "Look for upward leaf curling, yellowing, and stunted new growth.",
        "Inspect for whiteflies, which can transmit this virus.",
        "Monitor nearby plants and manage insect vectors using locally recommended practices.",
        "Seek agricultural confirmation; similar symptoms can have other causes.",
    ],

    "other": [
        "Inspect the plant and compare symptoms with reliable local crop guidance.",
        "Avoid applying pesticides based only on an automated prediction.",
        "Consider expert confirmation if symptoms persist or spread.",
    ],
}


def disease_key(class_name):
    """Map each model class to its recommendation category."""
    name = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")

    if "yellow_leaf_curl" in name:
        return "yellow_leaf_curl_virus"
    if "tomato_mosaic_virus" in name:
        return "tomato_mosaic_virus"
    if "spider_mites" in name:
        return "spider_mites"
    if "bacterial_spot" in name:
        return "bacterial_spot"
    if "early_blight" in name:
        return "early_blight"
    if "late_blight" in name:
        return "late_blight"
    if "leaf_mold" in name:
        return "leaf_mold"
    if "septoria_leaf_spot" in name:
        return "septoria_leaf_spot"
    if "target_spot" in name:
        return "target_spot"
    if "healthy" in name:
        return "healthy"

    return "other"


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def load_leaf(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    # EfficientNetV2 model was trained with images in the 0-255 range.
    array = np.asarray(resized, dtype=np.float32)[None, ...]

    return image, original, array


# =========================================================
# GRAD-CAM FOR THE 10-CLASS DISEASE MODEL
# =========================================================

def find_last_conv(layer):
    """Find a suitable final convolutional layer in the backbone."""
    for child in reversed(layer.layers):
        if isinstance(child, tf.keras.layers.Conv2D):
            return child

        if isinstance(child, tf.keras.Model):
            found = find_last_conv(child)
            if found is not None:
                return found

    return None


def make_gradcam(input_array, predicted_index):
    # The first nested model is the EfficientNet backbone.
    backbone = next(
        (
            layer for layer in disease_model.layers
            if isinstance(layer, tf.keras.Model)
        ),
        None,
    )

    if backbone is None:
        raise RuntimeError(
            "Could not find the EfficientNet backbone in the disease model."
        )

    last_conv = find_last_conv(backbone)

    if last_conv is None:
        raise RuntimeError(
            "Could not find a convolutional layer for Grad-CAM."
        )

    # Produce the convolutional activations and backbone features.
    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv.output, backbone.output],
    )

    # Collect the classification head that follows the backbone.
    head_layers = []
    found_backbone = False

    for layer in disease_model.layers:
        if layer is backbone:
            found_backbone = True
            continue

        if found_backbone:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(
            input_array,
            training=False,
        )

        predictions = features

        for layer in head_layers:
            predictions = layer(predictions)

        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    # Average gradients spatially to obtain channel importance weights.
    weights = tf.reduce_mean(gradients, axis=(1, 2))[0]
    activations = conv_outputs[0]

    heatmap = tf.reduce_sum(
        activations * weights[None, None, :],
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# =========================================================
# COMPLETE TOMATO LEAF ANALYSIS
# =========================================================

def analyze(image_path):
    path = Path(image_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image, original, array = load_leaf(path)

    # Disease prediction using the new 10-class model.
    disease_probabilities = disease_model.predict(
        array,
        verbose=0,
    )[0]

    disease_index = int(np.argmax(disease_probabilities))
    disease_name = disease_classes[disease_index]
    disease_confidence = float(disease_probabilities[disease_index])

    # Severity prediction using the existing five-class model.
    severity_probabilities = severity_model.predict(
        array,
        verbose=0,
    )[0]

    severity_index = int(np.argmax(severity_probabilities))
    severity_name = severity_classes[severity_index]
    severity_confidence = float(severity_probabilities[severity_index])

    health_score = HEALTH_SCORES[severity_index]

    key = disease_key(disease_name)
    advice = RECOMMENDATIONS[key]

    # Generate Grad-CAM for the predicted disease class.
    heatmap = make_gradcam(array, disease_index)

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize(
        image.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0

    colored_heatmap = plt.get_cmap("jet")(
        heatmap_array
    )[..., :3]

    overlay = np.clip(
        0.55 * (original / 255.0)
        + 0.45 * colored_heatmap,
        0,
        1,
    )

    # Save a two-panel explanation image.
    output_image = RESULTS / f"{path.stem}_full_analysis.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].imshow(original)
    axes[0].set_title("Original tomato leaf", fontsize=13)

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM: {disease_name}\n"
        f"Confidence: {disease_confidence:.1%}",
        fontsize=12,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_image,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Build the report.
    report_path = RESULTS / f"{path.stem}_analysis.txt"

    report = [
        "TOMATO LEAF ANALYSIS - PROTOTYPE",
        "=" * 38,
        f"Image: {path.name}",
        f"Disease prediction: {disease_name}",
        f"Disease model confidence: {disease_confidence:.2%}",
        f"Severity prediction: {severity_name}",
        f"Severity model confidence: {severity_confidence:.2%}",
        f"Prototype health score: {health_score}/100",
        "",
        "CARE AND PREVENTION SUGGESTIONS",
        *[f"- {item}" for item in advice],
        "",
        "LIMITATIONS",
        "- Predictions are experimental and may be wrong.",
        "- Disease classification and severity estimation are separate models.",
        "- The health score is a heuristic based on predicted severity.",
        "- Grad-CAM highlights influential regions; it does not prove causation.",
        "- Similar symptoms may occur in different diseases.",
        "- Confirm disease and treatment choices with reliable agricultural advice.",
    ]

    report_text = "\n".join(report)
    report_path.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print("\nGrad-CAM image saved:", output_image)
    print("Text report saved:", report_path)

    return {
        "disease": disease_name,
        "disease_confidence": disease_confidence,
        "severity": severity_name,
        "severity_confidence": severity_confidence,
        "health_score": health_score,
        "gradcam_path": str(output_image),
        "report_path": str(report_path),
        "report": report_text,
    }


if __name__ == "__main__":
    image_path = input(
        "Enter the full path to a tomato leaf image: "
    ).strip().strip('"')

    analyze(image_path)
