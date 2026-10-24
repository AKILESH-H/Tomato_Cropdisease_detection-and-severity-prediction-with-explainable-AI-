
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "disease_10class_manifest.csv"
MODEL_PATH = ROOT / "models" / "disease_model_10class.keras"
CLASSES_PATH = ROOT / "models" / "disease_classes_10class.json"
RESULTS = ROOT / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def main():
    df = pd.read_csv(MANIFEST)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("The test split is empty.")

    class_to_index = {name: i for i, name in enumerate(classes)}

    unknown = sorted(set(test_df["class_name"]) - set(classes))
    if unknown:
        raise ValueError(f"Unknown classes in test manifest: {unknown}")

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    paths = test_df["image_path"].astype(str).to_numpy()
    labels = test_df["class_name"].map(class_to_index).to_numpy(
        dtype=np.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Test images: {len(test_df)}")
    print("Evaluating model...")

    loss, accuracy = model.evaluate(ds, verbose=1)

    true_labels = []
    predicted_labels = []

    for images, batch_labels in ds:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        true_labels.extend(batch_labels.numpy().tolist())
        predicted_labels.extend(predictions.tolist())

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:\n")
    print(report)

    report_path = RESULTS / "disease_10class_classification_report.txt"
    report_path.write_text(
        f"Test images: {len(test_df)}\n"
        f"Test loss: {loss:.4f}\n"
        f"Test accuracy: {accuracy:.4%}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(classes))),
    )

    fig, ax = plt.subplots(figsize=(15, 12))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )
    display.plot(
        ax=ax,
        cmap="Purples",
        xticks_rotation=70,
        values_format="d",
        colorbar=True,
    )
    ax.set_title("Tomato Disease — 10-Class Test Confusion Matrix")
    fig.tight_layout()

    matrix_path = RESULTS / "disease_10class_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nReport saved: {report_path}")
    print(f"Confusion matrix saved: {matrix_path}")


if __name__ == "__main__":
    main()
