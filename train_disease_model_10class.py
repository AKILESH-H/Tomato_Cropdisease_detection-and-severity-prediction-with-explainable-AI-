
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "disease_10class_manifest.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODEL_DIR / "disease_model_10class.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_disease_model_10class.keras"
CLASSES_PATH = MODEL_DIR / "disease_classes_10class.json"
HISTORY_PATH = RESULTS_DIR / "disease_10class_training_history.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n"
            "Run: python src/prepare_10class_manifest.py"
        )

    df = pd.read_csv(MANIFEST)

    required = {"image_path", "class_name", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns: {required}")

    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths found in manifest.")

    missing = [p for p in df["image_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing. "
            f"Example: {missing[0]}"
        )

    return df


def make_dataset(frame, class_to_index, training=False):
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["class_name"].map(class_to_index).to_numpy(
        dtype=np.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(frame), 3000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    df = load_manifest()

    classes = sorted(df["class_name"].unique().tolist())
    if len(classes) != 10:
        raise ValueError(
            f"Expected 10 classes, but found {len(classes)}."
        )

    class_to_index = {
        name: index for index, name in enumerate(classes)
    }

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Training or validation split is empty.")

    print("Classes:")
    for index, name in enumerate(classes):
        print(f"  {index}: {name}")

    print(f"\nTraining images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {EPOCHS}")
    print("\nLoading image datasets...")

    train_ds = make_dataset(
        train_df, class_to_index, training=True
    )
    val_ds = make_dataset(
        val_df, class_to_index, training=False
    )

    # Keep the same general transfer-learning approach.
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        include_preprocessing=True,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        len(classes),
        activation="softmax",
        name="disease_probabilities",
    )(x)

    model = keras.Model(inputs, outputs, name="tomato_disease_10class")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(RESULTS_DIR / "disease_10class_training_log.csv")
        ),
    ]

    print("\nStarting 10-class training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint rather than blindly saving
    # the last epoch's weights.
    best_model = keras.models.load_model(BEST_MODEL_PATH)
    best_model.save(MODEL_PATH)

    CLASSES_PATH.write_text(
        json.dumps(classes, indent=2),
        encoding="utf-8",
    )

    HISTORY_PATH.write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_epoch = int(np.argmin(history.history["val_loss"]))
    best_val_accuracy = history.history["val_accuracy"][best_epoch]

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print(f"Final saved model: {MODEL_PATH}")
    print(f"Class names: {CLASSES_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print("\nYour original four-class model was not overwritten.")


if __name__ == "__main__":
    main()
