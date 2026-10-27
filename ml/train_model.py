
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "..", "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "..", "dataset", "val")

MODEL_OUTPUT_PATH = os.path.join(
    BASE_DIR, "..", "backend", "model",
    "plant_disease_model.keras"
)

GRAPH_OUTPUT_PATH = os.path.join(
    BASE_DIR, "accuracy_graph.png"
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 4


# ---------- Build CNN Model ----------
def build_model(num_classes):
    model = Sequential([
        tf.keras.Input(shape=(128, 128, 3)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------- Main ----------
def main():

    # Check dataset paths
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    # Load training dataset
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    # Load validation dataset
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())

    print("\nClass indices:", train_generator.class_indices)

    # Verify class consistency
    if train_generator.class_indices != val_generator.class_indices:
        raise ValueError(
            "Training and validation class folders do not match!"
        )

    num_classes = len(class_names)

    if num_classes != NUM_CLASSES:
        print(f"Detected {num_classes} classes instead of {NUM_CLASSES}")

    print("\nClasses:", class_names)
    print("Training images:", train_generator.samples)
    print("Validation images:", val_generator.samples)
    print("Total images:", train_generator.samples + val_generator.samples)

    # Build model
    model = build_model(num_classes)
    model.summary()

    # Create output directory
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    # Save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_OUTPUT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Stop if validation loss does not improve
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # ---------- Training ----------
    print("\nStarting model training...\n")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    # ---------- Accuracy ----------
    best_val_accuracy = max(history.history["val_accuracy"])
    final_train_accuracy = history.history["accuracy"][-1]

    print("\n========== TRAINING RESULTS ==========")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Final training accuracy: {final_train_accuracy * 100:.2f}%")

    # ---------- Load best model ----------
    best_model = load_model(MODEL_OUTPUT_PATH)

    # ---------- Validation Predictions ----------
    val_generator.reset()

    predictions = best_model.predict(
        val_generator,
        verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    # ---------- Classification Report ----------
    print("\n========== PERFORMANCE METRICS ==========")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(report)

    # Get weighted scores
    scores = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    precision = scores["weighted avg"]["precision"]
    recall = scores["weighted avg"]["recall"]
    f1_score = scores["weighted avg"]["f1-score"]

    print(f"Weighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall: {recall * 100:.2f}%")
    print(f"Weighted F1-score: {f1_score * 100:.2f}%")

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("Confusion matrix saved to:", cm_path)

    # ---------- Inference Time ----------
    sample = np.zeros((1, 128, 128, 3), dtype=np.float32)

    # Warm-up
    best_model.predict(sample, verbose=0)

    repetitions = 100
    start_time = time.perf_counter()

    for _ in range(repetitions):
        best_model.predict(sample, verbose=0)

    end_time = time.perf_counter()

    inference_time = (
        (end_time - start_time) / repetitions
    ) * 1000

    print(f"\nInference time per image: {inference_time:.2f} ms")

    # ---------- Accuracy Graph ----------
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(GRAPH_OUTPUT_PATH, dpi=300)
    plt.close()

    print("\nAccuracy graph saved to:", GRAPH_OUTPUT_PATH)

    # ---------- Loss Graph ----------
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    loss_path = os.path.join(BASE_DIR, "loss_graph.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print("Loss graph saved to:", loss_path)

    # ---------- Final Summary ----------
    print("\n========== FINAL SUMMARY ==========")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Final training accuracy: {final_train_accuracy * 100:.2f}%")
    print(f"Weighted precision: {precision * 100:.2f}%")
    print(f"Weighted recall: {recall * 100:.2f}%")
    print(f"Weighted F1-score: {f1_score * 100:.2f}%")
    print(f"Inference time: {inference_time:.2f} ms")
    print("Model saved at:", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
