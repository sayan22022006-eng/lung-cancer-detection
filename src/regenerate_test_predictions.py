import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "image_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_model.keras")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "test_predictions.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


print("=" * 60)
print("       REGENERATING TEST PREDICTIONS")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

test_df = df[df["split"] == "test"].copy()

print(f"Test images: {len(test_df)}")
print(f"Test patients: {test_df['patient_id'].nunique()}")

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, tf.cast(label, tf.float32)


paths = test_df["image_path"].values
labels = test_df["binary_label"].values.astype(np.float32)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (paths, labels)
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)

actual = labels.astype(int)


print("\nChecking probability distribution...")

print(f"Minimum probability: {probabilities.min():.6f}")
print(f"Maximum probability: {probabilities.max():.6f}")
print(f"Mean probability:    {probabilities.mean():.6f}")
print(f"Std probability:     {probabilities.std():.6f}")


accuracy = accuracy_score(
    actual,
    predictions
)

precision = precision_score(
    actual,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    actual,
    probabilities
)

cm = confusion_matrix(
    actual,
    predictions
)


print("\n")
print("=" * 60)
print("                 TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predictions,
        zero_division=0
    )
)


prediction_df = test_df[
    [
        "patient_id",
        "nodule_id",
        "image_path",
        "malignancy",
        "binary_label"
    ]
].copy()

prediction_df["prediction_probability"] = probabilities
prediction_df["predicted_label"] = predictions

prediction_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nPredictions saved to:")
print(OUTPUT_PATH)

print("\n")
print("=" * 60)
print("              REGENERATION COMPLETE")
print("=" * 60)
