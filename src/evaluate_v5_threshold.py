import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "image_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_model_v5.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    # Convert grayscale → RGB for MobileNetV2
    image = tf.image.grayscale_to_rgb(image)

    return image, label


def make_dataset(dataframe):
    paths = dataframe["image_path"].values
    labels = dataframe["binary_label"].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("V5 THRESHOLD ANALYSIS")
print("=" * 65)

df = pd.read_csv(CSV_PATH)

val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"\nValidation images: {len(val_df)}")
print(f"Test images:       {len(test_df)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V5 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

val_dataset = make_dataset(val_df)

val_probs = model.predict(
    val_dataset,
    verbose=1,
).flatten()

y_val = val_df["binary_label"].values

val_auc = roc_auc_score(
    y_val,
    val_probs,
)

print(f"\nValidation ROC-AUC: {val_auc:.4f}")


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

thresholds = np.arange(
    0.20,
    0.81,
    0.005,
)

results = []

for threshold in thresholds:

    predictions = (
        val_probs >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(
    best["threshold"]
)


# ============================================================
# VALIDATION RESULT
# ============================================================

print("\n" + "=" * 65)
print("BEST VALIDATION THRESHOLD")
print("=" * 65)

print(
    f"Threshold : {best_threshold:.3f}"
)

print(
    f"Accuracy  : {best['accuracy']:.4f}"
)

print(
    f"Precision : {best['precision']:.4f}"
)

print(
    f"Recall    : {best['recall']:.4f}"
)

print(
    f"F1 Score  : {best['f1']:.4f}"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

test_dataset = make_dataset(test_df)

test_probs = model.predict(
    test_dataset,
    verbose=1,
).flatten()

y_test = test_df["binary_label"].values


# ============================================================
# APPLY VALIDATION THRESHOLD
# ============================================================

test_predictions = (
    test_probs >= best_threshold
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_auc = roc_auc_score(
    y_test,
    test_probs,
)

cm = confusion_matrix(
    y_test,
    test_predictions,
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("V5 TEST RESULTS USING VALIDATION THRESHOLD")
print("=" * 65)

print(
    f"Threshold : {best_threshold:.3f}"
)

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\nProbability distribution:")

print(
    f"Minimum: {test_probs.min():.6f}"
)

print(
    f"Maximum: {test_probs.max():.6f}"
)

print(
    f"Mean:    {test_probs.mean():.6f}"
)

print(
    f"Std:     {test_probs.std():.6f}"
)


print("\nClass 0 mean probability:")

print(
    test_probs[y_test == 0].mean()
)


print("\nClass 1 mean probability:")

print(
    test_probs[y_test == 1].mean()
)


# ============================================================
# SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_path = os.path.join(
    BASE_DIR,
    "data",
    "v5_threshold_results.csv",
)

results_df.to_csv(
    threshold_path,
    index=False,
)

print("\nThreshold results saved to:")
print(threshold_path)


print("\n" + "=" * 65)
print("V5 THRESHOLD ANALYSIS COMPLETE")
print("=" * 65)
