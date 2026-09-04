import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "image_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_model_v5.keras")
PREDICTION_PATH = os.path.join(BASE_DIR, "data", "test_predictions_v5.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "training_history_v5.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("LUNG CANCER DETECTION - V5")
print("MobileNetV2 Fine-Tuning Experiment")
print("=" * 65)

df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"\nTrain images:      {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images:       {len(test_df)}")

print("\nClass distribution:")
print("Train:")
print(train_df["binary_label"].value_counts().sort_index())
print("Validation:")
print(val_df["binary_label"].value_counts().sort_index())
print("Test:")
print(test_df["binary_label"].value_counts().sort_index())

# ============================================================
# IMAGE PIPELINE
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    # MobileNetV2 expects 3 channels
    image = tf.image.grayscale_to_rgb(image)

    return image, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_df["binary_label"].values.astype(np.float32),
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["binary_label"].values.astype(np.float32),
    )
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["binary_label"].values.astype(np.float32),
    )
)

train_ds = (
    train_ds
    .shuffle(len(train_df), seed=SEED, reshuffle_each_iteration=True)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array([0, 1])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["binary_label"].values,
)

class_weights = {
    int(classes[i]): float(class_weights_array[i])
    for i in range(len(classes))
}

print("\nClass weights:")
print(class_weights)

# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.05),
    ],
    name="augmentation",
)

# ============================================================
# MOBILENETV2 BACKBONE
# ============================================================

print("\nLoading pretrained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Start frozen
base_model.trainable = False

# ============================================================
# V5 MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image",
)

x = augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x * 255.0
)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.35)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="malignancy_probability",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="lung_cancer_mobilenetv2_v5",
)

# ============================================================
# STAGE 1 - TRAIN CLASSIFIER
# ============================================================

print("\n" + "=" * 65)
print("STAGE 1 - TRAINING CLASSIFIER")
print("=" * 65)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.02
    ),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

stage1_checkpoint = os.path.join(
    BASE_DIR,
    "models",
    "lung_cancer_model_v5_stage1.keras",
)

callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(
        stage1_checkpoint,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weights,
    callbacks=callbacks_stage1,
    verbose=1,
)

# ============================================================
# STAGE 2 - FINE TUNE LAST MOBILE NET LAYERS
# ============================================================

print("\n" + "=" * 65)
print("STAGE 2 - FINE-TUNING MOBILENETV2")
print("=" * 65)

# Unfreeze MobileNetV2
base_model.trainable = True

# Freeze earlier layers.
# Fine-tune only the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable statistics.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable_layers = sum(
    1 for layer in base_model.layers if layer.trainable
)

print(f"Trainable MobileNetV2 layers: {trainable_layers}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.02
    ),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

stage2_checkpoint = os.path.join(
    BASE_DIR,
    "models",
    "lung_cancer_model_v5_stage2.keras",
)

callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(
        stage2_checkpoint,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    ),
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=callbacks_stage2,
    verbose=1,
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nFinal V5 model saved:")
print(MODEL_PATH)

# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

history_data = []

for epoch, values in enumerate(history1.history):
    pass

history1_df = pd.DataFrame(history1.history)
history1_df["stage"] = "stage1"
history1_df["epoch"] = range(1, len(history1_df) + 1)

history2_df = pd.DataFrame(history2.history)
history2_df["stage"] = "stage2"
history2_df["epoch"] = range(
    len(history1_df) + 1,
    len(history1_df) + len(history2_df) + 1,
)

history_df = pd.concat(
    [history1_df, history2_df],
    ignore_index=True,
)

history_df.to_csv(HISTORY_PATH, index=False)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("V5 TEST EVALUATION")
print("=" * 65)

probabilities = model.predict(
    test_ds,
    verbose=1,
).flatten()

y_test = test_df["binary_label"].values

# Default threshold only for initial evaluation.
predictions = (probabilities >= 0.50).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)
recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)
f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)
auc = roc_auc_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
)

print("\n" + "=" * 65)
print("V5 RESULTS")
print("=" * 65)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nProbability distribution:")
print(f"Minimum: {probabilities.min():.6f}")
print(f"Maximum: {probabilities.max():.6f}")
print(f"Mean:    {probabilities.mean():.6f}")
print(f"Std:     {probabilities.std():.6f}")

print("\nClass 0 mean probability:")
print(
    probabilities[y_test == 0].mean()
)

print("\nClass 1 mean probability:")
print(
    probabilities[y_test == 1].mean()
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

output_df = test_df.copy()

output_df["prediction_probability"] = probabilities
output_df["predicted_label"] = predictions

output_df.to_csv(
    PREDICTION_PATH,
    index=False,
)

print("\nPredictions saved to:")
print(PREDICTION_PATH)

print("\nTraining history saved to:")
print(HISTORY_PATH)

print("\n" + "=" * 65)
print("V5 TRAINING COMPLETE")
print("=" * 65)
