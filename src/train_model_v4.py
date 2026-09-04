import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "image_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lung_cancer_model_v4.keras"
)

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "test_predictions_v4.csv"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "training_history_v4.csv"
)

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15

SEED = 42


np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(
    os.path.join(BASE_DIR, "models"),
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("       LUNG CANCER MODEL TRAINING - V4")
print("              MobileNetV2 Transfer Learning")
print("=" * 65)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")

print(
    f"Total patients: "
    f"{df['patient_id'].nunique()}"
)


# ============================================================
# SPLITS
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "validation"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nDataset sizes:")

print(
    f"Training:   {len(train_df)}"
)

print(
    f"Validation: {len(val_df)}"
)

print(
    f"Test:       {len(test_df)}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTraining labels:")

print(
    train_df["binary_label"]
    .value_counts()
    .sort_index()
)

print("\nValidation labels:")

print(
    val_df["binary_label"]
    .value_counts()
    .sort_index()
)

print("\nTest labels:")

print(
    test_df["binary_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["binary_label"].values
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        weights
    )
}

print("\nClass weights:")

print(class_weights)


# ============================================================
# IMAGE LOADING
# ============================================================

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

    # Convert grayscale → 3 channels
    image = tf.image.grayscale_to_rgb(
        image
    )

    return (
        image,
        tf.cast(
            label,
            tf.float32
        )
    )


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = dataframe[
        "image_path"
    ].values

    labels = dataframe[
        "binary_label"
    ].values.astype(
        np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


print("\nCreating datasets...")

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)

test_dataset = create_dataset(
    test_df,
    training=False
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            mode="horizontal"
        ),

        layers.RandomRotation(
            0.03
        ),

        layers.RandomZoom(
            0.05
        ),

    ],

    name="data_augmentation"
)


# ============================================================
# BASE MODEL
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers initially

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding V4 model...")

inputs = keras.Input(
    shape=(
        224,
        224,
        3
    )
)


x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

x = keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.Dropout(
    0.30
)(
    x
)


x = layers.Dense(
    128,
    activation="relu"
)(
    x
)


x = layers.Dropout(
    0.25
)(
    x
)


outputs = layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)


model = keras.Model(
    inputs,
    outputs,
    name="lung_cancer_mobilenet_v4"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss=keras.losses.BinaryCrossentropy(),

    metrics=[

        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.Precision(
            name="precision"
        ),

        keras.metrics.Recall(
            name="recall"
        ),

        keras.metrics.AUC(
            name="auc"
        )
    ]
)


print("\nModel summary:")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_auc",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=4,

        restore_best_weights=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1
    )
]


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 65)
print("                 STARTING V4 TRAINING")
print("=" * 65)


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best V4 model...")

model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 65)
print("                 V4 TEST EVALUATION")
print("=" * 65)


results = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nTest metrics:")

for name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()


predictions = (
    probabilities >= 0.5
).astype(int)


actual = test_df[
    "binary_label"
].values.astype(int)


# ============================================================
# METRICS
# ============================================================

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
print("=" * 65)
print("                    FINAL V4 RESULTS")
print("=" * 65)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


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


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\nProbability distribution:")

print(
    f"Minimum: {probabilities.min():.6f}"
)

print(
    f"Maximum: {probabilities.max():.6f}"
)

print(
    f"Mean:    {probabilities.mean():.6f}"
)

print(
    f"Std:     {probabilities.std():.6f}"
)


print("\nClass 0 mean probability:")

print(
    probabilities[
        actual == 0
    ].mean()
)


print("\nClass 1 mean probability:")

print(
    probabilities[
        actual == 1
    ].mean()
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "patient_id",
        "nodule_id",
        "image_path",
        "malignancy",
        "binary_label"
    ]
].copy()


prediction_df[
    "prediction_probability"
] = probabilities


prediction_df[
    "predicted_label"
] = predictions


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 65)
print("                 V4 TRAINING COMPLETE")
print("=" * 65)

print("\nModel saved to:")

print(MODEL_PATH)

print("\nPredictions saved to:")

print(PREDICTION_PATH)

print("\nTraining history saved to:")

print(HISTORY_PATH)
