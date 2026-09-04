import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "data/image_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lung_cancer_model.keras")
PREDICTION_PATH = "data/test_predictions.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("        LUNG CANCER MODEL TRAINING - V3")
print("=" * 60)


# ============================================================
# READ DATASET
# ============================================================

print("\nReading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Total image records: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")

print("\nSplit distribution:")
print(df["split"].value_counts())


# ============================================================
# CHECK IMAGE FILES
# ============================================================

print("\nChecking image files...")

missing_files = []

for path in df["image_path"]:
    if not os.path.exists(path):
        missing_files.append(path)

print(f"Missing image files: {len(missing_files)}")

if missing_files:
    print("\nExample missing files:")
    for path in missing_files[:10]:
        print(path)

    raise FileNotFoundError(
        f"{len(missing_files)} image files are missing."
    )


# ============================================================
# SPLIT DATA
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nDataset sizes:")
print(f"Training:   {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")
print(train_df["binary_label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["binary_label"].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df["binary_label"].value_counts().sort_index())


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
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):
    """
    Loads a grayscale nodule image and converts it to
    a normalized 224x224 tensor.
    """

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


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(dataframe, training=False):

    paths = dataframe["image_path"].values
    labels = dataframe["binary_label"].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
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

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


print("\nCreating TensorFlow datasets...")

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
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD CNN
# ============================================================

print("\nBuilding local CNN model...")

model = keras.Sequential(
    [

        layers.Input(
            shape=(224, 224, 1)
        ),

        data_augmentation,

        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # BLOCK 3
        # ----------------------------------------------------

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # BLOCK 4
        # ----------------------------------------------------

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dropout(
            0.40
        ),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.30
        ),

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ],

    name="lung_cancer_cnn"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel architecture:\n")

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss=keras.losses.BinaryCrossentropy(),

    metrics=[
        "accuracy",

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

        patience=5,

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
print("=" * 60)
print("                 STARTING TRAINING")
print("=" * 60)

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

print("\nLoading best model...")

model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("                 TEST EVALUATION")
print("=" * 60)

results = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest results:")

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

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)

actual = test_df[
    "binary_label"
].values


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("             CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        actual,
        predictions,
        target_names=[
            "Non-malignant",
            "Malignant"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("                 CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    actual,
    predictions
)

print(cm)


# ============================================================
# ADDITIONAL METRICS
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

print("\n")
print("=" * 60)
print("                FINAL METRICS")
print("=" * 60)

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
# TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_path = "data/training_history.csv"

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("             TRAINING COMPLETE")
print("=" * 60)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nTest predictions saved to:")
print(PREDICTION_PATH)

print("\nTraining history saved to:")
print(history_path)

print("\n")
print("=" * 60)
print("                    DONE")
print("=" * 60)