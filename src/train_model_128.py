import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
layers = keras.layers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "image_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lung_cancer_model_128_improved.keras"
)

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "test_predictions_128_improved.csv"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "training_history_128_improved.csv"
)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 60)
print("LUNG CANCER DETECTION - IMPROVED 128x128 CNN")
print("=" * 60)

print(f"TensorFlow version: {tf.__version__}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print()


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Total image records: {len(df)}")
print()

print("Dataset columns:")
print(df.columns.tolist())
print()


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patient_id",
    "nodule_id",
    "image_path",
    "binary_label",
    "split"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in image_dataset.csv"
        )


# ============================================================
# 5. CONVERT IMAGE PATHS TO ABSOLUTE PATHS
# ============================================================

def make_absolute_path(path):
    path = str(path)

    if os.path.isabs(path):
        return path

    return os.path.join(BASE_DIR, path)


df["image_path"] = df["image_path"].apply(make_absolute_path)


# ============================================================
# 6. CHECK IMAGE FILES
# ============================================================

print("Checking image files...")

missing_files = []

for path in df["image_path"]:
    if not os.path.exists(path):
        missing_files.append(path)

if len(missing_files) > 0:
    print(f"WARNING: {len(missing_files)} image files are missing.")

    print("First few missing files:")

    for path in missing_files[:10]:
        print(path)

    raise FileNotFoundError(
        "Some image files are missing. Please check image_dataset.csv."
    )

else:
    print("All image files found.")


print()


# ============================================================
# 7. DISPLAY SPLIT INFORMATION
# ============================================================

print("Dataset split information:")

print(
    df.groupby("split")["image_path"]
    .count()
)

print()


# ============================================================
# 8. CREATE TRAIN / VALIDATION / TEST DATAFRAMES
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))
print()


# ============================================================
# 9. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("Training class distribution:")

print(
    train_df["binary_label"]
    .value_counts()
    .sort_index()
)

print()

print("Validation class distribution:")

print(
    val_df["binary_label"]
    .value_counts()
    .sort_index()
)

print()

print("Test class distribution:")

print(
    test_df["binary_label"]
    .value_counts()
    .sort_index()
)

print()


# ============================================================
# 10. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):
    """
    Loads a grayscale PNG image.

    Output shape:
        (128, 128, 1)

    Pixel values:
        0 to 1
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

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


# ============================================================
# 11. CREATE TF.DATA DATASET
# ============================================================

def create_dataset(dataframe, shuffle=False):
    paths = dataframe["image_path"].values
    labels = dataframe["binary_label"].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
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


# ============================================================
# 12. CREATE DATASETS
# ============================================================

print("Creating TensorFlow datasets...")

train_dataset = create_dataset(
    train_df,
    shuffle=True
)

val_dataset = create_dataset(
    val_df,
    shuffle=False
)

test_dataset = create_dataset(
    test_df,
    shuffle=False
)

print("Datasets created successfully.")
print()


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

classes = np.unique(
    train_df["binary_label"]
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["binary_label"]
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        class_weights_array
    )
}

print("Class weights:")
print(class_weights)
print()


# ============================================================
# 14. BUILD IMPROVED CNN MODEL
# ============================================================

print("Building improved 128x128 CNN...")


model = keras.Sequential(
    [

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        layers.Input(
            shape=(128, 128, 1)
        ),


        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        layers.RandomFlip(
            "horizontal"
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

        layers.Dropout(
            0.15
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

        layers.Dropout(
            0.20
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

        layers.Dropout(
            0.25
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

        layers.Dropout(
            0.30
        ),


        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.40
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ],

    name="lung_cancer_cnn_128_improved"
)


# ============================================================
# 15. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
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


# ============================================================
# 16. MODEL SUMMARY
# ============================================================

print()

model.summary()

print()


# ============================================================
# 17. CALLBACKS
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = keras.callbacks.EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# 18. TRAIN MODEL
# ============================================================

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print()

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 19. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    range(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

print()
print(
    f"Training history saved to: {HISTORY_PATH}"
)


# ============================================================
# 20. LOAD BEST MODEL
# ============================================================

print()
print("Loading best saved model...")

if os.path.exists(MODEL_PATH):

    model = keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Best model loaded successfully."
    )

else:

    print(
        "Best model file not found."
    )

    print(
        "Using current model."
    )


# ============================================================
# 21. EVALUATE ON TEST SET
# ============================================================

print()
print("=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

test_results = model.evaluate(
    test_dataset,
    verbose=1
)

print()


for metric_name, metric_value in zip(
    model.metrics_names,
    test_results
):

    print(
        f"{metric_name}: {metric_value:.4f}"
    )


# ============================================================
# 22. GET TEST PREDICTIONS
# ============================================================

print()
print("Generating test predictions...")

prediction_probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()


# ============================================================
# 23. TRUE LABELS
# ============================================================

y_true = test_df[
    "binary_label"
].values.astype(int)


# ============================================================
# 24. CONVERT PROBABILITIES TO LABELS
# ============================================================

THRESHOLD = 0.50

y_pred = (
    prediction_probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 25. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

try:

    auc = roc_auc_score(
        y_true,
        prediction_probabilities
    )

except ValueError:

    auc = float("nan")


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# 27. PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {auc:.4f}"
)

print()

print("Confusion Matrix:")

print(cm)

print()

print("Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Non-Malignant",
            "Malignant"
        ],
        zero_division=0
    )
)


# ============================================================
# 28. SAVE TEST PREDICTIONS
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
] = prediction_probabilities


prediction_df[
    "predicted_label"
] = y_pred


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


print(
    f"Test predictions saved to: {PREDICTION_PATH}"
)


# ============================================================
# 29. SHOW PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print()

print("Predicted classes:")

print(
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
)


print()

print("Prediction probability statistics:")

print(
    pd.Series(
        prediction_probabilities
    ).describe()
)


# ============================================================
# 30. CHECK SEPARATION BETWEEN CLASSES
# ============================================================

print()
print("=" * 60)
print("PROBABILITY BY ACTUAL CLASS")
print("=" * 60)

probability_analysis = pd.DataFrame(
    {
        "actual_label": y_true,
        "prediction_probability":
            prediction_probabilities
    }
)

print(
    probability_analysis
    .groupby("actual_label")
    ["prediction_probability"]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "max"
        ]
    )
)


# ============================================================
# 31. TRAINING SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

best_epoch = (
    history_df["val_auc"].idxmax() + 1
)

best_val_auc = (
    history_df["val_auc"].max()
)

best_val_accuracy = (
    history_df["val_accuracy"].max()
)

best_val_loss = (
    history_df["val_loss"].min()
)

print(
    f"Best epoch       : {best_epoch}"
)

print(
    f"Best val AUC     : {best_val_auc:.4f}"
)

print(
    f"Best val accuracy: {best_val_accuracy:.4f}"
)

print(
    f"Best val loss    : {best_val_loss:.4f}"
)


# ============================================================
# 32. FILE SUMMARY
# ============================================================

print()
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print()

print(
    f"Model:"
)

print(
    MODEL_PATH
)

print()

print(
    f"Predictions:"
)

print(
    PREDICTION_PATH
)

print()

print(
    f"Training history:"
)

print(
    HISTORY_PATH
)

print()

print("=" * 60)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 60)