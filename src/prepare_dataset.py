import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================
# PATHS
# ============================================

ANNOTATION_FILE = "data/nodule_annotations_v2.csv"

OUTPUT_DIR = "data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "prepared_dataset.csv"
)


# ============================================
# SETTINGS
# ============================================

RANDOM_STATE = 42


# ============================================
# READ ANNOTATIONS
# ============================================

print("Reading annotation dataset...")

df = pd.read_csv(ANNOTATION_FILE)

print(
    f"Original ROI records: {len(df)}"
)


# ============================================
# KEEP ONLY VALID MALIGNANCY LABELS
# ============================================

df = df[
    df["malignancy"].isin(
        [1, 2, 3, 4, 5]
    )
].copy()

print(
    f"Records with valid malignancy: {len(df)}"
)


# ============================================
# ONE RECORD PER NODULE
# ============================================
#
# A nodule can have multiple ROI records
# because the same nodule appears across
# multiple CT slices.
#
# We therefore group by:
#
# patient_id + nodule_id
#
# and keep the malignancy label.
#
# ============================================

print()
print("Creating nodule-level dataset...")


def majority_label(series):
    """
    Select the most common malignancy
    score for a nodule.
    """

    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return values.mode().iloc[0]


nodule_df = (
    df.groupby(
        [
            "patient_id",
            "nodule_id"
        ],
        as_index=False
    )
    .agg(
        malignancy=(
            "malignancy",
            majority_label
        ),

        series_uid=(
            "series_uid",
            "first"
        ),

        xml_file=(
            "xml_file",
            "first"
        ),

        radiologist=(
            "radiologist",
            "first"
        )
    )
)


print(
    f"Unique nodules: {len(nodule_df)}"
)


# ============================================
# CREATE NUMERIC LABEL
# ============================================

nodule_df["label"] = (
    nodule_df["malignancy"]
    .astype(int)
)


# ============================================
# CREATE BINARY LABEL
# ============================================
#
# For the first version of the model:
#
# 1,2,3 -> Lower suspicion
# 4,5   -> Higher suspicion
#
# This gives us a simpler binary
# classification problem.
#
# ============================================

nodule_df["binary_label"] = np.where(
    nodule_df["malignancy"] >= 4,
    1,
    0
)


# ============================================
# SHOW DISTRIBUTION
# ============================================

print()
print("============================================")
print("NODULE DISTRIBUTION")
print("============================================")

print(
    nodule_df["label"]
    .value_counts()
    .sort_index()
)


print()
print("Binary distribution:")

print(
    nodule_df["binary_label"]
    .value_counts()
    .sort_index()
)


# ============================================
# PATIENT-LEVEL SPLIT
# ============================================
#
# VERY IMPORTANT:
#
# We split patients, NOT individual images.
#
# This prevents images from the same patient
# appearing in both training and testing.
#
# ============================================

print()
print("Creating patient-level train/validation/test split...")


patients = nodule_df[
    "patient_id"
].unique()


print(
    f"Total patients: {len(patients)}"
)


# --------------------------------------------
# 70% TRAIN
# 30% TEMPORARY
# --------------------------------------------

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=RANDOM_STATE
)


# --------------------------------------------
# 15% VALIDATION
# 15% TEST
# --------------------------------------------

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_STATE
)


# ============================================
# ASSIGN SPLIT
# ============================================

nodule_df["split"] = "train"

nodule_df.loc[
    nodule_df["patient_id"].isin(
        val_patients
    ),
    "split"
] = "validation"

nodule_df.loc[
    nodule_df["patient_id"].isin(
        test_patients
    ),
    "split"
] = "test"


# ============================================
# ADD CROP DIRECTORY
# ============================================

nodule_df["crop_directory"] = (
    "data/nodule_crops/"
    + nodule_df["patient_id"]
)


# ============================================
# SAVE
# ============================================

nodule_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================
# RESULTS
# ============================================

print()
print("============================================")
print("       DATASET PREPARATION COMPLETE")
print("============================================")

print(
    f"Total nodules: {len(nodule_df)}"
)

print(
    f"Training patients: {len(train_patients)}"
)

print(
    f"Validation patients: {len(val_patients)}"
)

print(
    f"Test patients: {len(test_patients)}"
)

print()

print("Nodules per split:")

print(
    nodule_df["split"]
    .value_counts()
)


print()
print("Patients per split:")

print(
    nodule_df
    .groupby("split")["patient_id"]
    .nunique()
)


print()
print(
    f"Output file: {OUTPUT_FILE}"
)

print("============================================")
print("                 DONE")
print("============================================")