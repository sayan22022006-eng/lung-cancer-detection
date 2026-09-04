import os
import pandas as pd


# ============================================
# PATHS
# ============================================

ANNOTATION_FILE = "data/nodule_annotations_v2.csv"
PREPARED_FILE = "data/prepared_dataset.csv"

CROP_ROOT = "data/nodule_crops"
OUTPUT_FILE = "data/image_dataset.csv"


# ============================================
# READ DATA
# ============================================

print("Reading annotation data...")

annotations = pd.read_csv(ANNOTATION_FILE)
prepared = pd.read_csv(PREPARED_FILE)

print(f"Annotation ROI records: {len(annotations)}")
print(f"Prepared nodules: {len(prepared)}")


# ============================================
# ADD ORIGINAL ROW NUMBER
# ============================================

annotations["roi_number"] = annotations.index


# ============================================
# VALID MALIGNANCY
# ============================================

annotations = annotations[
    annotations["malignancy"].isin([1, 2, 3, 4, 5])
].copy()


# ============================================
# CREATE NODULE KEY
# ============================================

annotations["nodule_key"] = (
    annotations["patient_id"].astype(str)
    + "_"
    + annotations["nodule_id"].astype(str)
)

prepared["nodule_key"] = (
    prepared["patient_id"].astype(str)
    + "_"
    + prepared["nodule_id"].astype(str)
)


# ============================================
# GET SPLIT INFORMATION
# ============================================

split_map = prepared[
    [
        "nodule_key",
        "split",
        "binary_label"
    ]
].drop_duplicates("nodule_key")


# ============================================
# MATCH SPLIT TO ROI RECORDS
# ============================================

print()
print("Connecting ROI records to patient splits...")

annotations = annotations.merge(
    split_map,
    on="nodule_key",
    how="inner"
)

print(
    f"Matched ROI records: {len(annotations)}"
)


# ============================================
# CREATE IMAGE RECORDS
# ============================================

print()
print("Creating image dataset...")


records = []
missing = 0


for _, row in annotations.iterrows():

    patient_id = str(row["patient_id"])
    nodule_id = str(row["nodule_id"])
    roi_number = int(row["roi_number"])

    image_path = os.path.join(
        CROP_ROOT,
        patient_id,
        f"nodule_{nodule_id}_roi_{roi_number}.png"
    )


    if os.path.exists(image_path):

        records.append(
            {
                "patient_id": patient_id,
                "nodule_id": nodule_id,
                "image_path": image_path,
                "malignancy": int(row["malignancy"]),
                "binary_label": int(row["binary_label"]),
                "split": row["split"]
            }
        )

    else:

        missing += 1


# ============================================
# CREATE DATAFRAME
# ============================================

image_df = pd.DataFrame(records)


# ============================================
# REMOVE DUPLICATES
# ============================================

image_df = image_df.drop_duplicates(
    subset=["image_path"]
)


# ============================================
# SAVE
# ============================================

image_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================
# RESULTS
# ============================================

print()
print("============================================")
print("       IMAGE DATASET CREATED")
print("============================================")

print(
    f"Image records: {len(image_df)}"
)

print(
    f"Missing crops: {missing}"
)

print()

print("Images per split:")

if len(image_df) > 0:

    print(
        image_df["split"]
        .value_counts()
    )

print()

print("Malignancy distribution:")

if len(image_df) > 0:

    print(
        image_df["malignancy"]
        .value_counts()
        .sort_index()
    )

print()

print("Patients per split:")

if len(image_df) > 0:

    print(
        image_df
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