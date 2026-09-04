import os
import glob
import pandas as pd
import pydicom

# ==============================
# PATHS
# ==============================

DICOM_ROOT = "/Users/sayanghosh/Desktop/manifest-1787634840820/lidc_idri"
ANNOTATION_CSV = "data/nodule_annotations_v2.csv"
OUTPUT_CSV = "data/nodule_annotations_matched.csv"


# ==============================
# STEP 1: LOAD ANNOTATIONS
# ==============================

print("Reading annotation CSV...")

df = pd.read_csv(ANNOTATION_CSV)

print(f"Annotation records: {len(df)}")


# ==============================
# STEP 2: SCAN DICOM FILES
# ==============================

print("\nScanning DICOM files...")

dicom_files = glob.glob(
    os.path.join(DICOM_ROOT, "**", "*.dcm"),
    recursive=True
)

print(f"DICOM files found: {len(dicom_files)}")


# ==============================
# STEP 3: BUILD SOP UID MAP
# ==============================

print("\nReading DICOM SOP UIDs...")

sop_to_file = {}

for i, file_path in enumerate(dicom_files):

    try:
        ds = pydicom.dcmread(
            file_path,
            stop_before_pixels=True
        )

        sop_uid = getattr(ds, "SOPInstanceUID", None)

        if sop_uid:
            sop_to_file[sop_uid] = file_path

    except Exception as e:
        print(f"Could not read: {file_path}")
        print(f"Error: {e}")

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(dicom_files)} DICOM files")


print(f"\nUnique SOP UIDs found: {len(sop_to_file)}")


# ==============================
# STEP 4: MATCH ANNOTATIONS
# ==============================

print("\nMatching annotation SOP UIDs...")

df["dicom_file"] = df["image_sop_uid"].map(sop_to_file)

df["sop_match"] = df["dicom_file"].notna()


# ==============================
# STEP 5: RESULTS
# ==============================

matched = df["sop_match"].sum()
unmatched = len(df) - matched

print("\n============================================")
print("       SOP UID MATCHING RESULTS")
print("============================================")

print(f"Total annotation records: {len(df)}")
print(f"Matched records:          {matched}")
print(f"Unmatched records:        {unmatched}")

if len(df) > 0:
    percentage = (matched / len(df)) * 100
    print(f"Match percentage:         {percentage:.2f}%")

print("============================================")


# ==============================
# STEP 6: SAVE
# ==============================

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nOutput saved to:")
print(OUTPUT_CSV)

print("\nDONE")