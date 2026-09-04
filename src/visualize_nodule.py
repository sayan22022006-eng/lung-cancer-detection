import os
import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# PATHS
# ============================================

CSV_FILE = "data/nodule_annotations_matched.csv"
OUTPUT_DIR = "data/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# READ ANNOTATIONS
# ============================================

print("Reading annotation data...")

df = pd.read_csv(CSV_FILE)

print(f"Total annotation records: {len(df)}")


# ============================================
# SELECT ONE ANNOTATION
# ============================================

row = df.iloc[0]

patient_id = row["patient_id"]
nodule_id = row["nodule_id"]
dicom_file = row["dicom_file"]

print("\n============================================")
print("SELECTED NODULE")
print("============================================")

print(f"Patient:    {patient_id}")
print(f"Nodule ID:  {nodule_id}")
print(f"DICOM:      {dicom_file}")


# ============================================
# READ DICOM
# ============================================

print("\nReading DICOM slice...")

ds = pydicom.dcmread(dicom_file)

image = ds.pixel_array.astype(np.float32)


# ============================================
# CONVERT TO HU
# ============================================

slope = float(getattr(ds, "RescaleSlope", 1))
intercept = float(getattr(ds, "RescaleIntercept", 0))

image = image * slope + intercept


# ============================================
# GET X/Y COORDINATES
# ============================================

x_values = [
    int(x)
    for x in str(row["x_coordinates"]).split(",")
    if x.strip()
]

y_values = [
    int(y)
    for y in str(row["y_coordinates"]).split(",")
    if y.strip()
]


print(f"Number of annotation points: {len(x_values)}")


# ============================================
# CREATE VISUALIZATION
# ============================================

plt.figure(figsize=(8, 8))

plt.imshow(
    image,
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.plot(
    x_values,
    y_values,
    linewidth=2
)

plt.scatter(
    x_values,
    y_values,
    s=8
)

plt.title(
    f"{patient_id} | Nodule {nodule_id}"
)

plt.axis("off")


# ============================================
# SAVE IMAGE
# ============================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"{patient_id}_nodule_{nodule_id}.png"
)

plt.savefig(
    output_file,
    bbox_inches="tight",
    dpi=150
)

print("\n============================================")
print("VISUALIZATION COMPLETE")
print("============================================")

print(f"Saved to:")
print(output_file)

print("============================================")

plt.show()