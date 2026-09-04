import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image


# ============================================
# PATHS
# ============================================

CSV_FILE = "data/nodule_annotations_matched.csv"

OUTPUT_DIR = "data/nodule_crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# SETTINGS
# ============================================

# Larger crop than the original 64x64
CROP_SIZE = 128

HALF_SIZE = CROP_SIZE // 2


# ============================================
# READ ANNOTATIONS
# ============================================

print("Reading matched annotations...")

df = pd.read_csv(CSV_FILE)

print(f"Total ROI records: {len(df)}")


# ============================================
# HELPER: CONVERT CT TO DISPLAY IMAGE
# ============================================

def normalize_ct(image):

    # Lung window
    window_center = -600
    window_width = 1500

    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    image = np.clip(
        image,
        lower,
        upper
    )

    image = (
        (image - lower)
        / (upper - lower)
        * 255
    )

    return image.astype(np.uint8)


# ============================================
# HELPER: GET CENTER OF NODULE
# ============================================

def get_nodule_center(x_string, y_string):

    try:

        x_values = [
            float(x)
            for x in str(x_string).split(",")
            if x.strip()
        ]

        y_values = [
            float(y)
            for y in str(y_string).split(",")
            if y.strip()
        ]

        if not x_values or not y_values:
            return None, None

        center_x = int(
            round(np.mean(x_values))
        )

        center_y = int(
            round(np.mean(y_values))
        )

        return center_x, center_y

    except Exception:

        return None, None


# ============================================
# PROCESSING
# ============================================

print()
print("Creating 128x128 nodule image crops...")
print()


created = 0
skipped = 0


for index, row in df.iterrows():

    dicom_file = row["dicom_file"]


    # ----------------------------------------
    # Check DICOM path
    # ----------------------------------------

    if not isinstance(dicom_file, str):

        skipped += 1
        continue

    if not os.path.exists(dicom_file):

        print(
            f"Missing DICOM: {dicom_file}"
        )

        skipped += 1
        continue


    # ----------------------------------------
    # Get nodule center
    # ----------------------------------------

    center_x, center_y = get_nodule_center(
        row["x_coordinates"],
        row["y_coordinates"]
    )

    if center_x is None or center_y is None:

        skipped += 1
        continue


    # ----------------------------------------
    # Read DICOM
    # ----------------------------------------

    try:

        ds = pydicom.dcmread(
            dicom_file
        )

        image = ds.pixel_array.astype(
            np.float32
        )

    except Exception as e:

        print(
            f"Could not read DICOM: {dicom_file}"
        )

        print(e)

        skipped += 1
        continue


    # ----------------------------------------
    # Convert to Hounsfield Units
    # ----------------------------------------

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0
        )
    )

    image = (
        image * slope
        + intercept
    )


    # ----------------------------------------
    # Image dimensions
    # ----------------------------------------

    height, width = image.shape


    # ----------------------------------------
    # Check coordinates
    # ----------------------------------------

    if (
        center_x < 0
        or center_x >= width
        or center_y < 0
        or center_y >= height
    ):

        skipped += 1
        continue


    # ----------------------------------------
    # Normalize CT
    # ----------------------------------------

    image_8bit = normalize_ct(image)


    # ----------------------------------------
    # Create 128x128 crop
    # ----------------------------------------

    x1 = center_x - HALF_SIZE
    x2 = center_x + HALF_SIZE

    y1 = center_y - HALF_SIZE
    y2 = center_y + HALF_SIZE


    # ----------------------------------------
    # Handle image boundaries
    # ----------------------------------------

    crop = np.zeros(
        (
            CROP_SIZE,
            CROP_SIZE
        ),
        dtype=np.uint8
    )


    source_x1 = max(
        0,
        x1
    )

    source_x2 = min(
        width,
        x2
    )

    source_y1 = max(
        0,
        y1
    )

    source_y2 = min(
        height,
        y2
    )


    dest_x1 = source_x1 - x1
    dest_x2 = dest_x1 + (
        source_x2 - source_x1
    )

    dest_y1 = source_y1 - y1
    dest_y2 = dest_y1 + (
        source_y2 - source_y1
    )


    crop[
        dest_y1:dest_y2,
        dest_x1:dest_x2
    ] = image_8bit[
        source_y1:source_y2,
        source_x1:source_x2
    ]


    # ----------------------------------------
    # Patient / Nodule information
    # ----------------------------------------

    patient_id = str(
        row["patient_id"]
    )

    nodule_id = str(
        row["nodule_id"]
    )


    # ----------------------------------------
    # Patient directory
    # ----------------------------------------

    patient_dir = os.path.join(
        OUTPUT_DIR,
        patient_id
    )

    os.makedirs(
        patient_dir,
        exist_ok=True
    )


    # ----------------------------------------
    # Filename
    # ----------------------------------------

    filename = (
        f"nodule_{nodule_id}"
        f"_roi_{index}"
        f".png"
    )

    output_file = os.path.join(
        patient_dir,
        filename
    )


    # ----------------------------------------
    # Save crop
    # ----------------------------------------

    Image.fromarray(
        crop
    ).save(
        output_file
    )


    created += 1


    # ----------------------------------------
    # Progress
    # ----------------------------------------

    if created % 100 == 0:

        print(
            f"Crops created: {created}"
        )


# ============================================
# COMPLETE
# ============================================

print()
print("============================================")
print("       128x128 NODULE DATASET CREATED")
print("============================================")

print(
    f"ROI records processed: {len(df)}"
)

print(
    f"Image crops created:   {created}"
)

print(
    f"Records skipped:       {skipped}"
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print("============================================")
print("                 DONE")
print("============================================")