import csv
import os
from collections import Counter


# ============================================
# PATH
# ============================================

CSV_FILE = "data/nodule_annotations.csv"


# ============================================
# CHECK FILE
# ============================================

if not os.path.exists(CSV_FILE):
    print("ERROR: CSV file not found:")
    print(CSV_FILE)
    exit()


print("Reading:", CSV_FILE)


# ============================================
# READ CSV
# ============================================

with open(
    CSV_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print("Total records:", len(rows))


# ============================================
# UNIQUE PATIENTS
# ============================================

patients = set()

for row in rows:

    xml_file = row["xml_file"]

    # Extract patient ID from path
    parts = xml_file.split(os.sep)

    for part in parts:

        if part.startswith("LIDC-IDRI-"):
            patients.add(part)


print()
print("============================================")
print("PATIENT INFORMATION")
print("============================================")

print(
    "Unique patients:",
    len(patients)
)


# ============================================
# UNIQUE NODULES
# ============================================

unique_nodules = set()

for row in rows:

    patient = ""

    parts = row["xml_file"].split(os.sep)

    for part in parts:

        if part.startswith("LIDC-IDRI-"):
            patient = part
            break

    nodule_id = row["nodule_id"]

    unique_nodules.add(
        (patient, nodule_id)
    )


print(
    "Unique patient+nodule combinations:",
    len(unique_nodules)
)


# ============================================
# RECORDS PER PATIENT
# ============================================

patient_counts = Counter()

for row in rows:

    parts = row["xml_file"].split(os.sep)

    for part in parts:

        if part.startswith("LIDC-IDRI-"):

            patient_counts[part] += 1
            break


print()
print("============================================")
print("RECORDS PER PATIENT")
print("============================================")

for patient in sorted(patient_counts):

    print(
        patient,
        "->",
        patient_counts[patient],
        "records"
    )


# ============================================
# INCLUSION VALUES
# ============================================

inclusion_counts = Counter(
    row["inclusion"]
    for row in rows
)


print()
print("============================================")
print("INCLUSION VALUES")
print("============================================")

for value, count in inclusion_counts.items():

    print(
        repr(value),
        "->",
        count
    )


# ============================================
# MISSING VALUES
# ============================================

fields_to_check = [
    "nodule_id",
    "image_z_position",
    "image_sop_uid",
    "inclusion",
    "x_coordinates",
    "y_coordinates",
    "characteristics"
]


print()
print("============================================")
print("MISSING VALUES")
print("============================================")

for field in fields_to_check:

    missing = sum(
        1
        for row in rows
        if not row[field].strip()
    )

    print(
        field,
        "->",
        missing,
        "missing"
    )


# ============================================
# SAMPLE RECORDS
# ============================================

print()
print("============================================")
print("SAMPLE RECORDS")
print("============================================")

for i, row in enumerate(rows[:5], start=1):

    print()
    print("RECORD", i)

    print("Patient/XML:", row["xml_file"])
    print("Radiologist:", row["radiologist"])
    print("Nodule ID:", row["nodule_id"])
    print("Z position:", row["image_z_position"])
    print("SOP UID:", row["image_sop_uid"])
    print("Inclusion:", row["inclusion"])

    print(
        "X coordinates:",
        row["x_coordinates"][:100]
    )

    print(
        "Y coordinates:",
        row["y_coordinates"][:100]
    )

    print(
        "Characteristics:",
        row["characteristics"][:300]
    )


# ============================================
# FINISHED
# ============================================

print()
print("============================================")
print("CHECK COMPLETE")
print("============================================")