import os
import glob
import pydicom
import xml.etree.ElementTree as ET


# ============================================
# PATHS
# ============================================

DICOM_ROOT = os.path.expanduser(
    "~/Desktop/manifest-1787634840820/lidc_idri"
)

XML_ROOT = "annotations/tcia-lidc-xml"


# ============================================
# STEP 1: READ DICOM DATASET
# ============================================

print("Reading DICOM dataset...")

dicom_series = {}

patient_folders = glob.glob(
    os.path.join(DICOM_ROOT, "LIDC-IDRI-*")
)

print("Patient folders found:", len(patient_folders))


for patient_path in patient_folders:

    patient_id = os.path.basename(patient_path)

    # Search for ALL DICOM files inside this patient folder
    dcm_files = glob.glob(
        os.path.join(patient_path, "**", "*.dcm"),
        recursive=True
    )

    if not dcm_files:
        print("No DICOM files found:", patient_id)
        continue

    try:

        # Read first DICOM file only for metadata
        ds = pydicom.dcmread(
            dcm_files[0],
            stop_before_pixels=True
        )

        series_uid = ds.SeriesInstanceUID

        dicom_series[series_uid] = {
            "patient": patient_id,
            "folder": os.path.dirname(dcm_files[0]),
            "slices": len(dcm_files)
        }

    except Exception as e:

        print("Could not read:", patient_id)
        print("Error:", e)


print("DICOM series found:", len(dicom_series))


# ============================================
# STEP 2: FIND XML ANNOTATIONS
# ============================================

print("\nReading annotation files...")

xml_files = glob.glob(
    os.path.join(XML_ROOT, "**", "*.xml"),
    recursive=True
)

print("XML files found:", len(xml_files))


# ============================================
# STEP 3: MATCH XML WITH DICOM
# ============================================

print("\nMatching annotations with CT scans...")

matches = {}


for xml_file in xml_files:

    try:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        series_uid = None

        # Search XML for SeriesInstanceUid
        for element in root.iter():

            tag_name = element.tag.split("}")[-1]

            if tag_name in [
                "SeriesInstanceUid",
                "CTSeriesInstanceUid"
            ]:

                if element.text:

                    series_uid = element.text.strip()
                    break


        # Check whether this XML belongs to one
        # of our downloaded CT series

        if series_uid in dicom_series:

            patient_id = dicom_series[series_uid]["patient"]

            if patient_id not in matches:

                matches[patient_id] = []

            matches[patient_id].append(xml_file)


    except Exception as e:

        print("Could not read XML:")
        print(xml_file)
        print("Error:", e)


# ============================================
# STEP 4: DISPLAY RESULTS
# ============================================

print("\n")
print("============================================")
print("       ANNOTATION MATCHING RESULTS")
print("============================================")

total_patients = len(patient_folders)

patients_with_annotations = len(matches)

patients_without_annotations = (
    total_patients - patients_with_annotations
)

print("Total patients:", total_patients)

print(
    "Patients with annotations:",
    patients_with_annotations
)

print(
    "Patients without annotations:",
    patients_without_annotations
)


# ============================================
# STEP 5: SHOW MATCHED PATIENTS
# ============================================

print("\nMatched patients:\n")


for patient_id in sorted(matches):

    annotation_count = len(matches[patient_id])

    print(
        patient_id,
        "->",
        annotation_count,
        "annotation files"
    )


# ============================================
# STEP 6: SHOW PATIENTS WITHOUT ANNOTATIONS
# ============================================

print("\nPatients without annotations:\n")


matched_patient_ids = set(matches.keys())


for patient_path in sorted(patient_folders):

    patient_id = os.path.basename(patient_path)

    if patient_id not in matched_patient_ids:

        print(patient_id)


# ============================================
# FINISHED
# ============================================

print("\n")
print("============================================")
print("                 DONE")
print("============================================")