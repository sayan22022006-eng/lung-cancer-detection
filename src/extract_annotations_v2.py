import os
import glob
import csv
import pydicom
import xml.etree.ElementTree as ET


# ============================================
# PATHS
# ============================================

DICOM_ROOT = os.path.expanduser(
    "~/Desktop/manifest-1787634840820/lidc_idri"
)

XML_ROOT = "annotations/tcia-lidc-xml"

OUTPUT_DIR = "data"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nodule_annotations_v2.csv"
)


# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# XML HELPER
# ============================================

def clean_tag(tag):
    """
    Remove XML namespace from a tag.
    """

    return tag.split("}")[-1]


def get_text(element, tag_name):
    """
    Find the first occurrence of tag_name
    inside an XML element.
    """

    for child in element.iter():

        if clean_tag(child.tag) == tag_name:

            if child.text:
                return child.text.strip()

    return ""


# ============================================
# STEP 1
# BUILD DICOM SERIES DATABASE
# ============================================

print("Reading DICOM dataset...")

dicom_series = {}

patient_folders = glob.glob(
    os.path.join(DICOM_ROOT, "LIDC-IDRI-*")
)

print(
    "Patient folders found:",
    len(patient_folders)
)


for patient_path in patient_folders:

    patient_id = os.path.basename(patient_path)

    dcm_files = glob.glob(
        os.path.join(
            patient_path,
            "**",
            "*.dcm"
        ),
        recursive=True
    )

    if not dcm_files:
        continue

    try:

        # Read one DICOM file for metadata
        ds = pydicom.dcmread(
            dcm_files[0],
            stop_before_pixels=True
        )

        series_uid = str(
            ds.SeriesInstanceUID
        )

        dicom_series[series_uid] = {
            "patient_id": patient_id,
            "series_uid": series_uid,
            "folder": os.path.dirname(
                dcm_files[0]
            ),
            "slice_count": len(dcm_files)
        }

    except Exception as e:

        print(
            "Could not read:",
            patient_id
        )

        print(e)


print(
    "DICOM series found:",
    len(dicom_series)
)


# ============================================
# STEP 2
# FIND XML FILES
# ============================================

print()
print("Reading annotation files...")

xml_files = glob.glob(
    os.path.join(
        XML_ROOT,
        "**",
        "*.xml"
    ),
    recursive=True
)

print(
    "XML files found:",
    len(xml_files)
)


# ============================================
# OUTPUT COLUMNS
# ============================================

columns = [

    "patient_id",

    "series_uid",

    "xml_file",

    "radiologist",

    "nodule_id",

    "image_sop_uid",

    "image_z_position",

    "inclusion",

    "x_coordinates",

    "y_coordinates",

    "subtlety",

    "internal_structure",

    "calcification",

    "sphericity",

    "margin",

    "lobulation",

    "spiculation",

    "texture",

    "malignancy"

]


rows = []


# ============================================
# STEP 3
# PROCESS XML FILES
# ============================================

print()
print("Extracting annotation information...")


matched_xml = 0

nodule_count = 0

roi_count = 0


for xml_file in xml_files:

    try:

        tree = ET.parse(xml_file)

        root = tree.getroot()


        # ----------------------------------------
        # Find Series UID
        # ----------------------------------------

        series_uid = None

        for element in root.iter():

            tag = clean_tag(
                element.tag
            )

            if tag in [
                "SeriesInstanceUid",
                "CTSeriesInstanceUid"
            ]:

                if element.text:

                    series_uid = (
                        element.text.strip()
                    )

                    break


        # ----------------------------------------
        # Match against DICOM
        # ----------------------------------------

        if series_uid not in dicom_series:

            continue


        matched_xml += 1


        patient_id = dicom_series[
            series_uid
        ]["patient_id"]


        # ----------------------------------------
        # Reading sessions
        # ----------------------------------------

        for session in root.iter():

            if clean_tag(
                session.tag
            ) != "readingSession":

                continue


            radiologist = get_text(
                session,
                "servicingRadiologistID"
            )


            # ------------------------------------
            # Find nodules
            # ------------------------------------

            for nodule in session.iter():

                if clean_tag(
                    nodule.tag
                ) != "unblindedReadNodule":

                    continue


                nodule_count += 1


                nodule_id = get_text(
                    nodule,
                    "noduleID"
                )


                # --------------------------------
                # Find characteristics
                # --------------------------------

                characteristic_values = {

                    "subtlety": "",
                    "internal_structure": "",
                    "calcification": "",
                    "sphericity": "",
                    "margin": "",
                    "lobulation": "",
                    "spiculation": "",
                    "texture": "",
                    "malignancy": ""

                }


                for element in nodule.iter():

                    tag = clean_tag(
                        element.tag
                    )

                    if tag in characteristic_values:

                        if element.text:

                            characteristic_values[
                                tag
                            ] = element.text.strip()


                # --------------------------------
                # Find ROIs
                # --------------------------------

                for roi in nodule.iter():

                    if clean_tag(
                        roi.tag
                    ) != "roi":

                        continue


                    roi_count += 1


                    image_z = get_text(
                        roi,
                        "imageZposition"
                    )

                    image_sop_uid = get_text(
                        roi,
                        "imageSOP_UID"
                    )

                    inclusion = get_text(
                        roi,
                        "inclusion"
                    )


                    # ----------------------------
                    # Coordinates
                    # ----------------------------

                    x_coords = []

                    y_coords = []


                    for coord in roi.iter():

                        tag = clean_tag(
                            coord.tag
                        )


                        if tag == "xCoord":

                            if coord.text:

                                x_coords.append(
                                    coord.text.strip()
                                )


                        elif tag == "yCoord":

                            if coord.text:

                                y_coords.append(
                                    coord.text.strip()
                                )


                    # ----------------------------
                    # Save row
                    # ----------------------------

                    rows.append({

                        "patient_id":
                            patient_id,

                        "series_uid":
                            series_uid,

                        "xml_file":
                            xml_file,

                        "radiologist":
                            radiologist,

                        "nodule_id":
                            nodule_id,

                        "image_sop_uid":
                            image_sop_uid,

                        "image_z_position":
                            image_z,

                        "inclusion":
                            inclusion,

                        "x_coordinates":
                            ",".join(x_coords),

                        "y_coordinates":
                            ",".join(y_coords),

                        "subtlety":
                            characteristic_values[
                                "subtlety"
                            ],

                        "internal_structure":
                            characteristic_values[
                                "internal_structure"
                            ],

                        "calcification":
                            characteristic_values[
                                "calcification"
                            ],

                        "sphericity":
                            characteristic_values[
                                "sphericity"
                            ],

                        "margin":
                            characteristic_values[
                                "margin"
                            ],

                        "lobulation":
                            characteristic_values[
                                "lobulation"
                            ],

                        "spiculation":
                            characteristic_values[
                                "spiculation"
                            ],

                        "texture":
                            characteristic_values[
                                "texture"
                            ],

                        "malignancy":
                            characteristic_values[
                                "malignancy"
                            ]

                    })


    except Exception as e:

        print()
        print(
            "ERROR:",
            xml_file
        )

        print(e)


# ============================================
# STEP 4
# SAVE CSV
# ============================================

print()
print("Saving dataset...")


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================
# STEP 5
# SUMMARY
# ============================================

print()
print("============================================")
print("       ANNOTATION EXTRACTION V2")
print("============================================")

print(
    "DICOM series:",
    len(dicom_series)
)

print(
    "XML files:",
    len(xml_files)
)

print(
    "Matched XML files:",
    matched_xml
)

print(
    "Nodules found:",
    nodule_count
)

print(
    "ROI records:",
    roi_count
)

print(
    "CSV rows:",
    len(rows)
)

print()
print(
    "Output:",
    OUTPUT_FILE
)

print("============================================")
print("                 DONE")
print("============================================")