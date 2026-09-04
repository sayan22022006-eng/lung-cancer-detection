import os
import glob
import csv
import xml.etree.ElementTree as ET


# ============================================
# PATHS
# ============================================

XML_ROOT = "annotations/tcia-lidc-xml"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nodule_annotations.csv"
)


# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# FIND ALL XML FILES
# ============================================

xml_files = glob.glob(
    os.path.join(XML_ROOT, "**", "*.xml"),
    recursive=True
)

print("XML files found:", len(xml_files))


# ============================================
# HELPER FUNCTION
# ============================================

def get_text(element, tag_name):
    """
    Find the first XML element with the given
    tag name, ignoring XML namespaces.
    """

    for child in element.iter():

        current_tag = child.tag.split("}")[-1]

        if current_tag == tag_name:

            if child.text:
                return child.text.strip()

    return ""


# ============================================
# EXTRACT NODULE INFORMATION
# ============================================

rows = []

total_nodules = 0


for xml_file in xml_files:

    try:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # ----------------------------------------
        # Find reading sessions
        # ----------------------------------------

        for session in root.iter():

            session_tag = session.tag.split("}")[-1]

            if session_tag != "readingSession":
                continue


            # ------------------------------------
            # Radiologist
            # ------------------------------------

            radiologist = get_text(
                session,
                "servicingRadiologistID"
            )


            # ------------------------------------
            # Find nodules
            # ------------------------------------

            for nodule in session.iter():

                nodule_tag = nodule.tag.split("}")[-1]

                if nodule_tag != "unblindedReadNodule":
                    continue


                nodule_id = get_text(
                    nodule,
                    "noduleID"
                )


                # --------------------------------
                # Collect ROI information
                # --------------------------------

                rois = []

                for roi in nodule.iter():

                    roi_tag = roi.tag.split("}")[-1]

                    if roi_tag != "roi":
                        continue

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

                    x_coords = []
                    y_coords = []

                    for coord in roi.iter():

                        coord_tag = coord.tag.split("}")[-1]

                        if coord_tag == "xCoord":
                            if coord.text:
                                x_coords.append(
                                    coord.text.strip()
                                )

                        elif coord_tag == "yCoord":
                            if coord.text:
                                y_coords.append(
                                    coord.text.strip()
                                )


                    # ----------------------------
                    # Save ROI
                    # ----------------------------

                    rois.append({
                        "z": image_z,
                        "sop_uid": image_sop_uid,
                        "inclusion": inclusion,
                        "x": ",".join(x_coords),
                        "y": ",".join(y_coords)
                    })


                # --------------------------------
                # Extract characteristics
                # --------------------------------

                characteristics = ""

                for child in nodule.iter():

                    tag = child.tag.split("}")[-1]

                    if tag == "characteristics":

                        characteristics = ET.tostring(
                            child,
                            encoding="unicode"
                        )

                        break


                # --------------------------------
                # Create one row per ROI
                # --------------------------------

                for roi in rois:

                    rows.append({

                        "xml_file": xml_file,

                        "radiologist": radiologist,

                        "nodule_id": nodule_id,

                        "image_z_position": roi["z"],

                        "image_sop_uid": roi["sop_uid"],

                        "inclusion": roi["inclusion"],

                        "x_coordinates": roi["x"],

                        "y_coordinates": roi["y"],

                        "characteristics": characteristics

                    })

                    total_nodules += 1


    except Exception as e:

        print(
            "ERROR reading:",
            xml_file
        )

        print(e)


# ============================================
# SAVE CSV
# ============================================

columns = [
    "xml_file",
    "radiologist",
    "nodule_id",
    "image_z_position",
    "image_sop_uid",
    "inclusion",
    "x_coordinates",
    "y_coordinates",
    "characteristics"
]


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
# FINAL RESULT
# ============================================

print()
print("============================================")
print("       NODULE EXTRACTION COMPLETE")
print("============================================")

print(
    "XML files processed:",
    len(xml_files)
)

print(
    "Nodule/ROI records extracted:",
    total_nodules
)

print(
    "Output file:",
    OUTPUT_FILE
)

print("============================================")