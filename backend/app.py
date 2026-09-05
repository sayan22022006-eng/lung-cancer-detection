import os
import sys
import json
import sqlite3
import uuid
import tempfile
from datetime import datetime
from io import BytesIO
from html import escape

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file
)

from flask_cors import CORS
from werkzeug.utils import secure_filename

import pymupdf

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)


# ============================================================
# PROJECT PATH
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BACKEND_DIR
)

sys.path.insert(
    0,
    PROJECT_DIR
)


# ============================================================
# IMPORT MODELS
# ============================================================

from src.predict_image import (
    load_model,
    predict_image
)

from src.text_classifier.predict_text import (
    predict_text
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        BACKEND_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        BACKEND_DIR,
        "static"
    ),
    static_url_path="/static"
)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# STORAGE PATHS
# ============================================================

UPLOAD_DIR = os.path.join(
    BACKEND_DIR,
    "uploads"
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

DB_PATH = os.path.join(
    DATA_DIR,
    "prediction_history.db"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}

PDF_EXTENSION = ".pdf"

IMAGE_MODEL_NAME = (
    "lung_cancer_model_v5.keras"
)

TEXT_MODEL_NAME = (
    "tfidf_logistic_model.joblib"
)

IMAGE_THRESHOLD = 0.50

TEXT_THRESHOLD = 0.63

MIN_IMAGE_WIDTH = 200

MIN_IMAGE_HEIGHT = 200


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            saved_file TEXT NOT NULL,
            file_type TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL,
            model TEXT,
            details TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()

    print(
        "Prediction history database ready."
    )


def save_prediction(
    filename,
    saved_file,
    file_type,
    prediction,
    probability=None,
    model=None,
    details=None
):

    connection = get_db_connection()

    details_json = None

    if details is not None:

        details_json = json.dumps(
            details
        )

    cursor = connection.execute(
        """
        INSERT INTO predictions (
            filename,
            saved_file,
            file_type,
            prediction,
            probability,
            model,
            details,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            saved_file,
            file_type,
            prediction,
            probability,
            model,
            details_json,
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
    )

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


def get_prediction_history(
    limit=50
):

    connection = get_db_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            filename,
            saved_file,
            file_type,
            prediction,
            probability,
            model,
            details,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    connection.close()

    history = []

    for row in rows:

        item = dict(row)

        if item["details"]:

            try:

                item["details"] = json.loads(
                    item["details"]
                )

            except json.JSONDecodeError:

                pass

        history.append(item)

    return history


def delete_prediction_record(
    prediction_id
):

    connection = get_db_connection()

    row = connection.execute(
        """
        SELECT saved_file
        FROM predictions
        WHERE id = ?
        """,
        (prediction_id,)
    ).fetchone()

    if not row:

        connection.close()

        return None

    connection.execute(
        """
        DELETE FROM predictions
        WHERE id = ?
        """,
        (prediction_id,)
    )

    connection.commit()

    connection.close()

    return row["saved_file"]


# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_database()


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

print(
    "Loading lung cancer V5 model..."
)

model = load_model()

print(
    "V5 model loaded successfully."
)


# ============================================================
# WEBSITE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok",

        "image_model":
            IMAGE_MODEL_NAME,

        "text_model":
            TEXT_MODEL_NAME,

        "image_threshold":
            IMAGE_THRESHOLD,

        "text_threshold":
            TEXT_THRESHOLD,

        "model_loaded":
            model is not None

    })


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded."

        }), 400

    image = request.files["image"]

    if image.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No image selected."

        }), 400

    original_filename = image.filename

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in IMAGE_EXTENSIONS:

        return jsonify({

            "success": False,

            "error":
                (
                    "Unsupported file type. "
                    "Only PNG, JPG and JPEG "
                    "images are supported."
                )

        }), 400

    saved_filename = (
        f"{uuid.uuid4().hex}_"
        f"{secure_filename(original_filename)}"
    )

    saved_path = os.path.join(
        UPLOAD_DIR,
        saved_filename
    )

    try:

        image.save(
            saved_path
        )

        result = predict_image(
            saved_path,
            model=model
        )

        prediction_id = save_prediction(

            filename=original_filename,

            saved_file=os.path.relpath(
                saved_path,
                PROJECT_DIR
            ),

            file_type="image",

            prediction=result[
                "prediction"
            ],

            probability=result.get(
                "malignant_probability"
            ),

            model=IMAGE_MODEL_NAME,

            details=result

        )

        return jsonify({

            "success": True,

            "type": "image",

            "filename":
                original_filename,

            "history_id":
                prediction_id,

            "saved_file":
                os.path.relpath(
                    saved_path,
                    PROJECT_DIR
                ),

            "model":
                IMAGE_MODEL_NAME,

            "result":
                result

        }), 200

    except Exception as e:

        print(
            "IMAGE PREDICTION ERROR:",
            repr(e)
        )

        if os.path.exists(
            saved_path
        ):

            os.remove(
                saved_path
            )

        return jsonify({

            "success": False,

            "error":
                "Image prediction failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# PDF HELPER FUNCTIONS
# ============================================================

def extract_page_text(page):

    try:

        text = page.get_text(
            "text"
        )

        if text:

            return text.strip()

        return ""

    except Exception:

        return ""


def get_page_images(page):

    try:

        return page.get_images(
            full=True
        )

    except Exception:

        return []


def classify_pdf_text(text):

    if not text or not text.strip():

        return None

    try:

        return predict_text(
            text
        )

    except Exception as e:

        print(
            "TEXT CLASSIFICATION ERROR:",
            repr(e)
        )

        return {

            "error":
                "Text classification failed.",

            "details":
                str(e)

        }


def get_embedded_image_info(
    doc,
    image_info
):

    xref = image_info[0]

    try:

        image_data = (
            doc.extract_image(
                xref
            )
        )

        if not image_data:

            return None

        return {

            "xref":
                xref,

            "width":
                image_data.get(
                    "width",
                    0
                ),

            "height":
                image_data.get(
                    "height",
                    0
                ),

            "ext":
                image_data.get(
                    "ext",
                    "png"
                ),

            "image":
                image_data.get(
                    "image"
                )

        }

    except Exception as e:

        print(
            "IMAGE EXTRACTION ERROR:",
            repr(e)
        )

        return None


def predict_embedded_image(
    doc,
    image_info
):

    image_data = (
        get_embedded_image_info(
            doc,
            image_info
        )
    )

    if not image_data:

        return None

    width = image_data[
        "width"
    ]

    height = image_data[
        "height"
    ]

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):

        return {

            "status":
                "ignored",

            "reason":
                "Image too small.",

            "width":
                width,

            "height":
                height

        }

    image_bytes = (
        image_data["image"]
    )

    if not image_bytes:

        return None

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=(
                f".{image_data['ext']}"
            ),
            delete=False
        ) as temp_file:

            temp_file.write(
                image_bytes
            )

            temp_path = (
                temp_file.name
            )

        result = predict_image(
            temp_path,
            model=model
        )

        return {

            "status":
                "processed",

            "width":
                width,

            "height":
                height,

            "result":
                result

        }

    except Exception as e:

        print(
            "EMBEDDED IMAGE PREDICTION ERROR:",
            repr(e)
        )

        return {

            "status":
                "error",

            "error":
                "Image prediction failed.",

            "details":
                str(e)

        }

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


def render_pdf_page(
    doc,
    page_number
):

    temp_path = None

    try:

        page = doc[
            page_number
        ]

        matrix = pymupdf.Matrix(
            150 / 72,
            150 / 72
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp_file:

            pixmap.save(
                temp_file.name
            )

            temp_path = (
                temp_file.name
            )

        result = predict_image(
            temp_path,
            model=model
        )

        return result

    except Exception as e:

        print(
            "RENDERED PAGE PREDICTION ERROR:",
            repr(e)
        )

        return {

            "error":
                "Rendered page prediction failed.",

            "details":
                str(e)

        }

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# ============================================================
# PDF PREDICTION
# ============================================================

@app.route(
    "/api/predict-pdf",
    methods=["POST"]
)
def predict_pdf():

    if "pdf" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No PDF uploaded."

        }), 400

    pdf_file = request.files["pdf"]

    if pdf_file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No PDF selected."

        }), 400

    original_filename = (
        pdf_file.filename
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension != PDF_EXTENSION:

        return jsonify({

            "success": False,

            "error":
                "Only PDF files are supported."

        }), 400

    saved_filename = (
        f"{uuid.uuid4().hex}_"
        f"{secure_filename(original_filename)}"
    )

    saved_path = os.path.join(
        UPLOAD_DIR,
        saved_filename
    )

    document = None

    try:

        pdf_file.save(
            saved_path
        )

        document = pymupdf.open(
            saved_path
        )

        page_count = len(
            document
        )

        if page_count == 0:

            os.remove(
                saved_path
            )

            return jsonify({

                "success": False,

                "error":
                    "The PDF contains no pages."

            }), 400

        pages = []

        text_results = []

        image_results = []

        text_probabilities = []

        image_probabilities = []

        text_pages = 0

        image_pages = 0

        # ====================================================
        # PROCESS EVERY PAGE
        # ====================================================

        for page_index in range(
            page_count
        ):

            page_number = (
                page_index + 1
            )

            page = document[
                page_index
            ]

            text = extract_page_text(
                page
            )

            embedded_images = (
                get_page_images(
                    page
                )
            )

            page_result = {

                "page":
                    page_number,

                "text":
                    None,

                "images":
                    [],

                "rendered_page":
                    None

            }

            # =================================================
            # TEXT
            # =================================================

            if text:

                text_pages += 1

                text_result = (
                    classify_pdf_text(
                        text
                    )
                )

                page_result[
                    "text"
                ] = {

                    "characters":
                        len(text),

                    "prediction":
                        text_result

                }

                if (
                    text_result
                    and "probability"
                    in text_result
                    and "error"
                    not in text_result
                ):

                    text_results.append({

                        "page":
                            page_number,

                        "result":
                            text_result

                    })

                    text_probabilities.append(
                        text_result[
                            "probability"
                        ]
                    )

            # =================================================
            # EMBEDDED IMAGES
            # =================================================

            processed_images = []

            for (
                image_index,
                image_info
            ) in enumerate(
                embedded_images
            ):

                result = (
                    predict_embedded_image(
                        document,
                        image_info
                    )
                )

                if not result:

                    continue

                result[
                    "image_index"
                ] = (
                    image_index + 1
                )

                processed_images.append(
                    result
                )

                if (
                    result.get(
                        "status"
                    ) == "processed"
                    and "result"
                    in result
                ):

                    prediction = (
                        result[
                            "result"
                        ]
                    )

                    if (
                        "malignant_probability"
                        in prediction
                    ):

                        image_probabilities.append(
                            prediction[
                                "malignant_probability"
                            ]
                        )

                        image_results.append({

                            "page":
                                page_number,

                            "image_index":
                                image_index + 1,

                            "result":
                                prediction

                        })

            if processed_images:

                image_pages += 1

                page_result[
                    "images"
                ] = (
                    processed_images
                )

            # =================================================
            # IMAGE-ONLY / SCANNED PAGE
            # =================================================

            if (
                not text
                and not embedded_images
            ):

                rendered_result = (
                    render_pdf_page(
                        document,
                        page_index
                    )
                )

                if rendered_result:

                    page_result[
                        "rendered_page"
                    ] = (
                        rendered_result
                    )

                    if (
                        "malignant_probability"
                        in rendered_result
                        and "error"
                        not in rendered_result
                    ):

                        image_pages += 1

                        image_probabilities.append(
                            rendered_result[
                                "malignant_probability"
                            ]
                        )

                        image_results.append({

                            "page":
                                page_number,

                            "source":
                                "rendered_page",

                            "result":
                                rendered_result

                        })

            pages.append(
                page_result
            )

        # ====================================================
        # IMAGE SUMMARY
        # ====================================================

        image_summary = None

        if image_probabilities:

            average_probability = (
                sum(
                    image_probabilities
                )
                / len(
                    image_probabilities
                )
            )

            maximum_probability = max(
                image_probabilities
            )

            prediction = (

                "MALIGNANT"

                if average_probability
                >= IMAGE_THRESHOLD

                else

                "NON-MALIGNANT"

            )

            image_summary = {

                "prediction":
                    prediction,

                "average_malignant_probability":
                    average_probability,

                "maximum_malignant_probability":
                    maximum_probability,

                "images_processed":
                    len(
                        image_probabilities
                    ),

                "threshold":
                    IMAGE_THRESHOLD

            }

        # ====================================================
        # TEXT SUMMARY
        # ====================================================

        text_summary = None

        if text_probabilities:

            average_probability = (
                sum(
                    text_probabilities
                )
                / len(
                    text_probabilities
                )
            )

            prediction = (

                "PARENCHYMAL_LESION"

                if average_probability
                >= TEXT_THRESHOLD

                else

                "NO_PARENCHYMAL_LESION"

            )

            text_summary = {

                "prediction":
                    prediction,

                "average_probability":
                    average_probability,

                "pages_with_text_prediction":
                    len(
                        text_probabilities
                    ),

                "threshold":
                    TEXT_THRESHOLD

            }

        # ====================================================
        # DOCUMENT TYPE
        # ====================================================

        if (
            text_results
            and image_results
        ):

            document_type = "mixed"

        elif text_results:

            document_type = "text"

        elif image_results:

            document_type = "image"

        else:

            document_type = "unknown"

        # ====================================================
        # HISTORY
        # ====================================================

        history_prediction = "UNKNOWN"

        history_probability = None

        history_model = "NONE"

        if document_type == "image":

            history_prediction = (
                image_summary[
                    "prediction"
                ]
            )

            history_probability = (
                image_summary[
                    "average_malignant_probability"
                ]
            )

            history_model = (
                IMAGE_MODEL_NAME
            )

        elif document_type == "text":

            history_prediction = (
                text_summary[
                    "prediction"
                ]
            )

            history_probability = (
                text_summary[
                    "average_probability"
                ]
            )

            history_model = (
                TEXT_MODEL_NAME
            )

        elif document_type == "mixed":

            history_prediction = "MIXED"

            history_probability = None

            history_model = (
                f"{IMAGE_MODEL_NAME} + "
                f"{TEXT_MODEL_NAME}"
            )

        # ====================================================
        # DETAILS
        # ====================================================

        details = {

            "document_type":
                document_type,

            "pages_processed":
                page_count,

            "text_pages":
                text_pages,

            "image_pages":
                image_pages,

            "summary": {

                "image":
                    image_summary,

                "text":
                    text_summary

            },

            "text_results":
                text_results,

            "image_results":
                image_results,

            "pages":
                pages

        }

        prediction_id = save_prediction(

            filename=
                original_filename,

            saved_file=
                os.path.relpath(
                    saved_path,
                    PROJECT_DIR
                ),

            file_type="pdf",

            prediction=
                history_prediction,

            probability=
                history_probability,

            model=
                history_model,

            details=
                details

        )

        return jsonify({

            "success": True,

            "type": "pdf",

            "filename":
                original_filename,

            "history_id":
                prediction_id,

            "saved_file":
                os.path.relpath(
                    saved_path,
                    PROJECT_DIR
                ),

            "document_type":
                document_type,

            "pages_processed":
                page_count,

            "text_pages":
                text_pages,

            "image_pages":
                image_pages,

            "models": {

                "image_model":
                    IMAGE_MODEL_NAME,

                "text_model":
                    TEXT_MODEL_NAME,

                "image_threshold":
                    IMAGE_THRESHOLD,

                "text_threshold":
                    TEXT_THRESHOLD

            },

            "summary": {

                "image":
                    image_summary,

                "text":
                    text_summary

            },

            "text_results":
                text_results,

            "image_results":
                image_results,

            "pages":
                pages

        }), 200

    except Exception as e:

        print(
            "PDF PROCESSING ERROR:",
            repr(e)
        )

        if os.path.exists(
            saved_path
        ):

            os.remove(
                saved_path
            )

        return jsonify({

            "success": False,

            "error":
                "PDF processing failed.",

            "details":
                str(e)

        }), 500

    finally:

        if document:

            document.close()


# ============================================================
# REPORT GENERATION
#
# ALL THREE ENDPOINTS USE THE SAME FUNCTION
# ============================================================

@app.route(
    "/api/generate-report",
    methods=["POST"]
)
@app.route(
    "/api/generate-image-report",
    methods=["POST"]
)
@app.route(
    "/api/generate-pdf-report",
    methods=["POST"]
)
def generate_report():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "REPORT GENERATION REQUEST"
    )

    print(
        "=" * 60
    )

    try:

        # ====================================================
        # READ JSON
        # ====================================================

        data = request.get_json(
            silent=True
        )

        print(
            "Received JSON:",
            data is not None
        )

        if not data:

            print(
                "ERROR: No JSON data received."
            )

            return jsonify({

                "success": False,

                "error":
                    "No report data provided."

            }), 400

        # ====================================================
        # ACCEPT MULTIPLE FIELD NAMES
        # ====================================================

        report_type = (
            data.get("type")
            or data.get("report_type")
        )

        report_data = (
            data.get("data")
            or data.get("report_data")
            or {}
        )

        print(
            "Report type:",
            report_type
        )

        # ====================================================
        # AUTOMATICALLY DETECT TYPE FROM URL
        # ====================================================

        request_path = request.path

        if request_path == (
            "/api/generate-image-report"
        ):

            report_type = "image"

        elif request_path == (
            "/api/generate-pdf-report"
        ):

            report_type = "pdf"

        # ====================================================
        # VALIDATE TYPE
        # ====================================================

        if report_type not in {
            "image",
            "pdf"
        }:

            print(
                "ERROR: Invalid report type:",
                report_type
            )

            return jsonify({

                "success": False,

                "error":
                    (
                        "Invalid report type. "
                        "Expected 'image' or 'pdf'."
                    )

            }), 400

        # ====================================================
        # VALIDATE DATA
        # ====================================================

        if not isinstance(
            report_data,
            dict
        ):

            return jsonify({

                "success": False,

                "error":
                    "Invalid report data."

            }), 400

        # ====================================================
        # FILENAME
        # ====================================================

        filename = (

            report_data.get(
                "filename"
            )

            or report_data.get(
                "file_name"
            )

            or "analysis"

        )

        filename = os.path.basename(
            str(filename)
        )

        original_name = os.path.splitext(
            filename
        )[0]

        safe_report_name = (
            secure_filename(
                original_name
            )
            or "analysis"
        )

        report_filename = (
            f"{safe_report_name}_"
            f"analysis_report.pdf"
        )

        print(
            "Report filename:",
            report_filename
        )

        # ====================================================
        # CREATE PDF BUFFER
        # ====================================================

        buffer = BytesIO()

        pdf_document = SimpleDocTemplate(

            buffer,

            pagesize=A4,

            rightMargin=18 * mm,

            leftMargin=18 * mm,

            topMargin=18 * mm,

            bottomMargin=18 * mm

        )

        styles = (
            getSampleStyleSheet()
        )

        # ====================================================
        # REPORT STYLES
        # ====================================================

        title_style = ParagraphStyle(

            "ReportTitle",

            parent=styles["Title"],

            fontName="Helvetica-Bold",

            fontSize=22,

            leading=26,

            alignment=TA_CENTER,

            textColor=
                colors.HexColor(
                    "#172033"
                ),

            spaceAfter=8

        )

        subtitle_style = ParagraphStyle(

            "Subtitle",

            parent=styles["Normal"],

            fontName="Helvetica",

            fontSize=10,

            leading=14,

            alignment=TA_CENTER,

            textColor=
                colors.HexColor(
                    "#64748b"
                ),

            spaceAfter=18

        )

        heading_style = ParagraphStyle(

            "Heading",

            parent=styles["Heading2"],

            fontName="Helvetica-Bold",

            fontSize=13,

            leading=17,

            textColor=
                colors.HexColor(
                    "#172033"
                ),

            spaceBefore=14,

            spaceAfter=8

        )

        normal_style = ParagraphStyle(

            "NormalReport",

            parent=styles["Normal"],

            fontName="Helvetica",

            fontSize=10,

            leading=15,

            textColor=
                colors.HexColor(
                    "#334155"
                )

        )

        small_style = ParagraphStyle(

            "SmallReport",

            parent=styles["Normal"],

            fontName="Helvetica",

            fontSize=8,

            leading=12,

            textColor=
                colors.HexColor(
                    "#64748b"
                )

        )

        story = []

        # ====================================================
        # HEADER
        # ====================================================

        story.append(

            Paragraph(
                "Lung Cancer Detection",
                title_style
            )

        )

        story.append(

            Paragraph(
                "AI-Powered Medical Image and "
                "Radiology Report Analysis",
                subtitle_style
            )

        )

        story.append(

            HRFlowable(

                width="100%",

                thickness=1,

                color=
                    colors.HexColor(
                        "#e2e8f0"
                    ),

                spaceAfter=15

            )

        )

        story.append(

            Paragraph(
                "ACADEMIC / RESEARCH ANALYSIS REPORT",
                heading_style
            )

        )

        escaped_filename = escape(
            filename
        )

        story.append(

            Paragraph(
                f"<b>File:</b> "
                f"{escaped_filename}",
                normal_style
            )

        )

        story.append(
            Spacer(1, 8)
        )

        # ====================================================
        # IMAGE REPORT
        # ====================================================

        if report_type == "image":

            print(
                "Generating IMAGE report..."
            )

            result = (
                report_data.get(
                    "result",
                    {}
                )
            )

            if not isinstance(
                result,
                dict
            ):

                result = {}

            prediction = str(
                result.get(
                    "prediction",
                    "UNKNOWN"
                )
            )

            try:

                malignant_probability = float(
                    result.get(
                        "malignant_probability",
                        0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                malignant_probability = 0.0

            try:

                non_malignant_probability = float(
                    result.get(
                        "non_malignant_probability",
                        1 -
                        malignant_probability
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                non_malignant_probability = (
                    1 -
                    malignant_probability
                )

            try:

                threshold = float(
                    result.get(
                        "threshold",
                        IMAGE_THRESHOLD
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                threshold = IMAGE_THRESHOLD

            uncertainty = bool(
                result.get(
                    "uncertainty",
                    False
                )
            )

            story.append(

                Paragraph(
                    "CT Image Analysis",
                    heading_style
                )

            )

            image_data = [

                [
                    Paragraph(
                        "<b>Parameter</b>",
                        normal_style
                    ),

                    Paragraph(
                        "<b>Result</b>",
                        normal_style
                    )

                ],

                [
                    "Model Prediction",
                    escape(prediction)
                ],

                [
                    "Malignant Probability",
                    f"{malignant_probability * 100:.2f}%"
                ],

                [
                    "Non-Malignant Probability",
                    f"{non_malignant_probability * 100:.2f}%"
                ],

                [
                    "Decision Threshold",
                    f"{threshold:.2f}"
                ],

                [
                    "Uncertainty",
                    (
                        "Detected"
                        if uncertainty
                        else
                        "Not detected"
                    )
                ],

                [
                    "Model",
                    "MobileNetV2 V5"
                ]

            ]

            table = Table(

                image_data,

                colWidths=[
                    70 * mm,
                    95 * mm
                ]

            )

            table.setStyle(

                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#f1f5f9"
                        )
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor(
                            "#e2e8f0"
                        )
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )

                ])

            )

            story.append(
                table
            )

        # ====================================================
        # PDF REPORT
        # ====================================================

        elif report_type == "pdf":

            print(
                "Generating PDF analysis report..."
            )

            summary = (
                report_data.get(
                    "summary",
                    {}
                )
            )

            if not isinstance(
                summary,
                dict
            ):

                summary = {}

            document_type = (
                report_data.get(
                    "document_type"
                )
                or "unknown"
            )

            pages = (
                report_data.get(
                    "pages",
                    []
                )
            )

            image_results = (
                report_data.get(
                    "image_results",
                    []
                )
            )

            text_results = (
                report_data.get(
                    "text_results",
                    []
                )
            )

            image_summary = (
                summary.get(
                    "image"
                )
            )

            text_summary = (
                summary.get(
                    "text"
                )
            )

            story.append(

                Paragraph(
                    "PDF Document Analysis",
                    heading_style
                )

            )

            pdf_data = [

                [
                    Paragraph(
                        "<b>Parameter</b>",
                        normal_style
                    ),

                    Paragraph(
                        "<b>Result</b>",
                        normal_style
                    )

                ],

                [
                    "Document Type",
                    escape(
                        str(
                            document_type
                        ).title()
                    )
                ],

                [
                    "Total Pages",
                    str(
                        report_data.get(
                            "pages_processed",
                            len(pages)
                        )
                    )
                ],

                [
                    "Pages with Images",
                    str(
                        report_data.get(
                            "image_pages",
                            0
                        )
                    )
                ],

                [
                    "Pages with Text",
                    str(
                        report_data.get(
                            "text_pages",
                            0
                        )
                    )
                ]

            ]

            # ------------------------------------------------
            # IMAGE SUMMARY
            # ------------------------------------------------

            if isinstance(
                image_summary,
                dict
            ):

                image_prediction = (
                    image_summary.get(
                        "prediction"
                    )
                )

                image_probability = (
                    image_summary.get(
                        "average_malignant_probability"
                    )
                )

                images_processed = (
                    image_summary.get(
                        "images_processed"
                    )
                )

                if image_prediction:

                    pdf_data.append([

                        "Image Model Prediction",

                        escape(
                            str(
                                image_prediction
                            )
                        )

                    ])

                if isinstance(
                    image_probability,
                    (int, float)
                ):

                    pdf_data.append([

                        "Average Malignant Probability",

                        f"{image_probability * 100:.2f}%"

                    ])

                if images_processed is not None:

                    pdf_data.append([

                        "Images Processed",

                        str(
                            images_processed
                        )

                    ])

            # ------------------------------------------------
            # TEXT SUMMARY
            # ------------------------------------------------

            if isinstance(
                text_summary,
                dict
            ):

                text_prediction = (
                    text_summary.get(
                        "prediction"
                    )
                )

                text_probability = (
                    text_summary.get(
                        "average_probability"
                    )
                )

                text_pages_predicted = (
                    text_summary.get(
                        "pages_with_text_prediction"
                    )
                )

                if text_prediction:

                    pdf_data.append([

                        "Text Model Prediction",

                        escape(
                            str(
                                text_prediction
                            )
                        )

                    ])

                if isinstance(
                    text_probability,
                    (int, float)
                ):

                    pdf_data.append([

                        "Average Text Probability",

                        f"{text_probability * 100:.2f}%"

                    ])

                if text_pages_predicted is not None:

                    pdf_data.append([

                        "Text Pages Classified",

                        str(
                            text_pages_predicted
                        )

                    ])

            # ------------------------------------------------
            # MODELS
            # ------------------------------------------------

            if image_results:

                pdf_data.append([

                    "Image Model",

                    "MobileNetV2 V5"

                ])

            if text_results:

                pdf_data.append([

                    "Text Model",

                    "TF-IDF + Logistic Regression"

                ])

            table = Table(

                pdf_data,

                colWidths=[
                    70 * mm,
                    95 * mm
                ]

            )

            table.setStyle(

                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#f1f5f9"
                        )
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor(
                            "#e2e8f0"
                        )
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )

                ])

            )

            story.append(
                table
            )

            # ------------------------------------------------
            # IMAGE DETAILS
            # ------------------------------------------------

            if image_results:

                story.append(

                    Paragraph(
                        "Image Analysis Details",
                        heading_style
                    )

                )

                for item in image_results:

                    page_number = item.get(
                        "page",
                        "?"
                    )

                    image_index = item.get(
                        "image_index"
                    )

                    result = item.get(
                        "result",
                        {}
                    )

                    if not isinstance(
                        result,
                        dict
                    ):

                        result = {}

                    prediction = str(
                        result.get(
                            "prediction",
                            "UNKNOWN"
                        )
                    )

                    probability = result.get(
                        "malignant_probability"
                    )

                    if isinstance(
                        probability,
                        (int, float)
                    ):

                        probability_text = (
                            f"{probability * 100:.2f}%"
                        )

                    else:

                        probability_text = "N/A"

                    if image_index:

                        source_text = (
                            f"Page {page_number}, "
                            f"Image {image_index}"
                        )

                    else:

                        source_text = (
                            f"Page {page_number}"
                        )

                    story.append(

                        Paragraph(

                            f"<b>{escape(str(source_text))}</b> "
                            f"— {escape(prediction)} "
                            f"({probability_text} "
                            f"malignant probability)",

                            normal_style

                        )

                    )

                    story.append(
                        Spacer(1, 4)
                    )

            # ------------------------------------------------
            # TEXT DETAILS
            # ------------------------------------------------

            if text_results:

                story.append(

                    Paragraph(
                        "Text Analysis Details",
                        heading_style
                    )

                )

                for item in text_results:

                    page_number = item.get(
                        "page",
                        "?"
                    )

                    result = item.get(
                        "result",
                        {}
                    )

                    if not isinstance(
                        result,
                        dict
                    ):

                        result = {}

                    prediction = str(
                        result.get(
                            "prediction",
                            "UNKNOWN"
                        )
                    )

                    probability = result.get(
                        "probability"
                    )

                    if isinstance(
                        probability,
                        (int, float)
                    ):

                        probability_text = (
                            f"{probability * 100:.2f}%"
                        )

                    else:

                        probability_text = "N/A"

                    story.append(

                        Paragraph(

                            f"<b>Page {page_number}</b> "
                            f"— {escape(prediction)} "
                            f"({probability_text} probability)",

                            normal_style

                        )

                    )

                    story.append(
                        Spacer(1, 4)
                    )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        story.append(
            Spacer(1, 20)
        )

        story.append(

            Paragraph(
                "Important Research Disclaimer",
                heading_style
            )

        )

        story.append(

            Paragraph(

                "This report is generated by an "
                "academic/research prototype. The "
                "results are produced by machine-learning "
                "models trained on research datasets and "
                "are intended for experimentation and "
                "educational purposes only. This system "
                "is not a clinical diagnostic system and "
                "the output must not be used for medical "
                "decisions.",

                normal_style

            )

        )

        story.append(
            Spacer(1, 15)
        )

        story.append(

            Paragraph(

                "Generated by Lung Cancer Detection Project",

                small_style

            )

        )

        # ====================================================
        # BUILD PDF
        # ====================================================

        print(
            "Building PDF..."
        )

        pdf_document.build(
            story
        )

        buffer.seek(0)

        print(
            "PDF generated successfully."
        )

        print(
            "Returning:",
            report_filename
        )

        print(
            "=" * 60
        )

        return send_file(

            buffer,

            mimetype="application/pdf",

            as_attachment=True,

            download_name=
                report_filename

        )

    except Exception as e:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "REPORT GENERATION ERROR"
        )

        print(
            "=" * 60
        )

        print(
            "Exception:",
            repr(e)
        )

        import traceback

        traceback.print_exc()

        print(
            "=" * 60
        )

        return jsonify({

            "success": False,

            "error":
                "Report generation failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
def history():

    try:

        limit = request.args.get(
            "limit",
            default=50,
            type=int
        )

        limit = max(
            1,
            min(
                limit,
                100
            )
        )

        records = (
            get_prediction_history(
                limit
            )
        )

        return jsonify({

            "success": True,

            "count":
                len(records),

            "history":
                records

        }), 200

    except Exception as e:

        print(
            "HISTORY RETRIEVAL ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error":
                "Could not retrieve prediction history.",

            "details":
                str(e)

        }), 500


# ============================================================
# DELETE HISTORY RECORD
# ============================================================

@app.route(
    "/api/history/<int:prediction_id>",
    methods=["DELETE"]
)
def delete_history(
    prediction_id
):

    try:

        saved_file = (
            delete_prediction_record(
                prediction_id
            )
        )

        if saved_file is None:

            return jsonify({

                "success": False,

                "error":
                    "Prediction history record not found."

            }), 404

        file_path = os.path.join(
            PROJECT_DIR,
            saved_file
        )

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )

        return jsonify({

            "success": True,

            "message":
                "Prediction history deleted.",

            "deleted_id":
                prediction_id

        }), 200

    except Exception as e:

        print(
            "HISTORY DELETION ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error":
                "Could not delete prediction history.",

            "details":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LUNG CANCER DETECTION BACKEND"
    )

    print(
        "=" * 60
    )

    print(
        f"Image model: "
        f"{IMAGE_MODEL_NAME}"
    )

    print(
        f"Text model: "
        f"{TEXT_MODEL_NAME}"
    )

    print(
        f"Image threshold: "
        f"{IMAGE_THRESHOLD}"
    )

    print(
        f"Text threshold: "
        f"{TEXT_THRESHOLD}"
    )

    print(
        f"Upload directory: "
        f"{UPLOAD_DIR}"
    )

    print(
        f"History database: "
        f"{DB_PATH}"
    )

    print(
        "Server: "
        "http://127.0.0.1:5000"
    )

    print(
        "Website: "
        "http://127.0.0.1:5000/"
    )

    print(
        "Health: "
        "http://127.0.0.1:5000/api/health"
    )

    print(
        "Image prediction: "
        "http://127.0.0.1:5000/api/predict"
    )

    print(
        "PDF prediction: "
        "http://127.0.0.1:5000/api/predict-pdf"
    )

    print(
        "General report generation: "
        "http://127.0.0.1:5000/api/generate-report"
    )

    print(
        "Image report generation: "
        "http://127.0.0.1:5000/api/generate-image-report"
    )

    print(
        "PDF report generation: "
        "http://127.0.0.1:5000/api/generate-pdf-report"
    )

    print(
        "Prediction history: "
        "http://127.0.0.1:5000/api/history"
    )

    print(
        "=" * 60
        + "\n"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )