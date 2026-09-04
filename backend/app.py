import os
import sys
import json
import sqlite3
import uuid
import tempfile
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pymupdf


# ==================================================
# PROJECT PATH
# ==================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, PROJECT_DIR)


# ==================================================
# IMPORT MODELS
# ==================================================

from src.predict_image import load_model, predict_image
from src.text_classifier.predict_text import predict_text


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)
CORS(app)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==================================================
# STORAGE PATHS
# ==================================================

UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "prediction_history.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ==================================================
# CONSTANTS
# ==================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}

PDF_EXTENSION = ".pdf"

IMAGE_MODEL_NAME = "lung_cancer_model_v5.keras"
TEXT_MODEL_NAME = "tfidf_logistic_model.joblib"

IMAGE_THRESHOLD = 0.50
TEXT_THRESHOLD = 0.63

MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200


# ==================================================
# DATABASE
# ==================================================

def get_db_connection():
    """
    Create a connection to the SQLite prediction history database.
    """

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def init_database():
    """
    Create the prediction history table if it does not exist.
    """

    connection = get_db_connection()

    connection.execute("""
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
    """)

    connection.commit()
    connection.close()

    print("Prediction history database ready.")


def save_prediction(
    filename,
    saved_file,
    file_type,
    prediction,
    probability=None,
    model=None,
    details=None
):
    """
    Save one prediction to the history database.
    """

    connection = get_db_connection()

    details_json = None

    if details is not None:
        details_json = json.dumps(details)

    cursor = connection.execute("""
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
    """, (
        filename,
        saved_file,
        file_type,
        prediction,
        probability,
        model,
        details_json,
        datetime.now().isoformat(timespec="seconds")
    ))

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


def get_prediction_history(limit=50):
    """
    Return previous prediction records.
    """

    connection = get_db_connection()

    rows = connection.execute("""
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
    """, (limit,)).fetchall()

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


def delete_prediction_record(prediction_id):
    """
    Delete a prediction record and return its saved file path.
    """

    connection = get_db_connection()

    row = connection.execute("""
        SELECT saved_file
        FROM predictions
        WHERE id = ?
    """, (prediction_id,)).fetchone()

    if not row:

        connection.close()

        return None

    connection.execute("""
        DELETE FROM predictions
        WHERE id = ?
    """, (prediction_id,))

    connection.commit()

    connection.close()

    return row["saved_file"]


# ==================================================
# INITIALIZE DATABASE
# ==================================================

init_database()


# ==================================================
# LOAD IMAGE MODEL ONCE
# ==================================================

print("Loading lung cancer V5 model...")

model = load_model()

print("V5 model loaded successfully.")


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "image_model": IMAGE_MODEL_NAME,
        "text_model": TEXT_MODEL_NAME,
        "image_threshold": IMAGE_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "model_loaded": model is not None
    })


# ==================================================
# IMAGE PREDICTION
# ==================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image uploaded."
        }), 400

    image = request.files["image"]

    if image.filename == "":

        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400

    original_filename = image.filename

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in IMAGE_EXTENSIONS:

        return jsonify({
            "success": False,
            "error": (
                "Unsupported file type. "
                "Only PNG, JPG and JPEG images "
                "are supported."
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

        # --------------------------------------------------
        # SAVE UPLOADED FILE
        # --------------------------------------------------

        image.save(saved_path)

        # --------------------------------------------------
        # RUN MODEL
        # --------------------------------------------------

        result = predict_image(
            saved_path,
            model=model
        )

        # --------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------

        prediction_id = save_prediction(
            filename=original_filename,
            saved_file=os.path.relpath(
                saved_path,
                PROJECT_DIR
            ),
            file_type="image",
            prediction=result["prediction"],
            probability=result.get(
                "malignant_probability"
            ),
            model=IMAGE_MODEL_NAME,
            details=result
        )

        # --------------------------------------------------
        # RESPONSE
        # --------------------------------------------------

        return jsonify({
            "success": True,
            "type": "image",
            "filename": original_filename,
            "history_id": prediction_id,
            "saved_file": os.path.relpath(
                saved_path,
                PROJECT_DIR
            ),
            "model": IMAGE_MODEL_NAME,
            "result": result
        }), 200

    except Exception as e:

        print(
            "Image prediction error:",
            str(e)
        )

        # Remove uploaded file if prediction failed.
        if os.path.exists(saved_path):

            os.remove(saved_path)

        return jsonify({
            "success": False,
            "error": "Image prediction failed.",
            "details": str(e)
        }), 500


# ==================================================
# PDF HELPER FUNCTIONS
# ==================================================

def extract_page_text(page):
    """
    Extract text from a PDF page.
    """

    try:

        text = page.get_text("text")

        if text:
            return text.strip()

        return ""

    except Exception:

        return ""


def get_page_images(page):
    """
    Return embedded images from a PDF page.
    """

    try:

        return page.get_images(full=True)

    except Exception:

        return []


def classify_pdf_text(text):
    """
    Run the NLP model on extracted PDF text.
    """

    if not text or not text.strip():

        return None

    try:

        return predict_text(text)

    except Exception as e:

        print(
            "Text classification error:",
            str(e)
        )

        return {
            "error": "Text classification failed.",
            "details": str(e)
        }


def get_embedded_image_info(doc, image_info):
    """
    Extract metadata about one embedded PDF image.
    """

    xref = image_info[0]

    try:

        image_data = doc.extract_image(xref)

        if not image_data:

            return None

        return {
            "xref": xref,
            "width": image_data.get("width", 0),
            "height": image_data.get("height", 0),
            "ext": image_data.get("ext", "png"),
            "image": image_data.get("image")
        }

    except Exception as e:

        print(
            "Image metadata extraction error:",
            str(e)
        )

        return None


def predict_embedded_image(doc, image_info):
    """
    Extract one embedded PDF image and run V5.
    """

    image_data = get_embedded_image_info(
        doc,
        image_info
    )

    if not image_data:

        return None

    width = image_data["width"]
    height = image_data["height"]

    # Ignore tiny images.
    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):

        return {
            "status": "ignored",
            "reason": "Image too small.",
            "width": width,
            "height": height
        }

    image_bytes = image_data["image"]

    if not image_bytes:

        return None

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=f".{image_data['ext']}",
            delete=False
        ) as temp_file:

            temp_file.write(image_bytes)

            temp_path = temp_file.name

        result = predict_image(
            temp_path,
            model=model
        )

        return {
            "status": "processed",
            "width": width,
            "height": height,
            "result": result
        }

    except Exception as e:

        print(
            "Embedded image prediction error:",
            str(e)
        )

        return {
            "status": "error",
            "error": "Image prediction failed.",
            "details": str(e)
        }

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


def render_pdf_page(doc, page_number):
    """
    Render an image-only PDF page as PNG.
    """

    temp_path = None

    try:

        page = doc[page_number]

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

            pixmap.save(temp_file.name)

            temp_path = temp_file.name

        result = predict_image(
            temp_path,
            model=model
        )

        return result

    except Exception as e:

        print(
            "Rendered page prediction error:",
            str(e)
        )

        return {
            "error": "Rendered page prediction failed.",
            "details": str(e)
        }

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# ==================================================
# PDF PREDICTION
# ==================================================

@app.route("/api/predict-pdf", methods=["POST"])
def predict_pdf():

    if "pdf" not in request.files:

        return jsonify({
            "success": False,
            "error": "No PDF uploaded."
        }), 400

    pdf_file = request.files["pdf"]

    if pdf_file.filename == "":

        return jsonify({
            "success": False,
            "error": "No PDF selected."
        }), 400

    original_filename = pdf_file.filename

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension != PDF_EXTENSION:

        return jsonify({
            "success": False,
            "error": "Only PDF files are supported."
        }), 400

    # --------------------------------------------------
    # SAVE PERMANENT UPLOAD
    # --------------------------------------------------

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

        # --------------------------------------------------
        # SAVE PDF
        # --------------------------------------------------

        pdf_file.save(saved_path)

        # --------------------------------------------------
        # OPEN PDF
        # --------------------------------------------------

        document = pymupdf.open(saved_path)

        page_count = len(document)

        if page_count == 0:

            os.remove(saved_path)

            return jsonify({
                "success": False,
                "error": "The PDF contains no pages."
            }), 400

        # --------------------------------------------------
        # STORAGE
        # --------------------------------------------------

        pages = []

        text_results = []
        image_results = []

        text_probabilities = []
        image_probabilities = []

        text_pages = 0
        image_pages = 0

        # --------------------------------------------------
        # PROCESS EVERY PAGE
        # --------------------------------------------------

        for page_index in range(page_count):

            page_number = page_index + 1

            page = document[page_index]

            text = extract_page_text(page)

            embedded_images = get_page_images(page)

            page_result = {
                "page": page_number,
                "text": None,
                "images": [],
                "rendered_page": None
            }

            # ==================================================
            # TEXT PROCESSING
            # ==================================================

            if text:

                text_pages += 1

                text_result = classify_pdf_text(text)

                page_result["text"] = {
                    "characters": len(text),
                    "prediction": text_result
                }

                if (
                    text_result
                    and "probability" in text_result
                    and "error" not in text_result
                ):

                    text_results.append({
                        "page": page_number,
                        "result": text_result
                    })

                    text_probabilities.append(
                        text_result["probability"]
                    )

            # ==================================================
            # EMBEDDED IMAGE PROCESSING
            # ==================================================

            processed_images = []

            for image_index, image_info in enumerate(
                embedded_images
            ):

                result = predict_embedded_image(
                    document,
                    image_info
                )

                if not result:

                    continue

                result["image_index"] = image_index + 1

                processed_images.append(result)

                if (
                    result.get("status") == "processed"
                    and "result" in result
                ):

                    prediction = result["result"]

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
                            "page": page_number,
                            "image_index":
                                image_index + 1,
                            "result": prediction
                        })

            if processed_images:

                image_pages += 1

                page_result["images"] = processed_images

            # ==================================================
            # IMAGE-ONLY / SCANNED PAGE
            # ==================================================

            if (
                not text
                and not embedded_images
            ):

                rendered_result = render_pdf_page(
                    document,
                    page_index
                )

                if rendered_result:

                    page_result[
                        "rendered_page"
                    ] = rendered_result

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
                            "page": page_number,
                            "source": "rendered_page",
                            "result": rendered_result
                        })

            pages.append(page_result)

        # ==================================================
        # IMAGE SUMMARY
        # ==================================================

        image_summary = None

        if image_probabilities:

            average_probability = (
                sum(image_probabilities)
                / len(image_probabilities)
            )

            maximum_probability = max(
                image_probabilities
            )

            if average_probability >= IMAGE_THRESHOLD:

                prediction = "MALIGNANT"

            else:

                prediction = "NON-MALIGNANT"

            image_summary = {
                "prediction": prediction,
                "average_malignant_probability":
                    average_probability,
                "maximum_malignant_probability":
                    maximum_probability,
                "images_processed":
                    len(image_probabilities),
                "threshold": IMAGE_THRESHOLD
            }

        # ==================================================
        # TEXT SUMMARY
        # ==================================================

        text_summary = None

        if text_probabilities:

            average_probability = (
                sum(text_probabilities)
                / len(text_probabilities)
            )

            if average_probability >= TEXT_THRESHOLD:

                prediction = "PARENCHYMAL_LESION"

            else:

                prediction = "NO_PARENCHYMAL_LESION"

            text_summary = {
                "prediction": prediction,
                "average_probability":
                    average_probability,
                "pages_with_text_prediction":
                    len(text_probabilities),
                "threshold": TEXT_THRESHOLD
            }

        # ==================================================
        # DOCUMENT TYPE
        # ==================================================

        if text_results and image_results:

            document_type = "mixed"

        elif text_results:

            document_type = "text"

        elif image_results:

            document_type = "image"

        else:

            document_type = "unknown"

        # ==================================================
        # HISTORY RECORD
        # ==================================================

        history_prediction = "UNKNOWN"
        history_probability = None
        history_model = "NONE"

        if document_type == "image":

            history_prediction = image_summary["prediction"]

            history_probability = (
                image_summary[
                    "average_malignant_probability"
                ]
            )

            history_model = IMAGE_MODEL_NAME

        elif document_type == "text":

            history_prediction = text_summary["prediction"]

            history_probability = (
                text_summary[
                    "average_probability"
                ]
            )

            history_model = TEXT_MODEL_NAME

        elif document_type == "mixed":

            history_prediction = "MIXED"

            history_probability = None

            history_model = (
                f"{IMAGE_MODEL_NAME} + "
                f"{TEXT_MODEL_NAME}"
            )

        details = {
            "document_type": document_type,
            "pages_processed": page_count,
            "text_pages": text_pages,
            "image_pages": image_pages,
            "summary": {
                "image": image_summary,
                "text": text_summary
            },
            "text_results": text_results,
            "image_results": image_results,
            "pages": pages
        }

        prediction_id = save_prediction(
            filename=original_filename,
            saved_file=os.path.relpath(
                saved_path,
                PROJECT_DIR
            ),
            file_type="pdf",
            prediction=history_prediction,
            probability=history_probability,
            model=history_model,
            details=details
        )

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return jsonify({

            "success": True,

            "type": "pdf",

            "filename": original_filename,

            "history_id": prediction_id,

            "saved_file": os.path.relpath(
                saved_path,
                PROJECT_DIR
            ),

            "document_type": document_type,

            "pages_processed": page_count,

            "text_pages": text_pages,

            "image_pages": image_pages,

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
                "image": image_summary,
                "text": text_summary
            },

            "text_results": text_results,

            "image_results": image_results,

            "pages": pages

        }), 200

    except Exception as e:

        print(
            "PDF processing error:",
            str(e)
        )

        # Remove uploaded PDF if processing failed.
        if os.path.exists(saved_path):

            os.remove(saved_path)

        return jsonify({
            "success": False,
            "error": "PDF processing failed.",
            "details": str(e)
        }), 500

    finally:

        if document:

            document.close()


# ==================================================
# PREDICTION HISTORY
# ==================================================

@app.route("/api/history", methods=["GET"])
def history():

    try:

        limit = request.args.get(
            "limit",
            default=50,
            type=int
        )

        # Prevent unreasonable requests.
        limit = max(1, min(limit, 100))

        records = get_prediction_history(
            limit
        )

        return jsonify({
            "success": True,
            "count": len(records),
            "history": records
        }), 200

    except Exception as e:

        print(
            "History retrieval error:",
            str(e)
        )

        return jsonify({
            "success": False,
            "error": "Could not retrieve prediction history.",
            "details": str(e)
        }), 500


# ==================================================
# DELETE HISTORY RECORD
# ==================================================

@app.route(
    "/api/history/<int:prediction_id>",
    methods=["DELETE"]
)
def delete_history(prediction_id):

    try:

        saved_file = delete_prediction_record(
            prediction_id
        )

        if saved_file is None:

            return jsonify({
                "success": False,
                "error": "Prediction history record not found."
            }), 404

        # Convert stored relative path into absolute path.
        file_path = os.path.join(
            PROJECT_DIR,
            saved_file
        )

        if os.path.exists(file_path):

            os.remove(file_path)

        return jsonify({
            "success": True,
            "message": "Prediction history deleted.",
            "deleted_id": prediction_id
        }), 200

    except Exception as e:

        print(
            "History deletion error:",
            str(e)
        )

        return jsonify({
            "success": False,
            "error": "Could not delete prediction history.",
            "details": str(e)
        }), 500


# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("LUNG CANCER DETECTION BACKEND")
    print("=" * 60)

    print(
        f"Image model: {IMAGE_MODEL_NAME}"
    )

    print(
        f"Text model: {TEXT_MODEL_NAME}"
    )

    print(
        f"Image threshold: {IMAGE_THRESHOLD}"
    )

    print(
        f"Text threshold: {TEXT_THRESHOLD}"
    )

    print(
        f"Upload directory: {UPLOAD_DIR}"
    )

    print(
        f"History database: {DB_PATH}"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "Health: http://127.0.0.1:5000/api/health"
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
        "Prediction history: "
        "http://127.0.0.1:5000/api/history"
    )

    print("=" * 60 + "\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
