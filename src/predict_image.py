import os
import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lung_cancer_model_v5.keras"
)

IMG_SIZE = (224, 224)

# Primary classification threshold
THRESHOLD = 0.50

# Uncertainty zone around the decision boundary
UNCERTAINTY_LOW = 0.45
UNCERTAINTY_HIGH = 0.55


# ============================================================
# MODEL
# ============================================================

def load_model():
    """
    Load the V5 lung cancer classification model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess an image for the V5 model.

    The model expects:
        224 x 224
        3 channels (RGB)
        pixel values in [0, 1]
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path)

    # Convert grayscale/RGBA/etc. to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE)

    # Convert to numpy array
    image = np.array(image, dtype=np.float32)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path, model=None):
    """
    Predict whether an image is malignant or non-malignant.

    Returns:
        malignant_probability
        non_malignant_probability
        prediction
        confidence
        uncertainty
        threshold
    """

    # Load model only if one wasn't provided
    if model is None:
        model = load_model()

    # Preprocess image
    image = preprocess_image(image_path)

    # Generate prediction
    probability = float(
        model.predict(image, verbose=0)[0][0]
    )

    # Keep probability safely inside [0, 1]
    probability = max(0.0, min(1.0, probability))

    malignant_probability = probability
    non_malignant_probability = 1.0 - probability

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if UNCERTAINTY_LOW <= probability <= UNCERTAINTY_HIGH:

        prediction = "UNCERTAIN"

        # Distance from 0.5 converted to a simple confidence score.
        # This is NOT a medical confidence measure.
        confidence = 1.0 - (
            abs(probability - THRESHOLD) / 0.05
        )

        confidence = max(0.0, min(1.0, confidence))

        uncertainty = True

    elif probability >= THRESHOLD:

        prediction = "MALIGNANT"

        confidence = probability

        uncertainty = False

    else:

        prediction = "NON-MALIGNANT"

        confidence = non_malignant_probability

        uncertainty = False

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "malignant_probability": malignant_probability,
        "non_malignant_probability": non_malignant_probability,
        "prediction": prediction,
        "model_score": confidence,
        "uncertainty": uncertainty,
        "threshold": THRESHOLD
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python src/predict_image.py <image_path>"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = predict_image(image_path)

        print("\n" + "=" * 60)
        print("LUNG CANCER PREDICTION")
        print("=" * 60)

        print(
            f"Malignant probability : "
            f"{result['malignant_probability']:.4f}"
        )

        print(
            f"Non-malignant probability : "
            f"{result['non_malignant_probability']:.4f}"
        )

        print(
            f"Prediction : "
            f"{result['prediction']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence']:.4f}"
        )

        print(
            f"Uncertainty : "
            f"{result['uncertainty']}"
        )

        print(
            f"Threshold : "
            f"{result['threshold']:.2f}"
        )

        print("=" * 60)

    except Exception as e:

        print(f"\nError: {e}")
        sys.exit(1)