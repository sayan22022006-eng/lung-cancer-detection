# Lung Cancer Detection Project

A research and academic prototype for analyzing lung-related findings from medical images and radiology reports.

> **Important:** This project is for academic/research purposes only. It is not a clinical diagnostic system and must not be used for medical decisions.

---

## Project Overview

This project combines two machine-learning components with a Flask backend:

1. **Medical image classification**
   - Uses a MobileNetV2-based deep-learning model.
   - Trained using nodule crops derived from the LIDC-IDRI dataset.
   - Produces malignant/non-malignant predictions.

2. **Radiology report text classification**
   - Uses TF-IDF feature extraction.
   - Uses Logistic Regression for classification.
   - Detects whether a report contains a parenchymal lung lesion.

3. **PDF processing**
   - Extracts text from PDF files.
   - Extracts embedded images from PDFs.
   - Processes multi-page PDFs.
   - Supports image-only, text-only, and mixed PDFs.

4. **Prediction history**
   - Stores previous predictions in a local SQLite database.
   - Allows prediction history to be retrieved and deleted through the API.

The current project focuses on the **backend and machine-learning pipeline**.

---

## Features

- CT/nodule image classification
- Radiology report text classification
- PDF text extraction
- PDF image extraction
- Multi-page PDF processing
- Image-only PDF support
- Text-only PDF support
- Mixed PDF support
- Prediction history
- REST API using Flask
- CORS support
- Local SQLite storage
- Separate image and text ML models

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/sayan22022006-eng/lung-cancer-detection.git
cd lung-cancer-detection

2. Check Python Version

The project uses Python 3.12.

Check your version:

python3 --version

Expected:
Python 3.12.x

3. Create a Virtual Environment
python3 -m venv venv312

Activate it:

macOS / Linux
source venv312/bin/activate
Windows
venv312\Scripts\activate
4. Install Dependencies
python -m pip install --upgrade pip setuptools wheel

Then:

pip install -r requirements.txt
What Teammates Need to Download

For normal project execution, teammates only need:

Requirement	Needed?
Git repository	Yes
Python 3.12	Yes
Python packages	Yes
Trained image model	Already included
Trained text model	Already included
LIDC-IDRI raw dataset	No
DICOM files	No
XML annotation files	No
Nodule crop dataset	No
X-Raydar raw JSONL dataset	No
Virtual environment	No
SQLite database	Created locally
Uploaded files	Created locally

The large raw datasets are intentionally excluded from GitHub.

Model Files

The trained models are stored inside the models/ directory.

Image Model
models/lung_cancer_model_v5.keras

The image model is based on MobileNetV2.

Text Model
models/text/tfidf_logistic_model.joblib
models/text/tfidf_vectorizer.joblib
models/text/decision_threshold.txt

The text classifier uses:

TF-IDF + Logistic Regression
System Architecture
                    INPUT
                      |
          +-----------+-----------+
          |                       |
       IMAGE                     PDF
          |                       |
          |              +--------+--------+
          |              |                 |
          |            Text              Image
          |              |                 |
          |              v                 v
          |        Text Classifier    Image Classifier
          |              |                 |
          +--------------+-----------------+
                         |
                         v
                  Flask Backend API
                         |
              +----------+----------+
              |                     |
        Prediction Result      SQLite History
Technology Stack
Backend
Python
Flask
Flask-CORS
PyMuPDF
Machine Learning
TensorFlow
Keras
MobileNetV2
Scikit-learn
TF-IDF
Logistic Regression
Data Processing
NumPy
Pandas
SciPy
OpenCV
Pillow
Storage
SQLite
Development
VS Code
Git
GitHub
Project Structure
lung-cancer-project/
│
├── backend/
│   ├── app.py
│   └── uploads/
│
├── data/
│   ├── text/
│   ├── nodule_crops/
│   ├── nodule_crops_64/
│   └── ...
│
├── models/
│   ├── lung_cancer_model_v5.keras
│   └── text/
│       ├── tfidf_logistic_model.joblib
│       ├── tfidf_vectorizer.joblib
│       └── decision_threshold.txt
│
├── src/
│   ├── predict_image.py
│   ├── train_model.py
│   ├── train_model_v5.py
│   ├── evaluate_model.py
│   ├── evaluate_v5_threshold.py
│   ├── create_nodule_dataset.py
│   ├── create_image_dataset.py
│   ├── prepare_dataset.py
│   │
│   └── text_classifier/
│       ├── prepare_text_dataset.py
│       ├── train_text_model.py
│       ├── evaluate_text_threshold.py
│       └── predict_text.py
│
├── requirements.txt
├── API_DOCUMENTATION.md
├── README.md
└── .gitignore

Some dataset-generation files and raw datasets are intentionally excluded from the Git repository.

API Endpoints

The Flask backend runs on:

http://127.0.0.1:5000
Method	Endpoint	Purpose
GET	/api/health	Check backend and models
POST	/api/predict	Predict an uploaded image
POST	/api/predict-pdf	Process a PDF
GET	/api/history	Retrieve prediction history
DELETE	/api/history/<id>	Delete a history record
Starting the Backend

From the project root:

python backend/app.py

The backend should start on:

http://127.0.0.1:5000
Health Check

Run:

curl http://127.0.0.1:5000/api/health

A successful response contains information about:

image model
text model
image threshold
text threshold
model loading status

Example:

{
  "image_model": "lung_cancer_model_v5.keras",
  "image_threshold": 0.5,
  "model_loaded": true,
  "status": "ok",
  "text_model": "tfidf_logistic_model.joblib",
  "text_threshold": 0.63
}
Image Prediction

The image prediction endpoint is:

POST /api/predict

The uploaded file must be sent using the field:

image

Example:

curl -X POST \
  -F "image=@/path/to/image.jpg" \
  http://127.0.0.1:5000/api/predict

The API returns:

malignant probability
non-malignant probability
prediction
model score
threshold
uncertainty status
history ID

Example:

{
  "success": true,
  "type": "image",
  "model": "lung_cancer_model_v5.keras",
  "result": {
    "malignant_probability": 0.14,
    "non_malignant_probability": 0.86,
    "prediction": "NON-MALIGNANT",
    "model_score": 0.86,
    "threshold": 0.5,
    "uncertainty": false
  }
}
Image Model Decision Logic

The primary image classification threshold is:

0.50

The model produces a malignant probability.

Conceptually:

Probability >= 0.50
        |
        v
    MALIGNANT

Probability < 0.50
        |
        v
 NON-MALIGNANT

An uncertainty band is also used:

0.45 <= probability <= 0.55

Predictions inside this range are marked:

UNCERTAIN

This uncertainty rule is a prototype/UI handling mechanism and should not be interpreted as a medical confidence interval.

PDF Prediction

The PDF endpoint is:

POST /api/predict-pdf

Example:

curl -X POST \
  -F "pdf=@/path/to/file.pdf" \
  http://127.0.0.1:5000/api/predict-pdf

The backend analyzes each page independently.

PDF Types Supported
1. Image-only PDF

If the PDF contains images, the backend:

PDF
 |
 v
Extract images
 |
 v
Image model
 |
 v
Prediction
2. Text-only PDF

If the PDF contains radiology report text:

PDF
 |
 v
Extract text
 |
 v
TF-IDF
 |
 v
Logistic Regression
 |
 v
Prediction
3. Mixed PDF

A PDF may contain both:

radiology report text
medical images

The backend processes both components independently.

                 PDF
                  |
          +-------+-------+
          |               |
        TEXT            IMAGE
          |               |
          v               v
    Text Classifier   Image Classifier
          |               |
          +-------+-------+
                  |
                  v
             PDF Summary
Multi-page PDF Processing

Each page is processed separately.

For example:

Page 1 → Text/Image analysis
Page 2 → Text/Image analysis
Page 3 → Text/Image analysis
Page 4 → Text/Image analysis

The backend then generates an overall summary.

For image pages, the system calculates summary malignant probabilities.

For text pages, the system calculates summary lesion probabilities.

Text Classification

The text classifier is designed to identify whether a radiology report contains a:

PARENCHYMAL_LESION

or:

NO_PARENCHYMAL_LESION

The model uses:

Radiology Report
       |
       v
     TF-IDF
       |
       v
Logistic Regression
       |
       v
Prediction
Text Model Threshold

The selected text decision threshold is:

0.63

This threshold was selected using the validation set and then applied to the test set.

Example:

python src/text_classifier/predict_text.py \
"The lungs are clear. No focal pulmonary lesion identified."

Example result:

Prediction : NO_PARENCHYMAL_LESION
Probability: 0.0337
Threshold  : 0.63

Another example:

python src/text_classifier/predict_text.py \
"There is a focal parenchymal lesion in the right upper lung."

Example result:

Prediction : PARENCHYMAL_LESION
Probability: 0.8317
Prediction History

The backend stores prediction history in a local SQLite database.

The history contains information such as:

prediction ID
filename
file type
prediction
probability
model used
saved file
timestamp
detailed prediction information
Retrieve History
curl http://127.0.0.1:5000/api/history

Example:

{
  "success": true,
  "count": 1,
  "history": [
    {
      "id": 1,
      "filename": "example.jpg",
      "file_type": "image",
      "prediction": "NON-MALIGNANT",
      "probability": 0.14,
      "model": "lung_cancer_model_v5.keras"
    }
  ]
}
Delete History

Use:

DELETE /api/history/<id>

Example:

curl -X DELETE \
  http://127.0.0.1:5000/api/history/1

Deleting a history record also removes the associated locally stored uploaded file.

Image Model Details

The main image model is:

MobileNetV2

Input size:

224 × 224 × 3

The original grayscale medical image is converted to RGB before being passed to the model.

The model was trained using nodule crops generated from LIDC-IDRI annotations.

Training included:

transfer learning
conservative data augmentation
class weighting
label smoothing
dropout
L2 regularization
two-stage training
partial fine-tuning
Image Model Experimental Results

The V5 model achieved the following results on the held-out test set at the primary threshold of 0.50:

Metric	Result
Accuracy	75.42%
Precision	77.40%
Recall	85.93%
F1 Score	81.44%
ROC-AUC	79.73%

Confusion matrix:

                Predicted
              0          1

Actual 0     90         66
Actual 1     37        226

These are experimental research results and should not be interpreted as clinical performance.

Text Model Experimental Results

The text classifier was evaluated using a validation-selected threshold of 0.63.

Final test results:

Metric	Result
Accuracy	93.41%
Precision	51.43%
Recall	70.59%
F1 Score	59.50%
ROC-AUC	95.43%
PR-AUC	60.79%

Confusion matrix:

                Predicted
              0          1

Actual 0    3954        204
Actual 1      90        216

The relatively lower precision and F1 compared with accuracy show why accuracy alone should not be used to evaluate this classifier.

Dataset Information
LIDC-IDRI

The image model was developed using the LIDC-IDRI dataset.

The project uses:

CT images
radiologist annotations
nodule locations
malignancy ratings

The dataset was processed into image crops for model training.

The final prepared image dataset contains:

468 unique nodules
71 patients

The binary classification setup uses:

0 → Non-malignant
1 → Malignant

The binary label is derived from the malignancy rating.

Patient-level Data Splitting

To reduce data leakage, the dataset was divided at the patient level rather than randomly splitting individual images.

Current split:

Training patients   : 49
Validation patients  : 11
Test patients       : 11

This ensures that nodules belonging to the same patient do not appear across different dataset splits.

Text Dataset

The text classifier was developed using the X-Raydar annotated radiology report dataset.

The project uses report text and report-level findings to create a binary classification task for:

parenchymal_lesion

The raw dataset is not included in the Git repository.

Retraining the Image Model

If the image model needs to be retrained, the required raw datasets and derived files must be available locally.

The main training script is:

python src/train_model_v5.py

The trained model is saved as:

models/lung_cancer_model_v5.keras
Retraining the Text Model

Prepare the text dataset:

python src/text_classifier/prepare_text_dataset.py

Train the classifier:

python src/text_classifier/train_text_model.py

Evaluate the decision threshold:

python src/text_classifier/evaluate_text_threshold.py

Test the trained classifier:

python src/text_classifier/predict_text.py \
"Example radiology report text"
Troubleshooting
Python command not found

Try:

python3 --version

If Python 3.12 is installed:

python3.12 --version
Virtual environment is not activated

macOS/Linux:

source venv312/bin/activate

Check:

which python

It should point to the project virtual environment.

TensorFlow cannot be imported

Check:

python -c "import tensorflow as tf; print(tf.__version__)"

Expected:

2.16.2
OpenCV cannot be imported

Check:

python -c "import cv2; print(cv2.__version__)"
PyMuPDF cannot be imported

Use:

python -c "import pymupdf; print(pymupdf.__doc__[:100])"

The project uses:

import pymupdf

for PDF processing.

Git Workflow

Check the current status:

git status

Pull the latest changes:

git pull origin main

After making changes:

git add .

Create a commit:

git commit -m "Describe your changes"

Push:

git push origin main
API Documentation

Detailed API information is available in:

API_DOCUMENTATION.md

This file contains:

API endpoints
request formats
response formats
PDF processing behavior
error handling
testing examples
Project Limitations

This project currently has several important limitations:

The image model is a research prototype.
The image model was trained on LIDC-IDRI-derived nodule crops.
Arbitrary medical images may be outside the model's training distribution.
A chest X-ray should not be treated as equivalent to a CT nodule crop.
The text classifier detects a specific report finding and is not a complete lung-cancer diagnosis system.
PDF extraction quality depends on the PDF structure.
OCR is not currently implemented for scanned PDFs without extractable text.
Model probabilities should not be interpreted as medical certainty.