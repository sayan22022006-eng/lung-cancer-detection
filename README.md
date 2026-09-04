# Lung Cancer Detection Project

A research/academic prototype for analyzing lung-related findings from medical images and radiology reports.

> **Important:** This project is for academic/research purposes only. It is not a clinical diagnostic system and must not be used for medical decisions.

---

## 1. Project Features

- CT image classification using a MobileNetV2-based deep-learning model
- Radiology report text classification using TF-IDF + Logistic Regression
- PDF processing
- Multi-page PDF processing
- Image-only PDF support
- Text-only PDF support
- Mixed PDF support
- Prediction history using SQLite
- REST API using Flask
- CORS support for React frontend integration

---

## 2. Project Structure

```text
lung-cancer-detection/
│
├── backend/
│   └── app.py
│
├── data/
│   ├── image_dataset.csv
│   ├── prepared_dataset.csv
│   ├── test_predictions_v5.csv
│   ├── training_history_v5.csv
│   └── visualizations/
│
├── models/
│   ├── lung_cancer_model_v5.keras
│   └── text/
│       ├── decision_threshold.txt
│       ├── tfidf_logistic_model.joblib
│       └── tfidf_vectorizer.joblib
│
├── src/
│   ├── predict_image.py
│   ├── train_model_v5.py
│   └── text_classifier/
│       ├── predict_text.py
│       ├── prepare_text_dataset.py
│       ├── train_text_model.py
│       └── evaluate_text_threshold.py
│
├── API_DOCUMENTATION.md
├── requirements.txt
├── README.md
└── .gitignore

3. Prerequisites

Install:

Git
Python 3.12
pip

Check Python:

python3.12 --version
4. Clone the Repository
git clone https://github.com/sayan22022006-eng/lung-cancer-detection.git
cd lung-cancer-detection
5. Create Virtual Environment
macOS / Linux
python3.12 -m venv venv312
source venv312/bin/activate
Windows
py -3.12 -m venv venv312
venv312\Scripts\activate
6. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
7. Model Files

The trained models are already included in the repository.

Make sure these files exist:

models/
├── lung_cancer_model_v5.keras
└── text/
    ├── decision_threshold.txt
    ├── tfidf_logistic_model.joblib
    └── tfidf_vectorizer.joblib

You do not need to train the models before running the backend.

8. Run the Backend

From the project root:

python backend/app.py

The backend runs at:

http://127.0.0.1:5000

Keep this terminal running.

9. Test the Backend

Open another terminal and run:

curl http://127.0.0.1:5000/api/health

You should see:

{
  "status": "ok",
  "model_loaded": true
}
10. Image Prediction

Endpoint:

POST /api/predict

Example:

curl -X POST \
  -F "image=@/path/to/image.jpg" \
  http://127.0.0.1:5000/api/predict

Supported formats:

.jpg
.jpeg
.png

The response contains the prediction, probabilities, model score, threshold and uncertainty information.

11. PDF Prediction

Endpoint:

POST /api/predict-pdf

Example:

curl -X POST \
  -F "pdf=@/path/to/document.pdf" \
  http://127.0.0.1:5000/api/predict-pdf

The PDF processor supports:

Image-only PDFs
Text-only PDFs
Mixed PDFs
Multi-page PDFs

For mixed PDFs, images and text are processed separately.

12. Text Prediction

You can test the text classifier directly:

python src/text_classifier/predict_text.py "The lungs are clear. No focal pulmonary lesion identified."

Another example:

python src/text_classifier/predict_text.py "There is a focal parenchymal lesion in the right upper lung."
13. Prediction History

Prediction history is stored locally using SQLite.

Get history:

curl http://127.0.0.1:5000/api/history

Delete a prediction:

curl -X DELETE http://127.0.0.1:5000/api/history/1

The database is generated automatically when predictions are made.

14. React Frontend

The backend is designed to work with the React frontend.

Base URL:

http://127.0.0.1:5000

Image upload:

POST /api/predict

FormData field:

image

PDF upload:

POST /api/predict-pdf

FormData field:

pdf

History:

GET /api/history

See API_DOCUMENTATION.md for complete API information.

15. Files NOT Required for Normal Setup

If you only want to run the existing trained project, you do not need:

LIDC-IDRI DICOM files
Raw LIDC XML annotations
Nodule crop folders
X-Raydar raw dataset
Virtual environment
SQLite database
Uploaded files

These are excluded from GitHub intentionally.

16. Files Required for Retraining

Additional datasets are required if you want to retrain the models.

Image model

Requires:

LIDC-IDRI CT/DICOM data
LIDC annotations
Nodule data/crops
Text model

Requires:

X-Raydar annotated radiology-report dataset

These datasets are not included in this repository.

17. Training
Image model
python src/train_model_v5.py
Text dataset preparation
python src/text_classifier/prepare_text_dataset.py
Text model training
python src/text_classifier/train_text_model.py
Text threshold evaluation
python src/text_classifier/evaluate_text_threshold.py
18. Model Information
Image Model

Architecture:

MobileNetV2

Input:

224 x 224 RGB

Decision threshold:

0.50
Text Model

Architecture:

TF-IDF + Logistic Regression

Decision threshold:

0.63
19. Experimental Results
Image Model V5
Accuracy  : 75.42%
Precision : 77.40%
Recall    : 85.93%
F1 Score  : 81.44%
ROC-AUC   : 79.73%
Text Model
Accuracy  : 93.41%
Precision : 51.43%
Recall    : 70.59%
F1 Score  : 59.50%
ROC-AUC   : 95.43%
PR-AUC    : 60.79%

These are experimental research results and are not clinical performance measurements.

20. Troubleshooting
Python not found

Check:

python3.12 --version
Virtual environment not active

macOS/Linux:

source venv312/bin/activate

Windows:

venv312\Scripts\activate
Check installed packages
pip list
Check model files
ls -lh models
ls -lh models/text
21. Git Workflow

Before working:

git pull origin main

After making changes:

git status
git add .
git commit -m "Describe your changes"
git push

Do not commit:

DICOM files
XML annotations
Raw datasets
Nodule crops
Virtual environments
Uploaded medical files
SQLite databases
API keys or passwords
22. API Documentation

For detailed API endpoints, request formats, responses and React integration:

API_DOCUMENTATION.md