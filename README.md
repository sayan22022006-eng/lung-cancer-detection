# Lung Cancer Detection Project

> A research and academic prototype for analyzing lung-related findings from medical images and radiology reports.

> **Important:** This project is intended strictly for academic and research purposes. It is not a clinical diagnostic system and must not be used for medical decision-making.

---

## 1. Project Overview

| Item | Details |
|---|---|
| Project Name | Lung Cancer Detection |
| Project Type | Academic / Research Prototype |
| Primary Dataset | LIDC-IDRI |
| Image Model | MobileNetV2-based CNN |
| Text Model | TF-IDF + Logistic Regression |
| Image Input | Lung CT/nodule image |
| Text Input | Radiology report |
| Document Input | PDF |
| Backend | Flask |
| Database | SQLite |
| Programming Language | Python 3.12 |

The system is designed to process lung-related medical data through two independent machine-learning pipelines:

| Pipeline | Input | Output |
|---|---|---|
| Image Classification | CT/nodule image | Malignant / Non-Malignant |
| Text Classification | Radiology report | Parenchymal lesion / No parenchymal lesion |

The backend also supports PDF processing, allowing the system to inspect text and images contained inside PDF documents.

---

## 2. Key Features

| Feature | Description |
|---|---|
| CT Image Classification | Classifies lung/nodule images using the trained deep-learning model |
| Radiology Text Classification | Detects parenchymal-lesion-related findings from report text |
| PDF Processing | Processes medical PDF documents |
| Image PDF Processing | Extracts images from PDF pages |
| Text PDF Processing | Extracts text from PDF pages |
| Mixed PDF Processing | Processes both images and text from the same PDF |
| Multi-page Processing | Processes multiple PDF pages individually |
| Prediction History | Stores previous predictions in SQLite |
| REST API | Provides HTTP endpoints for prediction and history |
| Health Check | Reports model and backend status |
| Threshold-based Prediction | Uses separately selected thresholds for image and text models |

---

## 3. System Architecture

### 3.1 Overall Architecture

```text
                         Input
                           |
          +----------------+----------------+
          |                |                |
       Image             Text              PDF
          |                |                |
          v                v                v
   Image Preprocessing  Text Processing  PDF Processing
          |                |                |
          v                v                |
   MobileNetV2 Model   TF-IDF + LR Model   |
          |                |                |
          v                v                v
   Image Prediction   Text Prediction   Page Results
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Prediction Result
                           |
                           v
                     SQLite History

3.2 Processing Components
Component	Responsibility
Flask Backend	Handles API requests
Image Pipeline	Preprocesses images and runs image model
Text Pipeline	Processes radiology reports
PDF Pipeline	Extracts text and images from PDFs
Image Model	Predicts malignant probability
Text Model	Predicts parenchymal lesion probability
SQLite	Stores prediction history
4. Technology Stack
Category	Technology
Language	Python 3.12
Backend Framework	Flask
Cross-Origin Support	Flask-CORS
Deep Learning	TensorFlow 2.16.2
Image Model	MobileNetV2
Image Processing	Pillow, OpenCV
Numerical Computing	NumPy
Data Processing	Pandas
Machine Learning	Scikit-learn
Text Features	TF-IDF
Text Classifier	Logistic Regression
PDF Processing	PyMuPDF
Model Serialization	Joblib
Database	SQLite
Version Control	Git
Repository	GitHub
5. Project Structure
lung-cancer-project/
│
├── backend/
│   ├── app.py
│   └── uploads/
│
├── data/
│   ├── text/
│   │   └── xraydar-reports.jsonl
│   │
│   ├── image_dataset.csv
│   ├── prepared_dataset.csv
│   └── ...
│
├── models/
│   ├── lung_cancer_model_v5.keras
│   │
│   └── text/
│       ├── tfidf_logistic_model.joblib
│       ├── tfidf_vectorizer.joblib
│       └── decision_threshold.txt
│
├── src/
│   ├── create_image_dataset.py
│   ├── create_nodule_dataset.py
│   ├── evaluate_model.py
│   ├── evaluate_v5_threshold.py
│   ├── predict_image.py
│   ├── prepare_dataset.py
│   ├── regenerate_test_predictions.py
│   ├── train_model.py
│   ├── train_model_128.py
│   ├── train_model_v4.py
│   ├── train_model_v5.py
│   │
│   └── text_classifier/
│       ├── prepare_text_dataset.py
│       ├── train_text_model.py
│       ├── evaluate_text_threshold.py
│       └── predict_text.py
│
├── API_DOCUMENTATION.md
├── README.md
├── requirements.txt
└── .gitignore
6. Installation
6.1 Requirements
Requirement	Version / Information
Operating System	macOS / Linux / Windows
Python	3.12 recommended
Git	Required
RAM	Depends on model training workload
Internet	Required for initial dependency installation
6.2 Clone the Repository
git clone https://github.com/sayan22022006-eng/lung-cancer-detection.git
cd lung-cancer-detection
6.3 Create Virtual Environment
python3.12 -m venv venv312

Activate it:

source venv312/bin/activate
6.4 Install Dependencies
python -m pip install --upgrade pip setuptools wheel

Then:

pip install -r requirements.txt
6.5 Verify TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

Expected version:

2.16.2
7. Required Project Files

For normal project execution, the following files are required:

File / Folder	Required for Running Backend	Required for Retraining
backend/app.py	Yes	Yes
models/lung_cancer_model_v5.keras	Yes	No
models/text/tfidf_logistic_model.joblib	Yes	No
models/text/tfidf_vectorizer.joblib	Yes	No
models/text/decision_threshold.txt	Yes	No
requirements.txt	Yes	Yes
src/	Yes	Yes
data/image_dataset.csv	No	Yes
LIDC-IDRI raw DICOM data	No	Yes
X-Raydar raw JSONL	No	Yes
Generated image crops	No	Yes
7.1 Important

The large raw datasets are intentionally excluded from GitHub.

The trained models required to run the backend are stored in the repository.

Therefore, a teammate who only wants to run the existing system does not need to download the complete LIDC-IDRI dataset.

8. Running the Backend

From the project root:

source venv312/bin/activate

Start Flask:

python backend/app.py

The backend runs at:

http://127.0.0.1:5000
8.1 Health Check

Open:

http://127.0.0.1:5000/api/health

Example response:

{
  "image_model": "lung_cancer_model_v5.keras",
  "image_threshold": 0.5,
  "model_loaded": true,
  "status": "ok",
  "text_model": "tfidf_logistic_model.joblib",
  "text_threshold": 0.63
}
9. Machine Learning Models

The project currently contains two separate machine-learning models.

Model	Purpose	Algorithm
Image Model	Lung/nodule image classification	MobileNetV2
Text Model	Radiology report classification	TF-IDF + Logistic Regression
10. Image Classification Model
10.1 Model Architecture
Property	Value
Architecture	MobileNetV2
Learning Method	Transfer Learning
Input Size	224 × 224 × 3
Input Format	RGB
Output	Malignant probability
Primary Threshold	0.50
Loss	Binary Cross-Entropy
Regularization	Dropout + L2
Training	Two-stage
Fine-tuning	Partial MobileNetV2 layers
Label Smoothing	0.02

The original grayscale CT/nodule images are converted to RGB before being passed to MobileNetV2.

10.2 Image Prediction Classes
Model Output	Meaning
MALIGNANT	Probability is at or above 0.50
NON-MALIGNANT	Probability is below 0.50
UNCERTAIN	Probability falls within the configured uncertainty band
10.3 Image Threshold

The primary image classification threshold is:

0.50

The prototype also uses an uncertainty band:

0.45 - 0.55

This uncertainty rule is intended for prototype handling and should not be interpreted as a clinical confidence interval.

11. Image Model Experimental Results

The final V5 model was evaluated on the held-out test set.

Metric	Result
Accuracy	75.42%
Precision	77.40%
Recall	85.93%
F1 Score	81.44%
ROC-AUC	79.73%
11.1 Confusion Matrix
	Predicted Non-Malignant	Predicted Malignant
Actual Non-Malignant	90	66
Actual Malignant	37	226
11.2 Interpretation
Observation	Result
Correct non-malignant predictions	90
Incorrect malignant predictions for non-malignant cases	66
Missed malignant cases	37
Correct malignant predictions	226

These results are experimental results from the project dataset and should not be interpreted as clinical performance.

12. Text Classification Model

The text pipeline uses real radiology reports from the X-Raydar Annotated Radiology Reports dataset.

12.1 Text Processing
Component	Configuration
Feature Extraction	TF-IDF
N-grams	Unigrams + Bigrams
Lowercase	Enabled
Unicode Accent Removal	Enabled
Minimum Document Frequency	2
Maximum Document Frequency	95%
Sublinear TF	Enabled
Maximum Features	100,000
Classifier	Logistic Regression
Class Weight	Balanced
Solver	Liblinear
Maximum Iterations	1000
12.2 Target Label

The current text classifier predicts:

parenchymal_lesion
Output	Meaning
PARENCHYMAL_LESION	Report is classified as containing a parenchymal lesion
NO_PARENCHYMAL_LESION	Report is classified as not containing a parenchymal lesion
13. Text Dataset
Property	Value
Total Reports	29,756
Positive Reports	2,041
Negative Reports	27,715
Training Reports	20,829
Validation Reports	4,463
Test Reports	4,464
TF-IDF Vocabulary	46,814
13.1 Text Model Results
Metric	Result
Accuracy	93.41%
Precision	51.43%
Recall	70.59%
F1 Score	59.50%
ROC-AUC	95.43%
PR-AUC	60.79%
Decision Threshold	0.63
13.2 Final Confusion Matrix
	Predicted Negative	Predicted Positive
Actual Negative	3954	204
Actual Positive	90	216

The threshold of 0.63 was selected using the validation set and then evaluated on the held-out test set.

14. Text Classification Example

Example input:

The lungs are clear. No focal pulmonary lesion identified.

Example output:

Prediction : NO_PARENCHYMAL_LESION
Probability: 0.0337
Threshold  : 0.63

Example positive input:

There is a focal parenchymal lesion in the right upper lung.

Example output:

Prediction : PARENCHYMAL_LESION
Probability: 0.8317
Threshold  : 0.63
15. PDF Processing

The backend supports three main PDF document types.

PDF Type	Processing
Image PDF	Extracts and classifies images
Text PDF	Extracts and classifies text
Mixed PDF	Processes both images and text
15.1 PDF Processing Flow
PDF Upload
    |
    v
Read PDF Pages
    |
    +--------------------+
    |                    |
    v                    v
Extract Text       Extract Images
    |                    |
    v                    v
Text Classifier    Image Classifier
    |                    |
    +---------+----------+
              |
              v
        Page-level Results
              |
              v
        Document Summary
              |
              v
        SQLite History
16. Multi-page PDF Processing

Each PDF page is processed separately.

Page Content	Processing
Text	Text classifier
Embedded image	Image classifier
Both text and image	Both classifiers
No extractable content	Page rendering fallback

The backend returns page-level results as well as an overall document summary.

17. PDF Summary Logic
17.1 Image Summary

For image-containing PDFs, the backend calculates:

Value	Description
Average Probability	Average malignant probability across processed images
Maximum Probability	Highest malignant probability
Prediction	Based on the primary image threshold of 0.50
17.2 Text Summary

For text-containing PDFs:

Value	Description
Average Probability	Average text-model probability
Threshold	0.63
Prediction	Based on the text decision threshold
17.3 Mixed Documents

For mixed PDFs containing both text and images:

Document Type = MIXED

The backend preserves the separate image and text predictions instead of combining the two model probabilities into one artificial medical score.

18. REST API

The backend currently provides the following endpoints.

Method	Endpoint	Purpose
GET	/api/health	Backend and model health
POST	/api/predict	Predict an uploaded image
POST	/api/predict-pdf	Process a PDF
GET	/api/history	Retrieve prediction history
DELETE	/api/history/<id>	Delete a history record
19. Image Prediction API
Endpoint
POST /api/predict
Form Field
image
Example Request
curl -X POST \
  -F "image=@/path/to/image.jpg" \
  http://127.0.0.1:5000/api/predict
Example Response
{
  "filename": "example.jpg",
  "history_id": 1,
  "model": "lung_cancer_model_v5.keras",
  "result": {
    "malignant_probability": 0.1428,
    "model_score": 0.8572,
    "non_malignant_probability": 0.8572,
    "prediction": "NON-MALIGNANT",
    "threshold": 0.5,
    "uncertainty": false
  },
  "success": true,
  "type": "image"
}
20. PDF Prediction API
Endpoint
POST /api/predict-pdf
Form Field
pdf
Example Request
curl -X POST \
  -F "pdf=@/path/to/document.pdf" \
  http://127.0.0.1:5000/api/predict-pdf
Supported Processing
Input	Result
Image-only PDF	Image predictions
Text-only PDF	Text predictions
Mixed PDF	Image + text predictions
Multi-page PDF	Page-by-page processing
21. Prediction History API
21.1 Get History
GET /api/history

Example:

curl http://127.0.0.1:5000/api/history

The endpoint returns stored prediction records from SQLite.

21.2 Delete History
DELETE /api/history/<id>

Example:

curl -X DELETE \
  http://127.0.0.1:5000/api/history/1

Deleting a record also removes the corresponding uploaded file when applicable.

22. File Storage
Location	Purpose
backend/uploads/	Temporarily/stored uploaded prediction files
SQLite database	Prediction history
models/	Trained machine-learning models
data/	Dataset and derived training data

Uploaded files and local databases are excluded from Git through .gitignore.

23. Dataset Information
23.1 LIDC-IDRI

The image classification pipeline is based on the LIDC-IDRI dataset.

Property	Project Data
Annotation ROI Records	4,709
Unique Nodules Used	468
Patients Used	71
Training Nodules	344
Validation Nodules	58
Test Nodules	66
23.2 Binary Classification Labels
Label	Meaning
0	Non-malignant
1	Malignant

The binary label was derived from the malignancy annotation using:

Malignancy >= 4 → Malignant
Malignancy < 4  → Non-malignant
23.3 Patient-level Split

The dataset was divided at the patient level to reduce the possibility of patient-level leakage.

Split	Patients
Training	49
Validation	11
Test	11
24. Text Dataset Information

The text classifier uses the X-Raydar Annotated Radiology Reports dataset.

Property	Value
Dataset	X-Raydar Annotated Radiology Reports
Records	29,756
Language	English
Target Finding	parenchymal_lesion
Task	Binary text classification
Feature Method	TF-IDF
Classifier	Logistic Regression

The raw dataset is excluded from GitHub because it is unnecessary for normal backend execution and is used primarily for model development/retraining.

25. Retraining the Image Model

The image model can be retrained using the available training scripts.

Main V5 Training Script
python src/train_model_v5.py

The training process includes:

Stage	Description
Stage 1	MobileNetV2 base initially frozen
Stage 2	Selected layers fine-tuned
Augmentation	Conservative image augmentation
Class Weighting	Balanced classes
Regularization	Dropout + L2
Label Smoothing	0.02
Validation	Patient-level validation set

The final model is saved as:

models/lung_cancer_model_v5.keras

26. Retraining the Text Model

Prepare the text dataset:

python src/text_classifier/prepare_text_dataset.py

Train the text model:

python src/text_classifier/train_text_model.py

Evaluate and select the decision threshold:

python src/text_classifier/evaluate_text_threshold.py

The resulting files are:

models/text/tfidf_logistic_model.joblib
models/text/tfidf_vectorizer.joblib
models/text/decision_threshold.txt
27. Direct Prediction Scripts

27.1 Image Prediction
python src/predict_image.py "/path/to/image.jpg"
27.2 Text Prediction

python src/text_classifier/predict_text.py \
"The lungs are clear. No focal pulmonary lesion identified."

28. Troubleshooting

28.1 Python Command Not Found

Check Python:

python3 --version

If Python 3.12 is installed:

python3.12 --version

28.2 Virtual Environment Not Activated

Activate:

source venv312/bin/activate

Verify:

which python

The result should point inside the project virtual environment.

28.3 TensorFlow Import Error

Check:

python -c "import tensorflow as tf; print(tf.__version__)"

Expected:

2.16.2

If dependencies are missing:

pip install -r requirements.txt


28.4 OpenCV Import Error

Check:

python -c "import cv2; print(cv2.__version__)"
28.5 Model Not Found

Check:

ls -lh models/

The main image model should exist:

models/lung_cancer_model_v5.keras

Check the text models:

ls -lh models/text/

Expected files:

tfidf_logistic_model.joblib
tfidf_vectorizer.joblib
decision_threshold.txt
28.6 Backend Does Not Start

Run:

python backend/app.py

Then check:

http://127.0.0.1:5000/api/health


29. Git Workflow
Check Current Status
git status
Add Changes
git add .
Commit Changes
git commit -m "Describe the changes"
Push Changes
git push
Pull Latest Changes
git pull
View Commit History
git log --oneline


30. Important Files
File	Purpose
backend/app.py	Main Flask backend
src/predict_image.py	Image prediction logic
src/train_model_v5.py	V5 image model training
src/evaluate_v5_threshold.py	Image threshold evaluation
src/text_classifier/predict_text.py	Text prediction
src/text_classifier/train_text_model.py	Text model training
src/text_classifier/evaluate_text_threshold.py	Text threshold evaluation
src/text_classifier/prepare_text_dataset.py	Text dataset preparation
requirements.txt	Python dependencies
API_DOCUMENTATION.md	Complete API documentation
README.md	Project documentation


31. Project Status
Component	Status
LIDC-IDRI image dataset preparation	Complete
Image crop generation	Complete
Patient-level dataset splitting	Complete
Image model V5	Complete
Image model evaluation	Complete
Text dataset preparation	Complete
Text classifier	Complete
Text threshold selection	Complete
Image prediction API	Complete
PDF processing	Complete
Multi-page PDF processing	Complete
Mixed PDF processing	Complete
Prediction history	Complete
SQLite integration	Complete
REST API	Complete
API documentation	Complete
README documentation	Complete
32. Limitations
Limitation	Description
Research Dataset	The models are trained using research datasets
Limited Patient Count	The image model uses a relatively small patient-level subset
Class Imbalance	Malignancy classes are not perfectly balanced
Domain Shift	Performance may change on images from different sources
Image Input	The image model is designed around lung/nodule CT crops
Text Target	The text classifier detects parenchymal lesion findings, not cancer diagnosis directly
PDF Variability	PDF layouts and embedded content can vary
Clinical Validation	No clinical validation has been performed
OOD Inputs	Predictions on unrelated image types may be unreliable
