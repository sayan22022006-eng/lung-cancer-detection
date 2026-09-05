🫁 Lung Cancer Detection Project
A research prototype for lung cancer detection using both medical images and radiology report text.
This project combines deep learning (MobileNetV2) for image classification and TF‑IDF + Logistic Regression for text classification, wrapped in a Flask backend API.

📦 Installation
1. Clone Repository
bash
git clone https://github.com/sayan22022006-eng/lung-cancer-detection.git
cd lung-cancer-detection

2. Check Python Version
The project requires Python 3.12.

bash
python3 --version

3. Create Virtual Environment
bash
python3 -m venv venv312
Activate it:

macOS/Linux

bash
source venv312/bin/activate
Windows

bash
venv312\Scripts\activate

4. Install Dependencies
bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

👥 Team Setup
For normal execution, teammates only need:

Git repository ✅

Python 3.12 ✅

Python packages ✅

Trained models ✅ (already included)

Raw datasets ❌ (excluded from repo)

SQLite database ✅ (created locally)

🧠 Models
Image Model: models/lung_cancer_model_v5.keras

Based on MobileNetV2

Input size: 224 × 224 × 3

Trained on LIDC‑IDRI nodule crops

Text Model:
models/text/tfidf_logistic_model.joblib

models/text/tfidf_vectorizer.joblib

models/text/decision_threshold.txt

Uses TF‑IDF + Logistic Regression

🏗️ System Architecture
Code
INPUT
  |
  +-----------+-----------+
  |                       |
 IMAGE                   PDF
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

⚙️ Backend API
Runs on: http://127.0.0.1:5000

Endpoints
Method	Endpoint	Purpose
GET	/api/health	Check backend & models
POST	/api/predict	Predict uploaded image
POST	/api/predict-pdf	Process PDF (text/images)
GET	/api/history	Retrieve prediction history
DELETE	/api/history/<id>	Delete history record


Start backend:

python backend/app.py
📊 Model Performance
Image Model (V5)
Accuracy: 75.42%

Precision: 77.40%

Recall: 85.93%

F1 Score: 81.44%

ROC‑AUC: 79.73%

Text Model
Accuracy: 93.41%

Precision: 51.43%

Recall: 70.59%

F1 Score: 59.50%

ROC‑AUC: 95.43%

📂 Project Structure
Code
lung-cancer-project/
│
├── backend/                # Flask backend
│   ├── app.py
│   └── uploads/
│
├── data/                   # Dataset folders
│   ├── text/
│   ├── nodule_crops/
│   └── ...
│
├── models/                 # Trained models
│   ├── lung_cancer_model_v5.keras
│   └── text/
│       ├── tfidf_logistic_model.joblib
│       ├── tfidf_vectorizer.joblib
│       └── decision_threshold.txt
│
├── src/                    # Training & evaluation scripts
│   ├── predict_image.py
│   ├── train_model_v5.py
│   ├── evaluate_model.py
│   └── text_classifier/
│       ├── train_text_model.py
│       ├── predict_text.py
│
├── requirements.txt
├── API_DOCUMENTATION.md
├── README.md
└── .gitignore

⚠️ Limitations
Image model is a research prototype trained only on CT nodule crops.

Arbitrary medical images (e.g., chest X‑rays) are outside training distribution.

Text classifier detects parenchymal lesions only, not full diagnosis.

PDF extraction depends on structure; OCR not implemented for scanned PDFs.

Model outputs are not medical certainty — for research purposes only.

🚀 Quick Test
Health check:

bash
curl http://127.0.0.1:5000/api/health
Image prediction:

bash
curl -X POST -F "image=@/path/to/image.jpg" http://127.0.0.1:5000/api/predict
PDF prediction:

bash
curl -X POST -F "pdf=@/path/to/file.pdf" http://127.0.0.1:5000/api/predict-pdf