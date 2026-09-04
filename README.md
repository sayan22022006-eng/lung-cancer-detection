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
