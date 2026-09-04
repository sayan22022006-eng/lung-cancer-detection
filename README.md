# Lung Cancer Detection Project

A research/academic prototype for analyzing lung-related findings from medical images and radiology reports.

> **Important:** This project is for academic/research purposes only. It is not a clinical diagnostic system and must not be used for medical decisions.

## Features

- CT image classification using a MobileNetV2-based deep-learning model
- Radiology report text classification using TF-IDF + Logistic Regression
- PDF processing
- Multi-page PDF processing
- Image-only PDF support
- Text-only PDF support
- Mixed PDF support
- Prediction history
- Uploaded-file storage
- SQLite database for prediction history
- Flask REST API
- React frontend integration

## System Architecture

```text
React Frontend
      |
      | HTTP REST API
      v
Flask Backend
      |
      +-------------------+
      |                   |
      v                   v
CT Image Model       Text Model
MobileNetV2          TF-IDF + LR
      |                   |
      +---------+---------+
                |
                v
         Prediction Result
                |
        +-------+-------+
        |               |
        v               v
 Uploaded Files     SQLite Database
backend/uploads/    prediction_history.db
