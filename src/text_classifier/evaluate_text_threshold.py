import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

DATA_FILE = Path("data/text/lung_finding_dataset.csv")
MODEL_FILE = Path("models/text/tfidf_logistic_model.joblib")
VECTORIZER_FILE = Path("models/text/tfidf_vectorizer.joblib")

df = pd.read_csv(DATA_FILE).dropna(subset=["text", "label"])

X = df["text"].astype(str)
y = df["label"].astype(int)

# Recreate EXACT same split used during training
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Loading model...")
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

print(f"Validation reports: {len(X_val)}")
print(f"Test reports: {len(X_test)}")

X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

val_prob = model.predict_proba(X_val_tfidf)[:, 1]

print("\nValidation threshold search")
print("--------------------------------")
print("Threshold  Precision  Recall  F1")

best_threshold = 0.50
best_f1 = 0.0

for threshold in np.arange(0.10, 0.91, 0.01):

    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )

    recall = recall_score(
        y_val, val_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("--------------------------------")
print(f"Best validation threshold: {best_threshold:.2f}")
print(f"Best validation F1: {best_f1:.4f}")

# FINAL TEST — threshold chosen ONLY from validation
test_prob = model.predict_proba(X_test_tfidf)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy : {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, test_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nThreshold comparison:")
for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:

    pred = (test_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, pred, zero_division=0
    )

    recall = recall_score(
        y_test, pred, zero_division=0
    )

    f1 = f1_score(
        y_test, pred, zero_division=0
    )

    print(
        f"{threshold:.2f}       "
        f"{precision:.4f}    "
        f"{recall:.4f}  "
        f"{f1:.4f}"
    )

# Save threshold
threshold_file = Path("models/text/decision_threshold.txt")

with open(threshold_file, "w") as f:
    f.write(f"{best_threshold:.4f}")

print(f"\nSaved threshold to: {threshold_file}")
