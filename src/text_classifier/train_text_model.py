import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

DATA_FILE = Path("data/text/lung_finding_dataset.csv")
MODEL_DIR = Path("models/text")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

df = df.dropna(subset=["text", "label"])

X = df["text"].astype(str)
y = df["label"].astype(int)

print(f"Total reports: {len(df)}")
print(f"Positive: {(y == 1).sum()}")
print(f"Negative: {(y == 0).sum()}")

# Stratified split
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

print("\nDataset split:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# TF-IDF
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Train matrix: {X_train_tfidf.shape}")

# Logistic Regression
print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Validation
val_prob = model.predict_proba(X_val_tfidf)[:, 1]
val_pred = (val_prob >= 0.50).astype(int)

print("\nValidation results:")
print(f"Accuracy : {accuracy_score(y_val, val_pred):.4f}")
print(f"Precision: {precision_score(y_val, val_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_val, val_pred, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_val, val_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_val, val_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_val, val_prob):.4f}")

# Test
test_prob = model.predict_proba(X_test_tfidf)[:, 1]
test_pred = (test_prob >= 0.50).astype(int)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Accuracy : {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, test_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["No lesion", "Parenchymal lesion"],
    zero_division=0
))

# Save model and vectorizer
joblib.dump(model, MODEL_DIR / "tfidf_logistic_model.joblib")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

# Save test predictions
results = pd.DataFrame({
    "text": X_test.values,
    "actual": y_test.values,
    "probability": test_prob,
    "prediction": test_pred
})

results.to_csv(
    "data/text/text_test_predictions.csv",
    index=False
)

print("\nSaved:")
print(f"- {MODEL_DIR / 'tfidf_logistic_model.joblib'}")
print(f"- {MODEL_DIR / 'tfidf_vectorizer.joblib'}")
print("- data/text/text_test_predictions.csv")
