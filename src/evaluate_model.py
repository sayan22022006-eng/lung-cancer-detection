"""
Lung Cancer Model - Evaluation & Visualization
Generates:
1. Training vs validation accuracy
2. Training vs validation loss
3. ROC curve + AUC
4. Confusion matrix
5. Test prediction analysis
6. Overfitting analysis
7. Final evaluation summary

Run from project root:
    python3 src/evaluate_model.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "data" / "training_history.csv"
PREDICTION_PATH = BASE_DIR / "data" / "test_predictions.csv"
OUTPUT_DIR = BASE_DIR / "data" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("       LUNG CANCER MODEL EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def find_column(df, candidates, required=True):
    """Find a column using exact names first, then case-insensitive matching."""
    for name in candidates:
        if name in df.columns:
            return name

    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def save_fig(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ------------------------------------------------------------
# 1 & 2. TRAINING HISTORY
# ------------------------------------------------------------
print("\nReading training history...")

if not HISTORY_PATH.exists():
    raise FileNotFoundError(f"Missing: {HISTORY_PATH}")

history = pd.read_csv(HISTORY_PATH)

print("History columns:")
print(history.columns.tolist())

# Normalize possible epoch column
if "epoch" not in history.columns:
    history.insert(0, "epoch", np.arange(1, len(history) + 1))

train_acc = find_column(history, ["accuracy", "acc"])
val_acc = find_column(history, ["val_accuracy", "val_acc"])
train_loss = find_column(history, ["loss"])
val_loss = find_column(history, ["val_loss"])

# ----- Accuracy graph -----
plt.figure(figsize=(9, 6))
plt.plot(history["epoch"], history[train_acc], marker="o", linewidth=2, label="Training Accuracy")
plt.plot(history["epoch"], history[val_acc], marker="o", linewidth=2, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
save_fig("training_vs_validation_accuracy.png")

# ----- Loss graph -----
plt.figure(figsize=(9, 6))
plt.plot(history["epoch"], history[train_loss], marker="o", linewidth=2, label="Training Loss")
plt.plot(history["epoch"], history[val_loss], marker="o", linewidth=2, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training vs Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
save_fig("training_vs_validation_loss.png")

# ------------------------------------------------------------
# 3, 4 & 5. TEST PREDICTIONS
# ------------------------------------------------------------
print("\nReading test predictions...")

if not PREDICTION_PATH.exists():
    raise FileNotFoundError(f"Missing: {PREDICTION_PATH}")

pred = pd.read_csv(PREDICTION_PATH)

print("Prediction columns:")
print(pred.columns.tolist())

actual_col = find_column(
    pred,
    [
        "actual",
        "actual_label",
        "true_label",
        "y_true",
        "label",
        "binary_label",
        "target",
    ],
)

prob_col = find_column(
    pred,
    [
        "probability",
        "predicted_probability",
        "prediction_probability",
        "malignant_probability",
        "prob",
        "score",
        "prediction",
        "predicted_prob",
    ],
    required=False,
)

pred_label_col = find_column(
    pred,
    [
        "predicted_label",
        "prediction_label",
        "predicted_class",
        "y_pred",
        "predicted",
    ],
    required=False,
)

y_true = pd.to_numeric(pred[actual_col], errors="coerce").astype(int).to_numpy()

if prob_col is not None:
    y_prob = pd.to_numeric(pred[prob_col], errors="coerce").to_numpy()
    # If predictions are accidentally stored as 0/1 labels, this is still valid.
    y_prob = np.clip(y_prob, 0, 1)
else:
    y_prob = None

if pred_label_col is not None:
    y_pred = pd.to_numeric(pred[pred_label_col], errors="coerce").astype(int).to_numpy()
elif y_prob is not None:
    y_pred = (y_prob >= 0.5).astype(int)
else:
    raise ValueError(
        "Could not find predicted labels or probabilities in test_predictions.csv."
    )

# Remove invalid rows
valid = np.isfinite(y_true) & np.isfinite(y_pred)
if y_prob is not None:
    valid &= np.isfinite(y_prob)

y_true = y_true[valid]
y_pred = y_pred[valid]
if y_prob is not None:
    y_prob = y_prob[valid]

# ----- ROC -----
if y_prob is not None and len(np.unique(y_true)) == 2:
    auc_value = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    save_fig("roc_curve_auc.png")
else:
    auc_value = np.nan
    print("ROC curve skipped: probability column not available.")

# ----- Confusion matrix -----
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Test Set")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Non-malignant", "Malignant"])
plt.yticks(tick_marks, ["Non-malignant", "Malignant"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

threshold_cm = cm.max() / 2.0 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
        )

save_fig("confusion_matrix.png")

# ----- Test prediction analysis -----
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Prediction counts
actual_counts = pd.Series(y_true).value_counts().reindex([0, 1], fill_value=0)
pred_counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)

plt.figure(figsize=(9, 6))
x = np.arange(2)
width = 0.35
plt.bar(x - width / 2, actual_counts.values, width, label="Actual")
plt.bar(x + width / 2, pred_counts.values, width, label="Predicted")
plt.xticks(x, ["Non-malignant", "Malignant"])
plt.ylabel("Number of test samples")
plt.title("Test Prediction Analysis")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()

for i, value in enumerate(actual_counts.values):
    plt.text(i - width / 2, value + max(actual_counts.max() * 0.01, 1), str(value), ha="center")

for i, value in enumerate(pred_counts.values):
    plt.text(i + width / 2, value + max(pred_counts.max() * 0.01, 1), str(value), ha="center")

save_fig("test_prediction_analysis.png")

# ------------------------------------------------------------
# 6. OVERFITTING ANALYSIS
# ------------------------------------------------------------
best_val_acc_idx = history[val_acc].idxmax()
best_val_loss_idx = history[val_loss].idxmin()

final_train_acc = float(history[train_acc].iloc[-1])
final_val_acc = float(history[val_acc].iloc[-1])
final_train_loss = float(history[train_loss].iloc[-1])
final_val_loss = float(history[val_loss].iloc[-1])

best_val_acc = float(history[val_acc].max())
best_val_loss = float(history[val_loss].min())

accuracy_gap = final_train_acc - final_val_acc
loss_gap = final_val_loss - final_train_loss

# A simple, transparent heuristic:
# - Strong overfitting: train accuracy substantially above validation accuracy
#   AND validation loss clearly above training loss.
# - Mild/possible: moderate gap.
if accuracy_gap >= 0.10 and loss_gap >= 0.10:
    overfit_status = "STRONG EVIDENCE OF OVERFITTING"
elif accuracy_gap >= 0.05 or loss_gap >= 0.05:
    overfit_status = "POSSIBLE / MILD OVERFITTING"
else:
    overfit_status = "NO STRONG EVIDENCE OF OVERFITTING"

# Generalization assessment based on validation/test accuracy
generalization_gap = final_val_acc - accuracy

if accuracy >= 0.70 and not np.isnan(auc_value) and auc_value >= 0.70:
    recommendation = (
        "The current model is reasonably useful as a project baseline. "
        "Do not change the model only for the sake of changing it. "
        "If time permits, compare one stronger strategy (for example, "
        "patient-level aggregation or a better pretrained model) as an experiment."
    )
elif accuracy >= 0.70:
    recommendation = (
        "The current model has a useful accuracy level, but additional evaluation "
        "is recommended because ROC-AUC/probability quality should also be checked. "
        "A new model is optional rather than mandatory."
    )
else:
    recommendation = (
        "The current model is not yet strong enough for a final project result. "
        "Another training strategy/model should be considered after checking "
        "the data pipeline and patient-level evaluation."
    )

report_lines = [
    "=" * 60,
    "LUNG CANCER MODEL - EVALUATION REPORT",
    "=" * 60,
    "",
    "TRAINING HISTORY",
    f"Epochs recorded: {len(history)}",
    f"Best validation accuracy: {best_val_acc:.4f} (epoch {int(history['epoch'].iloc[best_val_acc_idx])})",
    f"Best validation loss: {best_val_loss:.4f} (epoch {int(history['epoch'].iloc[best_val_loss_idx])})",
    "",
    "FINAL EPOCH",
    f"Training accuracy:   {final_train_acc:.4f}",
    f"Validation accuracy: {final_val_acc:.4f}",
    f"Training loss:       {final_train_loss:.4f}",
    f"Validation loss:     {final_val_loss:.4f}",
    f"Accuracy gap:        {accuracy_gap:.4f}",
    f"Loss gap:            {loss_gap:.4f}",
    "",
    "TEST SET",
    f"Test samples: {len(y_true)}",
    f"Accuracy:  {accuracy:.4f}",
    f"Precision: {precision:.4f}",
    f"Recall:    {recall:.4f}",
    f"F1 Score:  {f1:.4f}",
    f"ROC-AUC:   {auc_value:.4f}" if not np.isnan(auc_value) else "ROC-AUC:   Not available",
    "",
    "CONFUSION MATRIX",
    f"[[TN={cm[0,0]}, FP={cm[0,1]}],",
    f" [FN={cm[1,0]}, TP={cm[1,1]}]]",
    "",
    "OVERFITTING ASSESSMENT",
    overfit_status,
    f"Validation-to-test accuracy difference: {generalization_gap:.4f}",
    "",
    "RECOMMENDATION",
    recommendation,
    "",
    "OUTPUT FILES",
    "training_vs_validation_accuracy.png",
    "training_vs_validation_loss.png",
    "roc_curve_auc.png (if probability column is available)",
    "confusion_matrix.png",
    "test_prediction_analysis.png",
    "evaluation_report.txt",
    "=" * 60,
]

report_path = OUTPUT_DIR / "evaluation_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

# Also save a compact metrics CSV
metrics_df = pd.DataFrame(
    {
        "metric": [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
            "final_train_accuracy",
            "final_validation_accuracy",
            "final_train_loss",
            "final_validation_loss",
            "accuracy_gap",
            "loss_gap",
        ],
        "value": [
            accuracy,
            precision,
            recall,
            f1,
            auc_value,
            final_train_acc,
            final_val_acc,
            final_train_loss,
            final_val_loss,
            accuracy_gap,
            loss_gap,
        ],
    }
)
metrics_df.to_csv(OUTPUT_DIR / "final_metrics.csv", index=False)

print("\n" + "\n".join(report_lines))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Non-malignant", "Malignant"],
        zero_division=0,
    )
)

print("\nAll evaluation files are in:")
print(OUTPUT_DIR)
print("\nDONE")
