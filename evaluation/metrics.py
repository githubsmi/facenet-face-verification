
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def choose_threshold(labels, scores):
    """Choose a threshold using the development set only."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        raise ValueError("Threshold selection requires both classes.")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.isfinite(thresholds)

    if not np.any(valid):
        raise ValueError("Could not determine a valid threshold.")

    # Youden's J statistic = TPR - FPR
    best_index = np.argmax((tpr - fpr)[valid])
    return float(thresholds[valid][best_index])


def compute_metrics(labels, scores, threshold):
    """Calculate binary face-verification metrics."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must have equal, nonzero lengths.")

    if len(np.unique(labels)) < 2:
        raise ValueError("Metrics require both positive and negative pairs.")

    # Higher cosine similarity means a stronger match.
    predictions = (scores >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    # Approximate EER from the closest FPR/FNR values.
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    return {
        "num_pairs": int(len(labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "far": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "frr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "eer_approx": eer,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def save_roc_plot(labels, scores, output_path):
    """Save the ROC curve as a PNG image."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Face Verification ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    temp_path = output_path.with_name(
        f"{output_path.stem}_new{output_path.suffix}"
    )

    try:
        fig.savefig(temp_path, dpi=160)
        plt.close(fig)
        temp_path.replace(output_path)
    finally:
        plt.close(fig)

def save_confusion_plot(y_true, scores, threshold, output_path):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Different person", "Same person"],
    )

    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Face Verification Confusion Matrix\nThreshold: {threshold:.4f}")
    fig.tight_layout()

    temp_path = output_path.with_name(
        f"{output_path.stem}_new{output_path.suffix}"
    )

    try:
        fig.savefig(temp_path, dpi=160)
        plt.close(fig)
        temp_path.replace(output_path)
    finally:
        plt.close(fig)
