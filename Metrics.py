"""
Metrics.py
==========
Evaluation metrics and paired cross-subject significance tests. These produce
the accuracy/F1 values and the p-value column of the cross-subject table:
collect each model's per-subject accuracy vector and compare it with MCL's
using a paired test (Wilcoxon signed-rank by default).

Pure scikit-learn/SciPy, so it runs without TensorFlow.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats


def classification_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f}


def evaluate_cross_subject(model, by_subject_data, test_subjects):
    """Accuracy and macro-F1 over held-out subjects.
    by_subject_data: dict[subject] -> (eeg, nirs, y_onehot)."""
    yt, yp = [], []
    for s in test_subjects:
        eeg, nirs, y = by_subject_data[s]
        yp.append(model.predict([eeg, nirs], verbose=0).argmax(1))
        yt.append(y.argmax(1))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    m = classification_metrics(yt, yp)
    return m["accuracy"], m["f1"]


def paired_significance(scores_model, scores_reference, method="wilcoxon"):
    a, b = np.asarray(scores_model, float), np.asarray(scores_reference, float)
    if a.shape != b.shape:
        raise ValueError("score vectors must align by subject")
    stat, p = (stats.wilcoxon(a, b) if method == "wilcoxon"
               else stats.ttest_rel(a, b))
    return {"method": method, "statistic": float(stat), "p_value": float(p)}


def significance_vs_reference(per_subject, reference="MCL", method="wilcoxon"):
    """per_subject: dict[model] -> per-subject accuracy vector.
    Returns dict[model] -> {statistic, p_value, method} (reference -> None)."""
    ref = per_subject[reference]
    return {m: (None if m == reference else paired_significance(v, ref, method))
            for m, v in per_subject.items()}


def format_p(p):
    return "<0.001" if p < 1e-3 else (f"{p:.3f}" if p < 1e-2 else f"{p:.2f}")
