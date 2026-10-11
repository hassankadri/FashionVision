"""Pure metrics: independent of model loading, data files, and report writing."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import CLASS_NAMES


def label_vector(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or not result.size or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty integer vector")
    if np.any((result < 0) | (result >= 10)):
        raise ValueError(f"{name} must contain canonical class IDs 0..9")
    return result


def calculate_metrics(y_true, *, values=None, predicted_labels=None,
                      class_names=CLASS_NAMES, value_kind="probabilities"):
    """Scores/probabilities are N x 10; columns follow class_names.

    Ties prefer the lower canonical class ID. Undefined class metrics are zero;
    macro averages always include all ten classes. Label-only Top-3 is unknown.
    """
    names = tuple(class_names)
    if len(names) != 10 or set(names) != set(CLASS_NAMES):
        raise ValueError("class_names must contain each canonical Fashion-MNIST class exactly once")
    truth = label_vector(y_true, "y_true")
    if (values is None) == (predicted_labels is None):
        raise ValueError("Provide exactly one of values or predicted_labels")
    top3 = None
    if values is not None:
        raw = np.asarray(values)
        if raw.shape != (len(truth), 10) or raw.dtype.kind not in "iuf":
            raise ValueError("Prediction values must be a numeric N x 10 matrix")
        scores = raw.astype(np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Prediction values must be finite")
        if value_kind not in ("probabilities", "scores"):
            raise ValueError("value_kind must be probabilities or scores")
        if value_kind == "probabilities":
            if np.any((scores < 0) | (scores > 1)) or not np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-5):
                raise ValueError("Probability rows must be in [0,1] and sum to one")
        scores = scores[:, [names.index(name) for name in CLASS_NAMES]]
        ranked = np.argsort(-scores, axis=1, kind="stable")
        predicted = ranked[:, 0]
        top3 = float(np.mean(np.any(ranked[:, :3] == truth[:, None], axis=1)))
    else:
        predicted = label_vector(predicted_labels, "predicted_labels")
        if predicted.shape != truth.shape:
            raise ValueError("Label prediction length must equal y_true length")
    accuracy = float(np.mean(predicted == truth))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=np.arange(10), zero_division=0)
    result = {"accuracy": accuracy, "top_1_accuracy": accuracy, "top_3_accuracy": top3,
              "evaluated_samples": len(truth),
              "confusion_matrix": confusion_matrix(truth, predicted, labels=np.arange(10)).tolist(),
              "per_class": [{"id": i, "label": name, "precision": float(precision[i]),
                             "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                            for i, name in enumerate(CLASS_NAMES)]}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            truth, predicted, labels=np.arange(10), average=average, zero_division=0)
        result.update({f"{average}_precision": float(p), f"{average}_recall": float(r),
                       f"{average}_f1": float(f)})
    return result
