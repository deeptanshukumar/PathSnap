"""Explicitly separated candidate, proxy pointwise, and direct route metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score


def candidate_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predicted = scores >= 0.5
    return {
        "candidate_accuracy": float(accuracy_score(labels, predicted)),
        "candidate_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "candidate_precision": float(precision_score(labels, predicted, zero_division=0)),
        "candidate_recall": float(recall_score(labels, predicted, zero_division=0)),
        "candidate_log_loss": float(log_loss(labels, np.column_stack([1 - scores, scores]), labels=[0, 1])),
        "candidate_confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def point_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    return {
        "proxy_top1_accuracy": float(np.mean(predicted == reference)),
        "proxy_arc_macro_f1": float(f1_score(reference, predicted, average="macro", zero_division=0)),
    }


def collapse(sequence: np.ndarray | list[int]) -> list[int]:
    values = np.asarray(sequence, dtype=int)
    if not len(values):
        return []
    return values[np.r_[True, values[1:] != values[:-1]]].tolist()


def ordered_lcs_f1(predicted: list[int], reference: list[int]) -> float:
    """Ordered sequence F1; repeated visits matter, no per-GPS reference is needed."""
    if not predicted or not reference:
        return 0.0
    previous = np.zeros(len(reference) + 1, dtype=np.int32)
    ref = np.asarray(reference)
    for arc in predicted:
        current = previous.copy()
        current[1:] = np.where(ref == arc, previous[:-1] + 1, previous[1:])
        current = np.maximum.accumulate(current)
        previous = current
    return float(2 * previous[-1] / (len(predicted) + len(reference)))


def route_metrics(predicted: np.ndarray | list[int], reference: np.ndarray, arcs: np.ndarray,
                  lengths: np.ndarray) -> dict:
    sequence = collapse(predicted)
    p, r = set(sequence), set(reference.tolist())
    p.discard(-1)
    intersection = p & r
    mass = lambda values: float(lengths[list(values)].sum()) if values else 0.0
    precision = mass(intersection) / max(mass(p), 1e-12)
    recall = mass(intersection) / max(mass(r), 1e-12)
    transitions = list(zip(sequence[:-1], sequence[1:]))
    continuity = float(np.mean([a >= 0 and b >= 0 and arcs[a, 1] == arcs[b, 0] for a, b in transitions])) if transitions else 1.0
    return {
        "route_length_precision": precision,
        "route_length_recall": recall,
        "route_length_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "ordered_route_lcs_f1": ordered_lcs_f1(sequence, collapse(reference)),
        "direct_arc_continuity": continuity,
        "predicted_sequence_arcs": len(sequence),
    }
