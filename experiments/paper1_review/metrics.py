"""Auditable external clustering metrics for fresh Paper 1 review experiments.

ARI, NMI, AMI and homogeneity/completeness/V-measure compare partitions directly.
ACC uses a one-to-one, maximum-count Hungarian assignment, never cluster purity.
Mapped F1 and recall use that same ACC-optimal assignment; they do not optimize
their own objectives. When ACC has multiple optimal assignments, these mapped
secondary scores need not be unique, and the output explicitly flags that fact.

Optional probabilities are cluster responsibilities, not calibrated class
probabilities. Only validity, occupancy and entropy diagnostics are computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

SCHEMA_VERSION = "paper1_review.clustering_metrics.v1"
DEFINITION_SOURCES = {
    "ari": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html",
    "nmi": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html",
    "ami": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html",
    "homogeneity_completeness_v_measure": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html",
    "hungarian": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html",
}


def _label(value: Any, name: str) -> tuple[Any, tuple[str, Any]]:
    """Keep string/numeric identities distinct without coercing mixed arrays."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value, ("string", value)
    if isinstance(value, numbers.Integral):
        # Integer labels can exceed float64's range without being invalid.
        value = int(value)
        return value, ("number", value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} contains a nonfinite label")
        if converted != value:
            raise ValueError(f"{name} contains a real label that cannot be represented exactly in the JSON audit; use a string label")
        value = converted
        # Ordinary numeric equality applies, e.g. integer 1 and float 1.0.
        return value, ("number", value)
    raise ValueError(f"{name} labels must be finite real numbers or strings; got {type(value).__name__}")


def _encode(values: Sequence[Any], name: str) -> tuple[list[Any], np.ndarray, list[tuple[str, Any]]]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    labels: list[Any] = []
    keys: list[tuple[str, Any]] = []
    lookup: dict[tuple[str, Any], int] = {}
    encoded = np.empty(array.size, dtype=np.int64)
    for index, value in enumerate(array):
        scalar, key = _label(value, name)
        if key not in lookup:
            lookup[key] = len(labels)
            labels.append(scalar)
            keys.append(key)
        encoded[index] = lookup[key]
    return labels, encoded, keys


def _entropy(masses: np.ndarray) -> float:
    positive = masses[masses > 0]
    return float(-np.sum(positive * np.log(positive)))


def _alternative_optimum(contingency: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> bool:
    """At most min(C,K) extra assignment solves; no factorial enumeration.

    Any different optimum omits an edge of the selected matching. Forbid each
    selected edge in turn and check the exact integer count of its alternative.
    A penalty below minus N prevents a forbidden edge whenever feasible.
    """
    best = int(contingency[rows, cols].sum())
    total = int(contingency.sum())
    for row, col in zip(rows, cols):
        altered = contingency.copy()
        altered[row, col] = -total - 1
        other_rows, other_cols = linear_sum_assignment(altered, maximize=True)
        if any(r == row and c == col for r, c in zip(other_rows, other_cols)):
            continue
        if int(contingency[other_rows, other_cols].sum()) == best:
            return True
    return False


def clustering_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    probabilities: Any | None = None,
    cluster_labels: Sequence[Any] | None = None,
    expected_n_clusters: int | None = None,
    probability_atol: float = 1e-6,
) -> dict[str, Any]:
    """Return a JSON-ready metric panel, mapping and per-class count audit.

    Labels may be strings or noncontiguous finite numbers. The observed truth
    classes define macro averaging. Unmatched predicted clusters are errors,
    not mapped many-to-one; unmatched truth classes have zero recall/F1.

    ``cluster_labels`` names all model components, including empty ones, and
    fixes probability-column order when probabilities are supplied. Without
    these names probability columns are explicitly unaligned to hard labels.
    ``expected_n_clusters`` can retain unnamed zero-occupancy slots. Neither
    empty slots nor responsibilities change the hard-label Hungarian matching.

    Label encoding follows first occurrence, making assignments invariant to
    renaming labels. Under tied ACC optima, observation reordering can select
    another equally good map; the output flags this limitation for F1/recall.
    """
    truth, true_ids, _ = _encode(y_true, "y_true")
    clusters, pred_ids, cluster_keys = _encode(y_pred, "y_pred")
    if len(true_ids) != len(pred_ids):
        raise ValueError("y_true and y_pred lengths differ")
    n = len(true_ids)
    c, k = len(truth), len(clusters)
    if expected_n_clusters is not None:
        if isinstance(expected_n_clusters, (bool, np.bool_)) or not isinstance(expected_n_clusters, numbers.Integral) or expected_n_clusters < k:
            raise ValueError("expected_n_clusters must be an integer at least the occupied cluster count")
        expected_n_clusters = int(expected_n_clusters)
    if not isinstance(probability_atol, numbers.Real) or not math.isfinite(probability_atol) or not 0 <= probability_atol < 1:
        raise ValueError("probability_atol must be finite and in [0, 1)")

    declared_labels = None
    declared_keys = None
    if cluster_labels is not None:
        declared_labels, declared_ids, declared_keys = _encode(cluster_labels, "cluster_labels")
        if len(declared_labels) != len(declared_ids):
            raise ValueError("cluster_labels must contain unique labels")
        if not set(cluster_keys).issubset(declared_keys):
            raise ValueError("y_pred contains a cluster absent from cluster_labels")
        if expected_n_clusters is not None and expected_n_clusters != len(declared_labels):
            raise ValueError("expected_n_clusters differs from cluster_labels length")
        expected_n_clusters = len(declared_labels)

    probability_panel = None
    if probabilities is not None:
        raw = np.asarray(probabilities)
        if raw.dtype.kind not in "biuf" or raw.ndim != 2 or raw.shape[0] != n or raw.shape[1] == 0:
            raise ValueError("probabilities must be a numeric N-by-K matrix")
        p = raw.astype(np.float64)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("probabilities must be finite and in [0, 1]")
        if p.shape[1] < k or (expected_n_clusters is not None and p.shape[1] != expected_n_clusters):
            raise ValueError("probability columns disagree with the declared/occupied cluster count")
        expected_n_clusters = p.shape[1]
        row_sum = p.sum(axis=1)
        mass_error = float(np.max(np.abs(row_sum - 1)))
        if mass_error > probability_atol:
            raise ValueError(f"probability rows do not sum to one within atol={probability_atol}; maximum error={mass_error}")
        # Explicit normalization of accepted floating-point row-sum errors.
        p = p / row_sum[:, None]
        soft_mass = p.mean(axis=0)
        row_entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, np.finfo(float).tiny)), 0), axis=1)
        soft_entropy = _entropy(soft_mass)
        probability_panel = {
            "interpretation": "cluster responsibilities only; no class calibration or class-probability score",
            "rows_valid": True,
            "row_sum_atol": float(probability_atol),
            "max_original_row_sum_error": mass_error,
            "normalization": "accepted rows normalized to unit mass for diagnostics",
            "column_labels": declared_labels,
            "alignment_to_hard_labels": "explicit cluster_labels" if declared_labels is not None else "unspecified; column order is not inferred from labels",
            "mean_component_mass": soft_mass.tolist(),
            "occupancy_entropy_nats": soft_entropy,
            "effective_component_count": float(np.exp(soft_entropy)),
            "max_component_share": float(soft_mass.max()),
            "mean_row_entropy_nats": float(row_entropy.mean()),
            "min_row_entropy_nats": float(row_entropy.min()),
            "max_row_entropy_nats": float(row_entropy.max()),
            "mean_max_cluster_probability": float(p.max(axis=1).mean()),
        }

    expected = expected_n_clusters if expected_n_clusters is not None else k
    contingency = np.zeros((c, k), dtype=np.int64)
    np.add.at(contingency, (true_ids, pred_ids), 1)
    rows, cols = linear_sum_assignment(contingency, maximize=True)
    ambiguous = _alternative_optimum(contingency, rows, cols)
    cluster_to_class = {int(col): int(row) for row, col in zip(rows, cols)}
    class_to_cluster = {row: col for col, row in cluster_to_class.items()}
    true_counts = contingency.sum(axis=1)
    pred_counts = contingency.sum(axis=0)
    per_class = []
    for i, label in enumerate(truth):
        j = class_to_cluster.get(i)
        tp = int(contingency[i, j]) if j is not None else 0
        predicted = int(pred_counts[j]) if j is not None else 0
        support = int(true_counts[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1 = 2 * tp / (support + predicted)
        per_class.append({
            "class_label": label, "support": support,
            "mapped_cluster_label": clusters[j] if j is not None else None,
            "has_assigned_cluster": j is not None,
            "mapped_predicted_count": predicted, "tp": tp,
            "fp": predicted - tp, "fn": support - tp,
            "precision": precision, "recall": recall, "f1": f1,
        })
    assignment = [{
        "cluster_label": label,
        "class_label": truth[cluster_to_class[j]] if j in cluster_to_class else None,
        "matched": j in cluster_to_class,
        "cluster_count": int(pred_counts[j]),
        "correct_count": int(contingency[cluster_to_class[j], j]) if j in cluster_to_class else 0,
    } for j, label in enumerate(clusters)]
    observed_count = dict(zip(cluster_keys, map(int, pred_counts)))
    if declared_labels is not None:
        occupancy = [{"cluster_label": label, "label_known": True, "count": observed_count.get(key, 0)} for label, key in zip(declared_labels, declared_keys)]
    else:
        occupancy = [{"cluster_label": label, "label_known": True, "count": int(count)} for label, count in zip(clusters, pred_counts)]
        occupancy += [{"cluster_label": None, "label_known": False, "count": 0} for _ in range(expected - k)]
    for item in occupancy:
        item["share"] = item["count"] / n
    hard_mass = pred_counts / n
    entropy = _entropy(hard_mass)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_ids, pred_ids)
    warnings = []
    if ambiguous:
        warnings.append("Multiple ACC-optimal maps exist. Mapped macro-F1, balanced accuracy and class-level scores describe the displayed map and may change under another ACC-optimal map.")
    return {
        "schema_version": SCHEMA_VERSION,
        "n_samples": n, "n_true_classes": c,
        "scores": {
            "ari": float(adjusted_rand_score(true_ids, pred_ids)),
            "nmi_arithmetic": float(normalized_mutual_info_score(true_ids, pred_ids, average_method="arithmetic")),
            "ami_arithmetic": float(adjusted_mutual_info_score(true_ids, pred_ids, average_method="arithmetic")),
            "hungarian_acc": int(contingency[rows, cols].sum()) / n,
            "homogeneity": float(homogeneity), "completeness": float(completeness), "v_measure": float(v_measure),
            "mapped_macro_f1": float(np.mean([item["f1"] for item in per_class])),
            "mapped_balanced_accuracy": float(np.mean([item["recall"] for item in per_class])),
        },
        "mapping": {
            "objective": "one-to-one maximum correct count (Hungarian ACC), not purity or F1 maximization",
            "solver": "scipy.optimize.linear_sum_assignment(maximize=True)",
            "ordering": "first occurrence of observed labels; empty components excluded from matching",
            "has_alternative_acc_optimum": ambiguous,
            "secondary_scores_depend_on_displayed_mapping": ambiguous,
            "unmatched_prediction_policy": "every observation assigned to an unmatched cluster is incorrect",
            "macro_average_classes": "all observed truth classes; no extra unknown-label class; zero division returns zero",
            "assignments": assignment,
            "unmatched_truth_classes": [truth[i] for i in range(c) if i not in class_to_cluster],
            "unmatched_predicted_clusters": [clusters[j] for j in range(k) if j not in cluster_to_class],
        },
        "per_class": per_class,
        "audit": {
            "contingency_rows_true_labels": truth,
            "contingency_columns_cluster_labels": clusters,
            "contingency_counts": contingency.tolist(),
            "sum_true_support": int(true_counts.sum()),
            "sum_cluster_counts": int(pred_counts.sum()),
            "sum_mapped_true_positives": sum(item["tp"] for item in per_class),
        },
        "occupancy": {
            "occupied_cluster_count": k, "expected_cluster_count": expected,
            "unoccupied_cluster_count": expected - k,
            "entropy_nats": entropy,
            "normalized_entropy": entropy / math.log(expected) if expected > 1 else 0.0,
            "effective_cluster_count": float(np.exp(entropy)),
            "max_cluster_share": float(hard_mass.max()),
            "components": occupancy,
        },
        "probabilities": probability_panel,
        "warnings": warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a newly generated NPZ with an explicit fresh-run manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Fresh prediction NPZ: targets, predictions, optional responsibilities and cluster_labels")
    parser.add_argument("--metadata", required=True, type=Path, help="JSON with fresh_run=true, run_id, protocol_version, source_sha256 and config_sha256")
    parser.add_argument("--output", required=True, type=Path, help="New JSON output path (existing files are not overwritten)")
    parser.add_argument("--expected-n-clusters", type=int)
    parser.add_argument("--probability-atol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata.read_text())
    if not isinstance(metadata, dict) or metadata.get("fresh_run") is not True:
        raise ValueError("metadata must declare fresh_run=true; this CLI is for new review predictions")
    for key in ("run_id", "protocol_version", "source_sha256", "config_sha256"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata requires a nonempty {key}")
    for key in ("source_sha256", "config_sha256"):
        if len(metadata[key]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in metadata[key]):
            raise ValueError(f"metadata {key} must be a SHA-256 hex digest")
    with np.load(args.input, allow_pickle=False) as arrays:
        if not {"targets", "predictions"}.issubset(arrays.files):
            raise ValueError("prediction NPZ requires targets and predictions")
        result = clustering_metrics(
            arrays["targets"], arrays["predictions"],
            probabilities=arrays["responsibilities"] if "responsibilities" in arrays.files else None,
            cluster_labels=arrays["cluster_labels"] if "cluster_labels" in arrays.files else None,
            expected_n_clusters=args.expected_n_clusters,
            probability_atol=args.probability_atol,
        )
    result["provenance"] = {
        "run_metadata": metadata,
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "metric_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "definition_sources": DEFINITION_SOURCES,
        "freshness_note": "Fresh-run origin is a caller provenance assertion, not inferred from prediction values.",
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
