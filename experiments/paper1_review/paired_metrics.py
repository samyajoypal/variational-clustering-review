"""Sample-paired robustness scores for fresh review experiments.

All differences use identical sample IDs and truth labels. A positive method-A
minus method-B difference-in-differences means A's score changes more favorably;
it does not merely mean A has a higher corrupted score. No seed-level inference
or confidence interval is performed here.

For inference corruption of the SAME checkpoint, both independently remapped
scores and clean-map-frozen classification scores are reported. Cluster IDs
are not treated as persistent identities across retrained models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Literal, Sequence

import numpy as np

from .metrics import _encode, _label, clustering_metrics

SCHEMA_VERSION = "paper1_review.paired_metrics.v1"
ROBUSTNESS_SCORES = (
    "ari", "hungarian_acc", "nmi_arithmetic", "ami_arithmetic",
    "mapped_macro_f1", "mapped_balanced_accuracy",
)
FROZEN_TO_REMAPPED = {
    "frozen_acc": "hungarian_acc",
    "frozen_macro_f1": "mapped_macro_f1",
    "frozen_balanced_accuracy": "mapped_balanced_accuracy",
}
FROZEN_MAP_SCORES = tuple(FROZEN_TO_REMAPPED)
ComparisonMode = Literal["inference_corruption", "retrained"]


@dataclass(frozen=True)
class PredictionBatch:
    """Predictions plus sample/checkpoint identities; labels never train a model.

    model_id should identify actual trained weights (prefer a checkpoint hash),
    not merely the method name. Equality is a caller provenance assertion: this
    module does not open a checkpoint to independently verify it.
    """

    sample_ids: Sequence[Any]
    y_true: Sequence[Any]
    y_pred: Sequence[Any]
    model_id: str
    probabilities: Any | None = None
    cluster_labels: Sequence[Any] | None = None
    expected_n_clusters: int | None = None


def _row_keys(values: Sequence[Any], name: str) -> list[tuple[str, Any]]:
    _, ids, keys = _encode(values, name)
    return [keys[index] for index in ids]


def _validated(batch: PredictionBatch, name: str) -> PredictionBatch:
    if not isinstance(batch, PredictionBatch):
        raise ValueError(f"{name} must be a PredictionBatch")
    if not isinstance(batch.model_id, str) or not batch.model_id.strip():
        raise ValueError(f"{name}.model_id must identify a checkpoint with a nonempty string")
    sample_labels, encoded_ids, _ = _encode(batch.sample_ids, f"{name}.sample_ids")
    if len(sample_labels) != len(encoded_ids):
        raise ValueError(f"{name}.sample_ids contains duplicates")
    _, truth, _ = _encode(batch.y_true, f"{name}.y_true")
    _, pred, _ = _encode(batch.y_pred, f"{name}.y_pred")
    if len(truth) != len(encoded_ids) or len(pred) != len(encoded_ids):
        raise ValueError(f"{name} sample IDs, truth and predictions must have equal lengths")
    probabilities = None
    if batch.probabilities is not None:
        probabilities = np.asarray(batch.probabilities)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(encoded_ids):
            raise ValueError(f"{name}.probabilities must have one row per sample")
    return replace(batch, sample_ids=np.asarray(batch.sample_ids, dtype=object),
                   y_true=np.asarray(batch.y_true, dtype=object),
                   y_pred=np.asarray(batch.y_pred, dtype=object), probabilities=probabilities)


def _align(reference: PredictionBatch, other: PredictionBatch, name: str) -> tuple[PredictionBatch, dict[str, Any]]:
    reference_keys = _row_keys(reference.sample_ids, "reference.sample_ids")
    other_keys = _row_keys(other.sample_ids, f"{name}.sample_ids")
    lookup = {key: index for index, key in enumerate(other_keys)}
    missing = set(reference_keys) - set(other_keys)
    extra = set(other_keys) - set(reference_keys)
    if missing or extra:
        raise ValueError(f"{name} sample IDs differ: {len(missing)} missing and {len(extra)} extra")
    order = np.array([lookup[key] for key in reference_keys], dtype=np.int64)
    truth = other.y_true[order]
    if _row_keys(reference.y_true, "reference.y_true") != _row_keys(truth, f"{name}.y_true"):
        raise ValueError(f"{name} y_true disagrees with reference after sample-ID alignment")
    reordered = not np.array_equal(order, np.arange(len(order)))
    aligned = replace(other, sample_ids=other.sample_ids[order], y_true=truth,
                      y_pred=other.y_pred[order],
                      probabilities=other.probabilities[order] if other.probabilities is not None else None)
    return aligned, {
        "n_samples": len(order),
        "reordered": reordered,
        "rows_in_reference_order": order.tolist() if reordered else "identity",
        "reference_sample_ids_sha256": hashlib.sha256(json.dumps(reference_keys, ensure_ascii=False).encode()).hexdigest(),
        "same_truth_after_alignment": True,
    }


def _panel(batch: PredictionBatch, probability_atol: float) -> dict[str, Any]:
    return clustering_metrics(batch.y_true, batch.y_pred, probabilities=batch.probabilities,
                              cluster_labels=batch.cluster_labels,
                              expected_n_clusters=batch.expected_n_clusters,
                              probability_atol=probability_atol)


def _difference(condition: dict[str, float], clean: dict[str, float], names: Sequence[str]) -> dict[str, float]:
    return {key: float(condition[key] - clean[key]) for key in names}


def _mapping_lookup(panel: dict[str, Any]) -> dict[tuple[str, Any], dict[str, Any]]:
    return {_label(item["cluster_label"], "cluster_label")[1]: item for item in panel["mapping"]["assignments"]}


def _mapping_changes(clean_panel: dict[str, Any], condition_panel: dict[str, Any]) -> dict[str, Any]:
    clean_map = _mapping_lookup(clean_panel)
    condition_map = _mapping_lookup(condition_panel)
    changed = []
    for key, item in clean_map.items():
        if key not in condition_map:
            continue
        other = condition_map[key]
        before = _label(item["class_label"], "class_label")[1] if item["matched"] else None
        after = _label(other["class_label"], "class_label")[1] if other["matched"] else None
        if before != after:
            changed.append({"cluster_label": item["cluster_label"], "clean_class_label": item["class_label"],
                            "condition_class_label": other["class_label"],
                            "clean_matched": item["matched"], "condition_matched": other["matched"]})
    introduced = [item["cluster_label"] for key, item in condition_map.items() if key not in clean_map]
    disappeared = [item["cluster_label"] for key, item in clean_map.items() if key not in condition_map]
    return {
        "any_map_change": bool(changed or introduced or disappeared),
        "changed_existing_cluster_mappings": changed,
        "new_occupied_condition_clusters": introduced,
        "no_longer_occupied_clusters": disappeared,
        "clean_acc_map_nonunique": clean_panel["mapping"]["has_alternative_acc_optimum"],
        "condition_acc_map_nonunique": condition_panel["mapping"]["has_alternative_acc_optimum"],
        "interpretation": "Compares displayed ACC-optimal maps; an optimum tie can change a map without changing the ACC objective.",
    }


def _frozen_map(clean_panel: dict[str, Any], condition_panel: dict[str, Any], condition: PredictionBatch) -> dict[str, Any]:
    classes = clean_panel["audit"]["contingency_rows_true_labels"]
    class_ids = {_label(value, "class_label")[1]: index for index, value in enumerate(classes)}
    clean_map = _mapping_lookup(clean_panel)
    counts = np.zeros((len(classes), len(classes) + 1), dtype=np.int64)
    unmapped: dict[tuple[str, Any], dict[str, Any]] = {}
    for truth, pred in zip(condition.y_true, condition.y_pred):
        truth_key = _label(truth, "y_true")[1]
        pred_label, pred_key = _label(pred, "y_pred")
        assignment = clean_map.get(pred_key)
        if assignment is not None and assignment["matched"]:
            target = class_ids[_label(assignment["class_label"], "class_label")[1]]
        else:
            target = len(classes)
            if pred_key not in unmapped:
                unmapped[pred_key] = {
                    "cluster_label": pred_label, "count": 0,
                    "reason": "unmatched_in_clean_map" if assignment is not None else "absent_from_clean_map",
                }
            unmapped[pred_key]["count"] += 1
        counts[class_ids[truth_key], target] += 1
    per_class = []
    for i, label in enumerate(classes):
        support = int(counts[i].sum())
        predicted = int(counts[:, i].sum())
        tp = int(counts[i, i])
        item = {
            "class_label": label, "support": support, "mapped_predicted_count": predicted,
            "tp": tp, "fp": predicted - tp, "fn": support - tp,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support, "f1": 2 * tp / (support + predicted),
        }
        clean_class = clean_panel["per_class"][i]
        item["recall_delta"] = item["recall"] - clean_class["recall"]
        item["f1_delta"] = item["f1"] - clean_class["f1"]
        per_class.append(item)
    # Frozen ACC is NOT optimized on condition, so use distinct score names.
    condition_scores = {
        "frozen_acc": sum(item["tp"] for item in per_class) / len(condition.y_true),
        "frozen_macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "frozen_balanced_accuracy": float(np.mean([item["recall"] for item in per_class])),
    }
    clean_scores = {key: clean_panel["scores"][source] for key, source in FROZEN_TO_REMAPPED.items()}
    remapped_condition = {key: condition_panel["scores"][source] for key, source in FROZEN_TO_REMAPPED.items()}
    return {
        "mapping_source": "clean data ACC-maximizing Hungarian map, held fixed",
        "score_key_note": "frozen_acc uses the clean Hungarian map; there is no condition-specific optimization",
        "clean_map_nonunique": clean_panel["mapping"]["has_alternative_acc_optimum"],
        "clean_scores": clean_scores, "condition_scores": condition_scores,
        "delta": _difference(condition_scores, clean_scores, FROZEN_MAP_SCORES),
        "condition_per_class": per_class,
        "unmapped_condition_clusters": list(unmapped.values()),
        "unmapped_condition_sample_count": int(counts[:, -1].sum()),
        "confusion_rows_true_labels": classes,
        "confusion_columns_mapped_labels": classes + [None],
        "last_confusion_column": "unmapped cluster; every entry is an error",
        "confusion_counts": counts.tolist(),
        "condition_remapped_minus_frozen": _difference(remapped_condition, condition_scores, FROZEN_MAP_SCORES),
    }


def paired_condition_metrics(
    clean: PredictionBatch,
    condition: PredictionBatch,
    *,
    mode: ComparisonMode,
    probability_atol: float = 1e-6,
) -> dict[str, Any]:
    """Align condition to clean IDs, score both and return condition-minus-clean.

    mode is deliberately required. Only inference_corruption reuses a clean
    map, and it requires equal checkpoint IDs. In retrained mode component IDs
    are unrelated even if their numeric values happen to be the same.
    """
    if mode not in {"inference_corruption", "retrained"}:
        raise ValueError("mode must be inference_corruption or retrained")
    clean = _validated(clean, "clean")
    condition = _validated(condition, "condition")
    if mode == "inference_corruption" and clean.model_id != condition.model_id:
        raise ValueError("inference_corruption requires the same model_id/checkpoint for clean and condition")
    condition, alignment = _align(clean, condition, "condition")
    clean_panel, condition_panel = _panel(clean, probability_atol), _panel(condition, probability_atol)
    frozen = _frozen_map(clean_panel, condition_panel, condition) if mode == "inference_corruption" else None
    return {
        "schema_version": SCHEMA_VERSION, "mode": mode,
        "model_ids": {"clean": clean.model_id, "condition": condition.model_id},
        "model_identity_note": "Checkpoint identity is asserted by caller model_id; weights are not opened here.",
        "alignment": alignment, "clean": clean_panel, "condition": condition_panel,
        "delta_definition": "condition score minus clean score; larger means less deterioration or more improvement",
        "delta": _difference(condition_panel["scores"], clean_panel["scores"], ROBUSTNESS_SCORES),
        "mapping_changes": _mapping_changes(clean_panel, condition_panel) if mode == "inference_corruption" else None,
        "clean_map_frozen": frozen,
        "frozen_map_policy": "same-checkpoint comparison" if frozen is not None else "not applied: cluster identities are not carried across retrained models",
    }


def compare_robustness(
    a_clean: PredictionBatch,
    a_condition: PredictionBatch,
    b_clean: PredictionBatch,
    b_condition: PredictionBatch,
    *,
    mode: ComparisonMode,
    method_a: str = "A",
    method_b: str = "B",
    probability_atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare paired deterioration, aligning all four batches to A's clean IDs."""
    if not isinstance(method_a, str) or not isinstance(method_b, str) or not method_a.strip() or not method_b.strip() or method_a == method_b:
        raise ValueError("method_a and method_b must be distinct nonempty names")
    batches = {"a_clean": a_clean, "a_condition": a_condition, "b_clean": b_clean, "b_condition": b_condition}
    batches = {name: _validated(batch, name) for name, batch in batches.items()}
    reference = batches["a_clean"]
    alignment = {}
    for name in batches:
        batches[name], alignment[name] = _align(reference, batches[name], name)
    a = paired_condition_metrics(batches["a_clean"], batches["a_condition"], mode=mode, probability_atol=probability_atol)
    b = paired_condition_metrics(batches["b_clean"], batches["b_condition"], mode=mode, probability_atol=probability_atol)
    frozen_gamma = None
    if mode == "inference_corruption":
        frozen_gamma = _difference(a["clean_map_frozen"]["delta"], b["clean_map_frozen"]["delta"], FROZEN_MAP_SCORES)
    return {
        "schema_version": SCHEMA_VERSION, "mode": mode,
        "method_a": method_a, "method_b": method_b,
        "alignment_reference": "method_a.clean", "alignment": alignment,
        "methods": {method_a: a, method_b: b},
        "clean_difference_a_minus_b": _difference(a["clean"]["scores"], b["clean"]["scores"], ROBUSTNESS_SCORES),
        "condition_difference_a_minus_b": _difference(a["condition"]["scores"], b["condition"]["scores"], ROBUSTNESS_SCORES),
        "gamma_definition": "(A_condition - A_clean) - (B_condition - B_clean)",
        "gamma_interpretation": "positive favors A's score change, not necessarily A's absolute condition score",
        "gamma": _difference(a["delta"], b["delta"], ROBUSTNESS_SCORES),
        "clean_map_frozen_gamma": frozen_gamma,
        "uncertainty_note": "One paired comparison only; no seed aggregation, significance test or confidence interval is implied.",
    }
