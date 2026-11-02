"""Fresh synthetic fixtures only: paired samples and persistent cluster IDs."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paper1_review.paired_metrics import PredictionBatch, compare_robustness, paired_condition_metrics


def batch(pred, *, truth=None, ids=None, model="checkpoint-A", **kwargs):
    if truth is None:
        truth = [0] * (len(pred) // 2) + [1] * (len(pred) - len(pred) // 2)
    if ids is None:
        ids = [f"sample-{i}" for i in range(len(pred))]
    return PredictionBatch(ids, truth, pred, model, **kwargs)


def test_higher_corrupted_score_does_not_imply_better_paired_robustness():
    a_clean = batch([0] * 5 + [1] * 5)
    a_condition = batch([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    b_clean = batch([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], model="checkpoint-B")
    b_condition = batch([0, 0, 0, 1, 1, 1, 1, 1, 1, 0], model="checkpoint-B")
    result = compare_robustness(a_clean, a_condition, b_clean, b_condition, mode="inference_corruption")
    assert result["condition_difference_a_minus_b"]["hungarian_acc"] == pytest.approx(0.1)
    assert result["clean_difference_a_minus_b"]["hungarian_acc"] == pytest.approx(0.4)
    assert result["gamma"]["hungarian_acc"] == pytest.approx(-0.3)
    assert result["condition_difference_a_minus_b"]["ari"] > 0
    assert result["gamma"]["ari"] < 0
    for name, value in result["gamma"].items():
        assert value == pytest.approx(result["condition_difference_a_minus_b"][name] - result["clean_difference_a_minus_b"][name])
    assert result["clean_map_frozen_gamma"]["frozen_acc"] == pytest.approx(-0.3)
    json.dumps(result, allow_nan=False)


def test_pure_cluster_label_permutation_preserves_partition_but_not_frozen_map():
    clean = batch([100, 100, 900, 900], truth=["cat", "cat", "dog", "dog"])
    condition = replace(clean, y_pred=[900, 900, 100, 100])
    result = paired_condition_metrics(clean, condition, mode="inference_corruption")
    assert result["delta"] == pytest.approx({name: 0 for name in result["delta"]})
    assert result["condition"]["scores"]["hungarian_acc"] == 1
    frozen = result["clean_map_frozen"]
    assert frozen["condition_scores"] == {"frozen_acc": 0, "frozen_macro_f1": 0, "frozen_balanced_accuracy": 0}
    assert frozen["delta"]["frozen_acc"] == -1
    assert frozen["condition_remapped_minus_frozen"]["frozen_acc"] == 1
    assert len(result["mapping_changes"]["changed_existing_cluster_mappings"]) == 2
    assert all(item["recall_delta"] == -1 for item in frozen["condition_per_class"])


def test_retrained_models_never_silently_reuse_clean_map():
    clean = batch([10, 10, 20, 20])
    condition = replace(clean, y_pred=[20, 20, 10, 10], model_id="retrained-checkpoint")
    result = paired_condition_metrics(clean, condition, mode="retrained")
    assert result["clean_map_frozen"] is None
    assert result["mapping_changes"] is None
    assert result["delta"]["hungarian_acc"] == 0
    with pytest.raises(ValueError, match="same model_id"):
        paired_condition_metrics(clean, condition, mode="inference_corruption")
    with pytest.raises(TypeError):
        paired_condition_metrics(clean, condition)


def test_newly_occupied_condition_cluster_is_an_error_under_clean_map():
    clean = batch(["a", "a", "b", "b"], cluster_labels=["a", "b", "previously-empty"])
    condition = replace(clean, y_pred=["a", "a", "previously-empty", "previously-empty"])
    result = paired_condition_metrics(clean, condition, mode="inference_corruption")
    frozen = result["clean_map_frozen"]
    assert result["condition"]["scores"]["hungarian_acc"] == 1
    assert frozen["condition_scores"]["frozen_acc"] == 0.5
    assert frozen["unmapped_condition_sample_count"] == 2
    assert frozen["unmapped_condition_clusters"] == [{"cluster_label": "previously-empty", "count": 2, "reason": "absent_from_clean_map"}]
    assert result["mapping_changes"]["new_occupied_condition_clusters"] == ["previously-empty"]
    assert result["mapping_changes"]["no_longer_occupied_clusters"] == ["b"]
    assert frozen["confusion_counts"] == [[2, 0, 0], [0, 0, 2]]


def test_cluster_unmatched_on_clean_stays_unmapped_even_if_condition_pure():
    clean = batch(["a", "a", "extra", "b"], truth=[0, 0, 0, 1])
    condition = replace(clean, y_pred=["extra", "extra", "extra", "b"])
    result = paired_condition_metrics(clean, condition, mode="inference_corruption")
    assert result["condition"]["scores"]["hungarian_acc"] == 1
    assert result["clean_map_frozen"]["condition_scores"]["frozen_acc"] == 0.25
    assert result["clean_map_frozen"]["unmapped_condition_clusters"][0]["reason"] == "unmatched_in_clean_map"


def test_condition_alignment_matches_original_order_and_audits_permutation():
    clean = batch(["x", "x", "y", "y"], ids=[104, 902, 12, 65], truth=["cat", "cat", "dog", "dog"])
    condition = replace(clean, sample_ids=[65, 104, 12, 902], y_true=["dog", "cat", "dog", "cat"], y_pred=["y", "x", "y", "x"], probabilities=[[0, 1], [1, 0], [0, 1], [1, 0]], cluster_labels=["x", "y"])
    result = paired_condition_metrics(clean, condition, mode="inference_corruption")
    assert result["alignment"]["reordered"]
    assert result["alignment"]["rows_in_reference_order"] == [1, 3, 2, 0]
    assert result["delta"] == pytest.approx({name: 0 for name in result["delta"]})
    assert result["clean_map_frozen"]["condition_scores"]["frozen_acc"] == 1
    assert result["condition"]["probabilities"]["rows_valid"]


def test_all_four_method_batches_align_before_comparison():
    clean = batch([0, 0, 1, 1])
    condition = replace(clean, y_pred=[0, 1, 1, 1])
    order = [3, 1, 0, 2]
    other_clean = PredictionBatch([clean.sample_ids[i] for i in order], [clean.y_true[i] for i in order], [clean.y_pred[i] for i in order], "checkpoint-B")
    other_condition = replace(other_clean, y_pred=[condition.y_pred[i] for i in order])
    result = compare_robustness(clean, condition, other_clean, other_condition, mode="inference_corruption")
    assert result["alignment"]["b_clean"]["reordered"]
    assert result["alignment"]["b_condition"]["reordered"]
    assert result["gamma"] == pytest.approx({name: 0 for name in result["gamma"]})
    assert result["clean_map_frozen_gamma"] == pytest.approx({name: 0 for name in result["clean_map_frozen_gamma"]})


@pytest.mark.parametrize("change,match", [
    ({"sample_ids": ["sample-0", "sample-1", "sample-2", "other"]}, "sample IDs differ"),
    ({"sample_ids": ["sample-0", "sample-0", "sample-2", "sample-3"]}, "duplicates"),
    ({"sample_ids": ["sample-0", "sample-1", "sample-2"]}, "equal lengths"),
    ({"sample_ids": ["sample-0", "sample-1", "sample-2", np.nan]}, "nonfinite"),
    ({"y_true": [0, 1, 0, 1]}, "y_true disagrees"),
    ({"y_pred": [0, 1, np.nan, 1]}, "nonfinite"),
    ({"model_id": ""}, "model_id"),
    ({"probabilities": [[1, 0]]}, "one row per sample"),
])
def test_pairing_rejects_missing_duplicate_or_inconsistent_samples(change, match):
    clean = batch([0, 0, 1, 1])
    with pytest.raises(ValueError, match=match):
        paired_condition_metrics(clean, replace(clean, **change), mode="inference_corruption")


def test_duplicate_ids_in_reference_and_cross_method_truth_mismatch_are_rejected():
    clean = batch([0, 0, 1, 1])
    with pytest.raises(ValueError, match="duplicates"):
        paired_condition_metrics(replace(clean, sample_ids=[0, 0, 2, 3]), clean, mode="retrained")
    with pytest.raises(ValueError, match="y_true disagrees"):
        compare_robustness(clean, clean, replace(clean, y_true=[0, 1, 0, 1]), clean, mode="retrained")


def test_ids_preserve_string_numeric_identity():
    clean = batch([0, 0, 1, 1], ids=[1, "1", 2, "2"])
    condition = replace(clean, sample_ids=["2", 2, "1", 1], y_true=[1, 1, 0, 0], y_pred=[1, 1, 0, 0])
    result = paired_condition_metrics(clean, condition, mode="inference_corruption")
    assert result["alignment"]["rows_in_reference_order"] == [3, 2, 1, 0]
    assert result["delta"]["ari"] == 0


def test_tied_clean_mapping_is_flagged_in_frozen_analysis():
    clean = batch([0] * 4 + [1] * 3 + [0] * 3 + [1] * 2, truth=["a"] * 7 + ["b"] * 5)
    result = paired_condition_metrics(clean, clean, mode="inference_corruption")
    assert result["clean_map_frozen"]["clean_map_nonunique"]
    assert result["mapping_changes"]["clean_acc_map_nonunique"]
    assert not result["mapping_changes"]["any_map_change"]
    assert result["clean_map_frozen"]["delta"] == pytest.approx({name: 0 for name in result["clean_map_frozen"]["delta"]})


def test_invalid_mode_and_duplicate_method_names_fail_explicitly():
    clean = batch([0, 0, 1, 1])
    with pytest.raises(ValueError, match="mode"):
        paired_condition_metrics(clean, clean, mode="automatic")
    with pytest.raises(ValueError, match="distinct"):
        compare_robustness(clean, clean, clean, clean, mode="retrained", method_a="same", method_b="same")

