"""Independent small examples for the fresh-review metric contract."""

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paper1_review.metrics import SCHEMA_VERSION, clustering_metrics, main


def test_perfect_string_and_noncontiguous_label_permutations():
    truth = [10, 10, -7, -7, 1001, 1001]
    clusters = ["blue", "blue", "red", "red", "green", "green"]
    result = clustering_metrics(truth, clusters)
    renamed = clustering_metrics(["a", "a", "b", "b", "c", "c"], [91, 91, 43, 43, -6, -6])
    assert result["scores"] == pytest.approx({key: 1 for key in result["scores"]})
    assert result["scores"] == renamed["scores"]
    assert result["mapping"]["has_alternative_acc_optimum"] is False
    assert result["audit"]["sum_mapped_true_positives"] == 6
    assert {item["cluster_label"]: item["class_label"] for item in result["mapping"]["assignments"]} == {"blue": 10, "red": -7, "green": 1001}
    json.dumps(result, allow_nan=False)


def test_row_order_and_label_rename_invariance_with_unique_map():
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([11, 11, 11, 55, 55, 55, 88, 88, 88, 88])
    result = clustering_metrics(truth, pred)
    order = np.array([8, 3, 2, 0, 9, 4, 1, 6, 7, 5])
    reordered = clustering_metrics(truth[order], pred[order])
    assert not result["mapping"]["has_alternative_acc_optimum"]
    assert result["scores"] == pytest.approx(reordered["scores"])


def test_collapsed_cluster_exposes_minority_class_failure():
    result = clustering_metrics(["major"] * 4 + ["minor-a", "minor-b"], [90] * 6, expected_n_clusters=3)
    assert result["scores"]["hungarian_acc"] == pytest.approx(2 / 3)
    assert result["scores"]["mapped_balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["scores"]["mapped_macro_f1"] == pytest.approx(0.8 / 3)
    assert result["scores"]["ari"] == 0
    assert result["scores"]["homogeneity"] == 0
    assert result["scores"]["completeness"] == 1
    assert result["occupancy"]["occupied_cluster_count"] == 1
    assert result["occupancy"]["unoccupied_cluster_count"] == 2
    assert result["occupancy"]["effective_cluster_count"] == 1
    assert result["occupancy"]["max_cluster_share"] == 1
    assert [x["count"] for x in result["occupancy"]["components"]] == [6, 0, 0]
    per_class = {x["class_label"]: x for x in result["per_class"]}
    assert per_class["major"]["tp"] == 4 and per_class["major"]["fp"] == 2
    assert per_class["minor-a"]["recall"] == 0
    assert not per_class["minor-a"]["has_assigned_cluster"]


def test_imbalanced_oversegmentation_is_not_purity():
    # Every cluster is pure, but several clusters belong to the same class.
    result = clustering_metrics(["a"] * 5 + ["b"], [10, 10, 20, 20, 30, 40])
    contingency = np.array(result["audit"]["contingency_counts"])
    purity = contingency.max(axis=0).sum() / contingency.sum()
    assert purity == 1
    assert result["scores"]["hungarian_acc"] == 0.5
    assert len(result["mapping"]["unmatched_predicted_clusters"]) == 2
    assert sum(x["tp"] for x in result["per_class"]) == 3
    assert sum(x["fn"] for x in result["per_class"]) == 3


def test_more_than_eight_classes_still_uses_one_to_one_accuracy():
    truth = np.repeat(np.arange(10) * 19, 5)
    pred = np.concatenate([np.array([2 * k] * 3 + [2 * k + 1] * 2) for k in range(10)])
    result = clustering_metrics(truth, pred)
    assert result["scores"]["hungarian_acc"] == pytest.approx(0.6)
    assert result["occupancy"]["occupied_cluster_count"] == 20
    assert len(result["mapping"]["unmatched_predicted_clusters"]) == 10


def test_fewer_clusters_than_classes_and_explicit_empty_components():
    result = clustering_metrics([1, 1, 2, 2, 3, 3], ["ab", "ab", "ab", "ab", "c", "c"], cluster_labels=["c", "unused", "ab"])
    assert result["scores"]["hungarian_acc"] == pytest.approx(2 / 3)
    assert result["scores"]["mapped_balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["scores"]["mapped_macro_f1"] == pytest.approx((2 / 3 + 1) / 3)
    assert len(result["mapping"]["unmatched_truth_classes"]) == 1
    assert [x["count"] for x in result["occupancy"]["components"]] == [2, 0, 4]
    assert result["audit"]["sum_cluster_counts"] == 6


def test_acc_ties_are_explicit_and_secondary_scores_use_displayed_map():
    truth = ["a"] * 7 + ["b"] * 5
    pred = [10] * 4 + [20] * 3 + [10] * 3 + [20] * 2
    result = clustering_metrics(truth, pred)
    assert result["mapping"]["has_alternative_acc_optimum"]
    assert result["mapping"]["secondary_scores_depend_on_displayed_mapping"]
    assert result["warnings"]
    assert result["scores"]["hungarian_acc"] == 0.5
    # Another ACC-optimal map has different F1/recall; there is no unique F1.
    assert any(result["scores"]["mapped_macro_f1"] == pytest.approx(value) for value in [0.5, (4 / 7 + 2 / 5) / 2])
    assert result["scores"]["mapped_macro_f1"] == pytest.approx(np.mean([x["f1"] for x in result["per_class"]]))


@pytest.mark.parametrize("matrix", [[[4, 3], [3, 2]], [[3, 1, 0], [0, 2, 2]], [[2, 0], [0, 3], [1, 1]], [[1]]])
def test_polynomial_tie_detection_matches_small_exhaustive_reference(matrix):
    matrix = np.array(matrix)
    truth, pred = [], []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            truth += [i] * int(matrix[i, j])
            pred += [j] * int(matrix[i, j])
    c, k = matrix.shape
    if c <= k:
        values = [sum(matrix[i, col] for i, col in enumerate(cols)) for cols in itertools.permutations(range(k), c)]
    else:
        values = [sum(matrix[row, j] for j, row in enumerate(rows)) for rows in itertools.permutations(range(c), k)]
    result = clustering_metrics(truth, pred)
    assert result["scores"]["hungarian_acc"] == pytest.approx(max(values) / matrix.sum())
    assert result["mapping"]["has_alternative_acc_optimum"] == (values.count(max(values)) > 1)


def test_arithmetic_information_scores_match_explicit_sklearn_definitions():
    truth = [0, 0, 0, 0, 1, 1, 2, 2]
    pred = [0, 0, 0, 1, 1, 2, 3, 3]
    result = clustering_metrics(truth, pred)["scores"]
    assert result["nmi_arithmetic"] == normalized_mutual_info_score(truth, pred, average_method="arithmetic")
    assert result["ami_arithmetic"] == adjusted_mutual_info_score(truth, pred, average_method="arithmetic")


def test_mixed_label_types_are_not_silently_string_coerced():
    result = clustering_metrics([1, 1, "1", "1"], ["alpha", "alpha", "beta", "beta"])
    assert result["n_true_classes"] == 2
    assert result["audit"]["contingency_rows_true_labels"] == [1, "1"]
    assert result["scores"]["hungarian_acc"] == 1


def test_numeric_label_audit_preserves_large_integers_and_exact_scalar_values():
    result = clustering_metrics([10**400, 10**400, np.longdouble(0.5)], ["a", "a", "b"])
    encoded = json.loads(json.dumps(result, allow_nan=False))
    assert encoded["audit"]["contingency_rows_true_labels"] == [10**400, 0.5]
    precise = np.longdouble("1.000000000000000001")
    if precise != float(precise):
        with pytest.raises(ValueError, match="JSON audit"):
            clustering_metrics([precise], [0])


def test_probability_diagnostics_preserve_zero_columns_without_class_calibration():
    result = clustering_metrics([0, 1], ["a", "b"], probabilities=[[0.8, 0.2, 0], [0.2, 0.8, 0]], cluster_labels=["a", "b", "empty"])
    panel = result["probabilities"]
    assert panel["mean_component_mass"] == [0.5, 0.5, 0]
    assert panel["effective_component_count"] == pytest.approx(2)
    assert panel["occupancy_entropy_nats"] == pytest.approx(np.log(2))
    assert result["occupancy"]["normalized_entropy"] == pytest.approx(np.log(2) / np.log(3))
    assert "class calibration" in panel["interpretation"]
    assert not any(key in panel for key in ("ece", "class_log_loss", "class_probabilities"))
    unaligned = clustering_metrics([0, 1], ["a", "b"], probabilities=[[1, 0], [0, 1]])
    assert unaligned["probabilities"]["column_labels"] is None
    assert "unspecified" in unaligned["probabilities"]["alignment_to_hard_labels"]


@pytest.mark.parametrize("truth,pred", [([], []), ([1], []), ([1, 2], [1]), ([[1]], [1]), ([np.nan], [1]), ([1], [np.inf]), ([None], [1]), ([complex(1, 2)], [1])])
def test_malformed_label_inputs_are_rejected(truth, pred):
    with pytest.raises(ValueError):
        clustering_metrics(truth, pred)


@pytest.mark.parametrize("kwargs", [
    {"expected_n_clusters": 1}, {"expected_n_clusters": 2.5}, {"expected_n_clusters": True},
    {"cluster_labels": ["a", "a"]}, {"cluster_labels": ["a", "missing"]},
    {"cluster_labels": ["a", "b"], "expected_n_clusters": 3},
    {"probabilities": [[0.3, 0.3], [0.5, 0.5]]},
    {"probabilities": [[1, 0]]}, {"probabilities": [[1], [1]]},
    {"probabilities": [[-0.1, 1.1], [0.5, 0.5]]},
    {"probabilities": [[np.nan, 0], [0, 1]]},
    {"probabilities": [["0.5", "0.5"], ["0.5", "0.5"]]},
    {"probability_atol": -1}, {"probability_atol": np.nan},
])
def test_malformed_component_and_probability_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        clustering_metrics([0, 1], ["a", "b"], **kwargs)


def test_probability_roundoff_policy_is_explicit():
    result = clustering_metrics([0, 1], [0, 1], probabilities=[[0.8, 0.2000001], [0, 1]])
    assert result["probabilities"]["max_original_row_sum_error"] > 0
    assert sum(result["probabilities"]["mean_component_mass"]) == pytest.approx(1)
    assert "normalized" in result["probabilities"]["normalization"]


def test_cli_only_reads_synthetic_fresh_predictions_and_writes_versioned_audit(tmp_path):
    predictions = tmp_path / "new_predictions.npz"
    np.savez(predictions, targets=np.array([11, 11, 99, 99]), predictions=np.array([7, 7, 3, 3]))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"fresh_run": True, "run_id": "synthetic-unit-test", "protocol_version": "test-v1", "source_sha256": "a" * 64, "config_sha256": "b" * 64}))
    output = tmp_path / "metrics.json"
    arguments = [str(predictions), "--metadata", str(metadata), "--output", str(output)]
    assert main(arguments) == 0
    record = json.loads(output.read_text())
    assert record["schema_version"] == SCHEMA_VERSION
    assert len(record["provenance"]["metric_implementation_sha256"]) == 64
    assert record["provenance"]["run_metadata"]["fresh_run"] is True
    assert record["scores"]["hungarian_acc"] == 1
    with pytest.raises(FileExistsError):
        main(arguments)
    metadata.write_text(json.dumps({"fresh_run": False}))
    with pytest.raises(ValueError, match="fresh_run"):
        main(arguments)


def test_cli_compatible_with_write_once_fresh_run_provenance(tmp_path):
    from experiments.paper1_review import metrics
    from experiments.paper1_review.provenance import finish_run, start_run

    directory = tmp_path / "fresh-synthetic-metric-integration"
    metadata = start_run(directory, stage="validation", config={"protocol_version": "synthetic-metric-integration-v1"}, sources=[Path(metrics.__file__)])
    predictions = directory / "predictions.npz"
    np.savez(predictions, sample_ids=np.arange(4), targets=np.array([0, 0, 1, 1]), predictions=np.array([11, 11, 77, 77]))
    output = directory / "metrics.json"
    assert main([str(predictions), "--metadata", str(directory / "run.json"), "--output", str(output)]) == 0
    record = json.loads(output.read_text())
    assert record["provenance"]["run_metadata"]["source_sha256"] == metadata["source_sha256"]
    assert record["provenance"]["run_metadata"]["config_sha256"] == metadata["config_sha256"]
    assert record["provenance"]["run_metadata"]["protocol_version"] == "synthetic-metric-integration-v1"
    finish_run(directory, metadata)
    finished = json.loads((directory / "run.json").read_text())
    assert finished["status"] == "completed"
    assert "metrics.json" in finished["artifact_sha256"]

