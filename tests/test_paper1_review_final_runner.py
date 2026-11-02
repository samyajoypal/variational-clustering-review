"""Fresh synthetic fixtures only; label artifacts do not exist."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paper1_review import final_runner as runner
from experiments.paper1_review.final_protocol import PROTOCOL, update_budget
from experiments.paper1_review.final_diagnostics import gradient_diagnostic, objective_terms
from experiments.paper1_review.data import _ids_hash
from experiments.paper1_review.models import dec_target_distribution
from experiments.paper1_review.provenance import sha256, write_json
from experiments.paper1_review.workload import METHODS


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_protocol():
    p = copy.deepcopy(PROTOCOL)
    p.update(batch_size=8, pretrain_epochs=2, training_epochs=2,
             diagnostic_epochs=[0, 1, 2], checkpoint_epochs=[0, 2],
             rolling_checkpoint_every_epochs=1, training_samples=2, prediction_samples=4,
             common_gradient_batch_size=4)
    p["architecture"].update(hidden_dims=[8], latent_dim=2)
    p["gmm"]["n_init"] = 2
    return p


def fixture_dataset(root, dataset="mnist"):
    root.mkdir(parents=True)
    manifest = {"schema_version": "paper1-review-dataset-v1", "status": "completed", "dataset": dataset,
                "n_features": 4, "n_clusters": 2, "splits": {}, "artifact_sha256": {}}
    for split, count in [("train", 30), ("validation", 8), ("test", 6)]:
        ids = np.array([f"{dataset}:{split}:{i}" for i in range(count)])
        x = np.random.default_rng(count).uniform(.05, .95, (count, 4)).astype(np.float32)
        if dataset == "reuters10k": x *= 8
        np.savez(root / f"{split}.npz", x=x, ids=ids)
        manifest["splits"][split] = {"inputs": f"{split}.npz", "labels": "ABSENT_LABELS.npz", "n_samples": count,
                                     "ids_sha256": _ids_hash(ids), "input_shape": list(x.shape)}
        manifest["artifact_sha256"][f"{split}.npz"] = sha256(root / f"{split}.npz")
    write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


@pytest.mark.parametrize("method", METHODS)
def test_full_matched_paths_without_labels_or_test_input_during_fit(tmp_path, monkeypatch, method):
    p = tiny_protocol()
    manifest = fixture_dataset(tmp_path / "data")
    original = runner.load_inputs
    reads = []
    def checked(path, split):
        reads.append(split)
        return original(path, split)
    monkeypatch.setattr(runner, "load_inputs", checked)
    initial, output = tmp_path / "initial", tmp_path / "fit"
    runner.initialize(manifest, initial, seed=101, device="cpu", protocol=p, validation_only=True)
    runner.train(manifest, initial, output, method=method, seed=101, device="cpu", protocol=p, validation_only=True)
    assert "test" not in reads
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "completed" and record["fixed_endpoint"] and not record["labels_loaded"]
    assert record["optimizer_updates"] == 8 and record["stage"] == "validation"
    batch_lengths = np.load(output / "batch_lengths.npy")
    order = np.fromfile(output / "sample_order.int32", dtype="<i4")
    assert len(batch_lengths) == 8 and batch_lengths.sum() == len(order) == 60
    with np.load(output / "sample_exposure.npz") as exposure:
        np.testing.assert_array_equal(exposure["count"], np.bincount(order, minlength=30))
    diagnostics = json.loads((output / "gradient_diagnostics.json").read_text())
    assert [row["clean_equivalent_epoch"] for row in diagnostics] == [0, 1, 2]
    assert all(not row["optimizer_update"] for row in diagnostics)
    if method not in ["dec_matched", "idec_matched"]:
        assert set(diagnostics[-1]["components"]) == {"reconstruction", "prior_cross_entropy", "negative_entropy"}
    else:
        assert "target_kl" in diagnostics[-1]["components"]
    checkpoint = torch.load(output / "latest_checkpoint.pt", weights_only=True)
    assert checkpoint["completed_updates"] == 8 and checkpoint["source_sha256"] == record["source_sha256"]
    if method == "vae_gmm": assert record["final_gmm"]["converged"]
    runner.predict(manifest, output, tmp_path / "prediction", device="cpu")
    assert "test" in reads
    with np.load(tmp_path / "prediction/predictions.npz") as pred:
        assert set(pred.files) == {"sample_ids", "predictions", "responsibilities", "cluster_labels"}
        np.testing.assert_allclose(pred["responsibilities"].sum(1), 1, atol=1e-5)
    with pytest.raises(FileExistsError):
        runner.train(manifest, initial, output, method=method, seed=101, device="cpu", protocol=p, validation_only=True)


@pytest.mark.parametrize("method", METHODS)
def test_diagnostic_objective_identity_and_no_rng_or_gradient_side_effects(method):
    p = tiny_protocol()
    model = runner.model_for(method, {"dataset": "mnist", "n_features": 4, "n_clusters": 2}, torch.device("cpu"), p).double()
    model.set_cluster_parameters([[-.6, .2], [.8, -.4]], [[.8, 1.2], [1., .7]], [.4, .6])
    x = torch.tensor([[.1, .3, .7, .4], [.6, .2, .5, .1]], dtype=torch.float64)
    target = dec_target_distribution(model.predict(x)) if model.is_dec else None
    torch.manual_seed(721)
    terms, scope = objective_terms(model, x, target)
    expected = sum(terms.values())
    torch.manual_seed(721)
    actual = model.training_loss(x, target)["loss"]
    torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-10)
    before = torch.get_rng_state().clone()
    diagnostic = gradient_diagnostic(model, x, seed=314, target=target)
    assert torch.equal(before, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(np.isfinite(row["preclip_norm"]) for row in diagnostic["components"].values())


def test_reuters_decoder_linear_and_exact_update_budget():
    p = tiny_protocol()
    model = runner.model_for("vae_gmm", {"dataset": "reuters10k", "n_features": 4, "n_clusters": 2}, torch.device("cpu"), p)
    assert isinstance(list(model.decoder.children())[-1], torch.nn.Linear)
    assert update_budget(54000, "pretrain") == 21100
    assert update_budget(54000, "training") == 42200
    assert PROTOCOL["structured"]["training_nodes"] == 128


def test_contamination_contract_shuffled_ids_and_equal_budget(tmp_path):
    from experiments.paper1_review.contamination import prepare, CONDITIONS
    clean = fixture_dataset(tmp_path / "data")
    view_path = prepare(clean, tmp_path / "view", condition=CONDITIONS[0], seed=101)
    view = runner.load_training_view(clean, view_path)
    assert view["clean_count"] == 30 and len(view["train"]) >= 30
    p = tiny_protocol()
    initial = tmp_path / "initial"
    runner.initialize(clean, initial, training_view=view_path, seed=101, device="cpu", protocol=p, validation_only=True)
    output = tmp_path / "fit"
    runner.train(clean, initial, output, method="vae_gmm", training_view=view_path, seed=101, device="cpu", protocol=p, validation_only=True)
    assert json.loads((output / "run.json").read_text())["optimizer_updates"] == update_budget(30, "training", p)
    with pytest.raises(ValueError, match="mismatch"):
        runner.verify_initialization(initial, runner.load_training_view(clean), 101, p, True)


def test_seed_and_development_initialization_rejected(tmp_path):
    p = tiny_protocol(); manifest = fixture_dataset(tmp_path / "data")
    initial = tmp_path / "initial"
    runner.initialize(manifest, initial, seed=101, device="cpu", protocol=p, validation_only=True)
    with pytest.raises(ValueError, match="mismatch"):
        runner.train(manifest, initial, tmp_path / "bad", method="vae_gmm", seed=102, device="cpu", protocol=p, validation_only=True)
    config = json.loads((initial / "config.json").read_text()); config["phase"] = "development"
    write_json(initial / "config.json", config)
    with pytest.raises(ValueError, match="development weights"):
        runner.verify_initialization(initial, runner.load_training_view(manifest), 101, p, True)


def test_numerical_rejection_preserved_and_no_test_read(tmp_path, monkeypatch):
    p = tiny_protocol(); manifest = fixture_dataset(tmp_path / "data")
    initial = tmp_path / "initial"
    runner.initialize(manifest, initial, seed=101, device="cpu", protocol=p, validation_only=True)
    monkeypatch.setattr(runner, "quadrature_check", lambda *args: {"passed": False})
    with pytest.raises(ArithmeticError, match="quadrature gate"):
        runner.train(manifest, initial, tmp_path / "failed", method="structured_q_student_prior", seed=101, device="cpu", protocol=p, validation_only=True)
    record = json.loads((tmp_path / "failed/run.json").read_text())
    assert record["status"] == "failed" and record["numerical_gate_passed"] is False
    assert (tmp_path / "failed/latest_checkpoint.pt").exists()


def test_custom_budget_cannot_be_presented_as_final(tmp_path):
    manifest = fixture_dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="altered settings are validation"):
        runner.initialize(manifest, tmp_path / "bad", seed=101, device="cpu", protocol=tiny_protocol())


@pytest.mark.parametrize("condition", runner.INFERENCE_CONDITIONS)
def test_corruptions_pair_across_methods_and_input_row_permutations(condition):
    x = torch.linspace(.1, .9, 24).reshape(6, 4)
    ids = np.array(["z", "b", "a", "d", "x", "c"])
    permutation = np.array([3, 0, 4, 2, 5, 1])
    original = runner.corrupt_inputs(x, ids, dataset="mnist", seed=101, condition=condition)
    reordered = runner.corrupt_inputs(x[permutation], ids[permutation], dataset="mnist", seed=101, condition=condition)
    torch.testing.assert_close(reordered, original[permutation], atol=0, rtol=0)
    assert (original >= 0).all() and (original <= 1).all()
    with pytest.raises(ValueError, match="unapproved"):
        runner.corrupt_inputs(x, ids, dataset="reuters10k", seed=101, condition=condition)
