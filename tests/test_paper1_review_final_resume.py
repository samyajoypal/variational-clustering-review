"""Exact continuation on new unlabeled fixtures, including CUDA when available."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments.paper1_review import final_runner as runner
from experiments.paper1_review.final_protocol import PROTOCOL
from experiments.paper1_review.provenance import sha256, write_json
from experiments.paper1_review.data import _ids_hash


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def protocol():
    p = copy.deepcopy(PROTOCOL)
    p.update(batch_size=8, pretrain_epochs=2, training_epochs=2,
             diagnostic_epochs=[0, 1, 2], checkpoint_epochs=[0, 1, 2],
             rolling_checkpoint_every_epochs=1, training_samples=2, prediction_samples=4,
             common_gradient_batch_size=4)
    p["architecture"].update(hidden_dims=[8], latent_dim=2)
    p["gmm"]["n_init"] = 2
    return p


def dataset(root):
    root.mkdir()
    info = {"schema_version": "paper1-review-dataset-v1", "status": "completed", "dataset": "mnist",
            "n_features": 4, "n_clusters": 2, "splits": {}, "artifact_sha256": {}}
    for split, size in [("train", 30), ("validation", 8)]:
        x = np.random.default_rng(size).uniform(.05, .95, (size, 4)).astype(np.float32)
        ids = np.array([f"mnist:{split}:{i}" for i in range(size)])
        name = split + ".npz"; np.savez(root / name, x=x, ids=ids)
        info["splits"][split] = {"inputs": name, "labels": "ABSENT_LABELS", "n_samples": size,
                                 "ids_sha256": _ids_hash(ids), "input_shape": list(x.shape)}
        info["artifact_sha256"][name] = sha256(root / name)
    write_json(root / "manifest.json", info)
    return root / "manifest.json"


def equal(left, right):
    if torch.is_tensor(left): assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): equal(a, b)
    else: assert left == right


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items()
                if k not in {"seconds", "elapsed_seconds", "timings"}}
    if isinstance(value, list): return [without_timing(v) for v in value]
    return value


def compare_endpoints(a, b):
    left = torch.load(a / "latest_checkpoint.pt", weights_only=True, map_location="cpu")
    right = torch.load(b / "latest_checkpoint.pt", weights_only=True, map_location="cpu")
    for key in ["model", "optimizer", "sampler", "torch_rng", "cuda_rng", "completed_updates", "phase"]:
        equal(left[key], right[key])
    equal(without_timing(left["progress"]), without_timing(right["progress"]))
    assert (a / "sample_order.int32").read_bytes() == (b / "sample_order.int32").read_bytes()
    np.testing.assert_array_equal(np.load(a / "batch_lengths.npy"), np.load(b / "batch_lengths.npy"))
    with np.load(a / "sample_exposure.npz") as x, np.load(b / "sample_exposure.npz") as y:
        for key in x.files: np.testing.assert_array_equal(x[key], y[key])
    record = json.loads((b / "run.json").read_text())
    assert record["status"] == "completed" and not record["labels_loaded"]
    assert (b / "resume_dependency.json").is_file()
    assert not (b / "checkpoints/epoch999.pt").exists()


def interrupt_at(monkeypatch, output, *, phase="epoch_boundary", update=4, uncommitted_tail=True):
    original = runner._checkpoint
    def interrupted(path, *args, **kwargs):
        original(path, *args, **kwargs)
        # Positional args: model, optimizer, sampler, update, record.
        if path.parent == output and args[3] == update and kwargs.get("phase", "epoch_boundary") == phase:
            if uncommitted_tail:
                kwargs["order_file"].write(np.array([123456], dtype="<i4").tobytes())
                (output / "checkpoints").mkdir(exist_ok=True)
                (output / "checkpoints/epoch999.pt").write_bytes(b"uncommitted future artifact")
            raise InterruptedError("synthetic preemption after committed boundary")
    monkeypatch.setattr(runner, "_checkpoint", interrupted)
    return original


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
def test_pretraining_resume_exact_with_partial_order_tail(tmp_path, monkeypatch, device):
    manifest = dataset(tmp_path / "data"); p = protocol()
    args = dict(seed=101, device=device, protocol=p, validation_only=True)
    uninterrupted, interrupted, resumed = [tmp_path / name for name in ["full", "interrupted", "resumed"]]
    runner.initialize(manifest, uninterrupted, **args)
    original = interrupt_at(monkeypatch, interrupted)
    with pytest.raises(InterruptedError): runner.initialize(manifest, interrupted, **args)
    monkeypatch.setattr(runner, "_checkpoint", original)
    runner.initialize(manifest, resumed, resume_from=interrupted / "latest_checkpoint.pt", **args)
    compare_endpoints(uninterrupted, resumed)
    equal(torch.load(uninterrupted / "autoencoder.pt", weights_only=True, map_location="cpu"),
          torch.load(resumed / "autoencoder.pt", weights_only=True, map_location="cpu"))
    with np.load(uninterrupted / "mixture.npz") as a, np.load(resumed / "mixture.npz") as b:
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    # Neither test inputs nor labels exist; fitting still succeeds.
    assert not (manifest.parent / "test.npz").exists()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["dec_matched", "gaussian_q_student_prior", "student_q_student_prior", "structured_q_student_prior"])
def test_training_resume_exact_for_dec_refresh_and_stochastic_families(tmp_path, monkeypatch, method, device):
    from experiments.paper1_review.contamination import prepare, CONDITIONS
    manifest = dataset(tmp_path / "data"); p = protocol()
    # For the approved structured contamination row a clean-equivalent epoch
    # ends inside a permutation; other methods retain their approved clean view.
    view = (prepare(manifest, tmp_path / "contaminated", condition=CONDITIONS[0], seed=101)
            if method == "structured_q_student_prior" else None)
    args = dict(seed=101, device=device, protocol=p, validation_only=True, training_view=view)
    initial = tmp_path / "initial"; runner.initialize(manifest, initial, **args)
    args["method"] = method
    uninterrupted, interrupted, resumed = [tmp_path / name for name in ["full", "interrupted", "resumed"]]
    runner.train(manifest, initial, uninterrupted, **args)
    original = interrupt_at(monkeypatch, interrupted)
    with pytest.raises(InterruptedError): runner.train(manifest, initial, interrupted, **args)
    monkeypatch.setattr(runner, "_checkpoint", original)
    before = sha256(interrupted / "run.json")
    runner.train(manifest, initial, resumed, resume_from=interrupted / "latest_checkpoint.pt", **args)
    assert sha256(interrupted / "run.json") == before
    compare_endpoints(uninterrupted, resumed)
    equal(torch.load(uninterrupted / "model.pt", weights_only=True, map_location="cpu"),
          torch.load(resumed / "model.pt", weights_only=True, map_location="cpu"))
    for name in ["training_curves.json", "gradient_diagnostics.json", "numerical_checks.json", "occupancy_diagnostics.json"]:
        equal(without_timing(json.loads((uninterrupted / name).read_text())),
              without_timing(json.loads((resumed / name).read_text())))
    for split in ["train", "validation"]:
        with np.load(uninterrupted / f"{split}_predictions.npz") as a, np.load(resumed / f"{split}_predictions.npz") as b:
            for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    assert [r["clean_equivalent_epoch"] for r in json.loads((resumed / "gradient_diagnostics.json").read_text())] == [0, 1, 2]
    assert (resumed / "checkpoints/epoch001.pt").is_file()
    with pytest.raises(FileExistsError): runner.train(manifest, initial, resumed, **args)


@pytest.mark.parametrize("failure_stage", ["pre_gmm", "endpoint_prediction"])
def test_endpoint_resume_refits_vae_gmm_from_pre_refit_state_only(tmp_path, monkeypatch, failure_stage):
    manifest = dataset(tmp_path / "data"); p = protocol()
    args = dict(seed=101, device="cpu", protocol=p, validation_only=True)
    initial = tmp_path / "initial"; runner.initialize(manifest, initial, **args)
    args["method"] = "vae_gmm"
    uninterrupted, interrupted, resumed = [tmp_path / name for name in ["full", "interrupted", "resumed"]]
    runner.train(manifest, initial, uninterrupted, **args)
    if failure_stage == "pre_gmm":
        original = interrupt_at(monkeypatch, interrupted, phase="optimization_complete", update=8)
        with pytest.raises(InterruptedError): runner.train(manifest, initial, interrupted, **args)
        assert not (interrupted / "model.pt").exists()
        monkeypatch.setattr(runner, "_checkpoint", original)
    else:
        original = runner._save_predictions
        def fail_write(*a, **kw): raise OSError("synthetic endpoint storage failure after GMM")
        monkeypatch.setattr(runner, "_save_predictions", fail_write)
        with pytest.raises(OSError): runner.train(manifest, initial, interrupted, **args)
        assert (interrupted / "model.pt").exists()
        monkeypatch.setattr(runner, "_save_predictions", original)
    fit = runner.fit_gmm; calls = []
    def counted(*a, **kw): calls.append(True); return fit(*a, **kw)
    monkeypatch.setattr(runner, "fit_gmm", counted)
    runner.train(manifest, initial, resumed, resume_from=interrupted / "latest_checkpoint.pt", **args)
    assert len(calls) == 1
    compare_endpoints(uninterrupted, resumed)
    equal(torch.load(uninterrupted / "model.pt", weights_only=True), torch.load(resumed / "model.pt", weights_only=True))


def test_resume_rejects_corrupt_incompatible_completed_and_numerically_failed_states(tmp_path, monkeypatch):
    manifest = dataset(tmp_path / "data"); p = protocol()
    args = dict(seed=101, device="cpu", protocol=p, validation_only=True)
    initial = tmp_path / "initial"; runner.initialize(manifest, initial, **args)
    completed = tmp_path / "completed"
    runner.train(manifest, initial, completed, method="dec_matched", **args)
    with pytest.raises(ValueError, match="Completed logical"):
        runner.train(manifest, initial, tmp_path / "bad_complete", method="dec_matched",
                     resume_from=completed / "latest_checkpoint.pt", **args)
    interrupted = tmp_path / "interrupted"; original = interrupt_at(monkeypatch, interrupted)
    with pytest.raises(InterruptedError): runner.train(manifest, initial, interrupted, method="dec_matched", **args)
    monkeypatch.setattr(runner, "_checkpoint", original)
    checkpoint = interrupted / "latest_checkpoint.pt"
    with pytest.raises(ValueError, match="mismatch"):
        runner.train(manifest, initial, tmp_path / "bad_method", method="idec_matched", resume_from=checkpoint, **args)
    # Hash protects the exact committed order, while uncommitted trailing bytes
    # are intentionally ignored (positive equivalence checks above).
    order_path = interrupted / "sample_order.int32"; order = order_path.read_bytes()
    order_path.write_bytes(bytes([order[0] ^ 1]) + order[1:])
    with pytest.raises(ValueError, match="prefix hash"):
        runner.train(manifest, initial, tmp_path / "bad_order", method="dec_matched", resume_from=checkpoint, **args)
    order_path.write_bytes(order)
    original_bytes = checkpoint.read_bytes(); checkpoint.write_bytes(original_bytes + b"corrupt")
    with pytest.raises(ValueError, match="checkpoint hash"):
        runner.train(manifest, initial, tmp_path / "bad_hash", method="dec_matched", resume_from=checkpoint, **args)
    checkpoint.write_bytes(original_bytes)
    monkeypatch.setattr(runner, "quadrature_check", lambda *a: {"passed": False})
    failed = tmp_path / "numerical_failed"
    with pytest.raises(ArithmeticError): runner.train(manifest, initial, failed, method="structured_q_student_prior", **args)
    state = torch.load(failed / "latest_checkpoint.pt", weights_only=True)
    assert not state["resumable"] and state["phase"] == "numerical_gate_failed"
    with pytest.raises(ValueError, match="Numerically rejected"):
        runner.train(manifest, initial, tmp_path / "bad_numerical", method="structured_q_student_prior",
                     resume_from=failed / "latest_checkpoint.pt", **args)
