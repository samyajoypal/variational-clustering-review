"""Independent equations, optimizer continuation and fresh label-free fixtures."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paper1_review import reference_dec_idec as ref
from experiments.paper1_review.data import _ids_hash
from experiments.paper1_review.provenance import sha256, write_json


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_protocol():
    p = ref.protocol_for(4)
    p.update(architecture=[4, 6, 5, 4, 2], n_clusters=2, batch_size=7,
             checkpoint_every=1, kmeans_n_init=2, assignment_tolerance=0.,
             dec_refresh=2, idec_refresh=2, weight_init_std=.2,
             pretrain_step=2, momentum_burnin=2)
    return p


def fixture_data(root: Path):
    root.mkdir()
    manifest = {"schema_version": "paper1-review-dataset-v1", "status": "completed", "dataset": "mnist",
                "n_features": 4, "n_clusters": 2, "splits": {}, "artifact_sha256": {}}
    for split, count in [("train", 19), ("validation", 5), ("test", 4)]:
        ids = np.array([f"{split}:{i}" for i in range(count)])
        x = np.random.default_rng(count).uniform(.1, .9, (count, 4)).astype(np.float32)
        path = root / f"{split}.npz"; np.savez(path, x=x, ids=ids)
        manifest["splits"][split] = {"inputs": path.name, "labels": "LABELS_DO_NOT_EXIST.npz",
                                     "n_samples": count, "ids_sha256": _ids_hash(ids), "input_shape": list(x.shape)}
        manifest["artifact_sha256"][path.name] = sha256(path)
    write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def load_latest(directory):
    pointer = json.loads((directory / "latest_checkpoint.json").read_text())
    return torch.load(directory / pointer["file"], weights_only=True)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(x, y)
    else: assert a == b


def test_source_fork_euclidean_is_half_element_mean_and_gradient():
    y = torch.tensor([[1., -2., 3.], [2., 3., -4.]], dtype=torch.double, requires_grad=True)
    loss = ref.caffe_euclidean(y, torch.zeros_like(y))
    assert loss.item() == pytest.approx(np.square(y.detach().numpy()).sum() / (2 * 6))
    loss.backward()
    torch.testing.assert_close(y.grad, y.detach() / 6)
    assert loss != .5 * y.square().sum(1).mean()


def test_caffe_sgd_exact_history_burnin_bias_and_lr_drop():
    w, b = torch.nn.Parameter(torch.tensor([1.], dtype=torch.double)), torch.nn.Parameter(torch.tensor([1.], dtype=torch.double))
    opt = ref.CaffeSGD(ref.caffe_groups([("weight", w), ("bias", b)]), lr=.1, momentum=.9,
                       burnin=2, step_size=2, gamma=.1)
    expected_w = expected_b = 1.; hw = hb = 0.
    for t, gradient in enumerate([2., -1., 3., 4.]):
        w.grad = torch.tensor([gradient], dtype=torch.double); b.grad = w.grad.clone()
        momentum = .9 * min(t / 2., 1.)
        rate = .1 * .1 ** (t // 2)
        hw = momentum * hw + rate * gradient
        hb = momentum * hb + 2 * rate * gradient
        expected_w -= hw; expected_b -= hb; opt.step()
        assert w.item() == pytest.approx(expected_w, abs=1e-14)
        assert b.item() == pytest.approx(expected_b, abs=1e-14)
    assert opt.param_groups[0]["iteration"] == 4
    # At the first decay the old history is *not* multiplied by the new rate.
    assert hw == pytest.approx(.0589)
    restored_w = torch.nn.Parameter(w.detach().clone()); restored_b = torch.nn.Parameter(b.detach().clone())
    restored = ref.CaffeSGD(ref.caffe_groups([("weight", restored_w), ("bias", restored_b)]))
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for v in [w, b, restored_w, restored_b]: v.grad = torch.ones_like(v)
    opt.step(); restored.step()
    assert_tree_equal(opt.state_dict(), restored.state_dict())
    torch.testing.assert_close(w, restored_w, rtol=0, atol=0)


def test_layerwise_targets_frozen_and_dropout_placement(monkeypatch):
    ae = ref.StackedAutoencoder([4, 5, 6, 3, 2])
    calls = []
    original = ref.F.dropout
    def dropout(x, p, training):
        calls.append(p)
        return original(x, p, training)
    monkeypatch.setattr(ref.F, "dropout", dropout)
    x = torch.ones(7, 4)
    for index, expected in [(0, [0., .2]), (1, [.2, .2]), (3, [.2])]:
        ae.zero_grad(set_to_none=True); calls.clear()
        reconstruction, target = ae.layerwise(x, index)
        assert calls == expected and not target.requires_grad
        ref.caffe_euclidean(reconstruction, target).backward()
        assert ae.encoder[index].weight.grad is not None
        for other in range(4):
            if other != index:
                assert ae.encoder[other].weight.grad is None and ae.decoder[other].weight.grad is None
    calls.clear(); ae(x)
    assert calls == []  # Joint fine-tuning has no dropout.


def test_dec_kl_matches_source_closed_form_latent_and_center_gradient():
    ae = ref.StackedAutoencoder([2, 2]).double()
    with torch.no_grad(): ae.encoder[0].weight.copy_(torch.eye(2)); ae.encoder[0].bias.zero_()
    centers = torch.tensor([[-.3,.8], [.9,-.4], [1.2,.6]], dtype=torch.double)
    model = ref.ReferenceClustering(ae, centers)
    x = torch.tensor([[.2,.3], [.9,.1], [-.7,1.2]], dtype=torch.double, requires_grad=True)
    p = torch.tensor([[.7,.2,.1], [.2,.5,.3], [.8,.1,.1]], dtype=torch.double)
    q = model.log_assignments(x).exp().detach()
    difference = x.detach()[:,None,:] - centers[None,:,:]
    coefficient = 2 * (p-q) / (1 + difference.square().sum(-1))
    expected_x = (coefficient[:,:,None] * difference).sum(1) / len(x)
    expected_mu = -(coefficient[:,:,None] * difference).sum(0) / len(x)
    loss = ref.clustering_loss(model, x, p, "dec")["loss"]
    expected_value = (p.numpy() * np.log(p.numpy() / q.numpy())).sum() / len(x)
    assert loss.item() == pytest.approx(expected_value, abs=1e-14)
    loss.backward()
    torch.testing.assert_close(x.grad, expected_x, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.centers.grad, expected_mu, rtol=1e-12, atol=1e-12)


def test_target_frequency_and_detach_and_idec_loss():
    q = torch.tensor([[.8,.2,0.], [.6,.4,0.], [.3,.7,0.]], requires_grad=True)
    target = ref.target_distribution(q)
    raw = q.detach().numpy()[:,:2] ** 2 / q.detach().numpy()[:,:2].sum(0)
    np.testing.assert_allclose(target[:,:2], raw / raw.sum(1, keepdims=True), atol=1e-7)
    assert not target.requires_grad and torch.equal(target[:,2], torch.zeros(3))
    with pytest.raises(ValueError): ref.target_distribution(q * 2)
    with pytest.raises(ValueError): ref.target_distribution(q * float("nan"))
    ae = ref.StackedAutoencoder([2, 2]).double()
    model = ref.ReferenceClustering(ae, torch.tensor([[-1.,0.], [1.,0.]], dtype=torch.double))
    x = torch.tensor([[.2,.5], [.8,.1]], dtype=torch.double)
    p = torch.tensor([[.7,.3], [.2,.8]], dtype=torch.double)
    result = ref.clustering_loss(model, x, p, "idec")
    torch.testing.assert_close(result["mse"], (ae(x)-x).square().sum()/x.numel())
    torch.testing.assert_close(result["loss"], result["mse"] + .1 * result["kl"])


def test_sampler_exact_wrap_short_and_state_restore():
    sampler = ref.SequentialSampler(5, 3, wrap=True)
    assert sampler.next().tolist() == [0,1,2]
    state = sampler.state_dict()
    assert sampler.next().tolist() == [3,4,0]
    sampler.load_state_dict(state)
    assert sampler.next().tolist() == [3,4,0]
    short = ref.SequentialSampler(6, 3, wrap=False)
    assert [short.next().tolist() for _ in range(3)] == [[0,1,2],[3,4,5],[0,1,2]]
    short = ref.SequentialSampler(5, 3, wrap=False)
    assert [short.next().tolist() for _ in range(3)] == [[0,1,2],[3,4],[0,1,2]]


@pytest.mark.parametrize("phase", ["dec", "idec"])
def test_full_fresh_pipeline_absent_labels_and_shared_initialization(tmp_path, monkeypatch, phase):
    manifest = fixture_data(tmp_path / "data"); p = tiny_protocol()
    reads, original = [], ref.load_inputs
    def checked(path, split):
        reads.append(split); assert split != "test"
        return original(path, split)
    monkeypatch.setattr(ref, "load_inputs", checked)
    initial, output = tmp_path / "initial", tmp_path / phase
    result = ref.execute("initialize", manifest, initial, seed=901, device="cpu", validation_updates=4, protocol=p)
    assert result["optimizer_updates"] == 20 and reads == ["train"]
    result = ref.execute(phase, manifest, output, seed=901, initial=initial, device="cpu", validation_updates=4, protocol=p)
    record = json.loads((output / "run.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert record["status"] == "completed" and record["stage"] == "validation"
    assert not record["labels_loaded"] and not record["native_reproduction"]
    assert result["optimizer_updates"] == 4 and config["effective_seed"] == 100901
    assert result["predictions_written"] == ["train", "validation"]
    assert result["initialization_sha256"] == sha256(initial / "initialization.pt")
    state = load_latest(output)
    assert state["sampler"]["draws"] == (28 if phase == "dec" else 26)
    assert state["optimizer"]["param_groups"][0].get("iteration") == (2 if phase == "dec" else None)
    for split in ["train", "validation"]:
        with np.load(output / f"{split}_predictions.npz") as pred:
            assert set(pred.files) == {"sample_ids", "predictions", "responsibilities", "cluster_labels"}
            np.testing.assert_allclose(pred["responsibilities"].sum(1), 1, atol=1e-6)
    with pytest.raises(FileExistsError):
        ref.execute(phase, manifest, output, seed=901, initial=initial, device="cpu", validation_updates=4, protocol=p)
    with pytest.raises(ValueError, match="Initialization.*mismatch"):
        ref.execute(phase, manifest, tmp_path / "wrong_seed", seed=902, initial=initial, device="cpu", validation_updates=4, protocol=p)
    with pytest.raises(ValueError, match="validation only"):
        ref.execute("initialize", manifest, tmp_path / "fake_final", seed=901, device="cpu", protocol=p)


class InterruptedForTest(Exception): pass


@pytest.mark.parametrize("phase", ["initialize", "dec", "idec"])
def test_resume_equals_uninterrupted_including_rng_optimizer_cursor(tmp_path, monkeypatch, phase):
    manifest = fixture_data(tmp_path / "data"); p = tiny_protocol()
    initial = None
    if phase != "initialize":
        initial = tmp_path / "initial"
        ref.execute("initialize", manifest, initial, seed=901, device="cpu", validation_updates=4, protocol=p)
    kwargs = dict(seed=901, device="cpu", validation_updates=4, protocol=p, initial=initial)
    complete, interrupted, continued = tmp_path / "complete", tmp_path / "interrupted", tmp_path / "continued"
    ref.execute(phase, manifest, complete, **kwargs)
    original = ref.save_checkpoint
    def interrupt(directory, **values):
        result = original(directory, **values)
        if values["progress"]["updates"] == 1 and values["progress"]["phase_index"] == 0:
            raise InterruptedForTest()
        return result
    monkeypatch.setattr(ref, "save_checkpoint", interrupt)
    with pytest.raises(InterruptedForTest): ref.execute(phase, manifest, interrupted, **kwargs)
    monkeypatch.setattr(ref, "save_checkpoint", original)
    assert json.loads((interrupted / "run.json").read_text())["status"] == "failed"
    assert torch.load(interrupted / "failure_state.pt", weights_only=True)["resumable"] is False
    ref.execute(phase, manifest, continued, resume_from=interrupted / "latest_checkpoint.json", **kwargs)
    first, second = load_latest(complete), load_latest(continued)
    for key in ["model", "optimizer", "sampler", "progress", "rng", "logs"]:
        assert_tree_equal(first[key], second[key])
    if phase == "initialize":
        assert_tree_equal(torch.load(complete / "initialization.pt", weights_only=True),
                          torch.load(continued / "initialization.pt", weights_only=True))
    with pytest.raises(ValueError, match="Resume config/source/data/seed/phase mismatch"):
        ref.execute(phase, manifest, tmp_path / "bad_resume", resume_from=interrupted / "latest_checkpoint.json",
                    **dict(kwargs, seed=902))


def test_resume_at_pretrain_phase_boundary_and_checksum_rejection(tmp_path, monkeypatch):
    manifest = fixture_data(tmp_path / "data"); p = tiny_protocol()
    complete, interrupted, resumed = [tmp_path / s for s in ["complete", "interrupted", "resumed"]]
    kwargs = dict(seed=901, device="cpu", validation_updates=4, protocol=p)
    ref.execute("initialize", manifest, complete, **kwargs)
    original = ref.save_checkpoint
    def interrupt(directory, **values):
        result = original(directory, **values)
        if values["retain"] and values["progress"]["phase_index"] == 0: raise InterruptedForTest()
        return result
    monkeypatch.setattr(ref, "save_checkpoint", interrupt)
    with pytest.raises(InterruptedForTest): ref.execute("initialize", manifest, interrupted, **kwargs)
    monkeypatch.setattr(ref, "save_checkpoint", original)
    ref.execute("initialize", manifest, resumed, resume_from=interrupted / "latest_checkpoint.json", **kwargs)
    assert_tree_equal(load_latest(complete)["model"], load_latest(resumed)["model"])
    state = load_latest(complete)
    pointer = json.loads((interrupted / "latest_checkpoint.json").read_text())
    path = interrupted / pointer["file"]
    with path.open("ab") as stream: stream.write(b"tampering")
    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        ref.load_checkpoint(path, state["config"], state["source_digest"])


def test_checkpoint_retention_and_initialization_tampering(tmp_path):
    manifest = fixture_data(tmp_path / "data"); p = tiny_protocol()
    initial = tmp_path / "initial"
    ref.execute("initialize", manifest, initial, seed=901, device="cpu", validation_updates=4, protocol=p)
    sidecars = [json.loads(x.read_text()) for x in initial.glob("checkpoint_*.sha256.json")]
    assert sum(s["retain"] for s in sidecars) == 5
    assert sum(not s["retain"] for s in sidecars) <= 2
    with (initial / "initialization.pt").open("ab") as stream: stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash verified"):
        ref.execute("dec", manifest, tmp_path / "bad", seed=901, device="cpu", initial=initial,
                    validation_updates=4, protocol=p)


def test_scale_linear_output_and_source_label_free_stopping(tmp_path, monkeypatch):
    manifest = fixture_data(tmp_path / "data"); p = tiny_protocol()
    scaled, _ = ref._arrays(manifest, "train")
    original = ref.load_inputs(manifest, "train")["x"]
    np.testing.assert_allclose(scaled, original * (255/50), rtol=1e-7)
    ae = ref.StackedAutoencoder([4,2])
    with torch.no_grad():
        ae.decoder[0].weight.zero_(); ae.decoder[0].bias.fill_(-3)
    assert (ae(torch.ones(2,4)) == -3).all()
    initial = tmp_path / "initial"
    ref.execute("initialize", manifest, initial, seed=901, device="cpu", validation_updates=4, protocol=p)
    def all_zero(model, x, device, batch_size=1024):
        return torch.tensor([[1.,0.]]).expand(len(x), -1).clone()
    monkeypatch.setattr(ref, "responsibilities", all_zero)
    p["assignment_tolerance"] = .001
    # The initializer's protocol must match; stopping is a real config parameter.
    initial = tmp_path / "initial_stopping"
    ref.execute("initialize", manifest, initial, seed=901, device="cpu", validation_updates=4, protocol=p)
    for phase, expected in [("dec", 0), ("idec", 2)]:
        result = ref.execute(phase, manifest, tmp_path / phase, seed=901, device="cpu", initial=initial,
                             validation_updates=4, protocol=p)
        assert result["optimizer_updates"] == expected
        assert result["stop_reason"] == "assignment_stability"


def test_checkpoint_binding_rejects_source_and_manifest_changes(tmp_path):
    manifest = fixture_data(tmp_path / "data"); p = tiny_protocol()
    output = tmp_path / "initial"
    ref.execute("initialize", manifest, output, seed=901, device="cpu", validation_updates=4, protocol=p)
    state = load_latest(output)
    for key, value in [("manifest_sha256", "changed-data"), ("phase", "idec"), ("effective_seed", 100902)]:
        config = dict(state["config"], **{key: value})
        with pytest.raises(ValueError, match="Resume config/source/data/seed/phase mismatch"):
            ref.load_checkpoint(output / "latest_checkpoint.json", config, state["source_digest"])
    with pytest.raises(ValueError, match="Resume config/source/data/seed/phase mismatch"):
        ref.load_checkpoint(output / "latest_checkpoint.json", state["config"], "changed-source")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("phase", ["dec", "idec"])
def test_cuda_source_equations_and_optimizer_step(phase):
    ref.setup(100901, "cuda")
    ae = ref.StackedAutoencoder([4, 6, 5, 2]).cuda()
    model = ref.ReferenceClustering(ae, torch.randn(3,2, device="cuda"))
    x = torch.rand(8,4,device="cuda")
    target = ref.target_distribution(model.log_assignments(x).exp())
    result = ref.clustering_loss(model, x, target, phase)
    norm = ref._finite_step(result["loss"], model, ref._cluster_optimizer(model, phase, ref.PROTOCOL))
    assert np.isfinite(norm)
