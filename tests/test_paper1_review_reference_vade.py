"""Independent equations, provenance gates and resume checks; synthetic inputs only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paper1_review import reference_vade as v
from experiments.paper1_review.provenance import finish_run, sha256, start_run


@pytest.fixture(autouse=True, scope="module")
def small_thread_count():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_bound_and_responsibilities_match_independent_distributions():
    dtype = torch.float64
    mean = torch.tensor([[.1, -.3], [.8, .4]], dtype=dtype)
    lv = torch.tensor([[-.5, .2], [.3, -.6]], dtype=dtype)
    z = mean + .4
    loc = torch.tensor([[-.5, .2], [1., -.1]], dtype=dtype)
    clv = torch.tensor([[.2, -.4], [-.3, .4]], dtype=dtype)
    logits = torch.tensor([-.3, .7], dtype=dtype)
    result = v.gaussian_categorical_bound(mean, lv, z, loc, clv, logits)
    components = torch.distributions.Independent(torch.distributions.Normal(loc, (clv/2).exp()), 1)
    q = torch.distributions.Independent(torch.distributions.Normal(mean[:, None], (lv[:, None]/2).exp()), 1)
    expected_gamma = (components.log_prob(z[:, None]) + logits.log_softmax(-1)).softmax(-1)
    component_kl = torch.distributions.kl_divergence(q, components)
    cat_kl = torch.distributions.kl_divergence(torch.distributions.Categorical(probs=expected_gamma),
                                             torch.distributions.Categorical(logits=logits))
    torch.testing.assert_close(result["responsibilities"], expected_gamma)
    torch.testing.assert_close(result["regularizer"], (expected_gamma*component_kl).sum(-1)+cat_kl)


@pytest.mark.parametrize("dimension", [1, 3, 10])
def test_mixture_weight_counted_once_and_identical_gaussian_zero_gap(dimension):
    mean = torch.zeros(2, dimension, dtype=torch.float64)
    logits = torch.tensor([.2, .8], dtype=torch.float64).log()
    result = v.gaussian_categorical_bound(mean, mean, mean+1., mean, mean, logits)
    torch.testing.assert_close(result["responsibilities"], torch.tensor([[.2,.8],[.2,.8]], dtype=torch.float64))
    torch.testing.assert_close(result["regularizer"], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)


def test_regularizer_has_correct_gradients_including_sampled_responsibilities():
    torch.manual_seed(314)
    arguments = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)*.4
                 for shape in [(2,2), (2,2), (2,2), (3,2), (3,2), (3,)]]
    function = lambda *args: v.gaussian_categorical_bound(*args)["regularizer"].sum()
    assert torch.autograd.gradcheck(function, tuple(arguments), eps=1e-6, atol=2e-5, rtol=2e-4)
    gradient_z = torch.autograd.grad(function(*arguments), arguments[2])[0]
    assert torch.linalg.vector_norm(gradient_z) > 1e-4  # a detached gamma fails this check


def test_bernoulli_reconstruction_is_feature_summed_and_stable():
    model = v.ReferenceVaDE(input_dim=4, hidden_dims=(6,), latent_dim=2, n_clusters=2).double()
    x = torch.tensor([[0.,1.,.2,.7]], dtype=torch.float64)
    mean, _ = model.encode(x)
    logits = model.decode_logits(mean)
    expected = torch.distributions.Bernoulli(logits=logits, validate_args=False).log_prob(x).sum(-1).neg().mean()
    torch.testing.assert_close(model.pretraining_loss(x), expected)
    variance_before = copy.deepcopy(model.log_variance_head.state_dict())
    optimizer = torch.optim.Adam(model.pretrain_parameters(), lr=.001)
    model.pretraining_loss(x).backward(); optimizer.step()
    for key, tensor in variance_before.items():
        torch.testing.assert_close(model.log_variance_head.state_dict()[key], tensor, rtol=0, atol=0)


def test_architecture_optimizer_partitions_schedule_and_uniform_initial_pi():
    model = v.ReferenceVaDE()
    assert [(m.in_features,m.out_features) for m in model.encoder if isinstance(m,torch.nn.Linear)] == [(784,500),(500,500),(500,2000)]
    assert model.mean_head.out_features == 10 and model.decoder_logits.out_features == 784
    model.set_initial_mixture(np.ones((10,10)), np.full((10,10), .7))
    torch.testing.assert_close(model.mixture_logits.softmax(-1), torch.full((10,), .1))
    network, mixture = v.make_optimizers(model)
    nids = {id(p) for group in network.param_groups for p in group["params"]}
    mids = {id(p) for group in mixture.param_groups for p in group["params"]}
    assert not nids & mids and nids | mids == {id(p) for p in model.parameters()}
    for optimizer in (network, mixture):
        assert optimizer.defaults["eps"] == 1e-4 and optimizer.defaults["lr"] == .002
        assert optimizer.defaults["weight_decay"] == 0
    assert v.learning_rate(0) == v.learning_rate(9) == .002
    assert v.learning_rate(10) == pytest.approx(.0018)
    assert v.learning_rate(219) > .0002
    assert v.learning_rate(220) == v.learning_rate(2999) == .0002


def test_source_pin_and_seed_gate(tmp_path, monkeypatch):
    assert v.verify_source_pin()["VaDE.py_sha256"] == v.SOURCE_SHA256
    source = tmp_path / "vendor"; source.mkdir()
    manifest = json.loads((v.SOURCE_DIRECTORY/"SOURCE_MANIFEST.json").read_text())
    (source/"SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    (source/"VaDE.py").write_text("modified")
    monkeypatch.setattr(v, "SOURCE_DIRECTORY", source)
    with pytest.raises(ValueError, match="source changed"):
        v.verify_source_pin()


def _configuration(tmp_path, phase="reference_training", updates=120):
    manifest = tmp_path/"manifest.json"
    manifest.write_text("{}")
    return v.configuration(manifest, 901, True, phase, updates)


def test_validation_cannot_use_final_seed_or_modified_final_budget(tmp_path):
    config = _configuration(tmp_path)
    assert config["seed"] == 100901 and config["base_seed"] == 901
    with pytest.raises(ValueError, match="approved base seeds"):
        v.configuration(tmp_path/"manifest.json", 123, False, "reference_training", 20)
    with pytest.raises(ValueError, match="restricted"):
        v.configuration(tmp_path/"manifest.json", 901, False, "reference_training", 5)


def _same_state(a,b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: _same_state(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): _same_state(x,y)
    else: assert a == b


@pytest.mark.parametrize("pretrain", [False, True])
def test_epoch_checkpoint_resume_is_bitwise_equivalent(tmp_path, monkeypatch, pretrain):
    monkeypatch.setattr(v, "BATCH_SIZE", 5)
    monkeypatch.setattr(v, "TRAIN_EPOCHS", 12)
    monkeypatch.setattr(v, "PRETRAIN_EPOCHS", 12)
    config = _configuration(tmp_path, "fresh_initialization" if pretrain else "reference_training")
    x = torch.rand(20,4, generator=torch.Generator().manual_seed(89))
    torch.manual_seed(17)
    base = v.ReferenceVaDE(4,(7,),2,2)
    uninterrupted, interrupted, continued = [copy.deepcopy(base) for _ in range(3)]
    out_full, out_first, out_resumed = [tmp_path/name for name in ("full","first","resumed")]
    for out in (out_full,out_first,out_resumed):
        start_run(out,stage="validation",config=config,sources=v.source_files())
    torch.manual_seed(1234); np.random.seed(1234)
    full_curves, full_updates = v._fit_epochs(uninterrupted,x,out_full,config,torch.device("cpu"),pretrain=pretrain)
    full_rng = torch.get_rng_state().clone()
    save = v._atomic_checkpoint
    def stop_after_checkpoint(path, payload):
        save(path,payload)
        if payload.get("epoch") == 10:
            raise RuntimeError("fixture preemption after durable epoch10")
    monkeypatch.setattr(v,"_atomic_checkpoint",stop_after_checkpoint)
    torch.manual_seed(1234); np.random.seed(1234)
    with pytest.raises(RuntimeError,match="fixture preemption"):
        v._fit_epochs(interrupted,x,out_first,config,torch.device("cpu"),pretrain=pretrain)
    monkeypatch.setattr(v,"_atomic_checkpoint",save)
    state = v._load_resume(out_first/"last.pt",config,torch.device("cpu"),out_resumed)
    resumed_curves, resumed_updates = v._fit_epochs(continued,x,out_resumed,config,torch.device("cpu"),pretrain=pretrain,resume_state=state)
    assert full_updates == resumed_updates == 48
    assert full_curves == resumed_curves
    _same_state(uninterrupted.state_dict(),continued.state_dict())
    _same_state(full_rng,torch.get_rng_state())
    full_saved = torch.load(out_full/"last.pt",weights_only=True)
    resumed_saved = torch.load(out_resumed/"last.pt",weights_only=True)
    _same_state(full_saved["optimizers"],resumed_saved["optimizers"])
    # A seed/config change and a corrupted checkpoint are not continuations.
    with pytest.raises(ValueError,match="configuration mismatch"):
        v._load_resume(out_first/"last.pt",dict(config,seed=100902),torch.device("cpu"),out_resumed)
    with (out_first/"last.pt").open("ab") as stream: stream.write(b"tamper")
    with pytest.raises(ValueError,match="hash, source, or runtime mismatch"):
        v._load_resume(out_first/"last.pt",config,torch.device("cpu"),out_resumed)


def test_prediction_rng_is_private_and_reproducible():
    model = v.ReferenceVaDE(4,(6,),2,2)
    model.set_initial_mixture([[-.5,0.],[.5,0.]],[[.4,.6],[.4,.6]])
    x = torch.rand(8,4)
    before = torch.get_rng_state().clone()
    first = v.sampled_predictions(model,x,torch.device("cpu"),51)
    assert torch.equal(before,torch.get_rng_state())
    np.testing.assert_array_equal(first,v.sampled_predictions(model,x,torch.device("cpu"),51))
    assert not np.array_equal(first,v.sampled_predictions(model,x,torch.device("cpu"),52))


def _fixture_manifest(root):
    root.mkdir()
    entries, hashes = {}, {}
    counts = {"train":32,"validation":5,"test":7}
    rng = np.random.RandomState(32)
    for split,count in counts.items():
        path = root/f"{split}.npz"
        x = rng.uniform(0,1,(count,1,28,28)).astype(np.float32)
        ids = np.asarray([f"fixture:{split}:{i}" for i in range(count)])
        np.savez(path,x=x,ids=ids)
        entries[split] = {"inputs":path.name,"n_samples":count,"input_shape":list(x.shape),
                          "ids_sha256":hashlib.sha256(json.dumps(ids.tolist(),separators=(",",":")).encode()).hexdigest()}
        hashes[path.name] = sha256(path)
    manifest = {"schema_version":"paper1-review-dataset-v1","protocol_version":"paper1-review-inductive-data-v1",
                "dataset":"mnist","status":"completed","n_features":784,"n_clusters":10,
                "sample_counts":counts,"splits":entries,"artifact_sha256":hashes}
    path = root/"manifest.json"; path.write_text(json.dumps(manifest))
    return path,counts


def test_synthetic_initialize_train_no_labels_no_test_and_write_once(tmp_path, monkeypatch):
    manifest,counts = _fixture_manifest(tmp_path/"data")
    monkeypatch.setattr(v,"MNIST_COUNTS",counts)
    cls = v.ReferenceVaDE
    monkeypatch.setattr(v,"ReferenceVaDE",lambda: cls(hidden_dims=(12,)))
    real_loader = v.load_inputs
    loaded = []
    def monitored(path,split):
        assert split != "test"
        loaded.append(split)
        return real_loader(path,split)
    monkeypatch.setattr(v,"load_inputs",monitored)
    initial, final = tmp_path/"initial",tmp_path/"trained"
    v.initialize(manifest,initial,901,device="cpu",validation=True,validation_updates=2)
    v.train(manifest,initial,final,901,device="cpu",validation=True,validation_updates=2)
    info = json.loads((final/"run.json").read_text())
    assert info["status"] == "completed" and info["stage"] == "validation"
    assert info["optimizer_updates"] == 2 and info["labels_loaded"] is False
    assert info["test_partition_loaded"] is False and loaded == ["train","train","validation"]
    with np.load(final/"validation_predictions.npz") as data:
        assert set(data.files) == {"ids","responsibilities","predictions"}
        assert len(data["ids"]) == 5
    with pytest.raises(FileExistsError):
        v.train(manifest,initial,final,901,device="cpu",validation=True,validation_updates=2)
    failure = tmp_path/"wrong-seed"
    with pytest.raises(ValueError,match="Initialization mismatch"):
        v.train(manifest,initial,failure,902,device="cpu",validation=True,validation_updates=2)
    failed = json.loads((failure/"run.json").read_text())
    assert failed["status"] == "failed" and failed["numerical_gate_passed"] is False
    assert not (failure/"final.pt").exists()
