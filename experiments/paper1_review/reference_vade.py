"""VaDE paper-equation port with a declared fresh inductive MNIST protocol.

This is not a literal legacy-code reproduction: mixture weights enter once,
mixture parameters are constrained by log/softmax parameterizations, and fresh
AE pretraining is explicitly specified. No evaluation labels are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture

from .data import load_inputs
from .provenance import ROOT, finish_run, sha256, start_run, write_json

PROTOCOL = "paper1-review-vade-reference-inductive-v1"
SOURCE_COMMIT = "eca52b4b57e32a68f578de7a54eda5d39907b0fe"
SOURCE_SHA256 = "2ff36c3bd4775489f357f179c9990fcbbabddd32374bbec6c2c7d578529a623f"
SOURCE_DIRECTORY = Path(__file__).parent / "vendor_sources/vade_eca52b4b"
MNIST_COUNTS = {"train": 54000, "validation": 6000, "test": 10000}
FINAL_SEEDS = (901, 902, 903)
VALIDATION_SEED_OFFSET = 100000
PRETRAIN_EPOCHS = 100
TRAIN_EPOCHS = 3000
BATCH_SIZE = 100
VALIDATION_UPDATES = 20
VALIDATION_TRAIN_LIMIT = 4096
VALIDATION_INPUT_LIMIT = 256
GMM_SETTINGS = dict(covariance_type="diag", n_init=1, tol=1e-3, max_iter=100,
                    reg_covar=1e-3, init_params="kmeans")


def verify_source_pin() -> dict:
    manifest = json.loads((SOURCE_DIRECTORY / "SOURCE_MANIFEST.json").read_text())
    if manifest.get("commit") != SOURCE_COMMIT:
        raise ValueError("Unexpected VaDE source commit")
    actual = sha256(SOURCE_DIRECTORY / "VaDE.py")
    listed = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    if actual != SOURCE_SHA256 or listed.get("VaDE.py") != SOURCE_SHA256:
        raise ValueError("Pinned VaDE source changed")
    return {"commit": SOURCE_COMMIT, "VaDE.py_sha256": actual,
            "source_manifest_sha256": sha256(SOURCE_DIRECTORY / "SOURCE_MANIFEST.json"),
            "url": f"https://github.com/slim1017/VaDE/blob/{SOURCE_COMMIT}/VaDE.py"}


def source_files() -> list[Path]:
    return [Path(__file__), Path(__file__).with_name("data.py"),
            SOURCE_DIRECTORY / "VaDE.py", SOURCE_DIRECTORY / "SOURCE_MANIFEST.json"]


def gaussian_component_log_prob(z: Tensor, loc: Tensor, log_variance: Tensor) -> Tensor:
    """Diagonal Normal log densities, with no mixture weights included."""
    return -.5 * (math.log(2 * math.pi) + log_variance[None]
                  + (z[:, None] - loc[None]).square() * (-log_variance[None]).exp()).sum(-1)


def log_responsibilities(z: Tensor, loc: Tensor, log_variance: Tensor,
                         logits: Tensor) -> Tensor:
    # π is counted once, correcting the legacy source's dimension-wise sum.
    log_joint = gaussian_component_log_prob(z, loc, log_variance) + logits.log_softmax(-1)
    return log_joint.log_softmax(-1)


def gaussian_categorical_bound(mean: Tensor, log_variance: Tensor, z: Tensor,
                               loc: Tensor, component_log_variance: Tensor,
                               logits: Tensor) -> dict[str, Tensor]:
    """Per-observation Gaussian-plus-categorical variational regularizer.

    Responsibilities use a differentiable sampled z, as in the VaDE recipe.
    The component Gaussian expectations are analytic. This is not the
    collapsed Gaussian-mixture KL, and no exact marginal KL is claimed.
    """
    log_gamma = log_responsibilities(z, loc, component_log_variance, logits)
    gamma = log_gamma.exp()
    component_kl = .5 * (component_log_variance[None] - log_variance[:, None] - 1
                        + (log_variance[:, None].exp() + (mean[:, None] - loc[None]).square())
                        * (-component_log_variance[None]).exp()).sum(-1)
    gaussian = (gamma * component_kl).sum(-1)
    categorical = (gamma * (log_gamma - logits.log_softmax(-1))).sum(-1)
    return {"regularizer": gaussian + categorical, "gaussian": gaussian,
            "categorical": categorical, "responsibilities": gamma}


def _hidden_network(widths) -> nn.Sequential:
    layers = []
    for before, after in zip(widths[:-1], widths[1:]):
        layers.extend([nn.Linear(before, after), nn.ReLU()])
    return nn.Sequential(*layers)


class ReferenceVaDE(nn.Module):
    """Source-shaped MLP with a diagonal Gaussian recognition distribution."""

    def __init__(self, input_dim=784, hidden_dims=(500, 500, 2000), latent_dim=10, n_clusters=10):
        super().__init__()
        if not hidden_dims:
            raise ValueError("At least one hidden layer is required")
        self.input_dim, self.latent_dim, self.n_clusters = input_dim, latent_dim, n_clusters
        self.hidden_dims = tuple(hidden_dims)
        self.encoder = _hidden_network((input_dim, *hidden_dims))
        self.mean_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_variance_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder_hidden = _hidden_network((latent_dim, *reversed(hidden_dims)))
        self.decoder_logits = nn.Linear(hidden_dims[0], input_dim)
        self.loc = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.component_log_variance = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.mixture_logits = nn.Parameter(torch.zeros(n_clusters))
        # Keras Dense defaults: Glorot uniform and zero biases. The untrained
        # variance head stays independent of AE pretraining, as in the source.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x)
        return self.mean_head(hidden), self.log_variance_head(hidden)

    def decode_logits(self, z: Tensor) -> Tensor:
        return self.decoder_logits(self.decoder_hidden(z))

    def mixture_parameters(self) -> list[nn.Parameter]:
        return [self.loc, self.component_log_variance, self.mixture_logits]

    def network_parameters(self) -> list[nn.Parameter]:
        mixture_ids = {id(p) for p in self.mixture_parameters()}
        return [p for p in self.parameters() if id(p) not in mixture_ids]

    def pretrain_parameters(self) -> list[nn.Parameter]:
        unused_ids = {id(p) for p in self.log_variance_head.parameters()}
        return [p for p in self.network_parameters() if id(p) not in unused_ids]

    def pretraining_loss(self, x: Tensor) -> Tensor:
        mean, _ = self.encode(x)
        return F.binary_cross_entropy_with_logits(self.decode_logits(mean), x, reduction="none").sum(-1).mean()

    def objective(self, x: Tensor, *, epsilon: Tensor | None = None) -> dict[str, Tensor]:
        mean, logvar = self.encode(x)
        epsilon = torch.randn_like(mean) if epsilon is None else epsilon
        z = mean + (.5 * logvar).exp() * epsilon
        reconstruction = F.binary_cross_entropy_with_logits(self.decode_logits(z), x,
                                                            reduction="none").sum(-1)
        panel = gaussian_categorical_bound(mean, logvar, z, self.loc,
                                           self.component_log_variance, self.mixture_logits)
        return {"loss": (reconstruction + panel["regularizer"]).mean(),
                "reconstruction": reconstruction.mean(), "regularizer": panel["regularizer"].mean(),
                "gaussian": panel["gaussian"].mean(), "categorical": panel["categorical"].mean()}

    @torch.no_grad()
    def set_initial_mixture(self, loc, covariance) -> None:
        loc = torch.as_tensor(loc, dtype=self.loc.dtype, device=self.loc.device)
        covariance = torch.as_tensor(covariance, dtype=self.loc.dtype, device=self.loc.device)
        if loc.shape != self.loc.shape or covariance.shape != self.loc.shape:
            raise ValueError("GMM shape mismatch")
        if not torch.isfinite(loc).all() or not torch.isfinite(covariance).all() or not (covariance > 0).all():
            raise ValueError("Invalid initial Gaussian mixture")
        self.loc.copy_(loc)
        self.component_log_variance.copy_(covariance.log())
        self.mixture_logits.zero_()  # The source does not copy fitted GMM weights.


def learning_rate(epoch_index: int) -> float:
    if epoch_index < 0:
        raise ValueError("Epoch index must be nonnegative")
    return max(.002 * .9 ** (epoch_index // 10), .0002)


def make_optimizers(model: ReferenceVaDE) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    options = dict(lr=.002, betas=(.9, .999), eps=1e-4, weight_decay=0.)
    return (torch.optim.Adam(model.network_parameters(), **options),
            torch.optim.Adam(model.mixture_parameters(), **options))


def _configure(seed: int, device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device unavailable; no CPU fallback")
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device(device)


def _input_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if (manifest.get("dataset") != "mnist" or manifest.get("status") != "completed"
            or manifest.get("n_features") != 784 or manifest.get("n_clusters") != 10
            or manifest.get("sample_counts") != MNIST_COUNTS
            or manifest.get("protocol_version") != "paper1-review-inductive-data-v1"):
        raise ValueError("The completed 54k/6k/10k inductive MNIST artifact is required")
    return manifest


def _inputs(path: Path, split: str, *, limit: int | None = None) -> tuple[Tensor, np.ndarray]:
    data = load_inputs(path, split)
    x = data["x"].reshape(len(data["x"]), -1)
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("MNIST must be finite /255 pixels in [0,1]; do not rescale again")
    return torch.from_numpy(x[:limit].copy()), data["ids"][:limit]


def _id_hash(ids: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":")).encode()).hexdigest()


def configuration(manifest_path: Path, seed: int, validation: bool, phase: str,
                  validation_updates: int) -> dict:
    if seed not in FINAL_SEEDS:
        raise ValueError("Use one of the approved base seeds 901, 902, 903")
    if validation_updates < 1 or (not validation and validation_updates != VALIDATION_UPDATES):
        raise ValueError("Update overrides are restricted to validation runs")
    if validation and validation_updates > 1000:
        raise ValueError("Validation is capped at 1000 updates and cannot masquerade as a final fit")
    effective_seed = seed + VALIDATION_SEED_OFFSET if validation else seed
    return {"protocol_version": PROTOCOL, "method": "vade", "phase": phase,
            "base_seed": seed, "seed": effective_seed, "validation": validation,
            "validation_update_limit": validation_updates if validation else None,
            "data_manifest": str(manifest_path.resolve()), "data_manifest_sha256": sha256(manifest_path),
            "source_pin": verify_source_pin(), "selection": "fixed endpoint; no outcome-based selection",
            "architecture": {"input": 784, "hidden": [500, 500, 2000], "latent": 10, "clusters": 10},
            "batch_size": BATCH_SIZE, "pretrain_epochs": PRETRAIN_EPOCHS, "training_epochs": TRAIN_EPOCHS,
            "ae_optimizer": {"name": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8},
            "ae_loss": "batch mean of feature-summed BCE; deterministic encoder mean",
            "posterior_variance_initialization": "independent Glorot head retained through AE pretraining",
            "training_optimizers": {"name": "separate network and mixture Adam", "lr": .002,
                                    "betas": [.9, .999], "eps": 1e-4, "weight_decay": 0.,
                                    "decay": ".9 at zero-based epochs10,20,...; floor.0002"},
            "gmm": {**GMM_SETTINGS, "random_state": effective_seed, "initial_mixture_weights": "uniform"},
            "training_mc_samples": 1, "gradient_clipping": None,
            "objective": "feature-summed Bernoulli NLL + analytic Gaussian/categorical regularizer; differentiable sampled-z responsibilities",
            "prediction": "one Gaussian posterior sample per observation; p(c|z) with pi once; fixed endpoint RNG",
            "prediction_seed": effective_seed + 50000,
            "labels_loaded": False, "test_loaded_during_optimization": False,
            "published_score_reproduction": False, "historical_weights_reused": False,
            "adaptations": ["54k/6k/10k inductive split instead of70k transductive fitting",
                            "fresh100-epoch AE schedule frozen because source pretraining is underdocumented",
                            "mixture pi counted once instead of source pi^latent_dim",
                            "positive component variances via exp; normalized weights via softmax",
                            "dimension-only Gaussian constants canceled algebraically",
                            "stable BCE-with-logits; PyTorch rather than Keras1/Theano Adam",
                            "explicit historical-GMM-like settings and source-like uniform initial mixture weights"]}


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    try:
        torch.save(payload, temporary)
        details = {"sha256": sha256(temporary), "port_sha256": sha256(Path(__file__)),
                   "data_module_sha256": sha256(Path(__file__).with_name("data.py")),
                   "provenance_module_sha256": sha256(Path(__file__).with_name("provenance.py")),
                   "torch_version": str(torch.__version__)}
        write_json(temporary_sidecar, details)
        # Preserve a previous valid pair if publication is interrupted between
        # the new checkpoint and its sidecar. At least one pair stays intact.
        if path.exists() and sidecar.exists():
            old_details = json.loads(sidecar.read_text())
            if sha256(path) == old_details.get("sha256"):
                previous = path.with_name(path.stem + "_previous" + path.suffix)
                for source, target in ((path, previous), (sidecar, previous.with_suffix(previous.suffix + ".sha256.json"))):
                    link = target.with_name(target.name + ".tmp")
                    link.unlink(missing_ok=True)
                    os.link(source, link)
                    os.replace(link, target)
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)


def _checkpoint(model, optimizers, config, epoch, updates, generator, curves) -> dict:
    numpy_state = np.random.get_state()
    return {"protocol_version": PROTOCOL, "config": config, "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers], "epoch": epoch, "updates": updates,
            "device_type": next(model.parameters()).device.type,
            "sampler_rng_state": generator.get_state(), "curves": curves,
            "checkpoint_boundary": "epoch boundary or completed validation update budget",
            "numpy_rng_state": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                                "pos": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict, generator: torch.Generator) -> None:
    generator.set_state(state["sampler_rng_state"].cpu())
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if state["cuda_rng_states"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA checkpoint requires CUDA for an equivalent continuation")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
    n = state["numpy_rng_state"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["pos"], n["has_gauss"], n["cached_gaussian"]))


def _load_resume(path: Path, config: dict, device: torch.device, output: Path) -> dict:
    if path.resolve().is_relative_to(output.resolve()):
        raise ValueError("Resume into a new attempt directory, not the checkpoint's own directory")
    details = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    expected = {"sha256": sha256(path), "port_sha256": sha256(Path(__file__)),
                "data_module_sha256": sha256(Path(__file__).with_name("data.py")),
                "provenance_module_sha256": sha256(Path(__file__).with_name("provenance.py")),
                "torch_version": str(torch.__version__)}
    if details != expected:
        raise ValueError("Resume checkpoint hash, source, or runtime mismatch")
    parent_run = json.loads((path.parent / "run.json").read_text())
    parent_config = json.loads((path.parent / "config.json").read_text())
    if not parent_run.get("fresh_run") or parent_run.get("protocol_version") != PROTOCOL:
        raise ValueError("Resume requires a fresh checkpoint from this campaign")
    if parent_run.get("stage") != ("validation" if config["validation"] else "final_benchmark"):
        raise ValueError("Resume stage mismatch")
    if parent_run["config_sha256"] != sha256(path.parent / "config.json"):
        raise ValueError("Parent run configuration changed")
    if parent_config != config:
        raise ValueError("Resume configuration mismatch: phase, seed, data, initialization or budget changed")
    state = torch.load(path, map_location=device, weights_only=True)
    required = {"optimizers", "epoch", "updates", "sampler_rng_state", "torch_rng_state", "cuda_rng_states", "numpy_rng_state", "curves"}
    if not required.issubset(state) or state.get("config") != config or state.get("protocol_version") != PROTOCOL:
        raise ValueError("Incomplete or incompatible resume checkpoint")
    if state.get("device_type") != device.type:
        raise ValueError("Resume device type changed; this is not an equivalent continuation")
    write_json(output / "resume_dependency.json", {"path": str(path.resolve()), "sha256": details["sha256"],
                                                   "parent_run_sha256": sha256(path.parent / "run.json"),
                                                   "completed_epochs": state["epoch"], "optimizer_updates": state["updates"],
                                                   "reuse_kind": "continuation of the same fresh campaign, not legacy weights"})
    return state


def _finite_parameters(model) -> bool:
    return bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all())


def _fit_epochs(model: ReferenceVaDE, x: Tensor, output: Path, config: dict,
                device: torch.device, *, pretrain: bool, resume_state: dict | None = None) -> tuple[list, int]:
    optimizers = ([torch.optim.Adam(model.pretrain_parameters(), lr=.001, eps=1e-8)] if pretrain
                  else list(make_optimizers(model)))
    epochs = PRETRAIN_EPOCHS if pretrain else TRAIN_EPOCHS
    cap = config["validation_update_limit"]
    if cap is not None:
        epochs = min(epochs, math.ceil(cap / math.ceil(len(x) / BATCH_SIZE)))
    generator = torch.Generator().manual_seed(config["seed"] + (10 if pretrain else 20))
    updates = 0
    curves = []
    start_epoch = 0
    if resume_state is not None:
        if len(resume_state["optimizers"]) != len(optimizers):
            raise ValueError("Resume optimizer partition mismatch")
        model.load_state_dict(resume_state["model"])
        for optimizer, saved in zip(optimizers, resume_state["optimizers"]):
            optimizer.load_state_dict(saved)
        start_epoch, updates, curves = resume_state["epoch"], resume_state["updates"], list(resume_state["curves"])
        if not 0 <= start_epoch <= epochs or (cap is not None and updates > cap):
            raise ValueError("Checkpoint exceeds the frozen training budget")
        expected_lr = .001 if pretrain else learning_rate(max(0, start_epoch-1))
        if any(group["lr"] != expected_lr for optimizer in optimizers for group in optimizer.param_groups):
            raise ValueError("Resume optimizer learning rate does not match its completed epoch")
        _restore_rng(resume_state, generator)
    model.train()
    for epoch in range(start_epoch, epochs):
        if cap is not None and updates >= cap:
            break
        if not pretrain:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(epoch)
        totals, observations = {}, 0
        for indices in torch.randperm(len(x), generator=generator).split(BATCH_SIZE):
            batch = x[indices].to(device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            panel = {"loss": model.pretraining_loss(batch)} if pretrain else model.objective(batch)
            if not torch.isfinite(panel["loss"]):
                raise FloatingPointError(f"Nonfinite objective at update{updates}")
            panel["loss"].backward()
            gradient_checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not bool(torch.stack(gradient_checks).all()):
                raise FloatingPointError(f"Nonfinite gradient at update{updates}")
            for optimizer in optimizers:
                optimizer.step()
            if not _finite_parameters(model):
                raise FloatingPointError(f"Nonfinite parameter at update{updates}")
            for name, value in panel.items():
                totals[name] = totals.get(name, 0.) + float(value.detach()) * len(batch)
            observations += len(batch)
            updates += 1
            if cap is not None and updates >= cap:
                break
        row = {"epoch": epoch + 1, "optimizer_updates": updates, "observations_this_epoch": observations,
               "learning_rate": optimizers[0].param_groups[0]["lr"],
               **{name: value / observations for name, value in totals.items()}}
        curves.append(row)
        write_json(output / "training_curves.json", curves)
        write_json(output / "progress.json", row)
        print(json.dumps(row), flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs or (cap is not None and updates >= cap):
            _atomic_checkpoint(output / "last.pt", _checkpoint(model, optimizers, config, epoch + 1, updates, generator, curves))
        if cap is not None and updates >= cap:
            break
    return curves, updates


@torch.no_grad()
def _embeddings(model, x, device) -> np.ndarray:
    model.eval()
    return torch.cat([model.encode(batch.to(device))[0].cpu() for batch in x.split(512)]).numpy()


def initialize(manifest_path: Path, output: Path, seed: int, *, device="cuda", validation=False,
               validation_updates=VALIDATION_UPDATES, resume_from: Path | None = None) -> None:
    config = configuration(manifest_path, seed, validation, "fresh_initialization", validation_updates)
    config["device_type"] = torch.device(device).type
    record = start_run(output, stage="validation" if validation else "final_benchmark",
                       config=config, sources=source_files())
    started = time.perf_counter()
    try:
        _input_manifest(manifest_path)
        target_device = _configure(config["seed"], device)
        x, ids = _inputs(manifest_path, "train", limit=VALIDATION_TRAIN_LIMIT if validation else None)
        model = ReferenceVaDE().to(target_device)
        resumed = _load_resume(resume_from, config, target_device, output) if resume_from is not None else None
        curves, updates = _fit_epochs(model, x, output, config, target_device, pretrain=True, resume_state=resumed)
        fitted = GaussianMixture(n_components=model.n_clusters, random_state=config["seed"],
                                 **GMM_SETTINGS).fit(_embeddings(model, x, target_device).astype(np.float64))
        if not fitted.converged_:
            raise RuntimeError("Training-input GMM did not converge; initialization is not accepted")
        model.set_initial_mixture(fitted.means_, fitted.covariances_)
        _atomic_checkpoint(output / "initial.pt", {"protocol_version": PROTOCOL, "config": config,
                                                   "model": model.state_dict(), "phase": "fresh_initialization"})
        np.savez_compressed(output / "gmm_initialization.npz", loc=fitted.means_, covariance=fitted.covariances_,
                            fitted_weights=fitted.weights_, initialized_weights=np.full(model.n_clusters, 1/model.n_clusters))
        finish_run(output, record, train_seconds=time.perf_counter()-started, optimizer_updates=updates,
                   epochs_completed=len(curves), train_count=len(x), training_ids_sha256=_id_hash(ids),
                   gmm_converged=True, gmm_iterations=int(fitted.n_iter_), numerical_gate_passed=True)
    except Exception as error:
        finish_run(output, record, status="failed", error=repr(error), numerical_gate_passed=False)
        raise


def _load_initial(initial: Path, config: dict, device: torch.device) -> dict:
    metadata = json.loads((initial / "run.json").read_text())
    initial_config = json.loads((initial / "config.json").read_text())
    if metadata.get("status") != "completed" or not metadata.get("fresh_run"):
        raise ValueError("A completed fresh VaDE initialization is required")
    if metadata.get("stage") != ("validation" if config["validation"] else "final_benchmark"):
        raise ValueError("Validation and final initializations cannot be mixed")
    for key in ("protocol_version", "seed", "base_seed", "data_manifest_sha256", "validation", "source_pin"):
        if initial_config.get(key) != config[key]:
            raise ValueError(f"Initialization mismatch: {key}")
    if initial_config.get("phase") != "fresh_initialization":
        raise ValueError("Initialization phase mismatch")
    for filename in ("config.json", "initial.pt"):
        if sha256(initial / filename) != metadata["artifact_sha256"].get(filename):
            raise ValueError(f"Initialization hash mismatch: {filename}")
    # Reject initializations produced by a changed version of this port.
    source_relative = str(Path(__file__).resolve().relative_to(ROOT))
    if metadata["source_file_sha256"].get(source_relative) != sha256(Path(__file__)):
        raise ValueError("Initialization was produced by a different port version")
    state = torch.load(initial / "initial.pt", map_location=device, weights_only=True)
    if state.get("protocol_version") != PROTOCOL or state.get("config") != initial_config:
        raise ValueError("Initialization checkpoint provenance mismatch")
    return state


@torch.no_grad()
def sampled_predictions(model: ReferenceVaDE, x: Tensor, device: torch.device, seed: int) -> np.ndarray:
    """Source-style single-draw native assignment; no GMM refit or labels."""
    model.eval()
    # Generate epsilon on CPU with a private RNG. Batching does not change it
    # and prediction does not consume the training/global RNG streams.
    generator = torch.Generator().manual_seed(seed)
    epsilon = torch.randn(len(x), model.latent_dim, generator=generator)
    probabilities = []
    for start in range(0, len(x), 512):
        mean, logvar = model.encode(x[start:start+512].to(device))
        z = mean + (.5*logvar).exp() * epsilon[start:start+512].to(device)
        probabilities.append(log_responsibilities(z, model.loc, model.component_log_variance,
                                                    model.mixture_logits).exp().cpu())
    result = torch.cat(probabilities).numpy()
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1., atol=1e-5):
        raise FloatingPointError("Invalid native prediction probabilities")
    return result


def train(manifest_path: Path, initial: Path, output: Path, seed: int, *, device="cuda", validation=False,
          validation_updates=VALIDATION_UPDATES, resume_from: Path | None = None) -> None:
    config = configuration(manifest_path, seed, validation, "reference_training", validation_updates)
    config["device_type"] = torch.device(device).type
    config["initial_run"] = str(initial.resolve())
    record = start_run(output, stage="validation" if validation else "final_benchmark",
                       config=config, sources=source_files())
    started = time.perf_counter()
    try:
        _input_manifest(manifest_path)
        target_device = _configure(config["seed"], device)
        state = _load_initial(initial, config, target_device)
        # Commit the dependency only after validation; this file is covered by
        # finish_run's artifact hashes along with all saved predictions.
        write_json(output / "initial_dependency.json", {"run_sha256": sha256(initial / "run.json"),
                                                         "checkpoint_sha256": sha256(initial / "initial.pt")})
        x, ids = _inputs(manifest_path, "train", limit=VALIDATION_TRAIN_LIMIT if validation else None)
        model = ReferenceVaDE().to(target_device)
        model.load_state_dict(state["model"])
        torch.manual_seed(config["seed"] + 100)
        resumed = _load_resume(resume_from, config, target_device, output) if resume_from is not None else None
        curves, updates = _fit_epochs(model, x, output, config, target_device, pretrain=False, resume_state=resumed)
        # Freeze the endpoint before loading any test inputs. Labels are never
        # loaded in this process, including after fitting.
        _atomic_checkpoint(output / "final.pt", {"protocol_version": PROTOCOL, "config": config,
                                                 "model": model.state_dict(), "optimizer_updates": updates,
                                                 "selection": "fixed final endpoint"})
        prediction_counts = {}
        for split_index, split in enumerate(("validation",) if validation else ("validation", "test")):
            xe, ids_e = _inputs(manifest_path, split, limit=VALIDATION_INPUT_LIMIT if validation else None)
            resp = sampled_predictions(model, xe, target_device, config["prediction_seed"] + split_index)
            np.savez_compressed(output / f"{split}_predictions.npz", ids=ids_e, responsibilities=resp,
                                predictions=resp.argmax(1))
            prediction_counts[split] = len(xe)
        finish_run(output, record, train_seconds=time.perf_counter()-started, optimizer_updates=updates,
                   epochs_completed=len(curves), train_count=len(x), training_ids_sha256=_id_hash(ids),
                   prediction_counts=prediction_counts, numerical_gate_passed=True,
                   test_partition_loaded=not validation, labels_loaded=False)
    except Exception as error:
        finish_run(output, record, status="failed", error=repr(error), numerical_gate_passed=False)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["initialize", "train"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=FINAL_SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--resume-from", type=Path, help="Same-campaign checkpoint; --output must be a new attempt directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--validation-updates", type=int, default=VALIDATION_UPDATES)
    args = parser.parse_args()
    if args.phase == "train" and args.initial is None:
        parser.error("train requires --initial pointing to this recipe's completed fresh initialization")
    if args.phase == "initialize" and args.initial is not None:
        parser.error("initialize cannot load old weights")
    kwargs = dict(device=args.device, validation=args.validation, validation_updates=args.validation_updates,
                  resume_from=args.resume_from)
    if args.phase == "initialize":
        initialize(args.manifest, args.output, args.seed, **kwargs)
    else:
        train(args.manifest, args.initial, args.output, args.seed, **kwargs)


if __name__ == "__main__":
    main()
