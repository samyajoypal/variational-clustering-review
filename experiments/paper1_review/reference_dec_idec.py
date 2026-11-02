"""Fresh inductive DEC/IDEC reference-recipe adaptations, not score reproductions.

The pinned DEC fork's Euclidean loss is 0.5 * element-mean squared error.
Its SGD history contains learning-rate-scaled gradients and is reset whenever
DEC starts a new target-refresh solver. These details differ from torch SGD.
Only verified input artifacts are read; labels are never loaded. Production
uses the approved 54k/6k/10k split and seeds 901--903. Short validation is a
separate stage and seed stream. Resume is explicit, into a new attempt folder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

from .data import load_inputs
from .provenance import ROOT, sha256, start_run, finish_run, write_json


PROTOCOL = {
    "protocol_version": "paper1-review-dec-idec-reference-v1",
    "architecture": [784, 500, 500, 2000, 10],
    "n_clusters": 10, "batch_size": 256,
    "layer_updates": 50000, "joint_updates": 100000,
    "layer_dropout": .2, "weight_init_std": .01,
    "pretrain_lr": .1, "pretrain_step": 20000, "pretrain_gamma": .1,
    "momentum": .9, "momentum_burnin": 1000, "bias_lr_multiplier": 2.,
    "dec_lr": .01, "dec_refresh": 160, "idec_refresh": 140,
    "cluster_max_updates": 20000, "assignment_tolerance": .001,
    "idec_lr": .001, "idec_betas": [.9, .999], "idec_epsilon": 1e-7,
    "idec_kl_weight": .1,
    "kmeans_n_init": 20, "kmeans_max_iter": 300, "kmeans_tol": 1e-4,
    "checkpoint_every": 1000,
    "source_pins": {"dec": "e551e01f90a3d67d2ef9c90e968c8faf5d8f857d",
                    "idec": "4a661c211f082022249758ab3312f7db21e2aa99"},
    "input_transform": "float32 prepared [0,1] pixels * (255/50)",
    "pretrain_loss": "0.5 * mean over observations and features of squared error",
    "dec_loss": "mean over observations of sum over clusters p*log(p/q); no extra lambda",
    "idec_loss": "feature-and-observation mean squared error + 0.1 * mean KL",
    "pretrain_sampler": "sequential fixed-full-batch with wrap; restart at each phase",
    "dec_sampler": "sequential fixed-full-batch with wrap; cursor survives target refresh",
    "idec_sampler": "sequential with short last batch; continuous Adam",
    "dec_optimizer": "Caffe LR-scaled momentum history; reset at every target refresh; no burnin",
    "selection": "label-free assignment-change stopping, otherwise fixed cap; final weights",
    "deviations": [
        "Inductive 54k optimization / 6k diagnostics / 10k endpoint evaluation instead of transductive 70k clustering",
        "Fresh seed-matched DEC-style AE and k-means shared between DEC and IDEC; no external historical weights",
        "DEC fixed refresh160 and declared20000-update safety cap; source loop is unbounded",
        "PyTorch kernels and RNG replace Caffe/Keras; no claim of identical historical trajectories",
        "KMeans algorithm lloyd and explicit tolerance/max_iter; original sklearn version is not recovered",
        "IDEC Adam epsilon explicitly1e-7; historical Keras runtime is not recovered",
        "Stable log probabilities and zero-log-zero convention; skip nonexistent empty IDEC batches",
        "No label logging, held-out checkpoint selection, or repeated Caffe test passes",
        "Endpoint predictions recomputed from final weights, rather than cached last-refresh assignments",
    ],
}


def source_files() -> list[Path]:
    vendor = Path(__file__).parent / "vendor_sources"
    paths = [Path(__file__), Path(__file__).with_name("data.py"),
             Path(__file__).with_name("provenance.py")]
    for directory in [vendor / "dec_e551e01f", vendor / "idec_4a661c21"]:
        paths.extend(p for p in directory.rglob("*") if p.is_file())
    return sorted(set(p.resolve() for p in paths))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def source_digest() -> str:
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in source_files()})


def protocol_for(validation_updates: int | None = None) -> dict:
    p = copy.deepcopy(PROTOCOL)
    if validation_updates is not None:
        if not 1 <= validation_updates <= 1000:
            raise ValueError("validation_updates must be in [1,1000]")
        p.update(layer_updates=validation_updates, joint_updates=validation_updates,
                 cluster_max_updates=validation_updates)
    return p


def setup(seed: int, device: str) -> torch.device:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    return target


def caffe_euclidean(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return .5 * (prediction - target).square().mean()


class StackedAutoencoder(nn.Module):
    def __init__(self, dimensions: list[int], *, dropout: float = .2, init_std: float = .01):
        super().__init__()
        if len(dimensions) < 2 or any(d <= 0 for d in dimensions):
            raise ValueError("Positive layer dimensions required")
        self.dimensions, self.dropout = list(dimensions), dropout
        self.encoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dimensions[:-1], dimensions[1:]))
        # decoder[i] is the paired inverse of encoder[i].
        self.decoder = nn.ModuleList(nn.Linear(b, a) for a, b in zip(dimensions[:-1], dimensions[1:]))
        for layer in [*self.encoder, *self.decoder]:
            nn.init.normal_(layer.weight, std=init_std); nn.init.zeros_(layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index < len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for index in reversed(range(len(self.decoder))):
            z = self.decoder[index](z)
            if index > 0:
                z = F.relu(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def layerwise(self, x: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            for previous in range(index):
                x = F.relu(self.encoder[previous](x))
            target = x.detach()
        h = self.encoder[index](F.dropout(target, self.dropout if index else 0., self.training))
        if index < len(self.encoder) - 1:
            h = F.dropout(F.relu(h), self.dropout, self.training)
        reconstruction = self.decoder[index](h)
        if index:
            reconstruction = F.relu(reconstruction)
        return reconstruction, target


class ReferenceClustering(nn.Module):
    def __init__(self, autoencoder: StackedAutoencoder, centers: torch.Tensor):
        super().__init__()
        self.autoencoder = autoencoder
        self.centers = nn.Parameter(centers.clone())

    def log_assignments(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        logits = -torch.log1p((z[:, None, :] - self.centers[None, :, :]).square().sum(-1))
        return torch.log_softmax(logits, dim=1)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("Finite nonnegative N-by-K responsibilities required")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Responsibilities must sum to one")
    with torch.no_grad():
        frequency = q.sum(0).clamp_min(torch.finfo(q.dtype).tiny)
        weights = q.square() / frequency
        return weights / weights.sum(1, keepdim=True)


def clustering_loss(model: ReferenceClustering, x: torch.Tensor, target: torch.Tensor,
                    method: str, *, gamma: float = .1) -> dict[str, torch.Tensor]:
    if method not in {"dec", "idec"}:
        raise ValueError("Expected dec or idec")
    p = target.detach()
    log_q = model.log_assignments(x)
    kl = (torch.special.xlogy(p, p) - p * log_q).sum(1).mean()
    mse = F.mse_loss(model.autoencoder(x), x) if method == "idec" else kl.new_zeros(())
    return {"loss": kl if method == "dec" else mse + gamma * kl, "kl": kl, "mse": mse}


class CaffeSGD(torch.optim.Optimizer):
    """Exact no-weight-decay update from the pinned fork's solver.cpp.

    h_t = momentum_t*h_(t-1) + lr_t*lr_mult*gradient_t; theta -= h_t.
    The history has update units, unlike torch.optim.SGD's gradient history.
    """
    def __init__(self, groups, *, lr=.1, momentum=.9, burnin=1000, step_size=20000, gamma=.1):
        super().__init__(groups, dict(lr=lr, momentum=momentum, burnin=burnin,
                                    step_size=step_size, gamma=gamma, lr_mult=1., iteration=0))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad(): closure()
        for group in self.param_groups:
            iteration = group["iteration"]
            rate = group["lr"] * group["gamma"] ** (iteration // group["step_size"])
            momentum = group["momentum"]
            if iteration < group["burnin"]:
                momentum *= iteration / group["burnin"]
            for parameter in group["params"]:
                if parameter.grad is None: continue
                if parameter.grad.is_sparse: raise ValueError("Sparse gradients unsupported")
                state = self.state[parameter]
                if "history" not in state: state["history"] = torch.zeros_like(parameter)
                history = state["history"]
                history.mul_(momentum).add_(parameter.grad, alpha=rate * group["lr_mult"])
                parameter.sub_(history)
            group["iteration"] = iteration + 1


def caffe_groups(named_parameters, bias_multiplier=2.):
    pairs = list(named_parameters)
    return [{"params": [p for name, p in pairs if not name.endswith("bias")], "lr_mult": 1.},
            {"params": [p for name, p in pairs if name.endswith("bias")], "lr_mult": bias_multiplier}]


class SequentialSampler:
    def __init__(self, size: int, batch_size: int, *, wrap: bool):
        if size <= 0 or batch_size <= 0: raise ValueError("Positive size and batch required")
        self.size, self.batch_size, self.wrap, self.cursor = size, batch_size, wrap, 0
        self.draws = 0

    def next(self) -> np.ndarray:
        if self.wrap:
            indices = (np.arange(self.batch_size) + self.cursor) % self.size
        else:
            indices = np.arange(self.cursor, min(self.cursor + self.batch_size, self.size))
        self.cursor = int((self.cursor + len(indices)) % self.size)
        self.draws += len(indices)
        return indices

    def state_dict(self):
        return dict(size=self.size, batch_size=self.batch_size, wrap=self.wrap,
                    cursor=self.cursor, draws=self.draws)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ["size", "batch_size", "wrap"]):
            raise ValueError("Sampler configuration mismatch")
        if not 0 <= state["cursor"] < self.size: raise ValueError("Invalid sampler cursor")
        self.cursor, self.draws = state["cursor"], state["draws"]


def _cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(_cpu(v) for v in value)
    return value


def rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device-count mismatch")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    a = state["numpy"]; np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), a[2], a[3], a[4]))


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def save_checkpoint(output: Path, *, model, optimizer, sampler, progress: dict,
                    config: dict, source_sha: str, logs: list, retain: bool = False) -> Path:
    """Immutable numbered generations; retain latest two plus phase boundaries.

    A completed hash sidecar precedes the atomic latest-pointer update. If killed
    during writing, the prior pointer/generation remains usable. No automatic
    continuation from an ambiguous or partially written checkpoint is attempted.
    """
    suffix = "end" if retain else "step"
    name = f"checkpoint_{progress['phase_index']:02d}_{progress['updates']:09d}_{suffix}.pt"
    path = output / name
    payload = {"schema_version": "paper1-reference-resume-v1", "config": config,
               "config_digest": digest(config), "source_digest": source_sha,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "sampler": sampler.state_dict(), "progress": progress, "rng": rng_state(), "logs": logs}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(_cpu(payload), temporary)
    os.replace(temporary, path)
    record = {"file": name, "sha256": sha256(path), "retain": retain,
              "config_digest": digest(config), "source_digest": source_sha}
    _atomic_json(path.with_suffix(".sha256.json"), record)
    _atomic_json(output / "latest_checkpoint.json", record)
    records = sorted(output.glob("checkpoint_*.sha256.json"))
    rolling = [p for p in records if not json.loads(p.read_text())["retain"]]
    for old in rolling[:-2]:
        (output / json.loads(old.read_text())["file"]).unlink(missing_ok=True)
        old.unlink()
    return path


def load_checkpoint(path: Path, config: dict, source_sha: str) -> dict:
    if path.suffix == ".json":
        pointer = json.loads(path.read_text()); path = path.parent / pointer["file"]
    sidecar = json.loads(path.with_suffix(".sha256.json").read_text())
    if sidecar["sha256"] != sha256(path): raise ValueError("Checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("schema_version") != "paper1-reference-resume-v1" or
        state.get("config_digest") != digest(config) or state.get("config") != config or
        state.get("source_digest") != source_sha):
        raise ValueError("Resume config/source/data/seed/phase mismatch")
    return state


def _arrays(manifest: Path, split: str):
    value = load_inputs(manifest, split)
    x = value["x"].reshape(len(value["x"]), -1)
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Reference MNIST requires prepared [0,1] inputs")
    return torch.from_numpy(x.copy()).mul_(255. / 50.), value["ids"]


@torch.no_grad()
def encoded(model: StackedAutoencoder, x: torch.Tensor, device, batch_size=1024):
    model.eval()
    return torch.cat([model.encode(batch.to(device)).cpu() for batch in x.split(batch_size)])


@torch.no_grad()
def responsibilities(model: ReferenceClustering, x: torch.Tensor, device, batch_size=1024):
    model.eval()
    q = torch.cat([model.log_assignments(batch.to(device)).exp().cpu() for batch in x.split(batch_size)])
    if not torch.isfinite(q).all() or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ArithmeticError("Invalid inference responsibilities")
    return q


def _finite_step(loss, model, optimizer):
    if not torch.isfinite(loss): raise ArithmeticError("Nonfinite objective")
    optimizer.zero_grad(set_to_none=True); loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in gradients): raise ArithmeticError("Nonfinite gradient")
    norm = torch.sqrt(sum(g.double().square().sum() for g in gradients)).item()
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise ArithmeticError("Nonfinite parameter")
    return norm


def _pretrain_optimizer(ae, phase, p):
    if phase < len(ae.encoder):
        named = [(f"encoder.{name}", v) for name, v in ae.encoder[phase].named_parameters()]
        named += [(f"decoder.{name}", v) for name, v in ae.decoder[phase].named_parameters()]
    else: named = list(ae.named_parameters())
    return CaffeSGD(caffe_groups(named, p["bias_lr_multiplier"]), lr=p["pretrain_lr"],
                    momentum=p["momentum"], burnin=p["momentum_burnin"],
                    step_size=p["pretrain_step"], gamma=p["pretrain_gamma"])


def _cluster_optimizer(model, method, p):
    if method == "idec":
        return torch.optim.Adam(model.parameters(), lr=p["idec_lr"], betas=tuple(p["idec_betas"]), eps=p["idec_epsilon"])
    named = [(name, v) for name, v in model.named_parameters() if ".decoder." not in name]
    return CaffeSGD(caffe_groups(named, p["bias_lr_multiplier"]), lr=p["dec_lr"],
                    momentum=p["momentum"], burnin=0, step_size=100000, gamma=.1)


def _initial_identity(config):
    return {k: v for k, v in config.items() if k not in {"phase", "initialization_sha256"}}


def _read_initial(initial: Path, config, source_sha):
    run = json.loads((initial / "run.json").read_text())
    checkpoint = initial / "initialization.pt"
    if (run.get("status") != "completed" or not run.get("fresh_run") or
        run.get("artifact_sha256", {}).get("initialization.pt") != sha256(checkpoint)):
        raise ValueError("Initialization must be completed, fresh and hash verified")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (value.get("kind") != "fresh_reference_dec_idec_initialization" or
        value.get("identity") != _initial_identity(config) or value.get("source_digest") != source_sha):
        raise ValueError("Initialization config/source/data/seed/stage mismatch; old weights forbidden")
    return value


def execute(phase: str, manifest: str | Path, output: str | Path, *, seed: int,
            initial: str | Path | None = None, device: str = "cuda",
            validation_updates: int | None = None, resume_from: str | Path | None = None,
            protocol: dict | None = None) -> dict:
    """Run one fresh phase. Custom dimensions/budgets are validation-only tests.

    Resume requires the same source/data/config/phase, and always a new output.
    Validation never reads official test inputs. Full endpoint inference reads
    train/validation/test inputs only after fitting; labels are physically separate.
    """
    if phase not in {"initialize", "dec", "idec"}: raise ValueError("Unknown phase")
    if seed not in [901, 902, 903]: raise ValueError("Reference seeds are 901,902,903")
    p = copy.deepcopy(protocol if protocol is not None else protocol_for(validation_updates))
    validation = validation_updates is not None
    if not validation and p != PROTOCOL: raise ValueError("Altered settings are validation only")
    if validation_updates is not None and not 1 <= validation_updates <= 1000:
        raise ValueError("validation_updates must be in [1,1000]")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    info = json.loads(manifest.read_text())
    if info["dataset"] != "mnist": raise ValueError("Reference validation is MNIST only")
    if not validation:
        if [info["splits"][s]["n_samples"] for s in ["train", "validation", "test"]] != [54000,6000,10000]:
            raise ValueError("Production requires approved MNIST54k/6k/10k split")
    config = {"protocol_version": p["protocol_version"], "phase": phase, "seed": seed,
              "effective_seed": seed + (100000 if validation else 0), "validation_only": validation,
              "protocol": p, "manifest_sha256": sha256(manifest), "native_reproduction": False}
    if phase != "initialize":
        if initial is None: raise ValueError("Fresh --initial directory required")
        initial = Path(initial).resolve()
        config["initialization_sha256"] = sha256(initial / "initialization.pt")
    elif initial is not None: raise ValueError("Initialization cannot import external weights")
    source_sha = source_digest()
    resumed = load_checkpoint(Path(resume_from).resolve(), config, source_sha) if resume_from else None
    initial_state = _read_initial(initial, config, source_sha) if phase != "initialize" else None
    target = setup(config["effective_seed"], device)
    x, ids = _arrays(manifest, "train")
    if x.shape[1] != p["architecture"][0] or info["n_clusters"] != p["n_clusters"]:
        raise ValueError("Input dimensions / cluster count mismatch")
    run = start_run(output, stage="validation" if validation else "final_benchmark", config=config, sources=source_files())
    run.update(manifest_path=str(manifest), reference_source_digest=source_sha,
               resume_from=str(resume_from) if resume_from else None,
               labels_loaded=False, native_reproduction=False,
               device=str(target), cuda_device_name=torch.cuda.get_device_name(target) if target.type == "cuda" else None)
    write_json(output / "run.json", run)
    begin = time.perf_counter()
    ae = StackedAutoencoder(p["architecture"], dropout=p["layer_dropout"], init_std=p["weight_init_std"]).to(target)
    logs = resumed["logs"] if resumed else []
    try:
        if phase == "initialize":
            if resumed: ae.load_state_dict(resumed["model"])
            n_layers = len(ae.encoder)
            first_phase = resumed["progress"]["phase_index"] if resumed else 0
            total = resumed["progress"]["total_updates"] if resumed else 0
            for index in range(first_phase, n_layers + 1):
                optimizer = _pretrain_optimizer(ae, index, p)
                sampler = SequentialSampler(len(x), p["batch_size"], wrap=True)
                completed = 0
                if resumed and index == first_phase:
                    optimizer.load_state_dict(resumed["optimizer"])
                    sampler.load_state_dict(resumed["sampler"])
                    completed = resumed["progress"]["updates"]
                    restore_rng(resumed["rng"])
                budget = p["layer_updates"] if index < n_layers else p["joint_updates"]
                ae.train()
                for update in range(completed, budget):
                    batch = x[sampler.next()].to(target)
                    ae.zero_grad(set_to_none=True)
                    reconstruction, reference = ae.layerwise(batch, index) if index < n_layers else (ae(batch), batch)
                    loss = caffe_euclidean(reconstruction, reference)
                    norm = _finite_step(loss, ae, optimizer); total += 1
                    progress = {"phase_index": index, "updates": update + 1, "total_updates": total}
                    boundary = update + 1 == budget
                    if boundary or (update + 1) % p["checkpoint_every"] == 0:
                        logs.append(dict(progress, loss=float(loss.detach()), preclip_gradient_norm=norm,
                                         sampler=sampler.state_dict()))
                        save_checkpoint(output, model=ae, optimizer=optimizer, sampler=sampler,
                                        progress=progress, config=config, source_sha=source_sha, logs=logs, retain=boundary)
                resumed = None
            features = encoded(ae, x, target).numpy()
            km = KMeans(n_clusters=p["n_clusters"], n_init=p["kmeans_n_init"],
                        max_iter=p["kmeans_max_iter"], tol=p["kmeans_tol"], algorithm="lloyd",
                        random_state=config["effective_seed"]).fit(features)
            initialization = {"kind": "fresh_reference_dec_idec_initialization",
                              "identity": _initial_identity(config), "source_digest": source_sha,
                              "autoencoder": _cpu(ae.state_dict()), "centers": torch.from_numpy(km.cluster_centers_),
                              "kmeans_labels": torch.from_numpy(km.labels_.astype(np.int64)),
                              "completed_pretrain_updates": total}
            torch.save(initialization, output / "initialization.pt")
            result = {"optimizer_updates": total, "pretrain_completed": True,
                      "kmeans_iterations": int(km.n_iter_), "kmeans_inertia": float(km.inertia_),
                      "kmeans_occupied": int(len(np.unique(km.labels_))), "predictions_written": []}
        else:
            ae.load_state_dict(initial_state["autoencoder"])
            model = ReferenceClustering(ae, initial_state["centers"].to(target)).to(target)
            optimizer = _cluster_optimizer(model, phase, p)
            sampler = SequentialSampler(len(x), p["batch_size"], wrap=phase == "dec")
            completed, stop_reason = 0, "update_cap"
            previous = torch.zeros(len(x), dtype=torch.long) if phase == "dec" else initial_state["kmeans_labels"].clone()
            target_p, last_refresh, last_change = None, -1, None
            if resumed:
                model.load_state_dict(resumed["model"]); optimizer.load_state_dict(resumed["optimizer"])
                sampler.load_state_dict(resumed["sampler"]); restore_rng(resumed["rng"])
                state = resumed["progress"]; completed = state["updates"]
                previous, target_p = state["previous"], state["target"]
                last_refresh, last_change = state["last_refresh"], state["last_change"]
                stop_reason = state.get("stop_reason", "update_cap")
            refresh = p[f"{phase}_refresh"]
            for update in range(completed, p["cluster_max_updates"]):
                if stop_reason == "assignment_stability": break
                if update % refresh == 0 and last_refresh != update:
                    q = responsibilities(model, x, target)
                    predicted = q.argmax(1)
                    last_change = float((predicted != previous).double().mean())
                    previous, target_p, last_refresh = predicted, target_distribution(q), update
                    logs.append({"refresh_update": update, "assignment_change": last_change,
                                 "occupied": int(predicted.unique().numel())})
                    if phase == "dec": optimizer = _cluster_optimizer(model, phase, p)
                    if last_change < p["assignment_tolerance"] and (phase == "dec" or update > 0):
                        stop_reason = "assignment_stability"; break
                model.train(); indices = sampler.next()
                terms = clustering_loss(model, x[indices].to(target), target_p[indices].to(target), phase,
                                        gamma=p["idec_kl_weight"])
                norm = _finite_step(terms["loss"], model, optimizer)
                completed = update + 1
                progress = {"phase_index": 0, "updates": completed, "total_updates": completed,
                            "previous": previous, "target": target_p, "last_refresh": last_refresh,
                            "last_change": last_change, "stop_reason": stop_reason}
                if completed % p["checkpoint_every"] == 0:
                    logs.append({"update": completed, **{k: float(v.detach()) for k,v in terms.items()},
                                 "preclip_gradient_norm": norm, "sampler": sampler.state_dict()})
                    save_checkpoint(output, model=model, optimizer=optimizer, sampler=sampler, progress=progress,
                                    config=config, source_sha=source_sha, logs=logs)
            progress = {"phase_index": 0, "updates": completed, "total_updates": completed,
                        "previous": previous, "target": target_p, "last_refresh": last_refresh,
                        "last_change": last_change, "stop_reason": stop_reason}
            save_checkpoint(output, model=model, optimizer=optimizer, sampler=sampler, progress=progress,
                            config=config, source_sha=source_sha, logs=logs, retain=True)
            torch.save(_cpu(model.state_dict()), output / "model.pt")
            splits = ["train", "validation"] if validation else ["train", "validation", "test"]
            for split in splits:
                values, sample_ids = (x, ids) if split == "train" else _arrays(manifest, split)
                q = responsibilities(model, values, target).numpy()
                np.savez_compressed(output / f"{split}_predictions.npz", sample_ids=sample_ids,
                                    predictions=q.argmax(1), responsibilities=q, cluster_labels=np.arange(p["n_clusters"]))
            result = {"optimizer_updates": completed, "stop_reason": stop_reason,
                      "last_assignment_change": last_change, "last_target_refresh": last_refresh,
                      "initialization_sha256": config["initialization_sha256"], "predictions_written": splits}
        write_json(output / "training_diagnostics.json", logs)
        result.update(wall_seconds=time.perf_counter() - begin, labels_loaded=False,
                      gradient_clipping=False, endpoint_selection="label_free_source_stability_or_fixed_cap")
        finish_run(output, run, **result)
        return result
    except BaseException as error:
        # Diagnostic only: a failed update may have nonfinite parameters. Never
        # allow this artifact to masquerade as a resumable valid generation.
        failure = {"schema_version": "paper1-reference-failure-state-v1", "resumable": False,
                   "config": config, "source_digest": source_sha,
                   "model": _cpu((model if phase != "initialize" and "model" in locals() else ae).state_dict()),
                   "optimizer": _cpu(optimizer.state_dict()) if "optimizer" in locals() else None,
                   "sampler": sampler.state_dict() if "sampler" in locals() else None,
                   "error": f"{type(error).__name__}: {error}"}
        temporary = output / "failure_state.pt.tmp"
        torch.save(failure, temporary); os.replace(temporary, output / "failure_state.pt")
        write_json(output / "training_diagnostics.json", logs)
        finish_run(output, run, status="failed", error=f"{type(error).__name__}: {error}",
                   wall_seconds=time.perf_counter() - begin,
                   resume_instruction="Use latest_checkpoint.json in a NEW output directory with identical config/source/data")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["initialize", "dec", "idec"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[901,902,903], required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-updates", type=int)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__": main()
