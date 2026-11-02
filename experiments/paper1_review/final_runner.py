"""Fresh fixed-budget matched fits; no label access or scheduler submission.

This implementation candidate does not discharge the separate 12 reference-run
gate. initialize/train never read test inputs; predict is a separate fixed-model
phase, also without labels. Changed numerical protocols require a new attempt.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import sklearn
from sklearn.mixture import GaussianMixture

from .data import load_inputs, _ids_hash
from .final_protocol import (PROTOCOL, PROTOCOL_VERSION, ALIASES, CONTAMINATION_CONDITIONS,
                             INFERENCE_CONDITIONS, protocol_digest, update_budget, validate_run)
from .final_diagnostics import gradient_diagnostic, parameter_diagnostics
from .models import create_model, dec_target_distribution
from .provenance import ROOT, sha256, start_run, finish_run, write_json


def setup(seed, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def sources():
    names = ["final_runner.py", "final_protocol.py", "final_diagnostics.py", "models.py", "site_parameterization.py", "data.py", "workload.py"]
    return [Path(__file__).with_name(n) for n in names] + [ROOT / "experiments/paper1_structured_inference" / n
              for n in ["structured.py", "gaussian_reference.py"]]


def _hardware(device):
    value = {"device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda,
             "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
             "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        value.update(gpu=p.name, compute_capability=[p.major, p.minor], total_gpu_mib=p.total_memory / 2**20,
                     peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    return value


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _array_inputs(path, metadata, split):
    entry = metadata["splits"][split]
    relative = entry["inputs"]
    artifact = (path.parent / relative).resolve()
    artifact.relative_to(path.parent.resolve())
    if sha256(artifact) != metadata["artifact_sha256"][relative]:
        raise ValueError("Training-view artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as saved:
        if set(saved.files) != {"x", "ids"}:
            raise ValueError("Training inputs must contain exactly x and ids, never labels/origins")
        arrays = {name: saved[name] for name in saved.files}
    x, ids = arrays["x"], arrays["ids"]
    if x.dtype != np.float32 or list(x.shape) != entry["input_shape"] or len(x) != entry["n_samples"] or _ids_hash(ids) != entry["ids_sha256"]:
        raise ValueError("Training-view shapes, dtype or IDs disagree with manifest")
    return arrays


def _validate_arrays(arrays, features):
    x, ids = arrays["x"], arrays["ids"]
    if x.ndim < 2 or len(x) < 1 or ids.ndim != 1 or len(ids) != len(x) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Nonempty inputs with unique aligned sample IDs required")
    x = x.reshape(len(x), -1)
    if x.shape[1] != features or not np.isfinite(x).all():
        raise ValueError("Input features or finiteness invalid")
    return torch.from_numpy(x), ids


def load_training_view(manifest_path, training_view=None):
    """Hash-checked clean/contaminated optimization inputs; labels never opened."""
    path = Path(manifest_path)
    clean = json.loads(path.read_text())
    clean_arrays = {split: load_inputs(path, split) for split in ["train", "validation"]}
    metadata = clean
    arrays = clean_arrays
    condition = "clean"
    if training_view is not None:
        vp = Path(training_view)
        metadata = json.loads(vp.read_text())
        condition = metadata.get("condition")
        if metadata.get("schema_version") != "paper1-review-training-view-v1" or metadata.get("status") != "completed":
            raise ValueError("Completed versioned training view required")
        if condition not in CONTAMINATION_CONDITIONS or metadata.get("clean_manifest_sha256") != sha256(path):
            raise ValueError("Unknown condition or mismatched clean reference")
        if metadata.get("clean_train_count") != len(clean_arrays["train"]["x"]) or any(metadata.get(k) != clean[k] for k in ["dataset", "n_features", "n_clusters"]):
            raise ValueError("Training-view clean count/dataset/dimensions mismatch")
        arrays = {split: _array_inputs(vp, metadata, split) for split in ["train", "validation"]}
        for split in arrays:
            clean_by_id = {str(value): i for i, value in enumerate(clean_arrays[split]["ids"])}
            rows = {str(value): i for i, value in enumerate(arrays[split]["ids"])}
            if not clean_by_id.keys() <= rows.keys():
                raise ValueError("Shared clean IDs must retain their original train/validation partition")
            ordered = np.array([rows[sample] for sample in clean_by_id])
            if not np.array_equal(arrays[split]["x"][ordered], clean_arrays[split]["x"]):
                raise ValueError("Shared clean observations changed in contaminated view")
    train, ids = _validate_arrays(arrays["train"], clean["n_features"])
    validation, validation_ids = _validate_arrays(arrays["validation"], clean["n_features"])
    if set(ids.tolist()) & set(validation_ids.tolist()):
        raise ValueError("Train and validation sample IDs overlap")
    common, common_ids = _validate_arrays(clean_arrays["train"], clean["n_features"])
    # Same fixed clean diagnostic observations across methods and conditions.
    order = sorted(range(len(common_ids)), key=lambda i: hashlib.sha256(str(common_ids[i]).encode()).digest())
    return {"manifest": clean, "train": train, "train_ids": ids, "validation": validation,
            "validation_ids": validation_ids, "clean_count": len(common), "condition": condition,
            "common_batch": common[order[:32]], "common_ids": common_ids[order[:32]],
            "clean_manifest_sha256": sha256(path), "view_sha256": sha256(Path(training_view)) if training_view else sha256(path),
            "view_seed": metadata.get("seed") if training_view else None,
            "view_manifest": str(Path(training_view).resolve()) if training_view else str(path.resolve())}


def model_for(method, manifest, device, protocol=PROTOCOL):
    p, s, architecture = protocol["objective"], protocol["structured"], protocol["architecture"]
    return create_model(ALIASES.get(method, method), manifest["n_features"], manifest["n_clusters"],
        latent_dim=architecture["latent_dim"], hidden_dims=tuple(architecture["hidden_dims"]),
        decoder_activation="identity" if manifest["dataset"] == "reuters10k" else "sigmoid",
        observation_variance=p["observation_variance"], beta=p["beta"], idec_gamma=p["idec_gamma"],
        df_init=p["df_init"], df_bounds=tuple(p["prior_df_bounds"]), posterior_df_bounds=tuple(p["recognition_df_bounds"]),
        log_variance_bounds=tuple(p["log_variance_bounds"]), n_samples=protocol["training_samples"],
        prediction_samples=protocol["prediction_samples"], nodes=s["training_nodes"], prediction_nodes=s["prediction_nodes"],
        standardized_bounds=tuple(s["bounds"]), structured_site_mode=s["site_mode"],
        structured_site_initial_precision=s["initial_precision"], structured_site_max_precision=s["max_precision"]).to(device)


class PermutationSampler:
    """Finite seeded passes, exact update budget and serializable next-batch state."""
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor, self.passes = 0, 0

    def next(self):
        if self.cursor >= len(self.order):
            self.order = torch.randperm(self.size, generator=self.generator)
            self.cursor, self.passes = 0, self.passes + 1
        result = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(result)
        return result

    def state(self):
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "cursor": self.cursor, "passes": self.passes, "rng": self.generator.get_state()}

    def restore(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Resume sampler configuration mismatch")
        order = state["order"].cpu()
        if (order.dtype != torch.long or len(order) not in {0, self.size}
                or not 0 <= state["cursor"] <= len(order)
                or (len(order) and not torch.equal(order.sort().values, torch.arange(self.size)))):
            raise ValueError("Invalid resume permutation/cursor")
        self.order, self.cursor, self.passes = order, state["cursor"], state["passes"]
        self.generator.set_state(state["rng"].cpu())


def _optimizer(parameters, settings):
    return torch.optim.Adam([p for p in parameters if p.requires_grad], lr=settings["lr"],
                            betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"])


def _runtime(device):
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "cuda_build": torch.version.cuda, "device": device.type,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def _prefix_hash(path, length, destination=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block: raise ValueError("Committed sample-order prefix is incomplete")
            digest.update(block)
            if destination is not None: destination.write(block)
            remaining -= len(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def _checkpoint(path, model, optimizer, sampler, update, record, *, progress, order_file=None,
                phase="epoch_boundary", resumable=True):
    """Commit a boundary plus the exact sample-order prefix; never a partial epoch.

    optimization_complete is before endpoint GMM/predictions. Replaying that
    deterministic postprocessing must start from this pre-GMM model.
    """
    if order_file is not None:
        order_file.flush(); os.fsync(order_file.fileno())
    order_path = path.parent / "sample_order.int32"
    if not order_path.exists(): order_path.touch()
    order_bytes = 4 * sum(progress["lengths"])
    retained = {str(p.relative_to(path.parent)): sha256(p)
                for p in sorted((path.parent / "checkpoints").glob("epoch*")) if p.is_file()}
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state(),
             "completed_updates": update, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
             "protocol_version": record["protocol_version"], "config_sha256": record["config_sha256"],
             "source_sha256": record["source_sha256"], "schema": "paper1-matched-resume-v1",
             "phase": phase, "resumable": resumable, "runtime": _runtime(next(model.parameters()).device),
             "progress": progress, "sample_order_bytes": order_bytes,
             "sample_order_sha256": _prefix_hash(order_path, order_bytes), "retained_artifacts": retained}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    receipt_path = path.with_suffix(".json")
    if path.exists() and receipt_path.exists() and json.loads(receipt_path.read_text())["sha256"] == sha256(path):
        for old, destination in [(path, path.with_name("previous_checkpoint.pt")),
                                 (receipt_path, path.with_name("previous_checkpoint.json"))]:
            backup = destination.with_name(destination.name + ".tmp")
            backup.unlink(missing_ok=True); os.link(old, backup); backup.replace(destination)
    temporary.replace(path)
    _atomic_json(receipt_path, {"sha256": sha256(path), "phase": phase, "resumable": resumable,
                               "completed_updates": update, "source_sha256": record["source_sha256"]})


def _load_resume(path, output, config, record, device, model, optimizer, sampler):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.parent == output: raise ValueError("Resume requires a new output attempt")
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["sha256"] != sha256(path): raise ValueError("Resume checkpoint hash mismatch")
    previous_run = json.loads((path.parent / "run.json").read_text())
    previous_config = json.loads((path.parent / "config.json").read_text())
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("schema") != "paper1-matched-resume-v1" or previous_config != config
            or sha256(path.parent / "config.json") != state["config_sha256"]
            or previous_run["config_sha256"] != state["config_sha256"]
            or state["config_sha256"] != record["config_sha256"]
            or state["source_sha256"] != record["source_sha256"]
            or previous_run["source_sha256"] != record["source_sha256"]
            or not previous_run.get("fresh_run") or previous_run["stage"] != record["stage"]
            or state["runtime"] != _runtime(device)):
        raise ValueError("Resume source/config/data/seed/method/stage/runtime mismatch")
    if previous_run.get("status") == "completed":
        raise ValueError("Completed logical fits cannot be resumed as additional attempts")
    if (not state["resumable"] or previous_run.get("numerical_failure", False)
            or state["phase"] not in {"epoch_boundary", "optimization_complete"}):
        raise ValueError("Numerically rejected checkpoint/lineage cannot resume")
    for key in ["phase", "resumable", "completed_updates", "source_sha256"]:
        if receipt[key] != state[key]: raise ValueError("Resume receipt/state mismatch")
    update = state["completed_updates"]; progress = state["progress"]
    per_epoch = math.ceil(config["clean_train_count"] / config["protocol"]["batch_size"])
    if (not 0 <= update <= config["optimizer_updates"] or update % per_epoch
            or len(progress["lengths"]) != update
            or state["sample_order_bytes"] != 4 * sum(progress["lengths"])
            or (state["phase"] == "optimization_complete" and update != config["optimizer_updates"])
            or len(progress["curves"]) != update // per_epoch):
        raise ValueError("Resume requires consistent completed epoch boundaries")
    exposure = progress["exposure"]
    if (exposure.dtype != torch.int64 or exposure.shape != (config["training_count"],)
            or (exposure < 0).any() or int(exposure.sum()) != sum(progress["lengths"])
            or any(not 0 < n <= config["protocol"]["batch_size"] for n in progress["lengths"])):
        raise ValueError("Resume sample-exposure counters are inconsistent")
    if _prefix_hash(path.parent / "sample_order.int32", state["sample_order_bytes"]) != state["sample_order_sha256"]:
        raise ValueError("Resume sample-order prefix hash mismatch")
    for relative, digest in state["retained_artifacts"].items():
        source = (path.parent / relative).resolve(); source.relative_to(path.parent)
        if sha256(source) != digest: raise ValueError("Resume retained artifact hash mismatch")
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    with (output / "sample_order.int32").open("wb") as stream:
        _prefix_hash(path.parent / "sample_order.int32", state["sample_order_bytes"], stream)
    model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
    sampler.restore(state["sampler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        if device.type != "cuda": raise ValueError("CUDA checkpoint requires CUDA resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    _atomic_json(output / "resume_dependency.json", {"checkpoint": str(path), "checkpoint_sha256": receipt["sha256"],
        "parent_run_sha256": sha256(path.parent / "run.json"), "completed_updates": update,
        "logical_fit_id": previous_run.get("logical_fit_id", str(path.parent)),
        "interpretation": "same fresh logical fit; continuation is not an additional independent replicate"})
    record["logical_fit_id"] = previous_run.get("logical_fit_id", str(path.parent))
    record["recovery_cost_note"] = "Trajectory timings retain committed work only; discarded partial-epoch/postprocessing work and downtime require scheduler/attempt accounting"
    write_json(output / "run.json", record)
    return state


@torch.no_grad()
def embeddings(model, x, device):
    return np.concatenate([model.encoder(batch.to(device)).cpu().numpy() for batch in x.split(512)])


def fit_gmm(model, view, seed, device, protocol):
    started = time.perf_counter()
    settings = {k: v for k, v in protocol["gmm"].items() if k != "selection"}
    gmm = GaussianMixture(n_components=view["manifest"]["n_clusters"], random_state=seed, **settings).fit(embeddings(model, view["train"], device))
    if not gmm.converged_:
        raise ArithmeticError("Train-only GMM did not converge within the fixed budget")
    return gmm, {"seconds": time.perf_counter() - started, "converged": bool(gmm.converged_),
                 "iterations": int(gmm.n_iter_), "training_lower_bound": float(gmm.lower_bound_)}


def _config(view, seed, phase, protocol, validation_only):
    if not validation_only and protocol != PROTOCOL:
        raise ValueError("Final fits require the frozen protocol; altered settings are validation only")
    if view["view_seed"] is not None and view["view_seed"] != seed:
        raise ValueError("Contamination view seed differs from final initialization/training seed")
    return {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_digest(protocol), "protocol": protocol,
            "phase": phase, "seed": seed, "dataset": view["manifest"]["dataset"], "condition": view["condition"],
            "clean_manifest_sha256": view["clean_manifest_sha256"], "training_view_sha256": view["view_sha256"],
            "training_view_manifest": view["view_manifest"], "clean_train_count": view["clean_count"],
            "training_count": len(view["train"]), "validation_count": len(view["validation"]),
            "training_ids_sha256": _ids_hash(view["train_ids"]), "validation_ids_sha256": _ids_hash(view["validation_ids"]),
            "common_gradient_ids": view["common_ids"].tolist(), "validation_only": validation_only,
            "labels_loaded": False, "test_partition_loaded": False, "native_reproduction": False,
            "initialization_from_development": False}


def _gradient_step(loss, model, optimizer, clip):
    if not torch.isfinite(loss):
        raise ArithmeticError("Nonfinite objective")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ArithmeticError("Nonfinite optimization gradient")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    optimizer.step()
    model.constrain_parameters()
    return norm


def initialize(manifest_path, output, *, seed, device="cuda", training_view=None, protocol=PROTOCOL,
               validation_only=False, resume_from=None):
    device = setup(seed, device)
    view = load_training_view(manifest_path, training_view)
    validate_run(view["manifest"]["dataset"], "structured_q_student_prior", seed, view["condition"])
    config = _config(view, seed, "fresh_final_shared_initialization", protocol, validation_only)
    budget = update_budget(view["clean_count"], "pretrain", protocol)
    config["optimizer_updates"] = budget
    output = Path(output)
    record = start_run(output, stage="validation" if validation_only else "final_benchmark", config=config, sources=sources())
    try:
        model = model_for("gaussian_q_gaussian_prior", view["manifest"], device, protocol)
        optimizer = _optimizer(list(model.encoder.parameters()) + list(model.decoder.parameters()), protocol["ae"])
        sampler = PermutationSampler(len(view["train"]), protocol["batch_size"], seed + 10)
        exposure = np.zeros(len(view["train"]), dtype=np.int64)
        curves, lengths = [], []
        completed, elapsed_before = 0, 0.
        per_epoch = math.ceil(view["clean_count"] / protocol["batch_size"])
        if resume_from is not None:
            saved = _load_resume(resume_from, output, config, record, device, model, optimizer, sampler)
            completed = saved["completed_updates"]; progress = saved["progress"]
            exposure = progress["exposure"].numpy().copy()
            curves, lengths = progress["curves"], progress["lengths"]
            elapsed_before = progress["elapsed_seconds"]
        _sync(device)
        start = time.perf_counter()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        def save_boundary(update, order_file, phase="epoch_boundary"):
            progress = {"exposure": torch.from_numpy(exposure.copy()), "curves": curves, "lengths": lengths,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - start}
            _checkpoint(output / "latest_checkpoint.pt", model, optimizer, sampler, update, record,
                        progress=progress, order_file=order_file, phase=phase)
        with (output / "sample_order.int32").open("ab" if resume_from else "wb") as order_file:
            save_boundary(completed, order_file, "optimization_complete" if completed == budget else "epoch_boundary")
            for update in range(completed, budget):
                indices = sampler.next()
                order_file.write(indices.numpy().astype("<i4").tobytes()); lengths.append(len(indices))
                exposure[indices.numpy()] += 1
                batch = view["train"][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model.decoder(model.encoder(batch)) - batch).square().mean()
                norm = _gradient_step(loss, model, optimizer, protocol["ae"]["gradient_clip"])
                if (update + 1) % per_epoch == 0:
                    epoch = (update + 1) // per_epoch
                    curves.append({"clean_equivalent_epoch": epoch, "updates": update + 1, "last_batch_mse": float(loss.detach()),
                                   "last_preclip_norm": norm, "seconds": elapsed_before + time.perf_counter() - start})
                    if epoch % protocol["rolling_checkpoint_every_epochs"] == 0:
                        save_boundary(update + 1, order_file)
                        write_json(output / "pretrain_curves.json", curves)
            save_boundary(budget, order_file, "optimization_complete")
        _sync(device)
        neural_seconds = elapsed_before + time.perf_counter() - start
        gmm, gmm_record = fit_gmm(model, view, seed, device, protocol)
        torch.save({"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict()}, output / "autoencoder.pt")
        np.savez(output / "mixture.npz", loc=gmm.means_, covariance_diag=gmm.covariances_, weights=gmm.weights_)
        np.save(output / "batch_lengths.npy", np.asarray(lengths, dtype=np.int32))
        np.savez_compressed(output / "sample_exposure.npz", sample_ids=view["train_ids"], count=exposure)
        write_json(output / "pretrain_curves.json", curves)
        finish_run(output, record, optimizer_updates=budget, neural_seconds=neural_seconds, gmm=gmm_record,
                   hardware=_hardware(device), labels_loaded=False, test_partition_loaded=False)
    except Exception as error:
        finish_run(output, record, status="failed", error=repr(error), hardware=_hardware(device),
                   numerical_failure=isinstance(error, ArithmeticError),
                   recovery="Resume last valid boundary into a new attempt; numerical failures remain rejected")
        raise
    return output


def verify_initialization(initial, view, seed, protocol, validation_only):
    initial = Path(initial)
    metadata = json.loads((initial / "run.json").read_text())
    config = json.loads((initial / "config.json").read_text())
    if metadata.get("status") != "completed" or not metadata.get("fresh_run") or config.get("phase") != "fresh_final_shared_initialization":
        raise ValueError("Completed fresh final initialization required; development weights forbidden")
    expected = {"protocol_sha256": protocol_digest(protocol), "seed": seed, "training_view_sha256": view["view_sha256"],
                "clean_manifest_sha256": view["clean_manifest_sha256"], "validation_only": validation_only}
    if any(config.get(k) != v for k, v in expected.items()) or sha256(initial / "config.json") != metadata["config_sha256"]:
        raise ValueError("Initialization seed/view/protocol/config mismatch")
    for path in sources():
        if metadata["source_file_sha256"].get(str(path.relative_to(ROOT))) != sha256(path):
            raise ValueError("Initialization source release differs; create a new versioned initialization")
    for name in ["autoencoder.pt", "mixture.npz"]:
        if sha256(initial / name) != metadata["artifact_sha256"][name]:
            raise ValueError("Initialization tensor hash mismatch")
    return metadata


@torch.no_grad()
def responsibilities(model, x, device, *, seed):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        value = torch.cat([model.predict(batch.to(device)).cpu() for batch in x.split(128)]).numpy()
    if not np.isfinite(value).all() or (value < 0).any() or np.max(abs(value.sum(1) - 1)) > 1e-5:
        raise ArithmeticError("Invalid cluster responsibilities")
    return value


@torch.no_grad()
def quadrature_check(model, x, device, protocol=PROTOCOL):
    """Both grid refinement and a wider refined reference; no adaptive selection."""
    if model.method != "structured_q_student_prior": return None
    from experiments.paper1_structured_inference.structured import CenteredGaussianSite, structured_posterior
    settings = protocol["structured"]
    # Deterministic spread of points, not a label-selected favorable batch.
    chosen = np.unique(np.linspace(0, len(x) - 1, min(len(x), 32), dtype=int))
    mean, lv = model._encoded(x[chosen].to(device))
    attained = model.make_site(mean, lv)
    site = CenteredGaussianSite(attained.center.double(), attained.precision.double())
    prior = copy.deepcopy(model.prior).double()
    variants = [(settings["training_nodes"], settings["bounds"]), (256, settings["bounds"]),
                (settings["reference_nodes"], settings["reference_bounds"]),
                (settings["refined_reference_nodes"], settings["refined_reference_bounds"])]
    qs = [structured_posterior(prior, site, nodes=n, grid="standardized_log_tau", standardized_bounds=tuple(bounds)) for n, bounds in variants]
    reference = qs[-1]
    rows = []
    limits = {"log_normalizer": settings["log_normalizer_atol"], "responsibilities": settings["responsibility_atol"], "kl_to_prior": settings["kl_atol"]}
    for (nodes, bounds), q in zip(variants, qs):
        errors = {name: float((getattr(q, name) - getattr(reference, name)).abs().max()) for name in limits}
        rows.append({"nodes": nodes, "bounds": bounds, "errors": errors})
    train_pass = all(rows[0]["errors"][k] <= v for k, v in limits.items())
    reference_pass = all(rows[-2]["errors"][k] <= v * settings["reference_tolerance_factor"] for k, v in limits.items())
    return {"rows": rows, "training_nodes_pass": train_pass, "prediction_nodes_pass": train_pass,
            "reference_refinement_pass": reference_pass, "passed": train_pass and reference_pass,
            "selected_positions": chosen.tolist(), "interpretation": settings["claim"]}


def _save_predictions(path, resp, ids):
    np.savez_compressed(path, sample_ids=ids, predictions=resp.argmax(1), responsibilities=resp,
                        cluster_labels=np.arange(resp.shape[1]))
    counts = np.bincount(resp.argmax(1), minlength=resp.shape[1])
    return {"count": len(resp), "occupied_clusters": int((counts > 0).sum()), "cluster_counts": counts.tolist(),
            "max_cluster_share": float(counts.max() / len(resp)),
            "mean_responsibility_entropy": float(-(resp * np.log(np.maximum(resp, np.finfo(resp.dtype).tiny))).sum(1).mean()),
            "probability_interpretation": "cluster responsibility diagnostic, not calibrated semantic probabilities"}


def train(manifest_path, initial, output, *, method, seed, device="cuda", training_view=None,
          protocol=PROTOCOL, validation_only=False, resume_from=None):
    device = setup(seed, device)
    view = load_training_view(manifest_path, training_view)
    validate_run(view["manifest"]["dataset"], method, seed, view["condition"])
    initial = Path(initial)
    initial_metadata = verify_initialization(initial, view, seed, protocol, validation_only)
    config = _config(view, seed, "fixed_final_matched_fit", protocol, validation_only)
    config.update(method=method, initial_run_sha256=sha256(initial / "run.json"), initial_tensor_sha256=initial_metadata["artifact_sha256"]["autoencoder.pt"])
    budget = update_budget(view["clean_count"], "training", protocol)
    config["optimizer_updates"] = budget
    output = Path(output)
    record = start_run(output, stage="validation" if validation_only else "final_benchmark", config=config, sources=sources())
    try:
        model = model_for(method, view["manifest"], device, protocol)
        write_json(output / "objective_metadata.json", model.objective_metadata)
        ae = torch.load(initial / "autoencoder.pt", map_location=device, weights_only=True)
        model.encoder.load_state_dict(ae["encoder"]); model.decoder.load_state_dict(ae["decoder"])
        with np.load(initial / "mixture.npz", allow_pickle=False) as mixture:
            model.set_cluster_parameters(mixture["loc"], mixture["covariance_diag"], mixture["weights"])
        torch.manual_seed(seed + 100)
        optimizer = _optimizer(model.parameters(), protocol["train"])
        sampler = PermutationSampler(len(view["train"]), protocol["batch_size"], seed + 200)
        per_epoch = math.ceil(view["clean_count"] / protocol["batch_size"])
        exposure = np.zeros(len(view["train"]), dtype=np.int64)
        curves, diagnostics, checks, lengths, occupancy = [], [], [], [], []
        completed, elapsed_before = 0, 0.
        timings = {"optimizer_updates_seconds": 0., "diagnostics_seconds": 0., "target_refresh_seconds": 0.}
        if resume_from is not None:
            saved = _load_resume(resume_from, output, config, record, device, model, optimizer, sampler)
            completed = saved["completed_updates"]; progress = saved["progress"]
            exposure = progress["exposure"].numpy().copy()
            curves, diagnostics, checks = progress["curves"], progress["diagnostics"], progress["checks"]
            lengths, occupancy, timings = progress["lengths"], progress["occupancy"], progress["timings"]
            elapsed_before = progress["elapsed_seconds"]
        gradient_batch = view["common_batch"][:protocol["common_gradient_batch_size"]].to(device)
        ids_to_row = {str(sample): i for i, sample in enumerate(view["train_ids"])}
        diagnostic_rows = torch.tensor([ids_to_row[str(sample)] for sample in view["common_ids"][:len(gradient_batch)]])
        clip = protocol["train"]["gradient_clip"]
        target = None
        start = time.perf_counter()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        checkpoint_directory = output / "checkpoints"; checkpoint_directory.mkdir(exist_ok=True)
        order_file = None

        def save_boundary(update, phase="epoch_boundary", resumable=True):
            progress = {"exposure": torch.from_numpy(exposure.copy()), "curves": curves, "lengths": lengths,
                        "diagnostics": diagnostics, "checks": checks, "occupancy": occupancy, "timings": timings,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - start}
            _checkpoint(output / "latest_checkpoint.pt", model, optimizer, sampler, update, record,
                        progress=progress, order_file=order_file, phase=phase, resumable=resumable)

        def audit(epoch, completed_updates):
            _sync(device); audit_started = time.perf_counter()
            diag_target = target[diagnostic_rows].to(device) if target is not None else None
            diagnostic = gradient_diagnostic(model, gradient_batch, seed=seed + 3000 + epoch, target=diag_target)
            diagnostic.update(clean_equivalent_epoch=epoch, completed_updates=completed_updates,
                              parameters=parameter_diagnostics(model, gradient_batch))
            diagnostics.append(diagnostic)
            write_json(output / "gradient_diagnostics.json", diagnostics)
            check = quadrature_check(model, view["validation"], device, protocol)
            checks.append({"clean_equivalent_epoch": epoch, "quadrature": check})
            write_json(output / "numerical_checks.json", checks)
            if check is not None and not check["passed"]:
                save_boundary(completed_updates, "numerical_gate_failed", False)
                raise ArithmeticError("Attained quadrature gate failed; retain attempt, no automatic grid/budget repair")
            _sync(device); timings["diagnostics_seconds"] += time.perf_counter() - audit_started

        def refresh_target():
            if not model.is_dec: return None
            _sync(device); target_started = time.perf_counter()
            value = dec_target_distribution(torch.from_numpy(responsibilities(model, view["train"], device, seed=seed + 4000)))
            _sync(device); timings["target_refresh_seconds"] += time.perf_counter() - target_started
            return value

        if resume_from is None:
            target = refresh_target()
            if 0 in protocol["diagnostic_epochs"]: audit(0, 0)
            if 0 in protocol["checkpoint_epochs"]:
                torch.save(model.state_dict(), checkpoint_directory / "epoch000.pt")
        elif completed == 0:
            # audit0 is committed; only the deterministic next-epoch target is
            # regenerated. Diagnostics must not be repeated on continuation.
            target = refresh_target()
        with (output / "sample_order.int32").open("ab" if resume_from else "wb") as order_file:
            save_boundary(completed, "optimization_complete" if completed == budget else "epoch_boundary")
            total, count, norm_sum, max_norm, clipped = {}, 0, 0., 0., 0
            for update in range(completed, budget):
                if update and update % per_epoch == 0: target = refresh_target()
                indices = sampler.next()
                order_file.write(indices.numpy().astype("<i4").tobytes()); lengths.append(len(indices))
                exposure[indices.numpy()] += 1
                _sync(device); step_started = time.perf_counter()
                batch = view["train"][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                panel = model.training_loss(batch, target=None if target is None else target[indices].to(device))
                norm = _gradient_step(panel["loss"], model, optimizer, clip)
                _sync(device); timings["optimizer_updates_seconds"] += time.perf_counter() - step_started
                count += len(batch); norm_sum += norm; max_norm = max(max_norm, norm); clipped += int(norm > clip)
                for key, value in panel.items(): total[key] = total.get(key, 0.) + float(value.detach()) * len(batch)
                if (update + 1) % per_epoch == 0:
                    epoch = (update + 1) // per_epoch
                    _sync(device)
                    row = {"clean_equivalent_epoch": epoch, "updates": update + 1, "examples_seen_this_window": count,
                           "preclip_norm_mean": norm_sum / per_epoch, "preclip_norm_max": max_norm,
                           "clipped_updates": clipped, "clip_fraction": clipped / per_epoch,
                           "elapsed_seconds": elapsed_before + time.perf_counter() - start, **{k: v / count for k, v in total.items()}}
                    curves.append(row); write_json(output / "training_curves.json", curves)
                    print(json.dumps(row), flush=True)
                    if epoch in protocol["diagnostic_epochs"]: audit(epoch, update + 1)
                    if epoch in protocol["checkpoint_epochs"]:
                        torch.save(model.state_dict(), checkpoint_directory / f"epoch{epoch:03d}.pt")
                        if method != "vae_gmm":
                            occ = _save_predictions(checkpoint_directory / f"epoch{epoch:03d}_validation_predictions.npz",
                                responsibilities(model, view["validation"], device, seed=seed + 1000), view["validation_ids"])
                            occupancy.append({"clean_equivalent_epoch": epoch, "split": "validation", **occ})
                            write_json(output / "occupancy_diagnostics.json", occupancy)
                    if epoch % protocol["rolling_checkpoint_every_epochs"] == 0:
                        save_boundary(update + 1)
                    total, count, norm_sum, max_norm, clipped = {}, 0, 0., 0., 0
            save_boundary(budget, "optimization_complete")
        _sync(device)
        training_seconds = elapsed_before + time.perf_counter() - start
        gmm_record = None
        if method == "vae_gmm":
            fitted, gmm_record = fit_gmm(model, view, seed, device, protocol)
            model.set_cluster_parameters(fitted.means_, fitted.covariances_, fitted.weights_)
        torch.save(model.state_dict(), output / "model.pt")
        prediction_start = time.perf_counter()
        for split in ["train", "validation"]:
            x = view[split]
            ids = view["train_ids" if split == "train" else "validation_ids"]
            occ = _save_predictions(output / f"{split}_predictions.npz", responsibilities(model, x, device, seed=seed + 1000), ids)
            occupancy.append({"clean_equivalent_epoch": protocol["training_epochs"], "split": split, "endpoint": True, **occ})
        write_json(output / "occupancy_diagnostics.json", occupancy)
        np.save(output / "batch_lengths.npy", np.asarray(lengths, dtype=np.int32))
        np.savez_compressed(output / "sample_exposure.npz", sample_ids=view["train_ids"], count=exposure)
        # Restore complete committed audit logs even when resuming at the
        # optimization endpoint and only replaying deterministic postprocessing.
        for filename, value in [("training_curves.json", curves), ("gradient_diagnostics.json", diagnostics),
                                ("numerical_checks.json", checks)]:
            write_json(output / filename, value)
        finish_run(output, record, optimizer_updates=budget, numerical_gate_passed=True,
                   training_seconds=training_seconds, final_gmm=gmm_record, prediction_seconds=time.perf_counter() - prediction_start,
                   timing_breakdown=timings,
                   timing_semantics="synchronized update cost includes device transfer; total training also includes diagnostics, target refresh, checkpoint I/O and Python overhead",
                   clipped_updates=sum(r["clipped_updates"] for r in curves), hardware=_hardware(device),
                   labels_loaded=False, test_partition_loaded=False, fixed_endpoint=True)
    except Exception as error:
        finish_run(output, record, status="failed", error=repr(error), numerical_gate_passed=False, hardware=_hardware(device),
                   numerical_failure=isinstance(error, ArithmeticError),
                   recovery="Resume last valid boundary into a new attempt; numerical failures remain rejected")
        raise
    return output


def corrupt_inputs(x, ids, *, dataset, seed, condition):
    """Paired image perturbations keyed by sorted sample IDs, never labels."""
    if condition == "clean": return x.clone()
    if condition not in INFERENCE_CONDITIONS or dataset not in PROTOCOL["datasets"][:3]:
        raise ValueError("Unknown or unapproved inference corruption")
    values = x.numpy()
    if values.min() < 0 or values.max() > 1: raise ValueError("Image corruptions require [0,1] inputs")
    key = int.from_bytes(hashlib.sha256(f"{dataset}|{condition}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261028, key]))
    order = np.argsort(ids, kind="stable")
    canonical = values[order]
    if condition.startswith("gaussian_sd"):
        corrupted = np.clip(canonical + rng.normal(0, float(condition.removeprefix("gaussian_sd")), canonical.shape), 0, 1)
    else:
        corrupted = canonical * (rng.random(canonical.shape) >= float(condition.removeprefix("dropout")))
    restored = np.empty_like(values); restored[order] = corrupted.astype(np.float32)
    return torch.from_numpy(restored)


def predict(manifest_path, fitted, output, *, split="test", device="cuda", corruption="clean"):
    fitted = Path(fitted)
    metadata = json.loads((fitted / "run.json").read_text())
    config = json.loads((fitted / "config.json").read_text())
    if metadata.get("status") != "completed" or config.get("phase") != "fixed_final_matched_fit" or not metadata.get("numerical_gate_passed"):
        raise ValueError("Completed numerically accepted fixed final fit required")
    if sha256(fitted / "config.json") != metadata["config_sha256"] or config["protocol_sha256"] != protocol_digest(config["protocol"]):
        raise ValueError("Fitted protocol/config hash mismatch")
    for source in sources():
        if metadata["source_file_sha256"].get(str(source.relative_to(ROOT))) != sha256(source):
            raise ValueError("Prediction source release differs from fitted release")
    if sha256(Path(manifest_path)) != config["clean_manifest_sha256"] or sha256(fitted / "model.pt") != metadata["artifact_sha256"]["model.pt"]:
        raise ValueError("Prediction manifest/checkpoint hash mismatch")
    if split not in {"train", "validation", "test"}: raise ValueError("Invalid split")
    if corruption != "clean" and (config["condition"] != "clean" or split != "test"):
        raise ValueError("Approved inference corruptions use clean-trained models on test inputs")
    device = setup(config["seed"], device)
    manifest = json.loads(Path(manifest_path).read_text())
    arrays = load_inputs(manifest_path, split)
    x, ids = _validate_arrays(arrays, manifest["n_features"])
    x = corrupt_inputs(x, ids, dataset=manifest["dataset"], seed=config["seed"], condition=corruption)
    model = model_for(config["method"], manifest, device, config["protocol"])
    model.load_state_dict(torch.load(fitted / "model.pt", map_location=device, weights_only=True))
    prediction_config = {"protocol_version": PROTOCOL_VERSION, "phase": "fixed_checkpoint_prediction", "split": split,
                         "model_sha256": sha256(fitted / "model.pt"), "fitted_run_sha256": sha256(fitted / "run.json"),
                         "clean_manifest_sha256": sha256(Path(manifest_path)), "labels_loaded": False,
                         "seed": config["seed"], "method": config["method"], "protocol_sha256": config["protocol_sha256"],
                         "corruption": corruption, "evaluated_input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                         "sample_ids_sha256": _ids_hash(ids)}
    output = Path(output)
    record = start_run(output, stage="validation" if config["validation_only"] else "final_benchmark", config=prediction_config, sources=sources())
    try:
        check = quadrature_check(model, x, device, config["protocol"])
        write_json(output / "numerical_checks.json", {"quadrature": check})
        if check is not None and not check["passed"]: raise ArithmeticError("Prediction quadrature gate failed")
        _save_predictions(output / "predictions.npz", responsibilities(model, x, device, seed=config["seed"] + 1000), ids)
        finish_run(output, record, labels_loaded=False, hardware=_hardware(device), checkpoint_sha256=prediction_config["model_sha256"])
    except Exception as error:
        finish_run(output, record, status="failed", error=repr(error)); raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["initialize", "train", "predict"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-view", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--fitted", type=Path)
    parser.add_argument("--method", choices=PROTOCOL["methods"])
    parser.add_argument("--seed", type=int, choices=PROTOCOL["seeds"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume-from", type=Path, help="Same-release epoch checkpoint; requires a NEW output attempt")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--corruption", choices=["clean", *INFERENCE_CONDITIONS], default="clean")
    args = parser.parse_args()
    if args.phase == "predict":
        if args.resume_from is not None: parser.error("Resume is only supported for initialize/train")
        if args.fitted is None: parser.error("predict requires --fitted")
        predict(args.manifest, args.fitted, args.output, split=args.split, device=args.device, corruption=args.corruption)
    else:
        if args.seed is None: parser.error("initialize/train require --seed")
        kwargs = dict(seed=args.seed, device=args.device, training_view=args.training_view, resume_from=args.resume_from)
        if args.phase == "initialize": initialize(args.manifest, args.output, **kwargs)
        else:
            if args.initial is None or args.method is None: parser.error("train requires --initial and --method")
            train(args.manifest, args.initial, args.output, method=args.method, **kwargs)


if __name__ == "__main__": main()
