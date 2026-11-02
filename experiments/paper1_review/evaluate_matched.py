"""CPU-only, write-once evaluation of the frozen matched campaign.

All available endpoints pass an anchored, label-blind preflight before the first
label read. Integrity violations abort even in partial mode. Missing/running/
failed jobs stay in a ledger; incomplete five-seed estimates are withheld.
No training code is imported, no checkpoint is deserialized, and no fit runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import t as student_t
import sklearn

from .data import _ids_hash, load_inputs, load_evaluation_labels
from .final_protocol import (PROTOCOL, PROTOCOL_VERSION, CONTAMINATION_CONDITIONS,
                             CONTAMINATION_METHODS, INFERENCE_CONDITIONS, protocol_digest)
from .metrics import clustering_metrics
from .paired_metrics import PredictionBatch, paired_condition_metrics, ROBUSTNESS_SCORES
from .provenance import sha256, write_json
from .workload import DATASETS, METHODS, SEEDS

SCHEMA = "paper1-matched-evaluation-v1"
ARTIFACT_HASH_WORKERS = 4
SOURCE_FILES = {
    "experiments/paper1_review/" + n for n in
    ("final_runner.py", "final_protocol.py", "final_diagnostics.py", "models.py",
     "site_parameterization.py", "data.py", "workload.py", "provenance.py")
} | {"experiments/paper1_structured_inference/" + n for n in ("structured.py", "gaussian_reference.py")}
CORE_CONTRASTS = [
    ("prior_family_under_gaussian_q", "gaussian_q_student_prior", "gaussian_q_gaussian_prior"),
    ("direct_recognition_family", "student_q_student_prior", "gaussian_q_student_prior"),
    ("structured_prior_family", "structured_q_student_prior", "structured_q_gaussian_prior"),
    ("structured_vs_direct_student", "structured_q_student_prior", "student_q_student_prior"),
] + [(f"{m}_vs_vae_gmm", m, "vae_gmm") for m in
     ("vade_matched", "dec_matched", "idec_matched", "pro_t_matched")]
CAVEATS = [
    "ARI is primary. ACC, arithmetic NMI and AMI are mandatory companions; no metric or seed selection.",
    "Five seeds are descriptive replications. Paired Student-t intervals use df=4, are unadjusted descriptive uncertainty, and are not significance tests or population guarantees. No p-values are computed.",
    "No cross-dataset pooling. Incomplete five-seed groups and paired contrasts have no aggregate estimate.",
    "Delta is condition minus clean; Gamma is delta(A)-delta(B), positive favoring A's change, not necessarily its condition score.",
    "Inference corruption reuses one checkpoint and reports both remapped and clean-map-frozen scores. Retrained model cluster IDs are never aligned by numeric ID.",
    "Mapped F1/recall use the displayed ACC-optimal map. Responsibility entropy is not calibrated semantic uncertainty; no class log-loss/Brier score.",
    "V-measure and arithmetic NMI coincide; balanced accuracy is mean class recall. These are not independent corroborating outcomes.",
    "This is the matched protocol, separate from the twelve source-recipe reference fits. VAE+GMM is its registered endpoint GMM, not a retrospectively chosen refit of every model.",
    "Fixed update budgets need not imply equal runtime or examples seen under appended contamination. Costs/exposure are reported separately; no cost-matched superiority claim.",
    "Attained quadrature checks cover saved diagnostic observations, not a certified continuous ELBO or uniform integration error bound.",
    "Finite df bounds [3,30] on Student priors are not a test of the Gaussian limit. Boundary/gradient diagnostics retain their original scope.",
    "Finite nonnegative responsibilities within the existing 1e-5 native/float64 row-mass tolerance are normalized in float64 for diagnostics; original files and hard assignments remain unchanged. Per-artifact roundoff and adjustment sizes are retained in evidence.",
]


def _json(path):
    def reject(value): raise ValueError(f"Nonfinite JSON constant {value}")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): reject(value)
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, list):
            for item in value: finite(item)
    value = json.loads(Path(path).read_text(), parse_constant=reject)
    finite(value)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _safe(root, name):
    p = (Path(root) / name).resolve()
    p.relative_to(Path(root).resolve())
    return p


def _anchor(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(Path(path)) != expected:
        raise ValueError(f"Trusted anchor hash mismatch: {path}")
    return _json(path)


class Paths:
    """Longest-prefix directory remapping; recorded remote identity is preserved."""
    def __init__(self, mappings=()):
        self.pairs = []
        for item in mappings:
            old, new = item.split("=", 1) if isinstance(item, str) else item
            old, new = Path(old), Path(new).resolve()
            if not old.is_absolute(): raise ValueError("Remote root must be absolute")
            if any(old == p[0] for p in self.pairs): raise ValueError("Duplicate path-map root")
            self.pairs.append((old, new))
        self.pairs.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    def __call__(self, value):
        p = Path(value)
        if not p.is_absolute() or ".." in p.parts: raise ValueError("Absolute canonical evidence path required")
        for old, new in self.pairs:
            if p.is_relative_to(old): return _safe(new, str(p.relative_to(old)))
        return p.resolve()


def validate_campaign(m):
    """Check the entire registered allocation, not only counts supplied by caller."""
    if m.get("schema") != "paper1-air-matched-campaign-v1":
        raise ValueError("Wrong campaign identity")
    if m.get("protocol_sha256") != protocol_digest() or m.get("labels_loaded") is not False:
        raise ValueError("Campaign protocol or label-blind declaration changed")
    expected = {(d, method, s, "clean") for d, method, s in product(DATASETS, METHODS, SEEDS)}
    expected |= {(d, method, s, c) for d, method, s, c in
                 product(DATASETS[:2], CONTAMINATION_METHODS, SEEDS, CONTAMINATION_CONDITIONS)}
    trains, initials, preds = m["trains"], m["initializations"], m["predictions"]
    keys = [(r["dataset"], r["method"], r["seed"], r["condition"]) for r in trains]
    if len(keys) != 300 or set(keys) != expected or len(initials) != 40 or len(preds) != 300:
        raise ValueError("Campaign roster differs from registered 300/40 allocation")
    if len({r["run_id"] for r in trains}) != 300: raise ValueError("Duplicate logical run ID")
    if set(m["inputs_sha256"]) != set(DATASETS): raise ValueError("Dataset anchors missing")
    seen, init_keys = [], set()
    for i, row in enumerate(initials):
        if row["index"] != i: raise ValueError("Initialization index mismatch")
        key = (row["dataset"], row["condition"], row["seed"])
        if key in init_keys: raise ValueError("Duplicate shared initializer")
        init_keys.add(key); seen.append(row["output"])
    for i, (row, pred) in enumerate(zip(trains, preds)):
        initial = initials[row["initial_index"]]
        if row["index"] != i or pred["index"] != i or pred["run_id"] != row["run_id"]:
            raise ValueError("Campaign fit/prediction index mismatch")
        if row["initial"] != initial["output"] or pred["fitted"] != row["output"] or pred["manifest"] != row["manifest"]:
            raise ValueError("Campaign dependency mismatch")
        for name in ("manifest", "training_view", "dataset", "condition", "seed"):
            if row[name] != initial[name]: raise ValueError("Initialization group mismatch")
        conditions = ["clean", *INFERENCE_CONDITIONS] if row["condition"] == "clean" and row["dataset"] in DATASETS[:3] else ["clean"]
        if pred["conditions"] != conditions or set(pred["outputs"]) != set(conditions):
            raise ValueError("Prediction condition allocation changed")
        seen += [row["output"], *pred["outputs"].values()]
    if len(set(seen)) != 1540: raise ValueError("Duplicate campaign output directories")
    if any(not Path(p).is_absolute() or not Path(p).is_relative_to(m["output_root"]) or ".." in Path(p).parts for p in seen):
        raise ValueError("Output outside campaign root")


class Verifier:
    def __init__(self, campaign, release, audit, paths, overrides):
        self.campaign, self.release, self.paths = campaign, release, paths
        if release.get("protocol") != PROTOCOL or release.get("protocol_sha256") != protocol_digest():
            raise ValueError("Release protocol differs from frozen matched protocol")
        self.sources = release.get("source_file_sha256", {})
        if set(self.sources) != SOURCE_FILES: raise ValueError("Release source inventory incomplete or changed")
        self.audit = {}
        self.audit_all_verified = audit.get("all_recorded_completed_artifacts_verified", False)
        for row in audit["runs"]:
            directory = row.get("directory", row.get("path"))
            if not directory or directory in self.audit: raise ValueError("Audit directory missing/duplicated")
            self.audit[directory] = row
        self.overrides = overrides or {}
        valid_paths = {r["output"] for r in campaign["initializations"] + campaign["trains"]}
        valid_paths |= {p for r in campaign["predictions"] for p in r["outputs"].values()}
        if not set(self.overrides) <= valid_paths: raise ValueError("Override for unknown logical output")
        targets = [v["directory"] for v in self.overrides.values()]
        if len(set(targets)) != len(targets): raise ValueError("Override attempts reused")
        self.cache, self.datasets, self.views, self.input_hashes = {}, {}, {}, {}
        self.ledger = []

    def chosen(self, original):
        entry = self.overrides.get(original)
        if entry:
            if not entry.get("reason") or not entry.get("audit_run_sha256"):
                raise ValueError("Attempt override needs reason and independently audited run hash")
            selected = entry["directory"]
            if self.audit.get(selected, {}).get("run_sha256") != entry["audit_run_sha256"]:
                raise ValueError("Override audit hash mismatch")
            return selected
        return original

    def _audit_hashes(self, remote, run=None):
        evidence = self.audit.get(remote)
        if evidence is None: raise ValueError(f"Completed/parent run lacks trusted completion audit: {remote}")
        hashes = evidence.get("artifact_sha256", evidence.get("recorded_run", {}).get("artifact_sha256", {}))
        # The global declaration explicitly covers completed runs, not an
        # interrupted parent whose artifacts need an independent per-run audit.
        fallback = self.audit_all_verified if run is not None and run.get("status") == "completed" else False
        if not hashes or evidence.get("all_artifacts_verified", evidence.get("verified", fallback)) is not True:
            raise ValueError("Independent audit does not attest artifact hashes")
        if run is not None and run.get("status") == "completed" and hashes != run.get("artifact_sha256"):
            raise ValueError("Trusted artifact inventory differs from completed run")
        return evidence, hashes

    def artifact(self, value, name, required=True):
        p = _safe(value["directory"], name)
        digest = value["hashes"].get(name)
        if not digest: raise ValueError(f"Unrecorded artifact {name}")
        if not p.is_file():
            if required: raise FileNotFoundError(p)
            return None
        if sha256(p) != digest: raise ValueError(f"Artifact changed: {p}")
        return p

    def base(self, remote, *, completed=True):
        key = (remote, completed)
        if key in self.cache: return self.cache[key]
        directory = self.paths(remote); run = _json(directory / "run.json")
        evidence, hashes = self._audit_hashes(remote, run)
        if sha256(directory / "run.json") != evidence.get("run_sha256"):
            raise ValueError("Run record differs from trusted completion audit")
        if (run.get("fresh_run") is not True or run.get("stage") != "final_benchmark" or
            run.get("legacy_outputs_used") is not False or run.get("protocol_version") != PROTOCOL_VERSION):
            raise ValueError("Wrong evidence stage/freshness/protocol")
        if completed and (run.get("status") != "completed" or run.get("labels_loaded") is not False):
            raise ValueError("Completed label-blind endpoint required")
        if run.get("numerical_failure") is True: raise ValueError("Rejected numerical lineage cannot be scored/resumed")
        if run.get("source_file_sha256") != self.sources or run.get("source_sha256") != _digest(self.sources):
            raise ValueError("Source inventory changed from trusted release")
        value = dict(remote=remote, directory=directory, run=run, hashes=hashes,
                     run_sha256=evidence["run_sha256"], locally_verified=[], remote_only=[])
        # Validate the entire inventory before any worker can open a file.
        # Resolve aliases as well as literal prefixes so a late malicious entry
        # cannot race ahead of this label barrier in a hashing worker.
        validated = []
        for name, digest in hashes.items():
            p = _safe(directory, name)
            relative = p.relative_to(directory)
            if (name.startswith("evaluation/") or
                    relative.parts and relative.parts[0] == "evaluation"):
                raise ValueError("Unexpected evaluation file in learner run")
            validated.append((name, digest, p))
        config = _json(self.artifact(value, "config.json"))
        if sha256(directory / "config.json") != run["config_sha256"] or config.get("labels_loaded") is not False:
            raise ValueError("Config hash/labels declaration mismatch")
        value["config"] = config
        # Only independent byte hashing runs concurrently. Results, exceptions
        # and evidence are consumed in the original inventory order, on this
        # thread. Every present artifact is read; no stat or digest cache exists.
        with ThreadPoolExecutor(max_workers=ARTIFACT_HASH_WORKERS) as pool:
            pending = {name: pool.submit(sha256, p) for name, _, p in validated if p.is_file()}
            for name, digest, p in validated:
                if name in pending:
                    if pending[name].result() != digest: raise ValueError(f"Artifact hash mismatch: {p}")
                    value["locally_verified"].append(name)
                else:
                    optional = name in {"latest_checkpoint.pt", "previous_checkpoint.pt"} or (name.startswith("checkpoints/") and name.endswith(".pt"))
                    if completed and not optional: raise FileNotFoundError(p)
                    value["remote_only"].append(name)
        for name, digest in self.sources.items():
            if hashes.get("source/" + name) != digest: raise ValueError("Source snapshot mismatch")
            self.artifact(value, "source/" + name)
        self.cache[key] = value
        return value

    def available(self, original, kind):
        remote = self.chosen(original); directory = self.paths(remote)
        if not (directory / "run.json").exists():
            self.ledger.append(dict(kind=kind, logical_output=original, attempt=remote, status="missing")); return None
        run = _json(directory / "run.json")
        if run.get("status") != "completed":
            self.ledger.append(dict(kind=kind, logical_output=original, attempt=remote,
                                    status=run.get("status", "unknown"), error=run.get("error"),
                                    numerical_failure=run.get("numerical_failure", False)))
            return None
        return self.base(remote)

    def dataset(self, path, dataset):
        if path in self.datasets: return self.datasets[path]
        local = self.paths(path)
        if sha256(local) != self.campaign["inputs_sha256"][dataset]: raise ValueError("Dataset manifest changed")
        meta = _json(local)
        if meta.get("dataset") != dataset: raise ValueError("Wrong dataset identity")
        result = dict(path=local, metadata=meta, splits={})
        for split in ("train", "validation", "test"):
            arrays = load_inputs(local, split); x, ids = arrays["x"], arrays["ids"]
            _unique_ids(ids)
            if not np.isfinite(x).all() or x.reshape(len(x), -1).shape[1] != meta["n_features"]:
                raise ValueError("Nonfinite/wrong-shaped inputs")
            result["splits"][split] = dict(ids=ids, count=len(ids), x_hash=hashlib.sha256(x.reshape(len(x), -1).tobytes()).hexdigest())
        id_sets = [set(v["ids"].tolist()) for v in result["splits"].values()]
        if any(a & b for a, b in combinations(id_sets, 2)): raise ValueError("Data split ID overlap")
        self.datasets[path] = result
        return result

    def view(self, row):
        key = (row["manifest"], row["training_view"])
        if key in self.views: return self.views[key]
        data = self.dataset(row["manifest"], row["dataset"])
        result = dict(data=data, sha256=self.campaign["inputs_sha256"][row["dataset"]],
                      splits={k: data["splits"][k] for k in ("train", "validation")}, manifest=None)
        if row["condition"] != "clean":
            p = self.paths(row["training_view"]); meta = _json(p)
            if (meta.get("status") != "completed" or meta.get("schema_version") != "paper1-review-training-view-v1" or
                any(meta.get(k) != row[k] for k in ("dataset", "condition", "seed")) or
                meta.get("clean_manifest_sha256") != result["sha256"] or meta.get("labels_loaded") is not False or
                meta.get("test_inputs_loaded") is not False or meta.get("model_weights_loaded") is not False or
                meta.get("source_split") != "train"):
                raise ValueError("Invalid contaminated view provenance")
            source_dataset = row["dataset"] if row["condition"].startswith("gaussian") else {"mnist":"fashion_mnist","fashion_mnist":"mnist"}[row["dataset"]]
            if (meta.get("source_dataset") != source_dataset or meta.get("appended_to_clean_ratio") != .1 or
                meta.get("clean_train_count") != data["splits"]["train"]["count"] or
                any(meta.get(k) != data["metadata"][k] for k in ("n_features","n_clusters"))):
                raise ValueError("Contamination source, dose or dimensions changed")
            if meta.get("source_manifest_sha256") != self.campaign["inputs_sha256"].get(meta.get("source_dataset")):
                raise ValueError("Contaminant source hash mismatch")
            splits = {}
            for split in ("train", "validation"):
                entry = meta["splits"][split]; file = _safe(p.parent, entry["inputs"])
                if sha256(file) != meta["artifact_sha256"][entry["inputs"]]: raise ValueError("View input hash mismatch")
                with np.load(file, allow_pickle=False) as a:
                    if set(a.files) != {"x", "ids"}: raise ValueError("View inputs contain unexpected fields")
                    ids, x = a["ids"], a["x"]
                _unique_ids(ids)
                if (len(ids) != entry["n_samples"] or _ids_hash(ids) != entry["ids_sha256"] or x.dtype != np.float32 or
                    list(x.shape) != entry["input_shape"] or x.shape[0] != len(ids) or
                    x.reshape(len(x),-1).shape[1] != data["metadata"]["n_features"] or not np.isfinite(x).all()):
                    raise ValueError("View inputs invalid")
                clean = load_inputs(data["path"], split); order = _align_ids(ids, clean["ids"], subset=True)
                if not np.array_equal(x[order], clean["x"]): raise ValueError("Shared clean inputs changed in contaminated view")
                counts = meta["realized_counts"][split]
                if counts.get("clean") != len(clean["ids"]) or counts.get("contaminants") != len(ids)-len(clean["ids"]):
                    raise ValueError("View realized counts mismatch")
                splits[split] = dict(ids=ids, count=len(ids))
            if set(splits["train"]["ids"]) & set(splits["validation"]["ids"]): raise ValueError("View split overlap")
            total_clean = sum(data["splits"][s]["count"] for s in ("train","validation"))
            if sum(v["count"] for v in splits.values()) - total_clean != total_clean // 10:
                raise ValueError("Actual appended contamination dose changed")
            result.update(sha256=sha256(p), splits=splits, manifest=meta, path=p)
        self.views[key] = result
        return result

    def config(self, value, row, phase):
        c, view = value["config"], self.view(row)
        n = view["data"]["splits"]["train"]["count"]
        budget = math.ceil(n / PROTOCOL["batch_size"]) * PROTOCOL["pretrain_epochs" if phase == "initialize" else "training_epochs"]
        expected = dict(protocol_version=PROTOCOL_VERSION, protocol_sha256=protocol_digest(), protocol=PROTOCOL,
                        phase="fresh_final_shared_initialization" if phase == "initialize" else "fixed_final_matched_fit",
                        seed=row["seed"], dataset=row["dataset"], condition=row["condition"], clean_train_count=n,
                        clean_manifest_sha256=self.campaign["inputs_sha256"][row["dataset"]], training_view_sha256=view["sha256"],
                        training_view_manifest=row["training_view"] or row["manifest"],
                        training_count=view["splits"]["train"]["count"], validation_count=view["splits"]["validation"]["count"],
                        validation_only=False, labels_loaded=False, test_partition_loaded=False,
                        native_reproduction=False, initialization_from_development=False, optimizer_updates=budget)
        for split in ("train", "validation"):
            expected["training_ids_sha256" if split == "train" else "validation_ids_sha256"] = _ids_hash(view["splits"][split]["ids"])
        if phase != "initialize": expected["method"] = row["method"]
        if any(c.get(k) != v for k, v in expected.items()): raise ValueError("Run configuration/seed/view/budget differs from frozen campaign")
        if value["run"].get("optimizer_updates") != budget or value["run"].get("test_partition_loaded") is not False:
            raise ValueError("Declared final update budget not completed")
        ids = view["data"]["splits"]["train"]["ids"]
        order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(str(ids[i]).encode()).digest())
        if c.get("common_gradient_ids") != ids[order[:32]].tolist(): raise ValueError("Common gradient batch changed")
        value.update(view=view, budget=budget)
        self.exposure(value)
        self.lineage(value, row["output"])
        return value

    def exposure(self, value):
        c = value["config"]; lengths = np.load(self.artifact(value, "batch_lengths.npy"), allow_pickle=False)
        order = np.memmap(self.artifact(value, "sample_order.int32"), dtype="<i4", mode="r")
        if (lengths.ndim != 1 or lengths.dtype.kind not in "iu" or len(lengths) != value["budget"] or
            (lengths <= 0).any() or (lengths > PROTOCOL["batch_size"]).any() or int(lengths.sum()) != len(order) or
            (order < 0).any() or (order >= c["training_count"]).any()):
            raise ValueError("Update lengths/sample-order invalid or short budget")
        with np.load(self.artifact(value, "sample_exposure.npz"), allow_pickle=False) as a:
            if set(a.files) != {"sample_ids", "count"}: raise ValueError("Exposure schema invalid")
            if not np.array_equal(a["sample_ids"], value["view"]["splits"]["train"]["ids"]) or not np.array_equal(a["count"], np.bincount(order, minlength=c["training_count"])):
                raise ValueError("Sample exposure differs from committed order")
            counts = a["count"]
        value["lengths"] = lengths
        value["exposure"] = dict(examples_seen=int(lengths.sum()), minimum=int(counts.min()), maximum=int(counts.max()),
                                  mean=float(counts.mean()), sample_order_sha256=value["hashes"]["sample_order.int32"])

    def lineage(self, value, original, seen=None):
        seen = set() if seen is None else seen
        remote = value["remote"]
        if remote in seen: raise ValueError("Cyclic resume lineage")
        seen.add(remote)
        depname = "resume_dependency.json"
        if depname not in value["hashes"]:
            if remote != original: raise ValueError("Training override lacks audited resume lineage")
            value["lineage"] = []; return
        dep = _json(self.artifact(value, depname)); checkpoint = self.paths(dep["checkpoint"])
        parent_remote = str(Path(dep["checkpoint"]).parent)
        parent = self.base(parent_remote, completed=False)
        if (parent["run"].get("status") == "completed" or parent["config"] != value["config"] or
            dep["parent_run_sha256"] != parent["run_sha256"] or
            parent["run"].get("source_sha256") != value["run"]["source_sha256"]):
            raise ValueError("Resume parent/source/config identity mismatch")
        receiptname = checkpoint.with_suffix(".json").name
        receipt = _json(self.artifact(parent, receiptname))
        if (receipt.get("sha256") != dep["checkpoint_sha256"] or
            parent["hashes"].get(checkpoint.name) != dep["checkpoint_sha256"] or receipt.get("resumable") is not True or
            receipt.get("phase") not in {"epoch_boundary", "optimization_complete"} or
            receipt.get("source_sha256") != value["run"]["source_sha256"] or
            receipt.get("completed_updates") != dep["completed_updates"]):
            raise ValueError("Resume checkpoint receipt/audit mismatch")
        if checkpoint.exists() and sha256(checkpoint) != dep["checkpoint_sha256"]: raise ValueError("Resume checkpoint bytes changed")
        u = dep["completed_updates"]; per_epoch = math.ceil(value["config"]["clean_train_count"] / PROTOCOL["batch_size"])
        if type(u) is not int or not 0 <= u <= value["budget"] or u % per_epoch:
            raise ValueError("Resume update is not a valid epoch boundary")
        if receipt["phase"] == "optimization_complete" and u != value["budget"]: raise ValueError("Forged completed-optimization receipt")
        if dep.get("logical_fit_id") != parent["run"].get("logical_fit_id", parent_remote): raise ValueError("Logical fit lineage mismatch")
        size = int(value["lengths"][:u].sum()) * 4
        parent_order = self.artifact(parent, "sample_order.int32")
        child_order = self.artifact(value, "sample_order.int32")
        if _prefix(parent_order, size) != _prefix(child_order, size): raise ValueError("Resume committed sample-order prefix changed")
        parent.update(budget=value["budget"], lengths=value["lengths"])
        self.lineage(parent, original, seen)
        value["lineage"] = parent["lineage"] + [dict(parent_directory=parent_remote, parent_run_sha256=parent["run_sha256"],
            checkpoint_sha256=dep["checkpoint_sha256"], checkpoint_verified_locally=checkpoint.exists(),
            completed_updates=u, committed_prefix_bytes=size, committed_prefix_sha256=_prefix(child_order, size),
            note="Source-pinned loader restores optimizer/RNG/sampler; evaluator validates receipts and byte prefix without deserializing model state")]

    def initialize(self, row):
        value = self.available(row["output"], "initialization")
        if value is None: return None
        self.config(value, row, "initialize")
        if value["run"].get("gmm", {}).get("converged") is not True: raise ValueError("Initial GMM did not converge")
        for name in ("autoencoder.pt", "mixture.npz", "pretrain_curves.json"): self.artifact(value, name)
        curves = _json(value["directory"] / "pretrain_curves.json")
        _curves(curves, PROTOCOL["pretrain_epochs"], value["budget"])
        return value

    def fit(self, row, initial):
        value = self.available(row["output"], "fit")
        if value is None: return None
        if initial is None:
            self.ledger.append(dict(kind="fit", logical_output=row["output"], status="blocked_missing_initialization")); return None
        self.config(value, row, "train")
        if (value["config"].get("initial_run_sha256") != initial["run_sha256"] or
            value["config"].get("initial_tensor_sha256") != initial["hashes"]["autoencoder.pt"]):
            raise ValueError("Initialization lineage forged/changed")
        run = value["run"]
        if run.get("fixed_endpoint") is not True or run.get("numerical_gate_passed") is not True: raise ValueError("Endpoint or numerical gate not attained")
        if row["method"] == "vae_gmm" and run.get("final_gmm", {}).get("converged") is not True: raise ValueError("Endpoint VAE GMM not converged")
        self.artifact(value, "model.pt")
        curves = _json(self.artifact(value, "training_curves.json")); _curves(curves, PROTOCOL["training_epochs"], value["budget"])
        checks = _json(self.artifact(value, "numerical_checks.json"))
        if [r["clean_equivalent_epoch"] for r in checks] != PROTOCOL["diagnostic_epochs"]: raise ValueError("Missing quadrature diagnostic endpoints")
        for check in checks: _quadrature(check["quadrature"], row["method"], value["config"]["validation_count"])
        gradients = _json(self.artifact(value, "gradient_diagnostics.json"))
        if [r["clean_equivalent_epoch"] for r in gradients] != PROTOCOL["diagnostic_epochs"]: raise ValueError("Gradient diagnostic endpoints missing")
        if any(r.get("optimizer_update") is not False for r in gradients): raise ValueError("Diagnostics modified optimization")
        components = {"target_kl"} if row["method"] == "dec_matched" else ({"target_kl", "reconstruction"} if row["method"] == "idec_matched" else {"reconstruction", "prior_cross_entropy", "negative_entropy"})
        for r in gradients:
            if set(r.get("components", {})) != components: raise ValueError("Missing objective-component gradient diagnostics")
            if r.get("completed_updates") != r["clean_equivalent_epoch"] * (value["budget"] // PROTOCOL["training_epochs"]):
                raise ValueError("Gradient diagnostic update mismatch")
        normalization = {}
        for split in ("train", "validation"):
            name = f"{split}_predictions.npz"
            prepared = _prediction(self.artifact(value, name), value["view"]["splits"][split]["ids"], value["view"]["data"]["metadata"]["n_clusters"])
            normalization[name] = dict(prepared["probability_normalization"], raw_artifact_sha256=value["hashes"][name])
        value.update(initial=initial, curves=curves, gradient_diagnostics=gradients, numerical_checks=checks,
                     probability_normalization=normalization)
        return value

    def input_hash(self, row, condition):
        key = (row["manifest"], row["seed"], condition)
        if key not in self.input_hashes:
            data = self.dataset(row["manifest"], row["dataset"])
            a = load_inputs(data["path"], "test"); values = a["x"].reshape(len(a["ids"]), -1)
            if condition != "clean":
                if values.min() < 0 or values.max() > 1: raise ValueError("Corruptions require image [0,1] inputs")
                code = int.from_bytes(hashlib.sha256(f"{row['dataset']}|{condition}".encode()).digest()[:4], "little")
                rng = np.random.default_rng(np.random.SeedSequence([row["seed"], 20261028, code]))
                order = np.argsort(a["ids"], kind="stable"); canonical = values[order]
                if condition.startswith("gaussian_sd"):
                    corrupted = np.clip(canonical + rng.normal(0, float(condition.removeprefix("gaussian_sd")), canonical.shape), 0, 1)
                else:
                    corrupted = canonical * (rng.random(canonical.shape) >= float(condition.removeprefix("dropout")))
                restored = np.empty_like(values); restored[order] = corrupted.astype(np.float32); values = restored
            self.input_hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
        return self.input_hashes[key]

    def prediction(self, row, predrow, condition, fitted):
        original = predrow["outputs"][condition]
        value = self.available(original, "prediction")
        if value is None: return None
        if fitted is None:
            self.ledger.append(dict(kind="prediction", logical_output=original, status="blocked_missing_fit")); return None
        c = value["config"]; data = self.dataset(row["manifest"], row["dataset"])
        expected = dict(protocol_version=PROTOCOL_VERSION, protocol_sha256=protocol_digest(), phase="fixed_checkpoint_prediction",
                        split="test", corruption=condition, seed=row["seed"], method=row["method"], labels_loaded=False,
                        model_sha256=fitted["hashes"]["model.pt"], fitted_run_sha256=fitted["run_sha256"],
                        clean_manifest_sha256=self.campaign["inputs_sha256"][row["dataset"]],
                        evaluated_input_sha256=self.input_hash(row, condition), sample_ids_sha256=_ids_hash(data["splits"]["test"]["ids"]))
        if any(c.get(k) != v for k, v in expected.items()): raise ValueError("Prediction/checkpoint/paired corruption input hash mismatch")
        if value["run"].get("checkpoint_sha256") != expected["model_sha256"]: raise ValueError("Prediction checkpoint mismatch")
        _quadrature(_json(self.artifact(value, "numerical_checks.json"))["quadrature"], row["method"], data["splits"]["test"]["count"])
        p = self.artifact(value, "predictions.npz")
        prepared = _prediction(p, data["splits"]["test"]["ids"], data["metadata"]["n_clusters"])
        value.update(prediction_path=p, fitted=fitted, row=row, condition=condition,
                     probability_normalization={"predictions.npz": dict(prepared["probability_normalization"],
                                                                       raw_artifact_sha256=value["hashes"]["predictions.npz"])})
        return value


def _prefix(path, size):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        remaining = size
        while remaining:
            block = f.read(min(remaining, 1024 * 1024))
            if not block: raise ValueError("Truncated sample-order prefix")
            h.update(block); remaining -= len(block)
    return h.hexdigest()


def _unique_ids(ids):
    if ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "USiu" or len(set(ids.tolist())) != len(ids):
        raise ValueError("Sample IDs must be nonempty, unique strings/integers")


def _align_ids(ids, reference, subset=False):
    _unique_ids(ids); _unique_ids(reference)
    if not set(reference.tolist()) <= set(ids.tolist()) or (not subset and len(ids) != len(reference)):
        raise ValueError("Prediction/input IDs differ")
    lookup = {v: i for i, v in enumerate(ids.tolist())}
    return np.array([lookup[v] for v in reference.tolist()], dtype=int)


def _prediction(path, expected_ids, k):
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != {"sample_ids", "predictions", "responsibilities", "cluster_labels"}: raise ValueError("Unexpected prediction schema")
        ids, pred, prob, labels = (a[n] for n in ("sample_ids", "predictions", "responsibilities", "cluster_labels"))
    order = _align_ids(ids, expected_ids)
    if (pred.shape != (len(ids),) or prob.shape != (len(ids), k) or
        not np.array_equal(labels, np.arange(k)) or prob.dtype.kind not in "fiu" or
        not np.isfinite(prob).all() or (prob < 0).any() or
        not np.allclose(prob.sum(1), 1, rtol=0, atol=1e-5) or
        not np.allclose(prob.sum(1, dtype=np.float64), 1, rtol=0, atol=1e-5) or
        not np.array_equal(pred, labels[prob.argmax(1)])):
        raise ValueError("Invalid probability rows/component labels/hard predictions")
    # The frozen runner accepts nonnegative float32 rows within 1e-5 of unit
    # mass. Summed quadrature weights can therefore exceed 1 by roundoff. Keep
    # both the native frozen gate and the existing float64 gate; only normalize
    # those accepted arrays, without changing the stored bytes or assignments.
    raw = prob.astype(np.float64)
    mass = raw.sum(1)
    normalized = raw / mass[:, None]
    if (not np.isfinite(normalized).all() or (normalized < 0).any() or (normalized > 1).any() or
            not np.array_equal(pred, labels[normalized.argmax(1)])):
        raise ValueError("Probability normalization changed hard predictions or validity")
    evidence = dict(policy="float64 row normalization after native and float64 mass gates",
                    raw_dtype=str(prob.dtype), normalization_dtype="float64", row_sum_atol=1e-5,
                    raw_minimum=float(raw.min()), raw_maximum=float(raw.max()),
                    raw_native_max_row_sum_error=float(np.abs(prob.sum(1)-1).max()),
                    raw_float64_max_row_sum_error=float(np.abs(mass-1).max()),
                    raw_entries_above_one=int((raw > 1).sum()),
                    max_absolute_adjustment=float(np.abs(normalized-raw).max()),
                    changed_rows=int(np.any(normalized != raw, axis=1).sum()),
                    normalized_max_row_sum_error=float(np.abs(normalized.sum(1)-1).max()),
                    raw_argmax_matches_saved=True, normalized_argmax_unchanged=True,
                    original_artifact_modified=False)
    return dict(ids=ids[order], predictions=pred[order], probabilities=normalized[order], cluster_labels=labels,
                probability_normalization=evidence)


def _curves(rows, epochs, budget):
    if len(rows) != epochs or [r.get("clean_equivalent_epoch") for r in rows] != list(range(1, epochs + 1)):
        raise ValueError("Training history incomplete")
    per_epoch = budget // epochs
    if [r.get("updates") for r in rows] != [i * per_epoch for i in range(1, epochs + 1)]: raise ValueError("Training history budget mismatch")
    for row in rows:
        if any(isinstance(v, (int, float)) and not np.isfinite(v) for v in row.values()): raise ValueError("Nonfinite training history")


def _quadrature(q, method, n_observations):
    if method != "structured_q_student_prior":
        if q is not None: raise ValueError("Unexpected quadrature gate")
        return
    if not q or any(q.get(k) is not True for k in ("passed", "training_nodes_pass", "prediction_nodes_pass", "reference_refinement_pass")):
        raise ValueError("Structured quadrature gate failed/missing")
    s = PROTOCOL["structured"]
    variants = [(128, s["bounds"]), (256, s["bounds"]), (512, s["reference_bounds"]), (1024, s["refined_reference_bounds"])]
    if [(r["nodes"], r["bounds"]) for r in q["rows"]] != variants: raise ValueError("Quadrature grid changed")
    limits = dict(log_normalizer=s["log_normalizer_atol"], responsibilities=s["responsibility_atol"], kl_to_prior=s["kl_atol"])
    for r in q["rows"]:
        if set(r["errors"]) != set(limits) or any(not np.isfinite(v) or v < 0 for v in r["errors"].values()): raise ValueError("Invalid quadrature errors")
    for index, factor in [(0, 1), (2, s["reference_tolerance_factor"])]:
        if any(q["rows"][index]["errors"][k] > v * factor for k, v in limits.items()): raise ValueError("Quadrature pass flag contradicts recorded errors")
    chosen = np.unique(np.linspace(0,n_observations-1,min(n_observations,32),dtype=int)).tolist()
    if q.get("selected_positions") != chosen: raise ValueError("Quadrature diagnostic positions changed")


def describe(values, *, interval=False):
    """Five prespecified seed replications; no p-value or interval clipping."""
    a = np.asarray(values, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all(): raise ValueError("Exactly five finite paired seed values required")
    sd = float(a.std(ddof=1))
    result = dict(values_by_seed=a.tolist(), mean=float(a.mean()), median=float(np.median(a)),
                  sample_sd=sd, minimum=float(a.min()), maximum=float(a.max()))
    if interval:
        radius = float(student_t.ppf(.975, 4) * sd / np.sqrt(5))
        result.update(descriptive_t_interval_95=[float(a.mean() - radius), float(a.mean() + radius)],
                      interval_note="mean ± t(.975,df=4)*sampleSD/sqrt(5); descriptive, unadjusted, no significance test")
    return result


def aggregate(rows, expected, group_fields, *, score_field="scores", paired=False):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in group_fields)].append(row)
    result = []
    for key in sorted(expected):
        group = groups[key]; seeds = sorted(r["seed"] for r in group)
        if len(set(seeds)) != len(seeds): raise ValueError("Duplicate seed inside aggregate")
        entry = {**dict(zip(group_fields, key)), "n": len(seeds), "seeds": seeds,
                 "missing_seeds": sorted(set(SEEDS) - set(seeds)),
                 "status": "complete_five_seed_group" if seeds == SEEDS else "incomplete_seed_set_aggregate_withheld"}
        if seeds == SEEDS:
            byseed = {r["seed"]: r for r in group}
            keys = set(group[0][score_field])
            if any(set(r[score_field]) != keys for r in group): raise ValueError("Different score panel within group")
            entry["scores"] = {k: describe([byseed[s][score_field][k] for s in SEEDS], interval=paired) for k in sorted(keys)}
        result.append(entry)
    return result


def _cost(value):
    run = value["run"]
    fields = ("optimizer_updates", "neural_seconds", "training_seconds", "prediction_seconds", "timing_breakdown",
              "gmm", "final_gmm", "clipped_updates", "hardware", "timing_semantics", "recovery_cost_note")
    return {**{k: run[k] for k in fields if k in run},
            "sample_exposure": value.get("exposure"), "resumed": bool(value.get("lineage")),
            "total_physical_cost_available": False if value.get("lineage") else None,
            "cost_note": "Saved matched-run neural/training times include committed predecessor time. Do not add parent cumulative time again; discarded work/downtime are not captured. Scheduler attempt accounting is required for total physical cost. Missing timing is unavailable, not zero.",
            "shared_initialization_not_charged_per_method": True}


def _public(value):
    return dict(directory=str(value["directory"]), recorded_remote_directory=value["remote"], run_sha256=value["run_sha256"],
                logical_identity={k:value["config"][k] for k in ("phase","dataset","method","seed","condition","corruption","split") if k in value["config"]},
                config_sha256=value["run"]["config_sha256"], source_sha256=value["run"]["source_sha256"],
                locally_verified_artifacts=value["locally_verified"], remote_only_artifacts=value["remote_only"],
                resume_lineage=value.get("lineage", []), cost=_cost(value),
                probability_normalization=value.get("probability_normalization", {}))


def origin_allocation(verifier, fitted):
    """Origin-stratified training allocation, never semantic accuracy of contaminants."""
    view = fitted["view"]
    if view["manifest"] is None: return None
    meta, p = view["manifest"], view["path"]
    name = "evaluation/train_origin.npz"; path = _safe(p.parent, name)
    if not path.exists(): return dict(status="pending_origin_artifact", path=str(path))
    if sha256(path) != meta["artifact_sha256"].get(name): raise ValueError("Training origin hash changed")
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != {"ids", "is_contaminant", "source_ids"}: raise ValueError("Training origin schema invalid")
        ids, mask, parents = a["ids"], a["is_contaminant"], a["source_ids"]
    target_ids = view["splits"]["train"]["ids"]
    order = _align_ids(ids, target_ids); mask, parents = mask[order], parents[order]
    if mask.dtype != bool or mask.shape != (len(ids),) or parents.shape != mask.shape: raise ValueError("Invalid origin flags")
    clean_ids = set(view["data"]["splits"]["train"]["ids"].tolist())
    if set(target_ids[~mask].tolist()) != clean_ids or not np.array_equal(parents[~mask], target_ids[~mask]):
        raise ValueError("Clean origin identity mismatch")
    expected = meta["realized_counts"]["train"]
    if int(mask.sum()) != expected["contaminants"] or int((~mask).sum()) != expected["clean"]:
        raise ValueError("Origin counts disagree with prepared view")
    source_path = verifier.paths(meta["source_manifest"])
    if sha256(source_path) != meta["source_manifest_sha256"]: raise ValueError("Origin source manifest changed")
    eligible = load_inputs(source_path,"train")["ids"]
    if len(set(parents[mask].tolist())) != int(mask.sum()) or not set(parents[mask].tolist()) <= set(eligible.tolist()):
        raise ValueError("Contaminant parents are duplicated or outside the source training pool")
    prediction = _prediction(verifier.artifact(fitted, "train_predictions.npz"), target_ids, view["data"]["metadata"]["n_clusters"])
    clusters = []
    for label in prediction["cluster_labels"]:
        assigned = prediction["predictions"] == label
        n, c = int(assigned.sum()), int((assigned & mask).sum())
        clusters.append(dict(cluster_label=int(label), total=n, clean=n-c, contaminants=c,
                             contaminant_fraction=c/n if n else None,
                             fraction_of_all_contaminants=c/int(mask.sum()) if mask.any() else None,
                             mean_responsibility_clean=float(prediction["probabilities"][~mask, label].mean()),
                             mean_responsibility_contaminant=float(prediction["probabilities"][mask, label].mean()) if mask.any() else None))
    return dict(status="completed", origin_sha256=sha256(path), training_prediction_sha256=fitted["hashes"]["train_predictions.npz"],
                probability_normalization=dict(prediction["probability_normalization"], raw_artifact_sha256=fitted["hashes"]["train_predictions.npz"]),
                clean_count=int((~mask).sum()), contaminant_count=int(mask.sum()), clusters=clusters,
                interpretation="Descriptive allocation of known training origins. No semantic labels, contamination detection threshold or calibrated probability claim.")


def _write_csv(path, rows, fields):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)


def _finish_analysis(output, provenance, *, inventory_only=False):
    write_json(output / "analysis_manifest.json", dict(schema_version=SCHEMA, status="completed",
               inventory_only=inventory_only,
               artifact_sha256={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}, provenance=provenance))


def evaluate(campaign_path, output, *, campaign_sha256, release_hashes, release_hashes_sha256,
             completion_audit, completion_audit_sha256, path_map=(), overrides=None, overrides_sha256=None,
             allow_incomplete=False, inventory_only=False, all_pairwise=False):
    """Public entry point. Hash anchors must come from independent release/retrieval evidence.

    Overrides JSON maps the original *remote output directory* to directory,
    reason and audit_run_sha256. Initial/train replacements must be continuations
    with a valid receipt/parent/prefix chain back to the original logical output.
    Prediction replacements are deterministic reruns tied to the selected fit.
    """
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    campaign = _anchor(campaign_path, campaign_sha256); validate_campaign(campaign)
    release = _anchor(release_hashes, release_hashes_sha256)
    audit = _anchor(completion_audit, completion_audit_sha256)
    replacements = _anchor(overrides, overrides_sha256) if overrides else {}
    verifier = Verifier(campaign, release, audit, Paths(path_map), replacements)
    output.mkdir(parents=True, exist_ok=False)
    stage = "preflight"
    provenance = dict(campaign_sha256=campaign_sha256, release_hashes_sha256=release_hashes_sha256,
                      completion_audit_sha256=completion_audit_sha256, overrides_sha256=overrides_sha256,
                      path_map=list(path_map), protocol_sha256=protocol_digest(),
                      artifact_hash_workers=ARTIFACT_HASH_WORKERS,
                      artifact_hash_execution="Bounded parallel file reads per uncached run; all inventory paths validated before worker submission; ordered evidence collection; no stat-based digest reuse or omitted artifact reads",
                      implementation_sha256={n: sha256(Path(__file__).with_name(n)) for n in
                                             ("evaluate_matched.py", "metrics.py", "paired_metrics.py", "data.py", "final_protocol.py", "workload.py")},
                      versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                      anchor_note="Hashes are integrity anchors supplied by an independent operator/audit; this is not cryptographic authorship authentication.")
    try:
        initial = {r["index"]: verifier.initialize(r) for r in campaign["initializations"]}
        fits, predictions = {}, {}
        for row, predrow in zip(campaign["trains"], campaign["predictions"]):
            fitted = verifier.fit(row, initial[row["initial_index"]]); fits[row["run_id"]] = fitted
            for condition in predrow["conditions"]:
                value = verifier.prediction(row, predrow, condition, fitted)
                if value is not None: predictions[row["run_id"], condition] = value
        inventory = dict(schema_version=SCHEMA, mode="inventory_only" if inventory_only else "evaluation_preflight", labels_loaded=False,
                         expected=dict(initializations=40, fits=300, predictions=1200),
                         verified=dict(initializations=sum(v is not None for v in initial.values()), fits=sum(v is not None for v in fits.values()), predictions=len(predictions)),
                         missing_or_failed=verifier.ledger, provenance=provenance,
                         initializations=[_public(v) for v in initial.values() if v], fits=[_public(v) for v in fits.values() if v],
                         predictions=[dict(run_id=k[0],inference_condition=k[1],**_public(v)) for k,v in predictions.items()])
        write_json(output / "inventory.json", inventory); write_json(output / "missing_or_failed.json", verifier.ledger)
        if inventory_only:
            _finish_analysis(output,provenance,inventory_only=True)
            return inventory
        if verifier.ledger and not allow_incomplete: raise ValueError("Incomplete campaign: strict mode refuses label access; inspect inventory/missing_or_failed.json")
        # Label barrier: every selected endpoint has passed the entire preflight.
        stage = "evaluation"
        truths = {}
        for value in predictions.values():
            path = value["row"]["manifest"]
            if path not in truths:
                data = verifier.datasets[path]; truth = load_evaluation_labels(data["path"], "test")
                _unique_ids(truth["ids"])
                if not np.array_equal(truth["ids"], data["splits"]["test"]["ids"]): raise ValueError("Evaluation labels/input IDs disagree")
                truths[path] = truth
        # Bounded cache avoids retaining all 1200 probability arrays in memory.
        from functools import lru_cache
        @lru_cache(maxsize=24)
        def batch(key):
            v = predictions[key]; truth = truths[v["row"]["manifest"]]
            k = v["fitted"]["view"]["data"]["metadata"]["n_clusters"]
            p = _prediction(v["prediction_path"], truth["ids"], k)
            return PredictionBatch(p["ids"], truth["labels"], p["predictions"], v["config"]["model_sha256"],
                                   probabilities=p["probabilities"], cluster_labels=p["cluster_labels"], expected_n_clusters=k)
        rows, row_lookup, expected_groups = [], {}, set()
        for fitrow, predrow in zip(campaign["trains"], campaign["predictions"]):
            for condition in predrow["conditions"]:
                group = (fitrow["dataset"], fitrow["method"], fitrow["condition"], condition)
                expected_groups.add(group); key = (fitrow["run_id"], condition)
                if key not in predictions: continue
                v, b = predictions[key], batch(key)
                panel = clustering_metrics(b.y_true, b.y_pred, probabilities=b.probabilities, cluster_labels=b.cluster_labels,
                                           expected_n_clusters=b.expected_n_clusters, probability_atol=1e-5)
                row = dict(run_id=fitrow["run_id"], dataset=fitrow["dataset"], method=fitrow["method"], seed=fitrow["seed"],
                           training_condition=fitrow["condition"], inference_condition=condition, split="test", metrics=panel,
                           scores=panel["scores"], evidence=_public(v), evaluated_input_sha256=v["config"]["evaluated_input_sha256"],
                           checkpoint_sha256=v["config"]["model_sha256"], initialization_run_sha256=v["fitted"]["initial"]["run_sha256"],
                           prediction_artifact_sha256=v["hashes"]["predictions.npz"])
                rows.append(row); row_lookup[key] = row
        paired, paired_lookup, expected_pairs = [], {}, set()
        fitlookup = {(r["dataset"],r["method"],r["seed"],r["condition"]):r for r in campaign["trains"]}
        for fitrow, predrow in zip(campaign["trains"], campaign["predictions"]):
            d,m,s,tc = (fitrow[k] for k in ("dataset","method","seed","condition"))
            cleanrow = fitlookup[d,m,s,"clean"]
            conditions = predrow["conditions"][1:] if tc == "clean" else [tc]
            for condition in conditions:
                mode = "inference_corruption" if tc == "clean" else "retrained"
                expected_pairs.add((d,m,mode,condition))
                clean_key = (cleanrow["run_id"], "clean")
                condition_key = (fitrow["run_id"], condition if mode == "inference_corruption" else "clean")
                if clean_key not in predictions or condition_key not in predictions: continue
                detail = paired_condition_metrics(batch(clean_key), batch(condition_key), mode=mode, probability_atol=1e-5)
                # Panels live once in evaluation rows. Frozen-map per-class detail
                # remains here because it differs from the native remapped panel.
                detail.pop("clean"); detail.pop("condition")
                scores = dict(detail["delta"])
                if detail["clean_map_frozen"] is not None:
                    scores.update(detail["clean_map_frozen"]["delta"])
                entry = dict(dataset=d,method=m,seed=s,condition=condition,clean_run_id=clean_key[0],
                             condition_run_id=condition_key[0], scores=scores, **detail)
                paired.append(entry); paired_lookup[d,m,s,mode,condition] = entry
        contrasts = list(CORE_CONTRASTS)
        if all_pairwise:
            used = {frozenset((a,b)) for _,a,b in contrasts}
            contrasts += [(f"supplemental_{a}_vs_{b}",a,b) for a,b in combinations(METHODS,2) if frozenset((a,b)) not in used]
        clean_contrasts, condition_contrasts, robustness_contrasts, expected_clean_contrasts, expected_robustness = [], [], [], set(), set()
        for name,a,b in contrasts:
            scope = "supplemental_all_pairwise" if name.startswith("supplemental_") else "prespecified_mechanism_or_named_comparison"
            for d in DATASETS:
                expected_clean_contrasts.add((d,name,a,b,scope))
                for s in SEEDS:
                    ka = (fitlookup[d,a,s,"clean"]["run_id"],"clean"); kb = (fitlookup[d,b,s,"clean"]["run_id"],"clean")
                    if ka in row_lookup and kb in row_lookup:
                        _same_input(row_lookup[ka],row_lookup[kb])
                        clean_contrasts.append(dict(dataset=d,contrast=name,method_a=a,method_b=b,scope=scope,seed=s,
                            scores={k:row_lookup[ka]["scores"][k]-row_lookup[kb]["scores"][k] for k in ROBUSTNESS_SCORES}))
                allowed = [("inference_corruption",c) for c in INFERENCE_CONDITIONS] if d in DATASETS[:3] else []
                if d in DATASETS[:2] and a in CONTAMINATION_METHODS and b in CONTAMINATION_METHODS:
                    allowed += [("retrained",c) for c in CONTAMINATION_CONDITIONS]
                for mode,condition in allowed:
                    expected_robustness.add((d,name,a,b,scope,mode,condition))
                    for s in SEEDS:
                        left,right = paired_lookup.get((d,a,s,mode,condition)),paired_lookup.get((d,b,s,mode,condition))
                        if left is None or right is None: continue
                        # paired_metrics already aligned identical test IDs/truth;
                        # both methods' actual input arrays must also be equal.
                        ic = condition if mode == "inference_corruption" else "clean"
                        for kind in ("clean_run_id","condition_run_id"):
                            _same_input(row_lookup[left[kind],"clean" if kind == "clean_run_id" else ic],
                                        row_lookup[right[kind],"clean" if kind == "clean_run_id" else ic])
                        scores = {k:left["scores"][k]-right["scores"][k] for k in left["scores"]}
                        d_clean = {k:row_lookup[left["clean_run_id"],"clean"]["scores"][k]-row_lookup[right["clean_run_id"],"clean"]["scores"][k] for k in ROBUSTNESS_SCORES}
                        d_condition = {k:row_lookup[left["condition_run_id"],ic]["scores"][k]-row_lookup[right["condition_run_id"],ic]["scores"][k] for k in ROBUSTNESS_SCORES}
                        condition_contrasts.append(dict(dataset=d,contrast=name,method_a=a,method_b=b,scope=scope,seed=s,
                            mode=mode,condition=condition,scores=d_condition,
                            definition="A_condition-B_condition; absolute conditional score difference, not robustness Gamma"))
                        robustness_contrasts.append(dict(dataset=d,contrast=name,method_a=a,method_b=b,scope=scope,seed=s,
                            mode=mode,condition=condition,scores=scores,
                            clean_difference_a_minus_b=d_clean,condition_difference_a_minus_b=d_condition,
                            gamma_definition="(A_condition-A_clean)-(B_condition-B_clean)",
                            difference_in_degradation={k:-v for k,v in scores.items()},
                            degradation_note="Degradation=clean-condition; positive difference-in-degradation means A deteriorated more. Gamma has the opposite sign."))
        allocations = []
        for fitrow in campaign["trains"]:
            fitted = fits[fitrow["run_id"]]
            if fitrow["condition"] != "clean" and fitted:
                allocations.append(dict(run_id=fitrow["run_id"], dataset=fitrow["dataset"], method=fitrow["method"], seed=fitrow["seed"],
                                        condition=fitrow["condition"], **origin_allocation(verifier,fitted)))
        summary = aggregate(rows, expected_groups, ("dataset","method","training_condition","inference_condition"))
        paired_summary = aggregate(paired,expected_pairs,("dataset","method","mode","condition"),paired=True)
        clean_summary = aggregate(clean_contrasts,expected_clean_contrasts,("dataset","contrast","method_a","method_b","scope"),paired=True)
        condition_summary = aggregate(condition_contrasts,expected_robustness,("dataset","contrast","method_a","method_b","scope","mode","condition"),paired=True)
        gamma_summary = aggregate(robustness_contrasts,expected_robustness,("dataset","contrast","method_a","method_b","scope","mode","condition"),paired=True)
        result = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), primary_metric="ari",
                      allow_incomplete=allow_incomplete, all_pairwise=all_pairwise, labels_loaded=True, stage="frozen_endpoint_analysis",
                      provenance=provenance, caveats=CAVEATS, counts=inventory["verified"], missing_or_failed=verifier.ledger,
                      rows=rows, summaries=summary, contrast_registry=[dict(name=n,method_a=a,method_b=b) for n,a,b in contrasts],
                      paired_condition_rows=paired, paired_condition_summaries=paired_summary,
                      clean_contrast_rows=clean_contrasts, clean_contrast_summaries=clean_summary,
                      condition_contrast_rows=condition_contrasts, condition_contrast_summaries=condition_summary,
                      robustness_contrast_rows=robustness_contrasts, robustness_contrast_summaries=gamma_summary,
                      originating_training_contaminant_diagnostics="training_origin_allocation.json",
                      cost_evidence="inventory.json: initializer costs recorded once, each fit carries its initializer identity")
        write_json(output / "evaluation.json", result)
        write_json(output / "training_origin_allocation.json", allocations)
        scalar_rows = [{**{k:r[k] for k in ("run_id","dataset","method","seed","training_condition","inference_condition")}, **r["scores"]} for r in rows]
        metric_names = sorted(rows[0]["scores"]) if rows else []
        _write_csv(output / "seed_metrics.csv", scalar_rows, ["run_id","dataset","method","seed","training_condition","inference_condition",*metric_names])
        for name,values in (("paired_seed_changes",paired),("clean_seed_contrasts",clean_contrasts),("condition_seed_contrasts",condition_contrasts),("robustness_seed_contrasts",robustness_contrasts)):
            flattened = [{**{k:r[k] for k in ("dataset","method","seed","mode","condition","contrast","method_a","method_b","scope") if k in r}, **r["scores"]} for r in values]
            fields = ["dataset","method","seed","mode","condition","contrast","method_a","method_b","scope"]
            score_names = sorted({k for r in values for k in r["scores"]})
            _write_csv(output / (name + ".csv"), flattened, fields+score_names)
        _finish_analysis(output,provenance)
        return result
    except Exception as error:
        write_json(output / "FAILED.json", dict(schema_version=SCHEMA,status="failed",stage=stage,
                   error=f"{type(error).__name__}: {error}", labels_may_have_been_loaded=stage=="evaluation",provenance=provenance,
                   missing_or_failed=verifier.ledger))
        raise


def _same_input(a,b):
    if a["evaluated_input_sha256"] != b["evaluated_input_sha256"]:
        raise ValueError("Cross-method paired corruption hash mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","release-hashes","completion-audit","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("campaign-sha256","release-hashes-sha256","completion-audit-sha256"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--path-map",action="append",default=[],metavar="REMOTE=LOCAL")
    p.add_argument("--overrides",type=Path); p.add_argument("--overrides-sha256")
    p.add_argument("--allow-incomplete",action="store_true"); p.add_argument("--inventory-only",action="store_true")
    p.add_argument("--all-pairwise",action="store_true")
    args=vars(p.parse_args()); campaign=args.pop("campaign"); output=args.pop("output")
    result=evaluate(campaign,output,**args)
    print(json.dumps(dict(output=str(output),schema_version=result["schema_version"],labels_loaded=result["labels_loaded"])))


if __name__=="__main__": main()
