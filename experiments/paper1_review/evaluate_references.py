"""Evaluate verified frozen reference endpoints, with labels read last.

These MNIST runs are source-recipe adaptations with different architectures,
objectives, budgets and inference rules, not a matched-method leaderboard.
Partial evaluation is explicit: missing runs remain in a ledger and incomplete
three-seed groups have no aggregate estimate. No checkpoint/model is executed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn

from .data import load_evaluation_labels, _ids_hash
from .metrics import clustering_metrics, DEFINITION_SOURCES, _encode
from .provenance import sha256, write_json

SCHEMA = "paper1-reference-endpoint-evaluation-v1"
SEEDS = (901, 902, 903)
METHODS = ("vade", "dec", "idec", "pro_t")
VERSIONS = {"vade": "paper1-review-vade-reference-inductive-v1",
            "dec": "paper1-review-dec-idec-reference-v1",
            "idec": "paper1-review-dec-idec-reference-v1",
            "pro_t": "paper1-reference-pro-t-20261028-v1"}
COUNTS = {"train": 54000, "validation": 6000, "test": 10000}
CAVEATS = [
    "ARI is primary; ACC, arithmetic NMI and arithmetic AMI are mandatory companions. No best-of-metrics ranking or winner selection.",
    "These are inductive source-recipe adaptations, not matched-budget/model comparisons or reproductions of historical published scores.",
    "VaDE uses one sampled Gaussian latent for native assignments; DEC/IDEC use deterministic embeddings; Pro-t native uses deterministic encoder means.",
    "Pro-t native Student assignments and final training-fitted GMM assignments are distinct outcomes, never selected by evaluation score.",
    "Three reference seeds are descriptive implementation checks; they are separate from the five fresh seeds of the matched core study.",
    "DEC and IDEC share fresh seed-matched pretraining. Initialization cost is reported as a shared dependency, not counted twice as independent work.",
    "ACC matching is evaluation-only. Mapped F1/recall use the same displayed ACC-optimal map; ambiguity is retained.",
    "Arithmetic NMI and V-measure are equivalent; balanced accuracy is mean per-class recall. These are not independent corroborating outcomes.",
    "Responsibility diagnostics do not establish calibrated class probabilities. No class Brier score/log-loss is computed.",
    "No robustness effect can be inferred from clean reference endpoints. Paired corruption metrics are inapplicable here.",
    "Unlike loss reductions and likelihoods are retained as method-specific diagnostics and are not compared as a common ELBO.",
]


def _safe(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    path.relative_to(directory.resolve())
    return path


def _require_hash(directory, record, name):
    path = _safe(directory, name)
    expected = record.get("artifact_sha256", {}).get(name)
    if not expected or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Required artifact missing/changed: {directory}/{name}")
    return path


def _base_verify(directory: Path, seed: int, manifest_sha: str, method: str) -> dict:
    run = json.loads((directory / "run.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if (run.get("status") != "completed" or run.get("stage") != "final_benchmark" or
        run.get("fresh_run") is not True or run.get("legacy_outputs_used") is not False or
        run.get("labels_loaded", config.get("labels_loaded")) is not False):
        raise ValueError(f"Not a completed, fresh, label-free final run: {directory}")
    if run.get("numerical_gate_passed") is False:
        raise ValueError("Failed numerical gate cannot be evaluated")
    if sha256(directory / "config.json") != run.get("config_sha256"):
        raise ValueError("Configuration hash mismatch")
    _require_hash(directory, run, "config.json")
    if config.get("protocol_version") != VERSIONS[method] or run.get("protocol_version") != VERSIONS[method]:
        raise ValueError("Unexpected reference protocol version")
    if config.get("effective_seed", config.get("seed")) != seed:
        raise ValueError("Wrong seed or validation seed stream")
    if config.get("validation", config.get("validation_only")) is not False:
        raise ValueError("Validation/development runs are not final evidence")
    if config.get("data_manifest_sha256", config.get("manifest_sha256")) != manifest_sha:
        raise ValueError("Training/evaluation dataset-manifest mismatch")
    sources = run.get("source_file_sha256", {})
    expected_source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if not sources or expected_source_digest != run.get("source_sha256"):
        raise ValueError("Source aggregate digest mismatch")
    required_port = f"experiments/paper1_review/reference_{'dec_idec' if method in {'dec','idec'} else method}.py"
    if required_port not in sources:
        raise ValueError("Missing trained-port source snapshot")
    for relative, expected in sources.items():
        snapshot = _require_hash(directory, run, "source/" + relative)
        if sha256(snapshot) != expected: raise ValueError("Source snapshot hash mismatch")
    # All retrieved declared artifacts are checked. Large omitted rolling
    # checkpoints are listed, not incorrectly claimed to be locally verified.
    present, omitted = [], []
    for name, expected in run["artifact_sha256"].items():
        path = _safe(directory, name)
        if path.is_file():
            if sha256(path) != expected: raise ValueError(f"Retrieved artifact changed: {path}")
            present.append(name)
        else: omitted.append(name)
    return {"directory": str(directory), "run": run, "config": config,
            "run_sha256": sha256(directory / "run.json"),
            "locally_verified_artifacts": present, "omitted_nonrequired_artifacts": omitted}


def _initialization(directory, method, seed, manifest_sha, fitted):
    value = _base_verify(directory, seed, manifest_sha, method)
    run, config = value["run"], value["config"]
    if method == "vade":
        if (config.get("phase") != "fresh_initialization" or config.get("pretrain_epochs") != 100 or
            run.get("epochs_completed") != 100 or run.get("optimizer_updates") != 54000 or
            run.get("train_count") != 54000 or run.get("gmm_converged") is not True):
            raise ValueError("Incomplete/wrong VaDE initialization")
        path = _require_hash(directory, run, "initial.pt")
        dependency = json.loads(_require_hash(Path(fitted["directory"]), fitted["run"], "initial_dependency.json").read_text())
        if dependency != {"run_sha256": value["run_sha256"], "checkpoint_sha256": sha256(path)}:
            raise ValueError("VaDE initialization dependency mismatch")
        _require_hash(directory, run, "training_curves.json")
    else:
        p = config.get("protocol", {})
        if (config.get("phase") != "initialize" or run.get("pretrain_completed") is not True or
            run.get("optimizer_updates") != 300000 or p.get("layer_updates") != 50000 or
            p.get("joint_updates") != 100000 or p.get("architecture") != [784,500,500,2000,10]):
            raise ValueError("Incomplete/wrong DEC-style initialization")
        path = _require_hash(directory, run, "initialization.pt")
        if fitted["config"].get("initialization_sha256") != sha256(path):
            raise ValueError("DEC/IDEC initialization dependency mismatch")
        _require_hash(directory, run, "training_diagnostics.json")
    if fitted["run"]["source_file_sha256"] != run["source_file_sha256"]:
        raise ValueError("Initialization/fitted source mismatch")
    value["checkpoint_sha256"] = sha256(path)
    return value


def _endpoint(directory, method, seed, manifest_sha, initial_directory=None):
    value = _base_verify(directory, seed, manifest_sha, method)
    run, config = value["run"], value["config"]
    if method == "vade":
        if (config.get("phase") != "reference_training" or config.get("training_epochs") != 3000 or
            config.get("batch_size") != 100 or run.get("epochs_completed") != 3000 or
            run.get("optimizer_updates") != 1620000 or run.get("train_count") != 54000):
            raise ValueError("VaDE fixed endpoint not completed")
        checkpoint, diagnostics = "final.pt", "training_curves.json"
        variants = ["native"]
    elif method in {"dec", "idec"}:
        p = config.get("protocol", {})
        if (config.get("phase") != method or p.get("architecture") != [784,500,500,2000,10] or
            p.get("cluster_max_updates") != 20000 or p.get("batch_size") != 256 or
            p.get(f"{method}_refresh") != (160 if method == "dec" else 140)):
            raise ValueError("DEC/IDEC final protocol mismatch")
        updates, reason = run.get("optimizer_updates"), run.get("stop_reason")
        if not isinstance(updates, int) or not 0 <= updates <= 20000:
            raise ValueError("Invalid clustering update count")
        if reason == "update_cap":
            if updates != 20000: raise ValueError("Clustering cap not attained")
        elif reason == "assignment_stability":
            change = run.get("last_assignment_change")
            if (not isinstance(change, (int,float)) or not 0 <= change < .001 or
                run.get("last_target_refresh") != updates or
                updates % (160 if method == "dec" else 140) or (method == "idec" and updates == 0)):
                raise ValueError("Invalid source-style stopping evidence")
        else: raise ValueError("Unregistered clustering endpoint rule")
        checkpoint, diagnostics = "model.pt", "training_diagnostics.json"
        variants = ["native"]
    else:
        p = config.get("settings", {})
        if (p.get("epochs") != 100 or p.get("batch_size") != 32 or p.get("overwrite_epoch") != 20 or
            run.get("completed_epochs") != 100 or run.get("optimizer_updates") != 168800 or
            run.get("overwrite_count") != 1 or run.get("fixed_endpoint") is not True or
            run.get("endpoint_gmm", {}).get("converged") is not True):
            raise ValueError("Pro-t fixed endpoint/overwrite not completed")
        checkpoint, diagnostics = "endpoint_checkpoint.pt", "training_history.json"
        variants = ["native", "gmm_refit"]
        for name in ["endpoint_train_gmm.npz", "training_ids.npz"]:
            _require_hash(directory, run, name)
    endpoint = _require_hash(directory, run, checkpoint)
    history = _require_hash(directory, run, diagnostics)
    value.update(checkpoint_sha256=sha256(endpoint), diagnostics_path=str(history), variants=variants)
    value["initialization"] = _initialization(initial_directory, method, seed, manifest_sha, value) if initial_directory else None
    return value


def _prediction(value, method, variant, split, manifest):
    name = f"{split}_{variant}_predictions.npz" if method == "pro_t" else f"{split}_predictions.npz"
    path = _require_hash(Path(value["directory"]), value["run"], name)
    with np.load(path, allow_pickle=False) as artifact:
        keys = set(artifact.files)
        id_key = "ids" if method == "vade" else "sample_ids"
        required = {id_key, "predictions", "responsibilities"}
        if keys != required | ({"cluster_labels"} if method != "vade" else set()):
            raise ValueError(f"Unexpected prediction schema: {path}")
        ids, predicted, probabilities = (artifact[id_key], artifact["predictions"], artifact["responsibilities"])
        labels = artifact["cluster_labels"] if "cluster_labels" in artifact else np.arange(10)
    unique, encoded, _ = _encode(ids, "prediction IDs")
    if len(unique) != len(encoded): raise ValueError("Duplicate prediction IDs")
    if len(ids) != manifest["splits"][split]["n_samples"]:
        raise ValueError("Prediction count differs from declared split")
    if (predicted.shape != (len(ids),) or probabilities.shape != (len(ids),10) or
        labels.shape != (10,) or len(set(labels.tolist())) != 10):
        raise ValueError("Prediction/component shapes invalid")
    if probabilities.dtype.kind not in "fiu" or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite/nonnumeric responsibilities")
    if (probabilities < 0).any() or (probabilities > 1).any(): raise ValueError("Invalid probability range")
    row_error = float(np.max(np.abs(probabilities.sum(1,dtype=np.float64)-1)))
    if row_error > 1e-5: raise ValueError("Invalid responsibility normalization")
    if not np.array_equal(predicted, labels[probabilities.argmax(1)]):
        raise ValueError("Hard predictions disagree with responsibility argmax")
    return {"path": str(path), "sha256": sha256(path), "ids": ids, "predictions": predicted,
            "probabilities": probabilities, "cluster_labels": labels,
            "prediction_rule": "frozen endpoint artifact; no refitting or re-sampling",
            "maximum_probability_row_sum_error": row_error}


def _keys(ids):
    _, encoded, unique = _encode(ids, "sample IDs")
    return [unique[index] for index in encoded]


def _align(prediction, truth):
    reference_keys, other_keys = _keys(truth["ids"]), _keys(prediction["ids"])
    if len(set(reference_keys)) != len(reference_keys): raise ValueError("Duplicate evaluation IDs")
    if set(reference_keys) != set(other_keys): raise ValueError("Prediction/evaluation IDs mismatch")
    lookup = {key: index for index,key in enumerate(other_keys)}
    order = np.asarray([lookup[key] for key in reference_keys])
    return order, {"reordered": bool(np.any(order != np.arange(len(order)))),
                   "n_samples": len(order), "evaluation_ids_sha256": _ids_hash(truth["ids"])}


def _cost(value):
    run = value["run"]
    fields = ["wall_seconds", "train_seconds", "training_seconds", "attempt_elapsed_seconds", "hardware",
              "optimizer_updates", "epochs_completed", "completed_epochs", "stop_reason", "last_assignment_change"]
    result = {key: run[key] for key in fields if key in run}
    result["missing_values_are_unavailable_not_zero"] = True
    return result


def _public_evidence(value):
    result = {k:v for k,v in value.items() if k not in {"run", "config", "initialization", "variants"}}
    result.update(config_sha256=value["run"]["config_sha256"], source_sha256=value["run"]["source_sha256"],
                  environment=value["run"].get("environment"), cost=_cost(value))
    if "diagnostics_path" in value:
        history = json.loads(Path(value["diagnostics_path"]).read_text())
        rows = history if isinstance(history,list) else []
        loss_rows = [row for row in rows if isinstance(row,dict) and
                     any(key in row for key in ["loss","reconstruction","bce","prior","kl","mse"])]
        result["method_specific_training_diagnostics"] = {
            "last_record":rows[-1] if rows else None,
            "last_record_with_objective":loss_rows[-1] if loss_rows else None,
            "interpretation":"Recorded training quantities at the displayed update/epoch; not newly recomputed endpoint likelihoods or cross-method comparable losses",
            "unavailable_reason":None if loss_rows else "No scalar objective persisted in retrieved diagnostic rows"}
    initial = value.get("initialization")
    if initial:
        result["initialization"] = _public_evidence(initial)
        result["initialization"]["sharing"] = "shared DEC/IDEC seed dependency" if value["config"].get("phase") in {"dec","idec"} else "VaDE seed dependency"
    return result


def summaries(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"],row["variant"],row["split"]), []).append(row)
    output = []
    for (method,variant,split), group in sorted(groups.items()):
        seeds = sorted(row["seed"] for row in group)
        entry = {"method":method,"variant":variant,"split":split,"seeds":seeds,"n":len(group),
                 "status":"complete_three_seed_group" if seeds == list(SEEDS) else "incomplete_seed_set_aggregate_withheld"}
        if seeds == list(SEEDS):
            entry["scores"] = {}
            for key in group[0]["metrics"]["scores"]:
                values = np.array([next(r for r in group if r["seed"] == seed)["metrics"]["scores"][key] for seed in SEEDS])
                entry["scores"][key] = {"values_by_seed":values.tolist(),"mean":float(values.mean()),
                    "median":float(np.median(values)),"sample_sd":float(values.std(ddof=1)),
                    "minimum":float(values.min()),"maximum":float(values.max())}
        output.append(entry)
    return output


def evaluate(root: str | Path, manifest_path: str | Path, output: str | Path, *,
             splits=("test",), allow_incomplete=False, missing_ledger=None, overrides=None,
             completion_audit=None):
    """Preflight all selected fits before the first evaluation-label read.

    ``overrides`` maps e.g. 'vade:902' to {directory, initialization}, permitting
    a verified resumed attempt without changing the method or seed roster.
    ``missing_ledger`` maps missing method:seed keys to scheduler/recovery notes.
    No failures are silently removed; complete but invalid artifacts abort.
    """
    begin = time.perf_counter()
    root, manifest_path, output = Path(root).resolve(), Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not splits or len(set(splits)) != len(splits) or any(s not in {"test","validation"} for s in splits):
        raise ValueError("Choose distinct test/validation endpoint splits")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != "paper1-review-dataset-v1" or manifest.get("status") != "completed" or
        manifest.get("dataset") != "mnist" or manifest.get("n_clusters") != 10 or
        {s:manifest["splits"][s]["n_samples"] for s in COUNTS} != COUNTS):
        raise ValueError("Expected frozen inductive MNIST54k/6k/10k manifest")
    manifest_sha = sha256(manifest_path)
    audit = json.loads(Path(completion_audit).read_text()) if completion_audit else None
    if audit is not None and audit.get("all_recorded_completed_artifacts_verified") is not True:
        raise ValueError("Remote completion/artifact audit did not pass")
    selected, missing = [], []
    overrides, missing_ledger = overrides or {}, missing_ledger or {}
    expected_keys = {f"{m}:{s}" for m in METHODS for s in SEEDS}
    if set(overrides)-expected_keys or set(missing_ledger)-expected_keys:
        raise ValueError("Unexpected method/seed in override or missing ledger")
    for method in METHODS:
        family = "dec_idec" if method in {"dec","idec"} else method
        for seed in SEEDS:
            key = f"{method}:{seed}"
            parent = root / f"reference_{family}_v1" / f"seed{seed}"
            spec = overrides.get(key, {})
            directory = Path(spec.get("directory", parent / (method if method in {"dec","idec"} else "train"))).resolve()
            initial = Path(spec.get("initialization", parent / "initialize")).resolve() if method != "pro_t" else None
            record = json.loads((directory/"run.json").read_text()) if (directory/"run.json").is_file() else {}
            if record.get("status") != "completed":
                missing.append({"method":method,"seed":seed,"directory":str(directory),
                    "recorded_status":record.get("status","not_retrieved"),
                    "run_sha256":sha256(directory/"run.json") if record else None,
                    "reason":missing_ledger.get(key, "No completed endpoint available; no score calculated"),
                    "error":record.get("error")})
                continue
            if key in missing_ledger: raise ValueError("Completed endpoint conflicts with missing-run ledger")
            value = _endpoint(directory,method,seed,manifest_sha,initial)
            if audit is not None:
                for proof,phase in [(value,method if method in {"dec","idec"} else "train"),
                                    (value.get("initialization"),"initialize")]:
                    if proof is None:continue
                    matches=[row for row in audit.get("runs",[]) if row.get("family")==family and
                             row.get("seed")==seed and row.get("phase")==phase and
                             row.get("run_sha256")==proof["run_sha256"]]
                    if not any(row.get("status")=="completed" and row.get("hash_errors")==[] for row in matches):
                        raise ValueError("Local run does not match verified remote completion receipt")
            predictions = {(variant,split):_prediction(value,method,variant,split,manifest)
                           for variant in value["variants"] for split in splits}
            selected.append((method,seed,value,predictions))
    if missing and not allow_incomplete:
        raise ValueError(f"{len(missing)} reference fits incomplete; labels remain unopened")
    if not selected: raise ValueError("No completed endpoints to evaluate")
    # No label artifact is opened above this boundary. All source/checkpoint,
    # stopping, split, hard-prediction and probability checks precede it.
    truths = {split:load_evaluation_labels(manifest_path,split) for split in splits}
    rows = []
    for method,seed,value,predictions in selected:
        for (variant,split),prediction in predictions.items():
            truth = truths[split]; order, alignment = _align(prediction,truth)
            panel = clustering_metrics(truth["labels"],prediction["predictions"][order],
                probabilities=prediction["probabilities"][order],cluster_labels=prediction["cluster_labels"],
                expected_n_clusters=10,probability_atol=1e-5)
            rows.append({"method":method,"variant":variant,"seed":seed,"split":split,
                         "metrics":panel,"alignment":alignment,
                         "prediction_artifact":{k:v for k,v in prediction.items() if k not in {"ids","predictions","probabilities","cluster_labels"}},
                         "evidence":_public_evidence(value)})
    result = {"schema_version":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
              "scope":"partial reference endpoint review" if missing else "complete reference endpoint review",
              "primary_metric":"ari","primary_split":"test","allow_incomplete":allow_incomplete,
              "expected_fit_count":12,"completed_fit_count":len(selected),"prediction_result_count":len(rows),
              "missing_runs":missing,"rows":rows,"summaries":summaries(rows),"caveats":CAVEATS,
              "inapplicable_metrics":{"robustness":"No paired clean/corrupt reference predictions",
                  "class_probability_scores":"Arbitrary cluster responsibilities are not calibrated class probabilities",
                  "common_ELBO":"Different model likelihoods and objective reductions",
                  "DEC_reconstruction_likelihood":"DEC clustering has no generative likelihood"},
              "provenance":{"root":str(root),"manifest_path":str(manifest_path),"manifest_sha256":manifest_sha,
                  "evaluation_labels":{s:{"path":manifest["splits"][s]["labels"],
                    "sha256":manifest["artifact_sha256"][manifest["splits"][s]["labels"]]} for s in splits},
                  "evaluator_sha256":sha256(Path(__file__)),"metric_implementation_sha256":sha256(Path(__file__).with_name("metrics.py")),
                  "data_loader_sha256":sha256(Path(__file__).with_name("data.py")),
                  "completion_audit":({"path":str(Path(completion_audit).resolve()),"sha256":sha256(Path(completion_audit))} if completion_audit else None),
                  "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
                  "definition_sources":DEFINITION_SOURCES,"evaluation_seconds":time.perf_counter()-begin}}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"evaluation.json",result)
    write_json(output/"missing_runs.json",missing)
    with (output/"seed_scores.csv").open("w",newline="") as stream:
        keys = ["method","variant","seed","split",*rows[0]["metrics"]["scores"],
                "occupied_clusters","effective_clusters","max_cluster_share","mapping_nonunique"]
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            panel=row["metrics"]
            writer.writerow({**{k:row[k] for k in ["method","variant","seed","split"]},**panel["scores"],
                "occupied_clusters":panel["occupancy"]["occupied_cluster_count"],
                "effective_clusters":panel["occupancy"]["effective_cluster_count"],
                "max_cluster_share":panel["occupancy"]["max_cluster_share"],
                "mapping_nonunique":panel["mapping"]["has_alternative_acc_optimum"]})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",action="append",choices=["test","validation"])
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--missing-ledger",type=Path,help="JSON mapping method:seed to scheduler/recovery note")
    parser.add_argument("--overrides",type=Path,help="JSON mapping method:seed to directory/initialization paths")
    parser.add_argument("--completion-audit",type=Path,help="Remote audit receipt anchoring each completed run.json hash")
    args=parser.parse_args()
    result=evaluate(args.root,args.manifest,args.output,splits=args.split or ["test"],
        allow_incomplete=args.allow_incomplete,
        missing_ledger=json.loads(args.missing_ledger.read_text()) if args.missing_ledger else None,
        overrides=json.loads(args.overrides.read_text()) if args.overrides else None,
        completion_audit=args.completion_audit)
    print(json.dumps({"output":str(args.output),"scope":result["scope"],
                      "completed_fit_count":result["completed_fit_count"],"missing_runs":result["missing_runs"]},indent=2))


if __name__ == "__main__": main()
