"""Summarize anchored, label-free diagnostics after strict endpoint evaluation.

Only JSON diagnostics/configuration/run records are read. The completed scorer's
anchored inventory supplies the full-artifact/lineage gate; this helper rehashes
the JSON it uses, never checkpoint files or sample arrays. No model is imported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from .evaluate_matched import (_anchor, _json, _quadrature, _safe, _curves,
                               Paths, validate_campaign, describe)
from .final_protocol import PROTOCOL, protocol_digest
from .provenance import sha256, write_json
from .workload import SEEDS

SCHEMA = "paper1-matched-diagnostic-summary-v1"
CAVEATS = [
    "Recorded diagnostics are observational summaries, not causal explanations of SGD or clustering performance.",
    "Prior df boundary counts use the original instrument's relative 1e-6 rule. A finite upper bound of 30 is not the Gaussian limit.",
    "Recognition df extrema and site quantities cover the fixed 32-observation clean diagnostic batch, not the whole dataset. No unrecorded boundary count is inferred.",
    "Component gradient norms are separately differentiated actual objective terms. They do not sum to the joint gradient norm; cancellation and direction are unmeasured.",
    "Clipping fractions count committed optimizer updates. No claim about an unclipped or retuned implementation follows from these diagnostics.",
    "Quadrature errors compare attained grids on selected observations with the 1024-node reference, not exact integrals or uniform error guarantees.",
    "Occupied clusters and responsibility entropy do not establish semantic class recovery or calibrated uncertainty.",
    "Recorded training time includes committed predecessor trajectory time on continuation. It excludes discarded/replayed work and scheduler downtime; physical cost needs separate scheduler accounting. Shared initialization is excluded here.",
    "Five-seed summaries are separate for each dataset, method and training condition. No cross-dataset pooled ranking or significance test is computed.",
]


def parameter_summary(parameters):
    """Preserve measured scalars and reduce matrices without new hit thresholds."""
    result = {k: v for k, v in parameters.items() if type(v) in (int, float)}
    if "prior_df" in parameters:
        a = np.asarray(parameters["prior_df"], dtype=float)
        w = np.asarray(parameters["component_weights"], dtype=float)
        if a.ndim != 1 or len(a) != len(w) or np.any(a <= 0) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid prior parameter diagnostic")
        result.update(prior_df_min=float(a.min()), prior_df_max=float(a.max()),
                      prior_df_mean=float(a.mean()), prior_df_weighted_mean=float(np.dot(a, w) / w.sum()))
        lo, hi = PROTOCOL["objective"]["prior_df_bounds"]
        if (parameters["df_lower_hits"] != int((a <= lo * (1 + 1e-6)).sum()) or
                parameters["df_upper_hits"] != int((a >= hi * (1 - 1e-6)).sum())):
            raise ValueError("Recorded df boundary counts contradict parameters")
    for name in ("scale_diagonal", "scale_condition_ratio", "component_weights"):
        if name in parameters:
            a = np.asarray(parameters[name], dtype=float)
            if not a.size or np.any(a < 0 if name == "component_weights" else a <= 0):
                raise ValueError("Invalid nonnegative/positive parameter diagnostic")
            result[name + "_min"] = float(a.min())
            result[name + "_max"] = float(a.max())
    return result


def occupancy_summary(row):
    counts = np.asarray(row["cluster_counts"])
    if counts.ndim != 1 or counts.dtype.kind not in "iu" or np.any(counts < 0) or counts.sum() != row["count"] or row["count"] <= 0:
        raise ValueError("Invalid occupancy counts")
    if int((counts > 0).sum()) != row["occupied_clusters"] or not math.isclose(float(counts.max()/counts.sum()), row["max_cluster_share"], abs_tol=1e-12):
        raise ValueError("Occupancy summaries contradict counts")
    p = counts[counts > 0] / counts.sum()
    return {**row, "effective_clusters_from_hard_counts": float(np.exp(-(p * np.log(p)).sum()))}


def quadrature_summary(q, method, count):
    _quadrature(q, method, count)
    if q is None: return None
    return dict(selected_observation_count=len(q["selected_positions"]), passed=True,
                errors_by_nodes={str(r["nodes"]): r["errors"] for r in q["rows"]},
                reference_nodes=1024, reference_bounds=PROTOCOL["structured"]["refined_reference_bounds"])


def fit_summary(row, run, config, curves, gradients, checks, occupancy):
    budget = config["optimizer_updates"]
    _curves(curves, PROTOCOL["training_epochs"], budget)
    epochs = PROTOCOL["diagnostic_epochs"]
    if [r["clean_equivalent_epoch"] for r in gradients] != epochs or [r["clean_equivalent_epoch"] for r in checks] != epochs:
        raise ValueError("Diagnostic epoch schedule changed")
    previous, clipped, weighted_norm, examples = 0, 0, 0., 0
    for item in curves:
        n = item["updates"] - previous; previous = item["updates"]
        if not 0 <= item["clipped_updates"] <= n or not math.isclose(item["clip_fraction"], item["clipped_updates"] / n, abs_tol=1e-12):
            raise ValueError("Invalid clipping count/fraction")
        clipped += item["clipped_updates"]; weighted_norm += n * item["preclip_norm_mean"]
        examples += item["examples_seen_this_window"]
    if clipped != run["clipped_updates"] or budget != run["optimizer_updates"]:
        raise ValueError("Committed clipping/update total changed")
    trajectory = []
    for diagnostic, check in zip(gradients, checks):
        epoch = diagnostic["clean_equivalent_epoch"]
        if diagnostic["optimizer_update"] is not False or diagnostic["completed_updates"] != epoch * (budget // PROTOCOL["training_epochs"]):
            raise ValueError("Invalid no-update gradient diagnostic")
        measured = parameter_summary(diagnostic["parameters"])
        expected_components = ({"target_kl"} if row["method"] == "dec_matched" else
                               {"target_kl", "reconstruction"} if row["method"] == "idec_matched" else
                               {"reconstruction", "prior_cross_entropy", "negative_entropy"})
        if set(diagnostic["components"]) != expected_components: raise ValueError("Objective diagnostic components changed")
        for name, component in diagnostic["components"].items():
            norms = component["group_norms"]
            if component["preclip_norm"] < 0 or any(v < 0 for v in norms.values()) or not math.isclose(
                    component["preclip_norm"] ** 2, sum(v*v for v in norms.values()), rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("Component/group gradient norms disagree")
            measured[f"gradient.{name}.preclip_norm"] = component["preclip_norm"]
            measured[f"objective.{name}.value"] = component["value"]
            for group, norm in norms.items(): measured[f"gradient.{name}.group.{group}"] = norm
        trajectory.append(dict(clean_equivalent_epoch=epoch, scalars=measured,
                               prior_df=diagnostic["parameters"].get("prior_df"),
                               component_weights=diagnostic["parameters"].get("component_weights"),
                               objective_scope=diagnostic["scope"],
                               quadrature=quadrature_summary(check["quadrature"], row["method"], config["validation_count"])))
    occupancies = [occupancy_summary(v) for v in occupancy]
    endpoints = {v["split"]: v for v in occupancies if v.get("endpoint") is True}
    if set(endpoints) != {"train", "validation"} or len([v for v in occupancies if v.get("endpoint")]) != 2:
        raise ValueError("Missing/duplicate endpoint occupancy")
    if any(v["clean_equivalent_epoch"] != PROTOCOL["training_epochs"] or v["count"] != config["training_count" if s == "train" else "validation_count"] for s, v in endpoints.items()):
        raise ValueError("Endpoint occupancy budget/count mismatch")
    scalars = dict(trajectory[-1]["scalars"], optimizer_updates=budget, clipped_updates=clipped,
                   clip_fraction=clipped / budget, preclip_norm_mean_over_updates=weighted_norm / budget,
                   preclip_norm_max_over_updates=max(c["preclip_norm_max"] for c in curves), examples_seen=examples)
    for key in ("training_seconds", "prediction_seconds"):
        if key in run: scalars[key] = run[key]
    for key, value in run.get("timing_breakdown", {}).items(): scalars["timing." + key] = value
    for split, item in endpoints.items():
        for key in ("occupied_clusters", "max_cluster_share", "mean_responsibility_entropy", "effective_clusters_from_hard_counts"):
            scalars[f"occupancy.{split}.{key}"] = item[key]
    applicable = [r["quadrature"] for r in trajectory if r["quadrature"] is not None]
    if applicable:
        for nodes in ("128", "256", "512"):
            for metric in applicable[0]["errors_by_nodes"][nodes]:
                scalars[f"quadrature.max_over_recorded_epochs.nodes{nodes}.{metric}"] = max(q["errors_by_nodes"][nodes][metric] for q in applicable)
    return dict(**{k: row[k] for k in ("run_id", "dataset", "method", "condition", "seed")},
                endpoint_scalars=scalars, diagnostic_trajectory=trajectory, occupancy_trajectory=occupancies,
                training_curves=curves, hardware=run.get("hardware"), resumed="resume_dependency.json" in run["artifact_sha256"],
                timing_semantics=run.get("timing_semantics"), recovery_cost_note=run.get("recovery_cost_note"))


def group_summaries(rows, keys, scalar_key):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        by_seed = {r["seed"]: r for r in values}
        if len(values) != 5 or set(by_seed) != set(SEEDS): raise ValueError("Incomplete/duplicate diagnostic five-seed group")
        metrics = set(values[0][scalar_key])
        if any(set(v[scalar_key]) != metrics for v in values): raise ValueError("Diagnostic scalar schema differs within group")
        result.append(dict(zip(keys, key), seeds=list(SEEDS),
                           scalars={m: describe([by_seed[s][scalar_key][m] for s in SEEDS]) for m in sorted(metrics)}))
    return result


def summarize(campaign, output, *, campaign_sha256, completion_audit, completion_audit_sha256,
              analysis_manifest, analysis_manifest_sha256, overrides=None, overrides_sha256=None, path_map=()):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    campaign = _anchor(campaign, campaign_sha256); validate_campaign(campaign)
    audit = _anchor(completion_audit, completion_audit_sha256)
    override = _anchor(overrides, overrides_sha256) if overrides else {}
    gate_path = Path(analysis_manifest); gate = _anchor(gate_path, analysis_manifest_sha256)
    if gate.get("status") != "completed" or gate.get("inventory_only") is not False:
        raise ValueError("Completed scored analysis gate required")
    inventory = _anchor(gate_path.parent / "inventory.json", gate["artifact_sha256"]["inventory.json"])
    provenance = gate["provenance"]
    if (provenance != inventory["provenance"] or provenance["campaign_sha256"] != campaign_sha256 or
            provenance["completion_audit_sha256"] != completion_audit_sha256 or provenance["overrides_sha256"] != overrides_sha256 or
            provenance["protocol_sha256"] != protocol_digest() or inventory["verified"] != dict(initializations=40, fits=300, predictions=1200) or inventory["missing_or_failed"]):
        raise ValueError("Strict complete analysis gate/provenance mismatch")
    records = {r.get("directory", r.get("path")): r for r in audit["runs"]}
    if len(records) != len(audit["runs"]): raise ValueError("Duplicate audit directory")
    gate_fits = {r["recorded_remote_directory"]: r for r in inventory["fits"]}
    if len(gate_fits) != 300: raise ValueError("Incomplete/duplicate fit inventory")
    paths = Paths(path_map); evidence = []; rows = []

    def read_run(remote):
        audited = records[remote]; directory = paths(remote)
        run = _anchor(directory / "run.json", audited["run_sha256"])
        hashes = audited.get("artifact_sha256", audited.get("recorded_run", {}).get("artifact_sha256"))
        if (run.get("status") != "completed" or run.get("labels_loaded") is not False or run.get("numerical_failure") is True or
                hashes != run.get("artifact_sha256") or audited.get("all_artifacts_verified", audit.get("all_recorded_completed_artifacts_verified")) is not True):
            raise ValueError("Completed audited label-free run required")
        used = {}
        def artifact(name):
            value = _anchor(_safe(directory, name), hashes[name]); used[name] = hashes[name]; return value
        config = artifact("config.json")
        if hashes["config.json"] != run["config_sha256"] or config.get("labels_loaded") is not False:
            raise ValueError("Configuration hash/label scope mismatch")
        evidence.append(dict(directory=remote, run_sha256=audited["run_sha256"], read_artifact_sha256=used))
        return run, config, artifact

    for row in campaign["trains"]:
        chosen = override.get(row["output"], {}).get("directory", row["output"])
        if chosen not in gate_fits or records[chosen]["run_sha256"] != gate_fits[chosen]["run_sha256"]:
            raise ValueError("Selected fit differs from completed analysis")
        run, config, read = read_run(chosen)
        if (config.get("protocol") != PROTOCOL or any(config.get(k) != row[k] for k in ("dataset", "method", "condition", "seed")) or
                run.get("fixed_endpoint") is not True or run.get("numerical_gate_passed") is not True):
            raise ValueError("Wrong fixed fit configuration")
        rows.append(fit_summary(row, run, config, read("training_curves.json"), read("gradient_diagnostics.json"),
                                read("numerical_checks.json"), read("occupancy_diagnostics.json")))
    trajectories = [{**{k: r[k] for k in ("dataset", "method", "condition", "seed")}, **d}
                    for r in rows for d in r["diagnostic_trajectory"]]
    groups = group_summaries(rows, ("dataset", "method", "condition"), "endpoint_scalars")
    trajectory_groups = group_summaries(trajectories, ("dataset", "method", "condition", "clean_equivalent_epoch"), "scalars")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=SCHEMA, status="completed", created_utc=datetime.now(timezone.utc).isoformat(),
                  labels_loaded=False, checkpoints_opened=False, scope="300 fit diagnostics; training validation quadrature, not test corruption quadrature",
                  provenance=dict(campaign_sha256=campaign_sha256, completion_audit_sha256=completion_audit_sha256,
                                  overrides_sha256=overrides_sha256, analysis_manifest_sha256=analysis_manifest_sha256,
                                  implementation_sha256=sha256(Path(__file__))),
                  counts=dict(fits=len(rows), endpoint_groups=len(groups), trajectory_groups=len(trajectory_groups)),
                  caveats=CAVEATS, rows=rows, endpoint_summaries=groups, trajectory_summaries=trajectory_groups, evidence=evidence)
    write_json(output / "diagnostics.json", result)
    write_json(output / "receipt.json", dict(schema_version=SCHEMA, status="completed", labels_loaded=False,
                                            diagnostics_sha256=sha256(output / "diagnostics.json"), provenance=result["provenance"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "completion-audit", "analysis-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("campaign-sha256", "completion-audit-sha256", "analysis-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--overrides", type=Path); parser.add_argument("--overrides-sha256")
    parser.add_argument("--path-map", action="append", default=[])
    result = summarize(**vars(parser.parse_args()))
    print(result["counts"])


if __name__ == "__main__": main()
