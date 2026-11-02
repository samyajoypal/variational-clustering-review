"""Render anchored matched-analysis summaries; never read labels or checkpoints.

ARI is prespecified as primary. Incomplete five-seed groups have no estimate.
This presentation layer trusts the externally supplied analysis-manifest digest,
not a digest discovered beside an untrusted evaluation file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from .final_protocol import CONTAMINATION_CONDITIONS, CONTAMINATION_METHODS, INFERENCE_CONDITIONS
from .workload import DATASETS, METHODS, SEEDS

SCHEMA = "paper1-matched-evaluation-v1"
REPORT_SCHEMA = "paper1-matched-report-v1"
EXPECTED_COUNTS = {"initializations": 40, "fits": 300, "predictions": 1200}
PRIMARY = ("ari", "hungarian_acc", "nmi_arithmetic", "ami_arithmetic")
ROBUST = PRIMARY + ("mapped_macro_f1", "mapped_balanced_accuracy")
ALL_METRICS = ROBUST + ("homogeneity", "completeness", "v_measure")
FROZEN = ("frozen_acc", "frozen_macro_f1", "frozen_balanced_accuracy")
CORE_CONTRASTS = [
    ("prior_family_under_gaussian_q", "gaussian_q_student_prior", "gaussian_q_gaussian_prior"),
    ("direct_recognition_family", "student_q_student_prior", "gaussian_q_student_prior"),
    ("structured_prior_family", "structured_q_student_prior", "structured_q_gaussian_prior"),
    ("structured_vs_direct_student", "structured_q_student_prior", "student_q_student_prior"),
] + [(f"{m}_vs_vae_gmm", m, "vae_gmm") for m in
     ("vade_matched", "dec_matched", "idec_matched", "pro_t_matched")]
METHOD_LABELS = dict(zip(METHODS, ["VaDE (matched)", "DEC (matched)", "IDEC (matched)",
    "VAE + GMM", "Pro-t (matched)", "Gaussian q / Gaussian prior", "Gaussian q / Student prior",
    "Student q / Student prior", "Site q / Gaussian prior", "Site q / Student prior"]))
SHORT_METHODS = dict(zip(METHODS, ["VaDE", "DEC", "IDEC", "VAE+GMM", "Pro-t",
    "Gq / G", "Gq / t", "tq / t", "Site / G", "Site / t"]))
DATA_LABELS = dict(zip(DATASETS, ["MNIST", "Fashion-MNIST", "USPS", "Reuters-10k"]))
METRIC_LABELS = dict(zip(ALL_METRICS + FROZEN, ["ARI", "ACC", "NMI (arithmetic)", "AMI (arithmetic)",
    "Mapped macro F1", "Mapped balanced accuracy", "Homogeneity", "Completeness", "V-measure",
    "Frozen-map ACC", "Frozen-map macro F1", "Frozen-map balanced accuracy"]))
FIELDS = {
    "summaries": ("dataset", "method", "training_condition", "inference_condition"),
    "paired_condition_summaries": ("dataset", "method", "mode", "condition"),
    "clean_contrast_summaries": ("dataset", "contrast", "method_a", "method_b", "scope"),
    "condition_contrast_summaries": ("dataset", "contrast", "method_a", "method_b", "scope", "mode", "condition"),
    "robustness_contrast_summaries": ("dataset", "contrast", "method_a", "method_b", "scope", "mode", "condition"),
}
COMPLETE = "complete_five_seed_group"
PENDING = "incomplete_seed_set_aggregate_withheld"
NOTES = [
    "ARI is primary; all prespecified companion metrics are retained. Method order is fixed, with no winner highlighting or metric selection.",
    "Only the five seeds 101--105 together define an aggregate. Missing groups are pending; available seeds are never averaged as a substitute.",
    "Absolute-score tables show mean (sample SD); seed-point plots show every seed and its mean. Paired intervals are evaluator-supplied descriptive, unadjusted 95% t intervals (df=4), not significance tests.",
    "Delta = condition minus clean. Plotted degradation = clean minus condition = -Delta. D_condition = A_condition - B_condition. Gamma = Delta_A - Delta_B; it is not D_condition.",
    "Inference-corruption maps are independently remapped unless explicitly labelled frozen-map. Frozen-map deltas apply only to the same checkpoint, not retrained models.",
    "Scores use unit fractions, not percentages. No intervals are clipped. No cross-dataset average is computed.",
    "V-measure equals arithmetic NMI; mapped balanced accuracy is mean class recall. These are not independent corroborating outcomes.",
    "This layer reads only anchored analysis JSON, never labels, inputs, predictions, or checkpoints. It does not repeat the upstream artifact audit.",
    "An inventory-only ledger describes the artifacts visible to that analysis snapshot, not live scheduler state or necessarily the whole HPC campaign.",
]


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _load_anchored(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("An explicit lowercase SHA256 trusted anchor is required")
    data = Path(path).read_bytes()
    if _hash_bytes(data) != expected:
        raise ValueError(f"Trusted anchor hash mismatch: {path}")
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(data, parse_constant=reject)
    def finite(v):
        if isinstance(v, float) and not math.isfinite(v): reject(v)
        if isinstance(v, dict):
            for item in v.values(): finite(item)
        elif isinstance(v, list):
            for item in v: finite(item)
    finite(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def expected_rosters(registry, all_pairwise=False):
    """Reconstruct the frozen allocation independently of supplied row counts."""
    contrasts = list(CORE_CONTRASTS)
    if all_pairwise:
        used = {frozenset((a, b)) for _, a, b in contrasts}
        contrasts += [(f"supplemental_{a}_vs_{b}", a, b) for a, b in combinations(METHODS, 2)
                      if frozenset((a, b)) not in used]
    if registry != [dict(name=n, method_a=a, method_b=b) for n, a, b in contrasts]:
        raise ValueError("Contrast registry/order differs from the prespecified panel")
    absolute = {(d, m, "clean", "clean") for d, m in product(DATASETS, METHODS)}
    absolute |= {(d, m, "clean", c) for d, m, c in product(DATASETS[:3], METHODS, INFERENCE_CONDITIONS)}
    absolute |= {(d, m, c, "clean") for d, m, c in product(DATASETS[:2], CONTAMINATION_METHODS, CONTAMINATION_CONDITIONS)}
    paired = {(d, m, "inference_corruption", c) for d, m, c in product(DATASETS[:3], METHODS, INFERENCE_CONDITIONS)}
    paired |= {(d, m, "retrained", c) for d, m, c in product(DATASETS[:2], CONTAMINATION_METHODS, CONTAMINATION_CONDITIONS)}
    clean, changed = set(), set()
    for n, a, b in contrasts:
        scope = "supplemental_all_pairwise" if n.startswith("supplemental_") else "prespecified_mechanism_or_named_comparison"
        for d in DATASETS:
            clean.add((d, n, a, b, scope))
            if d in DATASETS[:3]:
                changed |= {(d, n, a, b, scope, "inference_corruption", c) for c in INFERENCE_CONDITIONS}
            if d in DATASETS[:2] and a in CONTAMINATION_METHODS and b in CONTAMINATION_METHODS:
                changed |= {(d, n, a, b, scope, "retrained", c) for c in CONTAMINATION_CONDITIONS}
    return dict(summaries=absolute, paired_condition_summaries=paired,
                clean_contrast_summaries=clean, condition_contrast_summaries=changed,
                robustness_contrast_summaries=changed)


def _validate_description(value, paired):
    a = np.asarray(value.get("values_by_seed"), dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError("Exactly five finite seed values required in a complete group")
    stats = dict(mean=a.mean(), median=np.median(a), sample_sd=a.std(ddof=1), minimum=a.min(), maximum=a.max())
    if any(k not in value or not math.isclose(value[k], float(v), rel_tol=1e-10, abs_tol=1e-12) for k, v in stats.items()):
        raise ValueError("Summary statistics contradict the saved five-seed values")
    if paired:
        ci = np.asarray(value.get("descriptive_t_interval_95"), dtype=float)
        radius = 2.7764451051977987 * stats["sample_sd"] / math.sqrt(5)
        if ci.shape != (2,) or not np.allclose(ci, [stats["mean"] - radius, stats["mean"] + radius], rtol=1e-10, atol=1e-12):
            raise ValueError("Paired descriptive interval contradicts the five-seed values")


def validate_evaluation(value):
    if value.get("schema_version") != SCHEMA or value.get("stage") != "frozen_endpoint_analysis" or value.get("primary_metric") != "ari":
        raise ValueError("Unsupported evaluation schema, stage, or primary metric")
    counts = value.get("counts", {})
    if set(counts) != set(EXPECTED_COUNTS) or any(type(counts[k]) is not int or not 0 <= counts[k] <= v for k, v in EXPECTED_COUNTS.items()):
        raise ValueError("Invalid verified campaign counts")
    expected = expected_rosters(value.get("contrast_registry"), value.get("all_pairwise", False))
    lookup, pending, coverage = {}, [], {}
    for name, roster in expected.items():
        index = {}
        for row in value.get(name, []):
            key = tuple(row[f] for f in FIELDS[name])
            if key not in roster or key in index: raise ValueError(f"Unexpected or duplicate group in {name}")
            seeds = row.get("seeds")
            if not isinstance(seeds, list) or seeds != sorted(set(seeds)) or not set(seeds) <= set(SEEDS):
                raise ValueError("Invalid group seed identities")
            if row.get("n") != len(seeds) or row.get("missing_seeds") != sorted(set(SEEDS) - set(seeds)):
                raise ValueError("Group completeness contradicts its seed list")
            full = seeds == SEEDS
            if row.get("status") != (COMPLETE if full else PENDING): raise ValueError("Group status contradicts its seed list")
            if not full and "scores" in row: raise ValueError("Incomplete groups must not expose aggregate scores")
            if full:
                metrics = ALL_METRICS if name == "summaries" else ROBUST
                if name in ("paired_condition_summaries", "robustness_contrast_summaries") and row["mode"] == "inference_corruption":
                    metrics += FROZEN
                if set(row.get("scores", {})) != set(metrics): raise ValueError(f"Metric panel changed in {name}")
                for score in row["scores"].values(): _validate_description(score, name != "summaries")
            index[key] = row
        for key in sorted(roster):
            if key not in index:
                index[key] = dict(zip(FIELDS[name], key), status=PENDING, n=0, seeds=[], missing_seeds=SEEDS.copy())
            if index[key]["status"] != COMPLETE:
                pending.append(dict(section=name, **{k:v for k,v in index[key].items() if k != "scores"}))
        coverage[name] = dict(expected=len(roster), complete=sum(r["status"] == COMPLETE for r in index.values()))
        lookup[name] = index
    # Verify the algebra across paired panels, in seed order, without recomputing metrics.
    for key, gamma in lookup["robustness_contrast_summaries"].items():
        dc = lookup["condition_contrast_summaries"][key]
        d0 = lookup["clean_contrast_summaries"][key[:5]]
        if all(r["status"] == COMPLETE for r in (gamma, dc, d0)):
            for metric in ROBUST:
                a = np.asarray(dc["scores"][metric]["values_by_seed"]) - np.asarray(d0["scores"][metric]["values_by_seed"])
                if not np.allclose(a, gamma["scores"][metric]["values_by_seed"], rtol=1e-10, atol=1e-12):
                    raise ValueError("Gamma does not equal D_condition minus D_clean in paired seed order")
    partial = bool(pending or value.get("missing_or_failed") or counts != EXPECTED_COUNTS)
    return lookup, dict(expected=EXPECTED_COUNTS, verified=counts, group_coverage=coverage,
                        pending_groups=pending, missing_or_failed=value.get("missing_or_failed", []), partial=partial)


def _tex(value):
    return "".join({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in str(value))


def _cell(row, metric, interval=False, stack=True):
    if row is None or row["status"] != COMPLETE: return "Pending"
    v = row["scores"][metric]
    if interval:
        lo, hi = v["descriptive_t_interval_95"]
        if not stack: return f"${v['mean']:.3f}$ [{lo:.3f}, {hi:.3f}]"
        return "\\shortstack{" + f"${v['mean']:.3f}$" + r"\\{" + f"[{lo:.3f}, {hi:.3f}]" + "}}"
    return f"${v['mean']:.3f}$ ({v['sample_sd']:.3f})"


def _table(caption, headers, rows, partial=False):
    prefix = "PARTIAL: " if partial else ""
    body = [r"\begin{table}[htbp]", r"\centering\small", "\\caption{" + _tex(prefix + caption) + "}",
            r"\begin{tabularx}{\linewidth}{@{}l" + r">{\centering\arraybackslash}X" * (len(headers)-1) + "@{}}", r"\toprule",
            " & ".join(_tex(h) for h in headers) + r" \\", r"\midrule"]
    body += [" & ".join([_tex(r[0]), *r[1:]]) + r" \\" for r in rows]
    return "\n".join(body + [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])


def _condition_label(condition):
    if condition.startswith("gaussian_sd"): return "Gaussian noise SD " + condition[len("gaussian_sd"):]
    if condition.startswith("dropout"): return "pixel dropout " + condition[len("dropout"):]
    return {"gaussian_training_sd0.3_ratio0.1": "10% appended Gaussian contamination (SD 0.3)",
            "semantic_training_ratio0.1": "10% appended semantic contamination"}.get(condition, condition)


def _tables(output, lookup, partial):
    target = output / "tables"; target.mkdir()
    main, supplement, conditional, changes, contrasts = [], [], [], [], []
    for d in DATASETS:
        records = [lookup["summaries"][(d, m, "clean", "clean")] for m in METHODS]
        caption = DATA_LABELS[d] + ": clean test scores, five seeds; mean (sample SD)."
        main.append(_table(caption, ["Method", *[METRIC_LABELS[m] for m in PRIMARY]],
                           [[SHORT_METHODS[m], *[_cell(r, k) for k in PRIMARY]] for m,r in zip(METHODS,records)], partial))
        for metrics in (ALL_METRICS[:5], ALL_METRICS[5:]):
            supplement.append(_table(caption, ["Method", *[METRIC_LABELS[m] for m in metrics]],
                [[SHORT_METHODS[m], *[_cell(r, k) for k in metrics]] for m,r in zip(METHODS,records)], partial))
    absolute = lookup["summaries"]
    for d,tc,ic in sorted({(k[0],k[2],k[3]) for k in absolute if k[2:] != ("clean","clean")}):
        methods = METHODS if tc == "clean" else CONTAMINATION_METHODS
        records = [absolute[d,m,tc,ic] for m in methods]
        caption = f"{DATA_LABELS[d]}: training {_condition_label(tc)}, inference {_condition_label(ic)}. Absolute test scores, five seeds; mean (sample SD)."
        for metrics in (ALL_METRICS[:5], ALL_METRICS[5:]):
            conditional.append(_table(caption,["Method",*[METRIC_LABELS[m] for m in metrics]],
                [[SHORT_METHODS[m],*[_cell(r,k) for k in metrics]] for m,r in zip(methods,records)],partial))
    pairs = lookup["paired_condition_summaries"]
    groups = sorted({(k[0], k[2], k[3]) for k in pairs})
    for d, mode, c in groups:
        methods = METHODS if mode == "inference_corruption" else CONTAMINATION_METHODS
        caption = f"{DATA_LABELS[d]}: {mode}, {_condition_label(c)}. Delta = condition minus clean; mean (sample SD)."
        changes.append(_table(caption, ["Method", *[METRIC_LABELS[m] for m in PRIMARY]],
            [[SHORT_METHODS[m], *[_cell(pairs[d,m,mode,c], k) for k in PRIMARY]] for m in methods], partial))
    # Full paired detail is intentionally long-form: every metric and every interval.
    detail = [r"% Requires booktabs and longtable. All values are unit fractions."]
    for name in FIELDS:
        if name == "summaries": continue
        for key,row in sorted(lookup[name].items()):
            metrics = ROBUST + (FROZEN if name in ("paired_condition_summaries", "robustness_contrast_summaries") and row.get("mode") == "inference_corruption" else ())
            identity = DATA_LABELS[row["dataset"]] + ": " + (SHORT_METHODS[row["method"]] if "method" in row else
                SHORT_METHODS[row["method_a"]] + " minus " + SHORT_METHODS[row["method_b"]])
            if "condition" in row: identity += "; " + _condition_label(row["condition"])
            kind = {"paired_condition_summaries":"Paired changes (Delta)", "clean_contrast_summaries":"Clean differences (D clean)",
                    "condition_contrast_summaries":"Condition differences (D condition)", "robustness_contrast_summaries":"Differences in changes (Gamma)"}[name]
            caption = ("PARTIAL: " if partial else "") + kind + ". " + identity
            detail.extend([r"\begin{longtable}{lrr}", "\\caption{" + _tex(caption) + r"}\\", r"\toprule",
                           r"Metric & Mean (sample SD) & Mean [descriptive 95\% interval] \\", r"\midrule", r"\endfirsthead",
                           "\\caption[]{" + _tex(caption + " (continued)") + r"}\\",
                           r"\toprule Metric & Mean (sample SD) & Mean [descriptive 95\% interval] \\ \midrule\endhead"])
            detail += [_tex(METRIC_LABELS[k]) + " & " + _cell(row,k) + " & " + _cell(row,k,True,stack=False) + r" \\" for k in metrics]
            detail += [r"\bottomrule", r"\end{longtable}", ""]
    for d, mode, c in sorted({(k[0], k[5], k[6]) for k in lookup["condition_contrast_summaries"]}):
        rows = []
        for key, row in lookup["condition_contrast_summaries"].items():
            if (key[0],key[5],key[6]) != (d,mode,c): continue
            rows.append([SHORT_METHODS[row["method_a"]] + " - " + SHORT_METHODS[row["method_b"]],
                _cell(lookup["clean_contrast_summaries"][key[:5]], "ari", True), _cell(row,"ari",True),
                _cell(lookup["robustness_contrast_summaries"][key],"ari",True)])
        contrasts.append(_table(f"{DATA_LABELS[d]}: {mode}, {_condition_label(c)}. ARI paired contrasts; mean [descriptive 95% interval].",
                                ["A - B", "D clean", "D condition", "Gamma"], rows, partial))
    for name, content in (("clean_main.tex",main), ("clean_nine_metrics.tex",supplement),
                          ("conditional_nine_metrics.tex",conditional),
                          ("paired_changes_main.tex",changes), ("paired_contrasts_ari.tex",contrasts),
                          ("paired_all_metrics.tex",detail)):
        (target / name).write_text("% Generated from an explicitly SHA256-anchored analysis; no winner selection.\n" + "\n".join(content))


def degradation_series(rows):
    """Preserve dose order and missing gaps; reverse interval endpoints for -Delta."""
    means, lower, upper = [], [], []
    for row in rows:
        if row is None or row["status"] != COMPLETE:
            means.append(float("nan")); lower.append(float("nan")); upper.append(float("nan")); continue
        value = row["scores"]["ari"]; lo, hi = value["descriptive_t_interval_95"]
        means.append(-value["mean"]); lower.append(-hi); upper.append(-lo)
    return np.array(means), np.array(lower), np.array(upper)


def _figures(output, lookup, partial):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper1-matched-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 170})
    target = output / "figures"; target.mkdir()
    produced = []
    prefix = "PARTIAL - " if partial else ""
    note = "Five seeds; descriptive unadjusted 95% t intervals. No significance tests."
    def save(fig, name, title, foot, top=.94):
        fig.suptitle(prefix + title, fontsize=12)
        fig.text(.5,.015, foot + ("\nIncomplete groups withheld; see readiness.json." if partial else ""), ha="center", fontsize=9)
        fig.tight_layout(rect=(0,.09 if partial else .075,1,top))
        for ext in ("png","pdf"):
            fig.savefig(target / (name + "." + ext), bbox_inches="tight")
        plt.close(fig); produced.append(name)
    clean = lookup["summaries"]
    datasets = [d for d in DATASETS if any(clean[d,m,"clean","clean"]["status"] == COMPLETE for m in METHODS)]
    if datasets:
        columns = min(2, len(datasets)); rows = math.ceil(len(datasets) / columns)
        fig, axs = plt.subplots(rows,columns,figsize=(8, 3.8*rows), squeeze=False)
        for ax,d in zip(axs.flat,datasets):
            for j,m in enumerate(METHODS):
                row = clean[d,m,"clean","clean"]
                if row["status"] != COMPLETE:
                    ax.text(.99,j,"pending",transform=ax.get_yaxis_transform(),ha="right",color="0.5"); continue
                v = row["scores"]["ari"]
                ax.scatter(v["values_by_seed"], j + np.linspace(-.13,.13,5), s=18, color="0.55", alpha=.8)
                ax.scatter([v["mean"]],[j],marker="D",color="black",s=24)
            ax.set_yticks(range(len(METHODS)), [SHORT_METHODS[m] for m in METHODS]); ax.invert_yaxis()
            ax.set_title(DATA_LABELS[d]); ax.set_xlabel("Clean test ARI"); ax.grid(axis="x",alpha=.2)
        for ax in list(axs.flat)[len(datasets):]: ax.set_axis_off()
        save(fig,"clean_ari_seeds","Clean matched panel: ARI", "Dots: seeds 101--105 in fixed vertical order; diamond: five-seed mean.")
    pairs = lookup["paired_condition_summaries"]
    colors = plt.get_cmap("tab10")(np.arange(len(METHODS)))
    for d in DATASETS[:3]:
        if not any(k[0] == d and k[2] == "inference_corruption" and r["status"] == COMPLETE for k,r in pairs.items()): continue
        fig,axs = plt.subplots(1,2,figsize=(8,4.8))
        for ax,conditions,x,label in zip(axs,(INFERENCE_CONDITIONS[:3],INFERENCE_CONDITIONS[3:]),([.1,.2,.3],[.25,.5,.75]),("Gaussian pixel-noise SD","Pixel-dropout probability")):
            if not any(pairs[d,m,"inference_corruption",c]["status"] == COMPLETE for m,c in product(METHODS,conditions)):
                ax.text(.5,.5,"Pending\nNo complete five-seed groups",ha="center",va="center",transform=ax.transAxes,color="0.4")
                ax.set_title(label); ax.set_axis_off(); continue
            for j,m in enumerate(METHODS):
                ys,lo,hi = degradation_series([pairs[d,m,"inference_corruption",c] for c in conditions])
                if not np.isfinite(ys).any(): continue
                ax.errorbar(np.asarray(x)+(j-4.5)*(x[-1]-x[0])*.008,ys,yerr=np.vstack([ys-lo,hi-ys]),
                            color=colors[j],marker="o",markersize=3,linewidth=.9,capsize=2,label=SHORT_METHODS[m])
            ax.axhline(0,color="0.4",linewidth=.7); ax.set_xticks(x); ax.set_xlabel(label)
            ax.set_ylabel("ARI degradation:\nclean minus corrupted"); ax.grid(alpha=.15)
        from matplotlib.lines import Line2D
        handles=[Line2D([0],[0],color=colors[j],marker="o",linewidth=.9,markersize=3) for j in range(len(METHODS))]
        labels=[SHORT_METHODS[m]+(" (pending)" if not any(pairs[d,m,"inference_corruption",c]["status"]==COMPLETE for c in INFERENCE_CONDITIONS) else "") for m in METHODS]
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.90),ncol=5,fontsize=9,
                   columnspacing=1.0,handlelength=1.6,handletextpad=.5)
        # Reserve extra top space for a fixed-order method legend.
        save(fig,f"degradation_{d}",DATA_LABELS[d] + ": paired inference degradation",note,top=.80)
    def forest(ax, rows, labels, xlabel):
        for j,row in enumerate(rows):
            if row is None or row["status"] != COMPLETE:
                ax.text(.98,j,"pending",transform=ax.get_yaxis_transform(),ha="right",color="0.5"); continue
            v=row["scores"]["ari"]; lo,hi=v["descriptive_t_interval_95"]
            ax.errorbar(v["mean"],j,xerr=[[v["mean"]-lo],[hi-v["mean"]]],fmt="o",color="black",capsize=3,markersize=4)
        ax.axvline(0,color="0.5",linewidth=.8); ax.set_yticks(range(len(labels)),labels); ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.grid(axis="x",alpha=.2)
    for d in DATASETS:
        rows = [next(r for k,r in lookup["clean_contrast_summaries"].items() if k[0]==d and k[1]==n) for n,_,_ in CORE_CONTRASTS]
        if not any(r["status"]==COMPLETE for r in rows): continue
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        forest(ax,rows,[SHORT_METHODS[a]+" - "+SHORT_METHODS[b] for _,a,b in CORE_CONTRASTS],"D clean: ARI(A) minus ARI(B)")
        save(fig,f"clean_contrasts_{d}",DATA_LABELS[d]+": prespecified clean contrasts",note)
    for d in DATASETS[:2]:
        if not any(k[0]==d and k[2]=="retrained" and r["status"]==COMPLETE for k,r in pairs.items()): continue
        fig,axs=plt.subplots(1,2,figsize=(8,4))
        for ax,c in zip(axs,CONTAMINATION_CONDITIONS):
            forest(ax,[pairs[d,m,"retrained",c] for m in CONTAMINATION_METHODS],
                   [SHORT_METHODS[m] for m in CONTAMINATION_METHODS],"Delta ARI:\ncontaminated training minus clean training")
            ax.set_title(_condition_label(c),fontsize=9)
        save(fig,f"training_changes_{d}",DATA_LABELS[d]+": paired training-contamination changes",note)
    # Fixed core contrasts; optional all-pairwise expansions remain in the tables.
    dc=lookup["condition_contrast_summaries"]; gamma=lookup["robustness_contrast_summaries"]
    for d in DATASETS[:3]:
        families=[("gaussian", "inference_corruption",INFERENCE_CONDITIONS[:3]),("dropout","inference_corruption",INFERENCE_CONDITIONS[3:])]
        if d in DATASETS[:2]: families.append(("training","retrained",CONTAMINATION_CONDITIONS))
        for family,mode,conditions in families:
            selected=[(n,a,b) for n,a,b in CORE_CONTRASTS if mode != "retrained" or a in CONTAMINATION_METHODS and b in CONTAMINATION_METHODS]
            keys=[(d,n,a,b,"prespecified_mechanism_or_named_comparison",mode,c) for c in conditions for n,a,b in selected]
            if not any(dc[k]["status"]==COMPLETE or gamma[k]["status"]==COMPLETE for k in keys): continue
            fig,axs=plt.subplots(len(conditions),2,figsize=(8,2.9*len(conditions)+1),squeeze=False)
            for i,c in enumerate(conditions):
                keys=[(d,n,a,b,"prespecified_mechanism_or_named_comparison",mode,c) for n,a,b in selected]
                labels=[SHORT_METHODS[a]+" - "+SHORT_METHODS[b] for _,a,b in selected]
                forest(axs[i,0],[dc[k] for k in keys],labels,"D condition:\nARI(A) minus ARI(B)")
                forest(axs[i,1],[gamma[k] for k in keys],labels,"Gamma:\nDelta ARI(A) minus Delta ARI(B)")
                for ax in axs[i]: ax.set_title(_condition_label(c),fontsize=9)
            save(fig,f"condition_contrasts_{d}_{family}",DATA_LABELS[d]+": condition score and change contrasts",note)
    return produced


def generate_report(analysis_manifest, output, *, analysis_manifest_sha256, allow_partial=False, inventory_only=False):
    """Write once; authenticate inputs and validate all groups before rendering."""
    source=Path(analysis_manifest).resolve(); output=Path(output).resolve()
    manifest=_load_anchored(source,analysis_manifest_sha256)
    if manifest.get("schema_version")!=SCHEMA or manifest.get("status")!="completed" or type(manifest.get("inventory_only")) is not bool:
        raise ValueError("A completed evaluation analysis manifest is required")
    inventory_only=inventory_only or manifest["inventory_only"]
    name="inventory.json" if inventory_only else "evaluation.json"
    digest=manifest.get("artifact_sha256",{}).get(name)
    value=_load_anchored(source.parent/name,digest)
    if value.get("schema_version")!=SCHEMA or value.get("provenance")!=manifest.get("provenance"):
        raise ValueError("Analysis provenance/schema differs from its manifest")
    lookup=None
    if inventory_only:
        if value.get("labels_loaded") is not False or value.get("expected")!=EXPECTED_COUNTS:
            raise ValueError("Invalid label-blind inventory schema")
        verified=value.get("verified",{})
        if set(verified)!=set(EXPECTED_COUNTS) or any(type(verified[k]) is not int or not 0<=verified[k]<=v for k,v in EXPECTED_COUNTS.items()):
            raise ValueError("Invalid inventory counts")
        readiness=dict(expected=EXPECTED_COUNTS,verified=verified,missing_or_failed=value.get("missing_or_failed",[]),
                       partial=verified!=EXPECTED_COUNTS or bool(value.get("missing_or_failed")),
                       group_coverage={},pending_groups=[],scope="Artifacts visible in this anchored inventory snapshot; not live HPC status")
    else:
        lookup,readiness=validate_evaluation(value)
        if readiness["partial"] and not allow_partial:
            raise ValueError("Incomplete campaign/five-seed groups: explicit --allow-partial is required")
    output.mkdir(parents=True,exist_ok=False)
    provenance=dict(analysis_manifest=str(source),analysis_manifest_sha256=analysis_manifest_sha256,
                    input_artifact=name,input_artifact_sha256=digest,upstream=manifest.get("provenance"),
                    report_source_sha256=_hash_bytes(Path(__file__).read_bytes()),
                    workload_source_sha256=_hash_bytes(Path(__file__).with_name("workload.py").read_bytes()),
                    protocol_source_sha256=_hash_bytes(Path(__file__).with_name("final_protocol.py").read_bytes()))
    try:
        readiness.update(schema_version=REPORT_SCHEMA,inventory_only=inventory_only,labels_loaded=False,
                         primary_metric="ari",allow_partial=allow_partial,provenance=provenance)
        _write_json(output/"readiness.json",readiness)
        figures=[]
        has_scores=lookup is not None and any(v["complete"] for v in readiness["group_coverage"].values())
        if has_scores:
            _tables(output,lookup,readiness["partial"])
            figures=_figures(output,lookup,readiness["partial"])
        notes=NOTES+value.get("caveats",[])
        (output/"README.txt").write_text(("INVENTORY ONLY" if inventory_only else "PARTIAL REPORT" if readiness["partial"] else "COMPLETE MATCHED REPORT")+
            "\n\n"+"\n".join(notes)+"\n\nLaTeX fragments require booktabs, tabularx, array and longtable. They are not a standalone document.\n"+
            "Method key:\n"+"\n".join(f"{SHORT_METHODS[m]}: {METHOD_LABELS[m]} ({m})" for m in METHODS)+
            "\n\nNo score panels were emitted.\n"*(not has_scores))
        result=dict(schema_version=REPORT_SCHEMA,status="completed",created_utc=datetime.now(timezone.utc).isoformat(),
                    inventory_only=inventory_only,partial=readiness["partial"],labels_loaded=False,provenance=provenance,
                    figures=figures,score_panels_emitted=has_scores,versions=dict(python=platform.python_version(),numpy=np.__version__),
                    artifact_sha256={str(p.relative_to(output)):_hash_bytes(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()})
        if figures:
            import matplotlib
            result["versions"]["matplotlib"]=matplotlib.__version__
        _write_json(output/"report_manifest.json",result)
        return result
    except Exception as error:
        _write_json(output/"FAILED.json",dict(status="failed",error=f"{type(error).__name__}: {error}",provenance=provenance))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-manifest",type=Path,required=True)
    parser.add_argument("--analysis-manifest-sha256",required=True,help="Digest obtained from a trusted analysis-completion record, not discovered automatically")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--inventory-only",action="store_true",help="Render only the anchored inventory readiness ledger")
    args=parser.parse_args()
    result=generate_report(args.analysis_manifest,args.output,analysis_manifest_sha256=args.analysis_manifest_sha256,
                           allow_partial=args.allow_partial,inventory_only=args.inventory_only)
    print(json.dumps({k:result[k] for k in ("status","partial","inventory_only","score_panels_emitted")}))


if __name__=="__main__":
    main()
