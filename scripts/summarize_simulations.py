"""Read-only scientific synthesis of fresh S2/S3 simulation artifacts.

This script does not optimize models or change raw artifacts. Integrity checks
are owned by the parent audit; a run enters scientific summaries only when its
saved run status and all saved quality/publication flags pass. Source numerical
repairs may supersede these provisional summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def statistics(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a)>1 else None,
                se=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,
                min=float(a.min()), median=float(np.median(a)), max=float(a.max()),
                values=a.tolist())


def key(config):
    return tuple(config[k] for k in ("module", "truth", "noise_sd", "nuisance_dimension", "family"))


def labeled(k):
    return dict(zip(("module", "truth", "noise_sd", "nuisance_dimension", "family"), k))


def read_runs(source):
    accepted, failed = [], []
    for directory in sorted(source.glob("index*"), key=lambda p:int(p.name[5:])):
        config=json.loads((directory/"config.json").read_text())
        run=json.loads((directory/"run.json").read_text())
        result=json.loads((directory/"result.json").read_text()) if (directory/"result.json").exists() else {}
        item=dict(config=config,run=run,result=result,directory=str(directory))
        if (run["status"]=="completed" and run.get("quality_gate_passed") is True
                and run.get("publishable_numerical_result") is True
                and result.get("accuracy_passed") is True and result.get("stationarity_passed") is True
                and result.get("validation_only") is False and result.get("smoke_only") is False):
            accepted.append(item)
        else:
            diagnostics=[]
            for component in result.get("components",[]):
                for start in component["optimization"].get("starts",[]):
                    if not start["accuracy_passed"] or not start["stationarity_passed"]:
                        diagnostics.append(dict(component=component["component"],**start))
            failed.append(dict(**labeled(key(config)),index=config["index"],seed=config["seed"],
                               status=run["status"],error=run.get("error"),
                               accuracy_passed=result.get("accuracy_passed"),
                               stationarity_passed=result.get("stationarity_passed"),
                               failing_start_diagnostics=diagnostics))
    return accepted,failed


def posterior_dfs(item):
    c,r=item["config"],item["result"]
    if c["family"] not in ("student_q_student_prior","direct_student"):
        return []
    directory=Path(item["directory"]); n=r["actual_n_fit"]
    names=["fitted.npz"] if c["module"].startswith("S2") else ["component_0.npz","component_1.npz"]
    output=[]
    for name in names:
        p=np.load(directory/name)["parameters"]
        if not len(p):
            continue
        qnu=2+np.exp(p[-n:])
        output.append(dict(component_file=name,n=len(qnu),lower_bound_count=int(np.sum(qnu<=2.10001)),
                           upper_bound_count=int(np.sum(qnu>=199.99)),min=float(qnu.min()),
                           median=float(np.median(qnu)),max=float(qnu.max())))
    return output


def summarize(accepted,failed):
    groups=defaultdict(list)
    for item in accepted: groups[key(item["config"])].append(item)
    s2=[];s3=[]
    for k,items in groups.items():
        row=dict(**labeled(k),n=len(items),seeds=[i["config"]["seed"] for i in items],
                 indices=[i["config"]["index"] for i in items])
        if k[0].startswith("S2"):
            dfs=[i["result"]["finite_candidate_nu"] for i in items]
            row.update(finite_candidate_nu=statistics(dfs),
                       finite_upper_bound_count=sum(v>=199.99 for v in dfs),
                       finite_lower_bound_count=sum(v<=2.10001 for v in dfs),
                       selected_gaussian_count=sum(i["result"]["selected_gaussian_limit"] for i in items))
            row["seed_outcomes"]=[dict(seed=i["config"]["seed"],
                                        finite_candidate_nu=i["result"]["finite_candidate_nu"],
                                        selected_nu=i["result"]["selected_nu"],
                                        selected_gaussian_limit=i["result"]["selected_gaussian_limit"],
                                        heldout_ll=i["result"]["independent_heldout_log_likelihood"])
                                  for i in items]
            for field in ("finite_candidate_mean_gap","selected_mean_gap","independent_heldout_log_likelihood",
                          "finite_candidate_mean_objective","selected_mean_objective"):
                row[field]=statistics([i["result"][field] for i in items])
            row["finite_minus_gaussian_training_objective"]=statistics([
                i["result"]["finite_candidate_mean_objective"]-i["result"]["gaussian_limit_mean_objective"] for i in items])
            s2.append(row)
        else:
            for field in ("assignment_disagreement","responsibility_max_difference"):
                row[field]=statistics([i["result"][field] for i in items])
            for field in ("accuracy","ari","log_loss","predicted_component0_fraction"):
                row["fitted_"+field]=statistics([i["result"]["fitted"][field] for i in items])
                row["bayes_"+field]=statistics([i["result"]["exact_structured"][field] for i in items])
                row[field+"_minus_bayes"]=statistics([i["result"]["fitted"][field]-i["result"]["exact_structured"][field] for i in items])
            row["component_mean_gaps"]=[statistics([i["result"]["components"][j]["mean_gap"] for i in items]) for j in (0,1)]
            direction=[]
            for item in items:
                a=np.load(Path(item["directory"])/"assignments.npz")
                p=a["variational"].argmax(1);b=a["exact_structured"].argmax(1)
                direction.append(dict(seed=item["config"]["seed"],
                                      bayes0_to_variational1=float(np.mean((b==0)&(p==1))),
                                      bayes1_to_variational0=float(np.mean((b==1)&(p==0)))))
            row["assignment_direction"]=direction
            s3.append(row)
    byconfig={(i["config"]["module"],i["config"]["truth"],i["config"]["noise_sd"],
               i["config"]["nuisance_dimension"],i["config"]["family"],i["config"]["seed"]):i for i in accepted}
    paired=[]
    for k,items in groups.items():
        module,truth,sd,m,family=k
        if not (module=="S2_primary" and m==20 or module=="S3" and m==100):continue
        pairs=[]
        for item in items:
            seed=item["config"]["seed"]
            base=byconfig.get((module,truth,sd,0,family,seed))
            if base is None:continue
            r0,r1=base["result"],item["result"]
            assert r0["data_array_sha256"]["fit_x"]==r1["data_array_sha256"]["fit_x"]
            delta=dict(seed=seed,base_index=base["config"]["index"],nuisance_index=item["config"]["index"])
            if module=="S2_primary":
                assert r0["data_array_sha256"]["heldout_x"]==r1["data_array_sha256"]["heldout_x"]
                delta.update(finite_df_delta=r1["finite_candidate_nu"]-r0["finite_candidate_nu"],
                             heldout_ll_delta=r1["independent_heldout_log_likelihood"]-r0["independent_heldout_log_likelihood"],
                             finite_gap_delta=r1["finite_candidate_mean_gap"]-r0["finite_candidate_mean_gap"])
            else:
                delta.update(assignment_disagreement_delta=r1["assignment_disagreement"]-r0["assignment_disagreement"],
                             log_loss_delta=r1["fitted"]["log_loss"]-r0["fitted"]["log_loss"],
                             accuracy_delta=r1["fitted"]["accuracy"]-r0["fitted"]["accuracy"],
                             component0_fraction_delta=r1["fitted"]["predicted_component0_fraction"]-r0["fitted"]["predicted_component0_fraction"])
            pairs.append(delta)
        fields=[f for f in pairs[0] if f.endswith("delta")] if pairs else []
        paired.append(dict(**labeled(k),base_dimension=0,n_complete_pairs=len(pairs),planned_pairs=5,
                           pairs=pairs,statistics={f:statistics([p[f] for p in pairs]) for f in fields}))
    relative=[]
    for k,items in groups.items():
        if not k[0].startswith("S2") or k[4]=="structured_q_student_prior":continue
        pairs=[]
        for item in items:
            c=item["config"];r=item["result"]
            base=byconfig[("S2_primary",c["truth"],c["noise_sd"],0,"structured_q_student_prior",c["seed"])]
            assert base["result"]["data_array_sha256"]["heldout_x"]==r["data_array_sha256"]["heldout_x"]
            fitted=np.load(Path(item["directory"])/"fitted.npz")["heldout_log_evidence"]
            structured=np.load(Path(base["directory"])/"fitted.npz")["heldout_log_evidence"]
            difference=fitted-structured
            pairs.append(dict(seed=c["seed"],method_minus_structured_mean_ll=float(difference.mean()),
                              conditional_heldout_se=float(difference.std(ddof=1)/np.sqrt(len(difference)))))
        relative.append(dict(**labeled(k),paired=statistics([p["method_minus_structured_mean_ll"] for p in pairs]),
                             pairs=pairs,reference="same-seed structured m0 exact-model fit; nuisance-invariant"))
    rows=[]
    for item in accepted:
        c,r=item["config"],item["result"]
        rows.append(dict(**labeled(key(c)),index=c["index"],seed=c["seed"],posterior_dfs=posterior_dfs(item)))
    family_totals=[]
    for family in dict.fromkeys(row["family"] for row in s2):
        these=[row for row in s2 if row["family"]==family]
        family_totals.append(dict(family=family,n=sum(r["n"] for r in these),
                                 selected_gaussian_count=sum(r["selected_gaussian_count"] for r in these),
                                 finite_upper_bound_count=sum(r["finite_upper_bound_count"] for r in these),
                                 finite_lower_bound_count=sum(r["finite_lower_bound_count"] for r in these)))
    posterior_groups=defaultdict(lambda:dict(n=0,upper_bound_count=0,lower_bound_count=0))
    for row in rows:
        for q in row["posterior_dfs"]:
            entry=posterior_groups[key(row)]
            for field in entry:entry[field]+=q[field]
    posterior_totals=[dict(**labeled(k),**value) for k,value in posterior_groups.items()]
    noise_pairs=[]
    for k,items in groups.items():
        module,truth,sd,m,family=k
        if not module.startswith("S2") or truth!="t5" or sd==.5:continue
        pairs=[]
        for item in items:
            seed=item["config"]["seed"]
            base=byconfig.get((module,truth,.5,m,family,seed))
            if base is None:continue
            r0,r1=base["result"],item["result"]
            assert r0["data_array_sha256"]["fit_signal"]==r1["data_array_sha256"]["fit_signal"]
            s0=byconfig[("S2_primary",truth,.5,0,"structured_q_student_prior",seed)]["result"]
            s1=byconfig[("S2_primary",truth,sd,0,"structured_q_student_prior",seed)]["result"]
            pairs.append(dict(seed=seed,finite_df_delta=r1["finite_candidate_nu"]-r0["finite_candidate_nu"],
                              gaussian_selected_at_low_noise=r0["selected_gaussian_limit"],
                              gaussian_selected_at_high_noise=r1["selected_gaussian_limit"],
                              method_minus_structured_heldout_ll_delta=(r1["independent_heldout_log_likelihood"]-s1["independent_heldout_log_likelihood"])
                                -(r0["independent_heldout_log_likelihood"]-s0["independent_heldout_log_likelihood"])))
        noise_pairs.append(dict(**labeled(k),base_noise_sd=.5,n_complete_pairs=len(pairs),pairs=pairs,
                                finite_df_delta=statistics([p["finite_df_delta"] for p in pairs]),
                                method_minus_structured_heldout_ll_delta=statistics([p["method_minus_structured_heldout_ll_delta"] for p in pairs])))
    return dict(status="provisional; original final-release quality gates only, pending independent repaired quadrature audit",
                counts=dict(total=len(accepted)+len(failed),accepted=len(accepted),failed=len(failed),
                            accepted_s2=sum(i["config"]["module"].startswith("S2") for i in accepted),
                            accepted_s3=sum(i["config"]["module"]=="S3" for i in accepted)),
                failed_runs=failed,s2_groups=s2,s3_groups=s3,nuisance_pairs=paired,noise_pairs=noise_pairs,
                s2_relative_to_structured=relative,accepted_parameter_diagnostics=rows,
                s2_boundary_totals=family_totals,posterior_boundary_totals=posterior_totals,
                uncertainty="Group SE uses five independent seed means, or four where indicated; heldout conditional SE is not training uncertainty. No p-values or multiplicity-adjusted inferential claim.")


def text_report(summary):
    lines=["Fresh S2/S3 scientific simulation synthesis (provisional)",
           "Only saved successful final runs enter the numerical summaries; failed outputs are excluded.",
           json.dumps(summary["counts"]),"",
           "S2: df denotes finite candidate, G explicit Gaussian-limit selections, cap finite df>=199.99."]
    for row in summary["s2_groups"]:
        lines.append(f"{row['truth']} sd={row['noise_sd']:g} m={row['nuisance_dimension']} {row['family']}: "
                     f"n={row['n']}; df mean/median/range={row['finite_candidate_nu']['mean']:.5g}/"
                     f"{row['finite_candidate_nu']['median']:.5g}/[{row['finite_candidate_nu']['min']:.5g},{row['finite_candidate_nu']['max']:.5g}]; "
                     f"G={row['selected_gaussian_count']}/{row['n']}, cap={row['finite_upper_bound_count']}/{row['n']}; "
                     f"finite gap={row['finite_candidate_mean_gap']['mean']:.7g}; selected heldout LL={row['independent_heldout_log_likelihood']['mean']:.9f}")
    lines += ["", "Paired heldout LL difference, method minus same-seed structured (positive favors method):"]
    for row in summary["s2_relative_to_structured"]:
        z=row["paired"]
        lines.append(f"{row['truth']} sd={row['noise_sd']:g} m={row['nuisance_dimension']} {row['family']}: "
                     f"mean={z['mean']:.8g}, seed SE={z['se']:.5g}, n={z['n']}, values={z['values']}")
    lines += ["", "S2 prior-boundary totals (finite cap and selected Gaussian limit are distinct quantities):"]
    for row in summary["s2_boundary_totals"]:
        lines.append(f"{row['family']}: n={row['n']}, Gaussian-selected={row['selected_gaussian_count']}, "
                     f"finite-candidate upper cap={row['finite_upper_bound_count']}, lower cap={row['finite_lower_bound_count']}")
    lines += ["", "Direct Student posterior-boundary counts (S2 refers to finite candidate, including rejected candidates):"]
    for row in summary["posterior_boundary_totals"]:
        lines.append(f"{row['module']} {row['truth']} sd={row['noise_sd']:g} m={row['nuisance_dimension']}: "
                     f"upper={row['upper_bound_count']}/{row['n']}, lower={row['lower_bound_count']}/{row['n']}")
    lines += ["", "Paired noise comparisons (same latent-signal draws, base noiseSD0.5):"]
    for row in summary["noise_pairs"]:
        lines.append(f"{row['family']} m={row['nuisance_dimension']} noiseSD0.5->{row['noise_sd']:g}: "
                     f"n={row['n_complete_pairs']}, finite-df delta={row['finite_df_delta']['mean']:.8g}; "
                     f"change in method-minus-structured heldout LL={row['method_minus_structured_heldout_ll_delta']['mean']:.8g}")
    lines += ["", "S3: disagreement is against Bayes assignments; deltas are variational minus Bayes."]
    for row in summary["s3_groups"]:
        lines.append(f"{row['truth']} sd={row['noise_sd']:g} m={row['nuisance_dimension']} {row['family']}: "
                     f"n={row['n']}; disagreement={row['assignment_disagreement']['mean']:.6%}; "
                     f"accuracy delta={row['accuracy_minus_bayes']['mean']:.6%}; "
                     f"logloss delta={row['log_loss_minus_bayes']['mean']:.8g}; "
                     f"component0-fraction delta={row['predicted_component0_fraction_minus_bayes']['mean']:.6%}; "
                     f"component gaps={[v['mean'] for v in row['component_mean_gaps']]}")
    lines += ["", "Paired nuisance-dimension comparisons (exclude incomplete pairs):"]
    for row in summary["nuisance_pairs"]:
        lines.append(f"{row['truth']} sd={row['noise_sd']:g} m=0->{row['nuisance_dimension']} {row['family']}: "
                     f"pairs={row['n_complete_pairs']}/5; " + "; ".join(f"{k} mean={v['mean']:.8g}, SE={v['se']:.5g}" for k,v in row['statistics'].items()))
    lines += ["", "Excluded failed indices: "+(", ".join(str(r["index"]) for r in summary["failed_runs"]) or "none")]
    for failed in summary["failed_runs"]:
        lines.append(f"index{failed['index']}: error={failed['error']}; accuracy={failed['accuracy_passed']}; stationarity={failed['stationarity_passed']}")
    for row in summary["nuisance_pairs"]:
        if row["module"]=="S3" and row["n_complete_pairs"]<5:
            lines.append(f"Incomplete paired nuisance comparison: {row['truth']} noiseSD{row['noise_sd']:g} {row['family']}, {row['n_complete_pairs']}/5 pairs.")
    lines += [
              "", "Interpretation limits:",
              "Separate informative t5/noiseSD0.5 settings from high-noise settings; exact structured fitting can also select Gaussian with noisy finite data.",
              "Compare paired heldout differences at each noise and dimension; exact inference does not imply universal finite-sample predictive superiority.",
              "All Gaussian-limit candidate values use the exact Gaussian posterior; zero selected gap after selecting Gaussian is automatic and does not show a finite-t approximation was fixed.",
              "Coupled Gaussian-Gamma is already analytically nuisance-invariant; only m5 anchors were run, so empirical across-m invariance for that family is not estimated here.",
              "S3 knows the generative parameters and optimizes observation-specific conditionals; it is not learned/amortized clustering or a benchmark against VaDE/DEC.",
              "The scalar linear likelihood makes the structured posterior exact by construction. Gaussian-truth S3 controls invoke analytic posterior optima instead of testing optimization.",
              "Direct-t posterior df is capped at200, so its implemented finite-t family does not include Gaussian q for a finite Student prior; apparent small reversals need no contradiction.",
              "Finite prior candidates search df[2.1,200] plus an explicit Gaussian limit, not the whole interval(200,infinity). Two starts are not a global-optimum guarantee.",
              "Seed means are descriptive; use the accepted and paired sample counts above. This report asserts no significance test.",
              "Equal-df hard assignments, responsibility distortion, and component gaps are different diagnostics; shared gaps can cancel in allocation.",
              summary["status"],
              "", summary["uncertainty"]]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--status-note",help="Override the provenance/status sentence for a newly audited release.")
    args=parser.parse_args()
    accepted,failed=read_runs(args.source)
    summary=summarize(accepted,failed)
    if args.status_note:summary["status"]=args.status_note
    summary["source_directory"]=str(args.source.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"scientific_simulations.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (args.output/"scientific_simulations.txt").write_text(text_report(summary))
    print(json.dumps(summary["counts"]))


if __name__=="__main__":main()
