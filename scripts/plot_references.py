"""Static ARI/ACC figure from frozen evaluation.json; no labels or models read.

Run from the repository root:
  .venv/bin/python scripts/plot_references.py
Outputs are write-once. Complete-group summaries are read from the evaluation
record and checked against its displayed seed values; no summary is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper1_reference_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [("vade", "native", "VaDE"), ("dec", "native", "DEC"),
          ("idec", "native", "IDEC"), ("pro_t", "native", "Pro-t\nnative"),
          ("pro_t", "gmm_refit", "Pro-t\nGMM refit†")]
COLORS = {901: "#0072B2", 902: "#D55E00", 903: "#8B4C9E"}
JITTER = {901: -.16, 902: -.02, 903: .12}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(evaluation: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Figure directory already exists: {output}")
    result = json.loads(evaluation.read_text())
    if result["schema_version"] != "paper1-reference-endpoint-evaluation-v1" or result["primary_metric"] != "ari":
        raise ValueError("Unexpected evaluation protocol")
    if result["completed_fit_count"] != 12 or result["missing_runs"]:
        raise ValueError("Complete figure requires all12 verified fits")
    rows = [row for row in result["rows"] if row["split"] == "test"]
    summaries = {(s["method"],s["variant"]):s for s in result["summaries"] if s["split"] == "test"}
    if len({(r["method"],r["variant"],r["seed"]) for r in rows}) != len(rows):
        raise ValueError("Repeated seed predictions in evaluation")
    if len({r["metrics"]["n_samples"] for r in rows}) != 1:
        raise ValueError("Inconsistent test sample counts")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10.5,
                         "axes.labelsize":11.5, "axes.titlesize":12,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42,
                         "savefig.facecolor":"white", "axes.unicode_minus":False})
    fig, axes = plt.subplots(1,2,figsize=(11.8,5.8))
    fig.subplots_adjust(left=.068,right=.985,top=.755,bottom=.305,wspace=.19)
    plotted = []
    for ax,(score,title) in zip(axes,[("ari","(a) Adjusted Rand index"),("hungarian_acc","(b) Hungarian accuracy")]):
        ax.axvspan(3.5,4.5,color="#F1F3F5",zorder=0)
        ax.axvline(3.5,color="#B0B6BE",linewidth=1.,linestyle=(0,(3,3)),zorder=1)
        extrema = []
        for x,(method,variant,label) in enumerate(GROUPS):
            group = sorted((row for row in rows if (row["method"],row["variant"]) == (method,variant)),key=lambda r:r["seed"])
            if not group:raise ValueError(f"No completed rows for {method}/{variant}")
            seeds = [row["seed"] for row in group]
            values = [row["metrics"]["scores"][score] for row in group]
            if any(seed not in COLORS for seed in seeds) or not np.isfinite(values).all():
                raise ValueError("Unexpected seed/nonfinite metric")
            for row,value in zip(group,values):
                seed = row["seed"]
                ax.scatter(x+JITTER[seed],value,s=51,c=COLORS[seed],edgecolors="white",linewidths=.75,zorder=4)
            extrema.extend(values)
            summary = summaries[(method,variant)]
            record = {"panel":score,"method":method,"variant":variant,"seeds":seeds,"values":values}
            if summary["status"] == "complete_three_seed_group":
                if seeds != [901,902,903]:raise ValueError("Aggregate without all three seeds")
                mean,sd = summary["scores"][score]["mean"],summary["scores"][score]["sample_sd"]
                if not (math.isclose(mean,float(np.mean(values)),abs_tol=1e-12) and
                        math.isclose(sd,float(np.std(values,ddof=1)),abs_tol=1e-12)):
                    raise ValueError("Frozen summary disagrees with seed points")
                ax.errorbar(x+.25,mean,yerr=sd,fmt="D",markersize=4.6,color="#20262D",ecolor="#20262D",
                            capsize=4,elinewidth=1.25,capthick=1.25,zorder=3)
                extrema.extend([mean-sd,mean+sd])
                record.update(mean=mean,sample_sd=sd)
            elif "scores" in summary:
                raise ValueError("Incomplete group contains an aggregate")
            plotted.append(record)
        lower=math.floor((min(extrema)-.03)/.05)*.05
        upper=math.ceil((max(extrema)+.03)/.05)*.05
        ax.set_ylim(max(0.,lower),min(1.,upper));ax.set_xlim(-.5,4.5)
        ax.set_xticks(range(len(GROUPS)),[g[2] for g in GROUPS])
        ax.tick_params(axis="x",length=0,pad=10)
        ax.tick_params(axis="y",length=3,color="#7B8591")
        ax.yaxis.set_major_locator(MultipleLocator(.05));ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y",color="#DEE2E6",linewidth=.75,zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#A0A8B2");ax.spines["bottom"].set_color("#A0A8B2")
        ax.set_title(title,loc="left",pad=12,fontweight="bold")
        ax.text(4,.98,"DIAGNOSTIC",transform=ax.get_xaxis_transform(),ha="center",va="top",
                color="#606A76",fontsize=8.5,fontweight="normal")
    handles = [Line2D([],[],marker="o",linestyle="none",color=COLORS[seed],markersize=6.6,label=f"Seed {seed}") for seed in COLORS]
    handles.append(Line2D([],[],marker="D",linestyle="none",color="#20262D",markersize=5,label="Mean ± sample SD (3 seeds)"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.875),ncol=4,
               frameon=False,columnspacing=2.0,handletextpad=.6,fontsize=10)
    fig.suptitle("MNIST reference recipes: frozen test endpoints",x=.068,y=.98,ha="left",fontsize=16,fontweight="bold",color="#18212B")
    count=rows[0]["metrics"]["n_samples"]
    fig.text(.068,.92,f"Inductive 54k / 6k / 10k split  •  Test n = {count:,}  •  {result['completed_fit_count']} of 12 fits completed",
             ha="left",fontsize=10.8,color="#53606E")
    fig.text(.068,.195,"Each colored point is one seed. Black diamonds and error bars show mean ± sample SD only for complete three-seed groups.",fontsize=9.4,color="#37434F")
    fig.text(.068,.145,"All three prespecified seeds are complete. † Pro-t GMM refit is a separate diagnostic of the same trained models.",fontsize=9.4,color="#37434F")
    fig.text(.068,.095,"Architectures, objectives and training budgets differ. This is not a matched-comparison superiority test.",fontsize=9.4,color="#37434F")
    output.mkdir(parents=True,exist_ok=False)
    png,pdf = output/"reference_ari_acc.png",output/"reference_ari_acc.pdf"
    fig.savefig(png,dpi=300)
    fig.savefig(pdf,metadata={"Title":"MNIST reference recipes: frozen test endpoints",
                             "Subject":"Complete descriptive reference evaluation; all seed outcomes retained",
                             "Creator":"Matplotlib; reproducible reference_results_plot_complete_v2.py"})
    plt.close(fig)
    provenance={"schema_version":"paper1-reference-static-figure-v1","evaluation_path":str(evaluation.resolve()),
        "evaluation_sha256":file_hash(evaluation),"script_sha256":file_hash(Path(__file__)),
        "matplotlib_version":matplotlib.__version__,"numpy_version":np.__version__,
        "inputs":"Frozen evaluation JSON only; no labels, checkpoints or training files read",
        "plotted_data":plotted,"output_sha256":{p.name:file_hash(p) for p in [png,pdf]}}
    (output/"figure_provenance.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n")
    return png,pdf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",type=Path,default=ROOT/"results/paper1_review_20261029/reference_analysis_complete_v2/evaluation.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results/paper1_review_20261029/reference_figures_complete_v2")
    args=parser.parse_args()
    for path in plot(args.evaluation,args.output):print(path)


if __name__=="__main__":main()
