"""Reusable static scientific figures from scientific_simulations.py JSON.

Example:
  .venv/bin/python scripts/plot_simulations.py \
    --summary results/paper1_review_20261028/scientific_analysis/scientific_simulations.json \
    --output results/paper1_review_20261028/scientific_analysis --version interim_v1

The summary contains every plotted point; raw simulation files are not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import ScalarFormatter
import numpy as np


FAMILIES=["gaussian_q_student_prior","student_q_student_prior","structured_q_student_prior"]
NAMES={FAMILIES[0]:"Gaussian q",FAMILIES[1]:"Direct Student q",FAMILIES[2]:"Structured q",
       "gaussian":"Gaussian q","direct_student":"Direct Student q"}
COLORS={FAMILIES[0]:"#3464a4",FAMILIES[1]:"#ba6330",FAMILIES[2]:"#27836a",
        "gaussian":"#3464a4","direct_student":"#ba6330"}
MARKERS=["o","s","D","^","v"]


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":11,"axes.labelsize":10,"figure.titlesize":15,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.edgecolor":"#4b5260","axes.labelcolor":"#202630",
                         "text.color":"#202630","xtick.color":"#384252","ytick.color":"#384252",
                         "grid.color":"#dce1e6","grid.linewidth":.6,"pdf.fonttype":42,"ps.fonttype":42,
                         "savefig.facecolor":"white"})


def save(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def footer(summary,version,status_note):
    c=summary["counts"]
    complete=c["accepted"]==c["total"] and c["total"]>0 and c["failed"]==0
    conclusion=("Controlled known-model simulations; no trained-VAE benchmark claim." if complete
                else "Incomplete numerical panel; scientific results remain provisional.")
    return (f"{version.replace('_',' ').upper()} | {c['accepted']}/{c['total']} configurations passed; "
            f"{c['failed']} failures excluded. {status_note}\n"
            f"Points are individual seeds; short bars are seed means. {conclusion}")


def complete_s3_pairs(summary):
    """Require five matched seeds in every displayed noise/dimension/family cell."""
    rows={(r["noise_sd"],r["nuisance_dimension"],r["family"]):r
          for r in summary["s3_groups"] if r["truth"]=="t3_t30"}
    required={(sd,m,f) for sd in (.5,2.) for m in (0,100) for f in ("gaussian","direct_student")}
    if not required.issubset(rows):return False
    seed_sets=[set(rows[key]["seeds"]) for key in required]
    if not all(rows[key]["n"]==5 and len(rows[key]["assignment_direction"])==5 for key in required):return False
    if not all(seeds==seed_sets[0] and len(seeds)==5 for seeds in seed_sets):return False
    pairs={(r["noise_sd"],r["family"]):r for r in summary["nuisance_pairs"]
           if r["module"]=="S3" and r["truth"]=="t3_t30" and r["nuisance_dimension"]==100}
    return all((sd,f) in pairs and pairs[(sd,f)]["n_complete_pairs"]==5
               and pairs[(sd,f)]["planned_pairs"]==5
               for sd in (.5,2.) for f in ("gaussian","direct_student"))


def plot_s2(summary,output,version,status_note):
    rows=[r for r in summary["s2_groups"] if r["module"]=="S2_primary" and r["truth"]=="t5" and r["noise_sd"]==.5]
    groups={(r["family"],r["nuisance_dimension"]):r for r in rows}
    seeds=sorted({s["seed"] for r in rows for s in r["seed_outcomes"]})
    marker=dict(zip(seeds,MARKERS))
    fig=plt.figure(figsize=(11.8,6.4))
    grid=fig.add_gridspec(2,2,height_ratios=[.52,3.7],width_ratios=[1.2,1],
                          left=.09,right=.97,bottom=.30,top=.82,hspace=.06,wspace=.29)
    infinite=fig.add_subplot(grid[0,0]);ax=fig.add_subplot(grid[1,0],sharex=infinite)
    deltaax=fig.add_subplot(grid[:,1])
    fig.suptitle("Irrelevant latent coordinates distort learned tail weight",x=.09,y=.96,ha="left",fontweight="bold")
    fig.text(.09,.907,"S2: true Student t(5), unit signal variance, Gaussian observation noise SD 0.5",fontsize=10.5)
    infinite.set_title("A   Selected prior degrees of freedom",loc="left",pad=14)
    infinite.set_facecolor("#f1f3f6")
    infinite.set_ylim(-.5,.5);infinite.set_yticks([0],["Gaussian\nlimit"])
    infinite.tick_params(axis="x",which="both",bottom=False,labelbottom=False)
    infinite.tick_params(axis="y",length=0)
    infinite.spines["left"].set_visible(False);infinite.spines["bottom"].set_visible(False)
    ax.set_xlim(-.38,1.38);ax.set_yscale("log");ax.set_ylim(2.9,255)
    ax.set_yticks([3,5,10,30,100,200]);ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off();ax.set_xticks([0,1],["0","20"])
    ax.set_xlabel("Number of ignored latent coordinates, m")
    ax.set_ylabel("Finite degrees of freedom, ν")
    ax.grid(axis="y",alpha=.65)
    ax.axhline(5,color="#68717c",linewidth=.9,linestyle=(0,(3,3)),zorder=0)
    ax.text(.985,5.25,"true ν = 5",ha="right",fontsize=8.5,color="#68717c",transform=ax.get_yaxis_transform())
    ax.axhline(200,color="#9da5ae",linewidth=.8,linestyle=":",zorder=0)
    ax.text(.02,210,"finite search cap",fontsize=8,color="#68717c",transform=ax.get_yaxis_transform())
    for family_index,family in enumerate(FAMILIES):
        outcomes={m:{s["seed"]:s for s in groups[(family,m)]["seed_outcomes"]} for m in (0,20)}
        for seed_index,seed in enumerate(seeds):
            offset=(family_index-1)*.15+(seed_index-2)*.022
            points=[]
            for xpos,m in enumerate((0,20)):
                point=outcomes[m][seed]
                selected_gaussian=point["selected_gaussian_limit"]
                target=infinite if selected_gaussian else ax
                y=0 if selected_gaussian else point["finite_candidate_nu"]
                target.scatter(xpos+offset,y,s=41,marker=marker[seed],facecolor=COLORS[family],
                               edgecolor="white",linewidth=.55,zorder=4)
                points.append((xpos+offset,y,target))
            p0,p1=points
            if p0[2] is p1[2]:
                ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=COLORS[family],alpha=.22,linewidth=.8,zorder=1)
            else:
                fig.add_artist(ConnectionPatch((p0[0],p0[1]),(p1[0],p1[1]),coordsA="data",coordsB="data",
                                               axesA=p0[2],axesB=p1[2],color=COLORS[family],alpha=.19,
                                               linewidth=.8,zorder=1,clip_on=False))
    deltaax.set_title("B   Held-out effect of adding 20 coordinates",loc="left",pad=14)
    deltaax.axhline(0,color="#68717c",linewidth=.9,zorder=0)
    deltaax.grid(axis="y",alpha=.65)
    deltaax.set_ylabel("Paired Δ log likelihood (nats / observation)\nm = 20 minus m = 0")
    deltaax.set_xticks(range(3),["Gaussian q","Direct Student q","Structured q"])
    deltaax.set_xlim(-.5,2.5);deltaax.set_ylim(-.031,.004)
    plot_points=[]
    for j,family in enumerate(FAMILIES):
        row=next(r for r in summary["nuisance_pairs"] if r["module"]=="S2_primary" and r["truth"]=="t5" and
                 r["noise_sd"]==.5 and r["family"]==family)
        values=[]
        for point in row["pairs"]:
            seed=point["seed"];i=seeds.index(seed);value=point["heldout_ll_delta"];values.append(value)
            deltaax.scatter(j+(i-2)*.07,value,color=COLORS[family],s=46,marker=marker[seed],
                            edgecolors="white",linewidth=.55,zorder=4)
            plot_points.append(dict(family=family,seed=seed,heldout_ll_delta=value))
        mean=float(np.mean(values));deltaax.plot([j-.22,j+.22],[mean,mean],color=COLORS[family],linewidth=2.5,zorder=3)
        offset=.001 if j==0 else (-.0024 if mean<0 else .0012)
        deltaax.text(j,mean+offset,f"mean {mean:+.4f}",ha="center",fontsize=8.5,color=COLORS[family])
    family_handles=[Line2D([0],[0],color=COLORS[f],marker="o",linestyle="-",label=NAMES[f],markersize=5) for f in FAMILIES]
    fig.legend(handles=family_handles,loc="lower left",bbox_to_anchor=(.087,.155),frameon=False,ncol=3,columnspacing=1.6)
    seed_handles=[Line2D([0],[0],color="#6e7782",marker=marker[s],linestyle="",label=str(s),markersize=5) for s in seeds]
    fig.legend(handles=seed_handles,title="Seed",loc="lower right",bbox_to_anchor=(.973,.145),frameon=False,
               ncol=5,columnspacing=.85,handletextpad=.35,title_fontsize=8.5,fontsize=8.5)
    fig.text(.09,.095,"N = 512 fit observations; independent held-out N = 4,096. Identical observations are paired across m and families.",fontsize=8.5)
    fig.text(.09,.035,footer(summary,version,status_note),fontsize=8,color="#56606d",linespacing=1.4)
    save(fig,output/f"scientific_simulations_s2_{version}")
    return plot_points


def plot_s3(summary,output,version,status_note):
    rows=[r for r in summary["s3_groups"] if r["truth"]=="t3_t30"]
    lookup={(r["noise_sd"],r["nuisance_dimension"],r["family"]):r for r in rows}
    families=["gaussian","direct_student"]
    seeds=sorted({s["seed"] for r in rows for s in r["assignment_direction"]})
    # Missing seeds retain their fixed marker mapping rather than changing by panel.
    all_seeds=list(range(min(seeds),max(seeds)+1));marker=dict(zip(all_seeds,MARKERS))
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.8),sharey=True)
    fig.subplots_adjust(left=.09,right=.97,bottom=.34,top=.78,wspace=.22)
    fig.suptitle("Allocation bias favors the lighter-tailed component",x=.09,y=.96,ha="left",fontweight="bold")
    fig.text(.09,.905,"S3: known equal-weight mixture, ν = 3 versus 30; component means -1 and +1; unit component variances",fontsize=10.2)
    plotted=[]
    for panel,(ax,sd) in enumerate(zip(axes,(.5,2.))):
        ax.set_title(f"{'AB'[panel]}   Observation noise SD {sd:g}",loc="left",pad=14)
        ax.axhline(0,color="#68717c",linewidth=.9,zorder=0);ax.grid(axis="y",alpha=.65)
        ax.set_xlim(-.46,1.46);ax.set_ylim(-2.5,40)
        ax.set_xticks([0,1],["0","100"]);ax.set_xlabel("Number of ignored latent coordinates, m")
        for xpos,m in enumerate((0,100)):
            available=[lookup.get((sd,m,f)) for f in families]
            if all(r is None for r in available):
                ax.axvspan(xpos-.37,xpos+.37,color="#f2f3f5",zorder=0)
                ax.text(xpos,21,"MISSING\n0 / 5 accepted seeds\nfor both families\n\nIntegration accuracy\ngate failed",ha="center",va="center",
                        fontsize=9.5,color="#727b86",linespacing=1.5)
                continue
            for fj,family in enumerate(families):
                row=lookup.get((sd,m,family))
                if row is None:continue
                values=[]
                for p in row["assignment_direction"]:
                    seed=p["seed"];i=all_seeds.index(seed);value=100*p["bayes0_to_variational1"];values.append(value)
                    ax.scatter(xpos+(fj-.5)*.27+(i-2)*.03,value,s=44,color=COLORS[family],marker=marker[seed],
                               edgecolor="white",linewidth=.55,zorder=4)
                    plotted.append(dict(noise_sd=sd,m=m,family=family,seed=seed,heavy_to_light_percent=value,
                                        light_to_heavy_percent=100*p["bayes1_to_variational0"]))
                mean=float(np.mean(values));center=xpos+(fj-.5)*.27
                ax.plot([center-.11,center+.11],[mean,mean],color=COLORS[family],linewidth=2.5,zorder=3)
                # Small values use offset labels because a common linear axis preserves magnitude.
                offset=(2. if mean>25 else 3.7) if fj==0 else 1.5
                ax.annotate(f"{mean:.2f}%",(center,mean),xytext=(center,mean+offset),textcoords="data",
                            ha="center",fontsize=9,color=COLORS[family],
                            arrowprops=dict(arrowstyle="-",color=COLORS[family],lw=.6))
            ns=sorted({r["n"] for r in available if r is not None})
            ax.text(xpos,39,f"{','.join(map(str,ns))} / 5 seeds",ha="center",va="top",fontsize=9,color="#56606d")
    axes[0].set_ylabel("Observations reassigned from t(3) to t(30)\nrelative to exact Bayes assignment (%)")
    handles=[Line2D([0],[0],color=COLORS[f],marker="o",linestyle="-",label=NAMES[f],markersize=5) for f in families]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.087,.175),frameon=False,ncol=2,columnspacing=1.8)
    seed_handles=[Line2D([0],[0],color="#6e7782",marker=marker[s],linestyle="",label=str(s),markersize=5) for s in all_seeds]
    fig.legend(handles=seed_handles,title="Seed",loc="lower right",bbox_to_anchor=(.973,.16),frameon=False,
               ncol=5,columnspacing=.85,handletextpad=.35,title_fontsize=8.5,fontsize=8.5)
    reverse=sum(p["light_to_heavy_percent"] for p in plotted)
    direction_note="No accepted observation changes in the reverse direction." if reverse==0 else "Reverse-direction changes are recorded in the plot metadata."
    present={p["seed"] for r in rows if r["noise_sd"]==.5 for p in r["assignment_direction"]}
    missing=sorted(set(all_seeds)-present)
    missing_note=("Noise SD 0.5 excludes seed(s) "+", ".join(map(str,missing))+".") if missing else "Noise SD 0.5 includes all five seeds."
    fig.text(.09,.125,f"N = 2,000 per seed. {direction_note} {missing_note}",fontsize=8.5)
    complete=complete_s3_pairs(summary)
    pair_note=("5/5 paired seeds at both noise levels and both nuisance dimensions, for both inference families."
               if complete else "Paired coverage is incomplete; use the summary JSON for accepted counts by noise, dimension and family.")
    fig.text(.09,.092,pair_note,fontsize=8.5)
    fig.text(.09,.032,footer(summary,version,status_note),fontsize=8,color="#56606d",linespacing=1.4)
    save(fig,output/f"scientific_simulations_s3_{version}")
    return plotted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--version",default="interim_v1")
    parser.add_argument("--status-note",default=None)
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text());args.output.mkdir(parents=True,exist_ok=True);style()
    expected=[args.output/f"scientific_simulations_{panel}_{args.version}.{extension}"
              for panel in ("s2","s3") for extension in ("png","pdf")]
    expected.append(args.output/f"scientific_simulations_plot_{args.version}.json")
    if any(path.exists() for path in expected):raise FileExistsError("Use a new version/output; existing figures are immutable.")
    if args.status_note is None:
        args.status_note=("Complete repaired numerical panel." if summary["counts"]["failed"]==0
                          and summary["counts"]["accepted"]==summary["counts"]["total"]
                          else "Numerical repair rerun pending.")
    points=dict(s2=plot_s2(summary,args.output,args.version,args.status_note),s3=plot_s3(summary,args.output,args.version,args.status_note))
    metadata=dict(summary=str(args.summary.resolve()),summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                  version=args.version,status_note=args.status_note,counts=summary["counts"],plotted_points=points,
                  complete_s3_paired_coverage=complete_s3_pairs(summary),
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=args.output/f"scientific_simulations_plot_{args.version}.json"
    path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),version=args.version,pdf_count=2,png_count=2)))


if __name__=="__main__":main()
