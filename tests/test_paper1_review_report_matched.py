"""Publication tests use invented unit-test summaries, never real observations."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from experiments.paper1_review import report_matched as report


def description(base, paired=False, slope=1):
    values=base+slope*np.array([-.02,-.01,0,.01,.02])
    sd=float(values.std(ddof=1))
    result=dict(values_by_seed=values.tolist(),mean=float(values.mean()),median=float(np.median(values)),
                sample_sd=sd,minimum=float(values.min()),maximum=float(values.max()))
    if paired:
        radius=2.7764451051977987*sd/math.sqrt(5)
        result["descriptive_t_interval_95"]=[result["mean"]-radius,result["mean"]+radius]
    return result


def evaluation(*, full=False, empty=False):
    registry=[dict(name=n,method_a=a,method_b=b) for n,a,b in report.CORE_CONTRASTS]
    value=dict(schema_version=report.SCHEMA,stage="frozen_endpoint_analysis",primary_metric="ari",
               all_pairwise=False,contrast_registry=registry,provenance={"unit_test":"synthetic; not empirical evidence"},
               counts=dict(report.EXPECTED_COUNTS) if full else dict(initializations=1,fits=1,predictions=1),
               missing_or_failed=[] if full else [{"reason":"synthetic incomplete fixture"}],caveats=[])
    rosters=report.expected_rosters(registry)
    for name,roster in rosters.items():
        entries=[]
        for key in sorted(roster):
            row=dict(zip(report.FIELDS[name],key))
            selected=(key[0]=="mnist" and (
                name=="summaries" and key[1:]==("vae_gmm","clean","clean") or
                name=="paired_condition_summaries" and key[1:]==("vae_gmm","inference_corruption","gaussian_sd0.1") or
                name=="clean_contrast_summaries" and key[1]==report.CORE_CONTRASTS[0][0] or
                name in ("condition_contrast_summaries","robustness_contrast_summaries") and
                key[1]==report.CORE_CONTRASTS[0][0] and key[-1]=="gaussian_sd0.1"))
            complete=(full or selected) and not empty
            row.update(status=report.COMPLETE if complete else report.PENDING,n=5 if complete else 0,
                       seeds=list(report.SEEDS) if complete else [],missing_seeds=[] if complete else list(report.SEEDS))
            if complete:
                metrics=report.ALL_METRICS if name=="summaries" else report.ROBUST
                if name in ("paired_condition_summaries","robustness_contrast_summaries") and row["mode"]=="inference_corruption":
                    metrics+=report.FROZEN
                base={"summaries":.5,"paired_condition_summaries":-.1,"clean_contrast_summaries":.03,
                      "condition_contrast_summaries":.01,"robustness_contrast_summaries":-.02}[name]
                row["scores"]={k:description(base,name!="summaries",slope=2 if name=="condition_contrast_summaries" else 1) for k in metrics}
            entries.append(row)
        value[name]=entries
    return value


def anchored(tmp_path,value,*,inventory=False):
    root=tmp_path/"analysis"; root.mkdir()
    name="inventory.json" if inventory else "evaluation.json"
    artifact=root/name; artifact.write_text(json.dumps(value,allow_nan=False))
    manifest=dict(schema_version=report.SCHEMA,status="completed",inventory_only=inventory,
                  artifact_sha256={name:hashlib.sha256(artifact.read_bytes()).hexdigest()},provenance=value["provenance"])
    path=root/"analysis_manifest.json"; path.write_text(json.dumps(manifest))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_registered_rosters_and_nine_metric_panel():
    lookup,readiness=report.validate_evaluation(evaluation(full=True))
    assert not readiness["partial"]
    assert {k:len(v) for k,v in lookup.items()}==dict(summaries=240,paired_condition_summaries=200,
        clean_contrast_summaries=32,condition_contrast_summaries=156,robustness_contrast_summaries=156)
    assert len(next(iter(lookup["summaries"].values()))["scores"])==9


def test_strict_partial_rejected_before_output_created(tmp_path):
    path,digest=anchored(tmp_path,evaluation())
    with pytest.raises(ValueError,match="allow-partial"):
        report.generate_report(path,tmp_path/"report",analysis_manifest_sha256=digest)
    assert not (tmp_path/"report").exists()


@pytest.mark.parametrize("change",["manifest","evaluation"])
def test_explicit_anchor_chain_rejects_tampering(tmp_path,change):
    path,digest=anchored(tmp_path,evaluation())
    modified=path if change=="manifest" else path.parent/"evaluation.json"
    modified.write_text(modified.read_text()+" ")
    with pytest.raises(ValueError,match="hash mismatch"):
        report.generate_report(path,tmp_path/"report",analysis_manifest_sha256=digest,allow_partial=True)


def test_incomplete_group_estimate_is_rejected():
    value=evaluation(); row=next(r for r in value["summaries"] if r["status"]==report.COMPLETE)
    row.update(n=4,seeds=[101,102,103,104],missing_seeds=[105],status=report.PENDING)
    with pytest.raises(ValueError,match="must not expose"):
        report.validate_evaluation(value)


@pytest.mark.parametrize("change",["mean","interval","gamma","duplicate","metric"])
def test_invalid_scientific_summaries_rejected(change):
    value=evaluation()
    name="robustness_contrast_summaries" if change=="gamma" else "paired_condition_summaries"
    row=next(r for r in value[name] if r["status"]==report.COMPLETE)
    if change=="mean": row["scores"]["ari"]["mean"]+=.1
    elif change=="interval": row["scores"]["ari"]["descriptive_t_interval_95"][0]+=.1
    elif change=="gamma": row["scores"]["ari"]=description(.4,paired=True)
    elif change=="duplicate": value[name].append(copy.deepcopy(row))
    else: del row["scores"]["frozen_acc"]
    with pytest.raises(ValueError): report.validate_evaluation(value)


def test_missing_group_not_hidden_by_complete_counts():
    value=evaluation(full=True); value["summaries"].pop()
    _,readiness=report.validate_evaluation(value)
    assert readiness["partial"]
    assert len(readiness["pending_groups"])==1
    assert readiness["pending_groups"][0]["missing_seeds"]==report.SEEDS


def test_degradation_sign_interval_order_and_missing_gap():
    row=dict(status=report.COMPLETE,scores={"ari":description(-.1,True)})
    y,lo,hi=report.degradation_series([row,None,row])
    assert np.isnan(y[1]) and np.isnan(lo[1]) and np.isnan(hi[1])
    assert y[0]==pytest.approx(.1)
    source=row["scores"]["ari"]["descriptive_t_interval_95"]
    assert lo[0]==pytest.approx(-source[1]) and hi[0]==pytest.approx(-source[0])


def test_complete_print_panel_preserves_roster_seed_points_and_source_values(tmp_path,monkeypatch):
    """Display changes must retain all registered plots and every clean seed."""
    from matplotlib.figure import Figure
    lookup,_=report.validate_evaluation(evaluation(full=True))
    before=copy.deepcopy(lookup)
    seen={}
    def inspect(figure,path,**kwargs):
        if Path(path).suffix != '.pdf': return
        name=Path(path).stem;seen[name]=tuple(figure.get_size_inches())
        assert figure.get_size_inches()[0] <= 8
        if name=='clean_ari_seeds':
            assert len(figure.axes)==4
            for axis,dataset in zip(figure.axes,report.DATASETS):
                assert [t.get_text() for t in axis.get_yticklabels()]==[report.SHORT_METHODS[m] for m in report.METHODS]
                assert len(axis.collections)==2*len(report.METHODS)
                for j,method in enumerate(report.METHODS):
                    expected=lookup['summaries'][dataset,method,'clean','clean']['scores']['ari']
                    assert axis.collections[2*j].get_offsets()[:,0].tolist()==expected['values_by_seed']
                    assert axis.collections[2*j+1].get_offsets()[0,0]==expected['mean']
    monkeypatch.setattr(Figure,'savefig',inspect)
    produced=report._figures(tmp_path,lookup,False)
    expected=['clean_ari_seeds',*[f'degradation_{d}' for d in report.DATASETS[:3]],
              *[f'clean_contrasts_{d}' for d in report.DATASETS],
              *[f'training_changes_{d}' for d in report.DATASETS[:2]],
              *[f'condition_contrasts_{d}_{family}' for d in report.DATASETS[:3]
                for family in (['gaussian','dropout','training'] if d in report.DATASETS[:2] else ['gaussian','dropout'])]]
    assert produced==expected and list(seen)==expected
    assert lookup==before


def test_conditional_only_scores_are_rendered_without_fake_figures(tmp_path):
    value=evaluation(empty=True)
    row=next(r for r in value["summaries"] if (r["dataset"],r["method"],r["training_condition"],r["inference_condition"])==
             ("mnist","vae_gmm","clean","gaussian_sd0.1"))
    row.update(status=report.COMPLETE,n=5,seeds=list(report.SEEDS),missing_seeds=[],
               scores={m:description(.321) for m in report.ALL_METRICS})
    path,digest=anchored(tmp_path,value)
    result=report.generate_report(path,tmp_path/"report",analysis_manifest_sha256=digest,allow_partial=True)
    assert result["score_panels_emitted"] and result["figures"]==[]
    assert "0.321" in (tmp_path/"report/tables/conditional_nine_metrics.tex").read_text()


def test_empty_partial_has_readiness_but_no_score_panels(tmp_path):
    path,digest=anchored(tmp_path,evaluation(empty=True))
    output=tmp_path/"report"
    result=report.generate_report(path,output,analysis_manifest_sha256=digest,allow_partial=True)
    assert not result["score_panels_emitted"] and not result["figures"]
    assert not (output/"figures").exists() and not (output/"tables").exists()
    assert (output/"readiness.json").is_file()
    with pytest.raises(FileExistsError):
        report.generate_report(path,output,analysis_manifest_sha256=digest,allow_partial=True)


def test_inventory_only_never_reads_evaluation_or_renders(tmp_path,monkeypatch):
    value=dict(schema_version=report.SCHEMA,labels_loaded=False,expected=report.EXPECTED_COUNTS,
               verified=dict(initializations=1,fits=0,predictions=0),missing_or_failed=[],provenance={})
    path,digest=anchored(tmp_path,value,inventory=True)
    (path.parent/"evaluation.json").write_text("not JSON and must never be opened")
    def forbidden(*args,**kwargs): raise AssertionError("Score rendering forbidden for inventory")
    monkeypatch.setattr(report,"_figures",forbidden); monkeypatch.setattr(report,"_tables",forbidden)
    result=report.generate_report(path,tmp_path/"report",analysis_manifest_sha256=digest)
    assert result["inventory_only"] and result["labels_loaded"] is False
    ledger=json.loads((tmp_path/"report/readiness.json").read_text())
    assert "not live HPC status" in ledger["scope"]


def test_partial_tables_and_pdf_smoke(tmp_path):
    path,digest=anchored(tmp_path,evaluation())
    output=tmp_path/"report"
    result=report.generate_report(path,output,analysis_manifest_sha256=digest,allow_partial=True)
    assert result["partial"] and len(result["figures"])==4
    text=(output/"tables/clean_main.tex").read_text()
    assert "PARTIAL:" in text and "Pending" in text and "0.500" in text
    assert all(label in text for label in ["VaDE","DEC","IDEC","VAE+GMM","Pro-t","Site / t"])
    assert "V-measure" in (output/"tables/clean_nine_metrics.tex").read_text()
    assert "Frozen-map ACC" in (output/"tables/paired_all_metrics.tex").read_text()
    assert "D condition" in (output/"tables/paired_contrasts_ari.tex").read_text()
    for relative,digest in result["artifact_sha256"].items():
        assert hashlib.sha256((output/relative).read_bytes()).hexdigest()==digest
    for pdf in (output/"figures").glob("*.pdf"):
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert pdf.with_suffix(".png").is_file()
        if shutil.which("pdfinfo"):
            info=subprocess.run(["pdfinfo",str(pdf)],check=True,capture_output=True,text=True).stdout
            assert "Pages:           1" in info
    if shutil.which("pdftotext"):
        extracted=subprocess.run(["pdftotext",str(output/"figures/degradation_mnist.pdf"),"-"],check=True,capture_output=True,text=True).stdout
        assert "No complete five-seed groups" in extracted


def test_supplemental_all_pairwise_registry_is_explicit():
    registry=[dict(name=n,method_a=a,method_b=b) for n,a,b in report.CORE_CONTRASTS]
    with pytest.raises(ValueError,match="registry"):
        report.expected_rosters(registry,all_pairwise=True)


def test_provenance_mismatch_rejected(tmp_path):
    value=evaluation(); path,_=anchored(tmp_path,value)
    manifest=json.loads(path.read_text()); manifest["provenance"]={"different":True}
    path.write_text(json.dumps(manifest)); digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="provenance"):
        report.generate_report(path,tmp_path/"report",analysis_manifest_sha256=digest,allow_partial=True)


@pytest.mark.skipif(shutil.which("pdflatex") is None,reason="LaTeX compiler unavailable")
def test_every_table_format_compiles_within_a4_article_width(tmp_path):
    lookup,_=report.validate_evaluation(evaluation(full=True))
    report._tables(tmp_path,lookup,False)
    fragments=[]
    for path in sorted((tmp_path/"tables").glob("*.tex")):
        ending=r"\end{longtable}" if path.name=="paired_all_metrics.tex" else r"\end{table}"
        fragments.append(path.read_text().split(ending)[0]+ending)
    source=(r"\documentclass[11pt,a4paper]{article}"+"\n"+
        r"\usepackage[T1]{fontenc}\usepackage[margin=25mm]{geometry}\usepackage{lmodern}"+
        r"\usepackage{booktabs,tabularx,array,longtable}\setlength{\tabcolsep}{5pt}"+
        r"\renewcommand{\arraystretch}{1.12}\begin{document}"+"\n"+
        "\n\\clearpage\n".join(fragments)+"\n\\end{document}\n")
    path=tmp_path/"table_layout.tex"; path.write_text(source)
    for _ in range(2):
        subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error","-output-directory",str(tmp_path),str(path)],
                       capture_output=True,text=True,check=True)
    assert "Overfull" not in (tmp_path/"table_layout.log").read_text()
    assert (tmp_path/"table_layout.pdf").read_bytes().startswith(b"%PDF-")
