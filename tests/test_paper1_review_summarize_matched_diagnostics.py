import copy
import json
from pathlib import Path

import pytest

from experiments.paper1_review import summarize_matched_diagnostics as sd
from experiments.paper1_review.final_protocol import PROTOCOL, protocol_digest
from experiments.paper1_review.provenance import sha256, write_json
from campaign_fixture import make_manifest, BASE


def ingredients(method="gaussian_q_student_prior"):
    row = dict(run_id="synthetic", dataset="mnist", method=method, condition="clean", seed=101)
    run = dict(optimizer_updates=200, clipped_updates=100, training_seconds=20., artifact_sha256={})
    config = dict(optimizer_updates=200, training_count=6, validation_count=4)
    curves = [dict(clean_equivalent_epoch=e, updates=e, clipped_updates=int(e <= 100), clip_fraction=float(e <= 100),
                   preclip_norm_mean=3., preclip_norm_max=4., examples_seen_this_window=6) for e in range(1,201)]
    params = dict(prior_df=[3.,30.], component_weights=[.25,.75], df_lower_hits=1, df_upper_hits=1,
                  scale_diagonal=[[1.,2.],[2.,4.]], scale_condition_ratio=[2.,2.])
    names = {"target_kl"} if method == "dec_matched" else {"target_kl","reconstruction"} if method == "idec_matched" else {"reconstruction","prior_cross_entropy","negative_entropy"}
    gradients = [dict(clean_equivalent_epoch=e, completed_updates=e, optimizer_update=False, parameters=copy.deepcopy(params),
                      scope="synthetic diagnostic only", components={n:dict(value=-1.,preclip_norm=5.,group_norms={"encoder":3.,"decoder":4.}) for n in names}) for e in PROTOCOL["diagnostic_epochs"]]
    q = None
    if method == "structured_q_student_prior":
        p=PROTOCOL["structured"]
        q=dict(passed=True,training_nodes_pass=True,prediction_nodes_pass=True,reference_refinement_pass=True,selected_positions=list(range(4)),
               rows=[dict(nodes=n,bounds=b,errors=dict(log_normalizer=0.,responsibilities=0.,kl_to_prior=0.)) for n,b in
                     [(128,p["bounds"]),(256,p["bounds"]),(512,p["reference_bounds"]),(1024,p["refined_reference_bounds"])]])
    checks=[dict(clean_equivalent_epoch=e,quadrature=copy.deepcopy(q)) for e in PROTOCOL["diagnostic_epochs"]]
    occupancy=[dict(clean_equivalent_epoch=200,split=s,endpoint=True,count=n,cluster_counts=[n//2,n//2],occupied_clusters=2,
                    max_cluster_share=.5,mean_responsibility_entropy=.2) for s,n in [("train",6),("validation",4)]]
    return row,run,config,curves,gradients,checks,occupancy


def test_measured_df_clipping_gradient_and_occupancy_scope():
    result=sd.fit_summary(*ingredients())
    s=result["endpoint_scalars"]
    assert s["clip_fraction"] == .5
    assert s["examples_seen"] == 1200
    assert s["prior_df_mean"] == 16.5
    assert s["prior_df_weighted_mean"] == 23.25
    assert s["gradient.reconstruction.preclip_norm"] == 5
    assert s["occupancy.train.effective_clusters_from_hard_counts"] == pytest.approx(2)
    assert len(result["diagnostic_trajectory"]) == 8
    assert all(x["quadrature"] is None for x in result["diagnostic_trajectory"])
    assert "recognition_df_upper_hits" not in s


@pytest.mark.parametrize("damage",["boundary","clipping","gradient","occupancy","schedule","quadrature"])
def test_contradictory_recorded_diagnostics_fail(damage):
    a=list(ingredients("structured_q_student_prior"))
    if damage=="boundary": a[4][-1]["parameters"]["df_upper_hits"]=0
    elif damage=="clipping": a[3][0]["clip_fraction"]=0.
    elif damage=="gradient": a[4][-1]["components"]["reconstruction"]["preclip_norm"]=7.
    elif damage=="occupancy": a[6][0]["occupied_clusters"]=1
    elif damage=="schedule": a[4].pop()
    else: a[5][-1]["quadrature"]["rows"][0]["errors"]["log_normalizer"]=1.
    with pytest.raises(ValueError): sd.fit_summary(*a)


def test_groups_require_five_unique_seeds_and_identical_scalar_sets():
    rows=[dict(dataset="mnist",seed=s,values={"x":float(s)}) for s in sd.SEEDS]
    assert sd.group_summaries(rows,("dataset",),"values")[0]["scalars"]["x"]["mean"] == 103.
    with pytest.raises(ValueError): sd.group_summaries(rows[:-1],("dataset",),"values")
    rows[-1]["values"]["extra"]=0.
    with pytest.raises(ValueError): sd.group_summaries(rows,("dataset",),"values")


@pytest.fixture
def full_gate(tmp_path):
    campaign=make_manifest(); records=[];fits=[]
    for row in campaign["trains"]:
        directory=tmp_path/"fits"/str(Path(row["output"]).relative_to(BASE));directory.mkdir(parents=True)
        _,run,config,curves,gradients,checks,occ=ingredients(row["method"])
        config.update(protocol=PROTOCOL,labels_loaded=False,**{k:row[k] for k in ("dataset","method","condition","seed")})
        for name,value in [("config.json",config),("training_curves.json",curves),("gradient_diagnostics.json",gradients),("numerical_checks.json",checks),("occupancy_diagnostics.json",occ)]:write_json(directory/name,value)
        run.update(status="completed",labels_loaded=False,fixed_endpoint=True,numerical_gate_passed=True,config_sha256=sha256(directory/"config.json"),
                   artifact_sha256={p.name:sha256(p) for p in directory.iterdir()})
        # A checkpoint is deliberately absent: the helper must not open it.
        run["artifact_sha256"]["model.pt"]="0"*64
        write_json(directory/"run.json",run)
        records.append(dict(directory=row["output"],run_sha256=sha256(directory/"run.json"),artifact_sha256=run["artifact_sha256"],all_artifacts_verified=True))
        fits.append(dict(recorded_remote_directory=row["output"],run_sha256=sha256(directory/"run.json")))
    write_json(tmp_path/"campaign.json",campaign);write_json(tmp_path/"audit.json",dict(runs=records))
    prov=dict(campaign_sha256=sha256(tmp_path/"campaign.json"),completion_audit_sha256=sha256(tmp_path/"audit.json"),overrides_sha256=None,protocol_sha256=protocol_digest())
    write_json(tmp_path/"inventory.json",dict(provenance=prov,verified=dict(initializations=40,fits=300,predictions=1200),missing_or_failed=[],fits=fits))
    write_json(tmp_path/"analysis_manifest.json",dict(status="completed",inventory_only=False,provenance=prov,artifact_sha256={"inventory.json":sha256(tmp_path/"inventory.json")}))
    args=dict(campaign=tmp_path/"campaign.json",campaign_sha256=prov["campaign_sha256"],completion_audit=tmp_path/"audit.json",completion_audit_sha256=prov["completion_audit_sha256"],
              analysis_manifest=tmp_path/"analysis_manifest.json",analysis_manifest_sha256=sha256(tmp_path/"analysis_manifest.json"),path_map=[f"{BASE}={tmp_path/'fits'}"])
    return args


def test_full_label_free_gate_no_checkpoints_needed_and_write_once(full_gate,tmp_path):
    value=sd.summarize(output=tmp_path/"out",**full_gate)
    assert value["counts"] == dict(fits=300,endpoint_groups=60,trajectory_groups=480)
    assert value["labels_loaded"] is False and value["checkpoints_opened"] is False
    assert len(value["evidence"])==300
    with pytest.raises(FileExistsError): sd.summarize(output=tmp_path/"out",**full_gate)
    file=next((tmp_path/"fits").rglob("gradient_diagnostics.json"));file.write_text("[]")
    with pytest.raises(ValueError,match="hash mismatch"):sd.summarize(output=tmp_path/"changed",**full_gate)
    assert not (tmp_path/"changed").exists()


def test_completed_manifest_anchor_required(full_gate,tmp_path):
    args=dict(full_gate,analysis_manifest_sha256="0"*64)
    with pytest.raises(ValueError,match="anchor hash mismatch"):sd.summarize(output=tmp_path/"bad",**args)
    manifest=full_gate["analysis_manifest"];value=json.loads(manifest.read_text());value["inventory_only"]=True;write_json(manifest,value)
    args=dict(full_gate,analysis_manifest_sha256=sha256(manifest))
    with pytest.raises(ValueError,match="scored analysis gate"):sd.summarize(output=tmp_path/"inventory",**args)
