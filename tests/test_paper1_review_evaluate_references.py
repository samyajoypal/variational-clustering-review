"""Synthetic endpoint artifacts only; no real benchmark labels are read."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paper1_review import evaluate_references as ev
from experiments.paper1_review.data import _ids_hash
from experiments.paper1_review.provenance import sha256, write_json


def dataset(root):
    root.mkdir()
    manifest={"schema_version":"paper1-review-dataset-v1","status":"completed","dataset":"mnist",
              "n_clusters":10,"n_features":784,"splits":{},"artifact_sha256":{}}
    for split,count in ev.COUNTS.items():
        ids=np.array([f"{split}:{i}" for i in range(count)])
        manifest["splits"][split]={"inputs":f"UNOPENABLE_{split}_inputs.npz","labels":f"{split}_labels.npz",
            "n_samples":count,"ids_sha256":_ids_hash(ids),"input_shape":[count,1,28,28]}
        if split!="train":
            np.savez_compressed(root/f"{split}_labels.npz",ids=ids,labels=np.arange(count)%10)
            manifest["artifact_sha256"][f"{split}_labels.npz"]=sha256(root/f"{split}_labels.npz")
    write_json(root/"manifest.json",manifest)
    return root/"manifest.json"


def save_record(directory,method,config,extra):
    directory.mkdir(parents=True,exist_ok=True)
    source_name=f"experiments/paper1_review/reference_{'dec_idec' if method in {'dec','idec'} else method}.py"
    source=directory/"source"/source_name;source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text("# SYNTHETIC TEST SOURCE: never executed, not benchmark evidence\n")
    sources={source_name:sha256(source)}
    write_json(directory/"config.json",config)
    record={"schema_version":"paper1-review-run-v2","status":"completed","stage":"final_benchmark",
            "fresh_run":True,"legacy_outputs_used":False,"labels_loaded":False,
            "protocol_version":ev.VERSIONS[method],"config_sha256":sha256(directory/"config.json"),
            "source_file_sha256":sources,
            "source_sha256":hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
            "artifact_sha256":{str(p.relative_to(directory)):sha256(p) for p in directory.rglob("*") if p.is_file() and p.name!="run.json"},
            **extra}
    record["artifact_sha256"]["omitted_rolling_checkpoint.pt"]="0"*64
    write_json(directory/"run.json",record)


def refresh_hash(directory,name):
    record=json.loads((directory/"run.json").read_text())
    record["artifact_sha256"][name]=sha256(directory/name)
    if name=="config.json":record["config_sha256"]=sha256(directory/name)
    write_json(directory/"run.json",record)


def create_fit(root,manifest,method,seed,*,reorder=False,collapse_native=False):
    family="dec_idec" if method in {"dec","idec"} else method
    parent=root/f"reference_{family}_v1"/f"seed{seed}"
    directory=parent/(method if method in {"dec","idec"} else "train")
    directory.mkdir(parents=True)
    initial=parent/"initialize"
    common={"protocol_version":ev.VERSIONS[method],"seed":seed,"validation_only":False,"manifest_sha256":sha256(manifest)}
    p={"architecture":[784,500,500,2000,10],"layer_updates":50000,"joint_updates":100000,
       "cluster_max_updates":20000,"batch_size":256,"dec_refresh":160,"idec_refresh":140}
    if method=="vade":
        common={"protocol_version":ev.VERSIONS[method],"seed":seed,"validation":False,"data_manifest_sha256":sha256(manifest),"labels_loaded":False}
        initial.mkdir()
        (initial/"initial.pt").write_bytes(b"SYNTHETIC TEST INITIALIZER")
        write_json(initial/"training_curves.json",[{"epoch":100,"loss":.2}])
        save_record(initial,method,dict(common,phase="fresh_initialization",pretrain_epochs=100),
                    {"epochs_completed":100,"optimizer_updates":54000,"train_count":54000,"gmm_converged":True})
        # Match the actual VaDE initialization schema: labels flag is in config.
        record=json.loads((initial/"run.json").read_text());record.pop("labels_loaded");write_json(initial/"run.json",record)
        write_json(directory/"initial_dependency.json",{"run_sha256":sha256(initial/"run.json"),"checkpoint_sha256":sha256(initial/"initial.pt")})
        config=dict(common,phase="reference_training",training_epochs=3000,batch_size=100)
        extra={"epochs_completed":3000,"optimizer_updates":1620000,"train_count":54000}
        checkpoint,history="final.pt","training_curves.json"
    elif method in {"dec","idec"}:
        if not initial.exists():
            initial.mkdir();(initial/"initialization.pt").write_bytes(b"SYNTHETIC SHARED AE")
            write_json(initial/"training_diagnostics.json",[{"phase_index":4,"updates":100000,"loss":.1}])
            save_record(initial,method,dict(common,phase="initialize",protocol=p),{"pretrain_completed":True,"optimizer_updates":300000})
        config=dict(common,phase=method,protocol=p,initialization_sha256=sha256(initial/"initialization.pt"))
        extra={"optimizer_updates":20000,"stop_reason":"update_cap"}
        checkpoint,history="model.pt","training_diagnostics.json"
    else:
        config=dict(common,effective_seed=seed,settings={"epochs":100,"batch_size":32,"overwrite_epoch":20})
        extra={"completed_epochs":100,"optimizer_updates":168800,"overwrite_count":1,"fixed_endpoint":True,"endpoint_gmm":{"converged":True}}
        checkpoint,history="endpoint_checkpoint.pt","training_history.json"
        np.savez(directory/"endpoint_train_gmm.npz",test_fixture_only=np.array([1]))
        np.savez(directory/"training_ids.npz",sample_ids=np.array(["fixture_only"]))
    (directory/checkpoint).write_bytes(b"SYNTHETIC TEST ENDPOINT: NOT A TRAINED MODEL")
    write_json(directory/history,[{"epoch":100,"loss":.25}])
    for split in ["test","validation"]:
        count=ev.COUNTS[split];ids=np.array([f"{split}:{i}" for i in range(count)])
        for variant in (["native","gmm_refit"] if method=="pro_t" else ["native"]):
            prediction=(np.arange(count)+3)%10
            if collapse_native and variant=="native":prediction[:]=0
            q=np.eye(10,dtype=np.float32)[prediction]
            if reorder:ids_out,prediction,q=ids[::-1],prediction[::-1],q[::-1]
            else:ids_out=ids
            filename=f"{split}_{variant}_predictions.npz" if method=="pro_t" else f"{split}_predictions.npz"
            value={"ids" if method=="vade" else "sample_ids":ids_out,"predictions":prediction,"responsibilities":q}
            if method!="vade":value["cluster_labels"]=np.arange(10)
            np.savez_compressed(directory/filename,**value)
    save_record(directory,method,config,extra)
    return directory


def forbid_labels(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("Evaluation labels opened before gate passed")
    monkeypatch.setattr(ev,"load_evaluation_labels",forbidden)


def test_partial_distinct_pro_t_variants_and_full_metric_panel(tmp_path):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    create_fit(root,manifest,"pro_t",901,reorder=True,collapse_native=True)
    result=ev.evaluate(root,manifest,tmp_path/"evaluation",allow_incomplete=True,
                       missing_ledger={"vade:902":"preempted; fixed-protocol resume running"})
    assert result["completed_fit_count"]==1 and len(result["missing_runs"])==11
    native,refit=result["rows"]
    assert native["variant"]=="native" and refit["variant"]=="gmm_refit"
    assert native["metrics"]["scores"]["ari"]==0 and refit["metrics"]["scores"]["ari"]==1
    assert native["metrics"]["occupancy"]["unoccupied_cluster_count"]==9
    assert native["alignment"]["reordered"] and refit["alignment"]["reordered"]
    assert len(refit["metrics"]["scores"])==9 and len(refit["metrics"]["per_class"])==10
    assert all("scores" not in summary for summary in result["summaries"])
    assert "omitted_rolling_checkpoint.pt" in native["evidence"]["omitted_nonrequired_artifacts"]
    assert (tmp_path/"evaluation/seed_scores.csv").is_file()
    with pytest.raises(FileExistsError):ev.evaluate(root,manifest,tmp_path/"evaluation",allow_incomplete=True)


def test_eleven_completed_fits_keep_missing_seed_and_withhold_only_vade_summary(tmp_path):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    for method in ev.METHODS:
        for seed in ev.SEEDS:
            if (method,seed)!=("vade",902):create_fit(root,manifest,method,seed)
    result=ev.evaluate(root,manifest,tmp_path/"evaluation",allow_incomplete=True)
    assert result["completed_fit_count"]==11 and len(result["rows"])==14
    assert [(r["method"],r["seed"]) for r in result["missing_runs"]]==[("vade",902)]
    for group in result["summaries"]:
        if group["method"]=="vade":assert "scores" not in group and group["seeds"]==[901,903]
        else:
            assert group["n"]==3
            assert group["scores"]["ari"]["sample_sd"]==0


def test_incomplete_default_never_opens_labels(tmp_path,monkeypatch):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    create_fit(root,manifest,"dec",901)
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError,match="incomplete; labels remain unopened"):
        ev.evaluate(root,manifest,tmp_path/"out")


@pytest.mark.parametrize("failure",["validation","source","checkpoint","budget","initializer","probabilities","argmax","duplicate_ids"])
def test_invalid_completed_fit_never_opens_labels(tmp_path,monkeypatch,failure):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    directory=create_fit(root,manifest,"dec",901)
    if failure=="validation":
        record=json.loads((directory/"run.json").read_text());record["stage"]="validation";write_json(directory/"run.json",record)
    elif failure=="source":
        next((directory/"source").rglob("*.py")).write_text("CHANGED")
    elif failure=="checkpoint":(directory/"model.pt").write_bytes(b"CHANGED")
    elif failure=="budget":
        record=json.loads((directory/"run.json").read_text());record["optimizer_updates"]=100;write_json(directory/"run.json",record)
    elif failure=="initializer":(directory.parent/"initialize/initialization.pt").write_bytes(b"CHANGED")
    else:
        name="test_predictions.npz"
        with np.load(directory/name) as archive:values={k:archive[k] for k in archive.files}
        if failure=="probabilities":values["responsibilities"][0,0]=np.nan
        elif failure=="argmax":values["predictions"][0]=(values["predictions"][0]+1)%10
        else:values["sample_ids"][1]=values["sample_ids"][0]
        np.savez_compressed(directory/name,**values);refresh_hash(directory,name)
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError):ev.evaluate(root,manifest,tmp_path/"out",allow_incomplete=True)


def test_id_mismatch_rejected_and_collapsed_predictions_not_filtered(tmp_path):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    directory=create_fit(root,manifest,"pro_t",901,collapse_native=True)
    name="test_native_predictions.npz"
    with np.load(directory/name) as archive:values={k:archive[k] for k in archive.files}
    values["sample_ids"]=values["sample_ids"].astype("U50");values["sample_ids"][0]="wrong:ID"
    np.savez_compressed(directory/name,**values);refresh_hash(directory,name)
    with pytest.raises(ValueError,match="IDs mismatch"):
        ev.evaluate(root,manifest,tmp_path/"out",allow_incomplete=True)
    assert not (tmp_path/"out").exists()


def test_completed_endpoint_conflict_and_source_style_stopping(tmp_path,monkeypatch):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    directory=create_fit(root,manifest,"idec",901)
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError,match="conflicts with missing-run ledger"):
        ev.evaluate(root,manifest,tmp_path/"out",allow_incomplete=True,missing_ledger={"idec:901":"failed"})
    record=json.loads((directory/"run.json").read_text())
    record.update(stop_reason="assignment_stability",optimizer_updates=141,last_assignment_change=0.,last_target_refresh=141)
    write_json(directory/"run.json",record)
    with pytest.raises(ValueError,match="stopping evidence"):
        ev.evaluate(root,manifest,tmp_path/"out",allow_incomplete=True)


def test_remote_completion_receipt_mismatch_blocks_label_access(tmp_path,monkeypatch):
    manifest=dataset(tmp_path/"data");root=tmp_path/"runs"
    create_fit(root,manifest,"pro_t",901)
    audit=tmp_path/"audit.json"
    write_json(audit,{"all_recorded_completed_artifacts_verified":True,
        "runs":[{"family":"pro_t","seed":901,"phase":"train","status":"completed","hash_errors":[],"run_sha256":"wrong"}]})
    forbid_labels(monkeypatch)
    with pytest.raises(ValueError,match="verified remote completion receipt"):
        ev.evaluate(root,manifest,tmp_path/"out",allow_incomplete=True,completion_audit=audit)
