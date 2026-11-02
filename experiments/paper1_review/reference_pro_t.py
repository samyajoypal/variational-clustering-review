"""Pro-t recovered-recipe architecture, fixed-endpoint inductive adaptation.

The hash-pinned MIT model28 source is inspected before execution and is used
unchanged, including its epsilon conventions and detached sampled assignments.
Only fresh initialization or verified checkpoints from this exact campaign are
accepted. Labels are never opened. See protocol() for every declared adaptation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import sklearn
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture

from .data import load_inputs, _ids_hash
from .provenance import ROOT, sha256, start_run, finish_run, write_json

VERSION="paper1-reference-pro-t-20261028-v1"
VENDOR=Path(__file__).parent/"vendor_sources/pro_t_11966cd0"
MODEL_SOURCE=VENDOR/"src/tvae/reference/model28.py"
MODEL_SHA256="45cc4eae53ddb5e529c3d8b292ef81f5a56d766bb23aca5e310cd6c629a525b2"
COMMIT="11966cd0a512dfeaeae636aebf7df80832e985b4"
REFERENCE=VENDOR/"experiments/reference/mnist/v4.1_tVaDE_ModeB"
PROTOCOL={
    "protocol_version":VERSION,"label":"Pro-t recovered-recipe architecture, fixed-endpoint inductive adaptation",
    "dataset":"mnist","split_counts":{"train":54000,"validation":6000,"test":10000},
    "latent_dim":8,"n_clusters":10,"epochs":100,"batch_size":32,"drop_last":False,
    "learning_rate":.001,"weight_decay":.0001,"optimizer":"AdamW","adam_betas":[.9,.999],
    "adam_eps":1e-8,"nu_init":5.,"gradient_clip":5.,"epsilon":1e-6,
    "reconstruction":"pixel-and-batch-summed BCE of sigmoid output clamped to[1e-6,1-1e-6]",
    "prior":"5 times batch-mean author three-term Gaussian/Gamma/categorical regularizer",
    "prior_weight":5.,"pi_balance_weight":0.,"overwrite_epoch":20,
    "gmm":{"covariance_type":"full","n_init":10,"random_state":0,"tol":.001,
           "reg_covar":1e-6,"max_iter":100,"init_params":"kmeans"},
    "selection":"fixed epoch100; no label or heldout-loss selection",
    "native_prediction":"Student-mixture assignment at deterministic encoder mean",
    "refit_prediction":"separate full-GMM fitted on final training encoder means",
    "overwrite":"once after epoch20; copy means, diagonal full-GMM covariance as scale, and log weights; preserve df and optimizer moments",
    "source_commit":COMMIT,"source_model_sha256":MODEL_SHA256,
    "source_url":f"https://github.com/xuzhikangnba/torch-tvae/blob/{COMMIT}/src/tvae/reference/model28.py",
    "source_license":"MIT; Copyright(c)2026 torch-tvae contributors; full LICENSE included in every source snapshot",
    "adaptations":["54k/6k/10k inductive split replaces recovered48k/12k",
                   "fixed endpoint replaces current source heldout-reconstruction checkpoint selection and recovered summary best-refit-accuracy reporting",
                   "no heldout forward passes or label metrics during optimization",
                   "skip diagnostic-only GMM fits except epoch20 and endpoint",
                   "fresh seeds901-903; no downloaded weights, old features, or old fitted GMM",
                   "deterministic kernels and independent permutation generator; source objective and architecture unchanged",
                   "last16-example training batch retained under54k split, with source loss reductions unchanged"],
    "limitations":["source epsilon terms intentionally preserved, including unmodified logvar entropy",
                   "sampled responsibilities are detached exactly as in source; no claim of unbiased gradient of a fully differentiated collapsed ELBO",
                   "prior Student scale copied directly from GMM covariance, as in source; no covariance-matching correction",
                   "recipe adaptation, not reproduction of a published score"],
}


def protocol():
    return copy.deepcopy(PROTOCOL)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def source_audit():
    manifest=json.loads((VENDOR/"SOURCE_MANIFEST.json").read_text())
    if manifest["commit"]!=COMMIT or sha256(MODEL_SOURCE)!=MODEL_SHA256:
        raise ValueError("Pinned Pro-t model/source commit mismatch")
    selected=[MODEL_SOURCE,REFERENCE/"config.py",REFERENCE/"train.py",REFERENCE/"training/trainer.py",
              REFERENCE/"utils/clustering.py",VENDOR/"LICENSE"]
    pins={entry["path"]:entry["sha256"] for entry in manifest["files"]}
    for path in selected:
        if sha256(path)!=pins[str(path.relative_to(VENDOR))]:
            raise ValueError(f"Pinned source changed: {path}")
    return {"commit":COMMIT,"verified_files":{str(p.relative_to(ROOT)):sha256(p) for p in selected},
            "execution":"only model28.py is executed; inspected imports math/torch, class/function declarations, no file/network/subprocess/checkpoint loading",
            "vendor_historical_results_or_weights_read":False}


def sources():
    return [Path(__file__),Path(__file__).with_name("data.py"),VENDOR/"SOURCE_MANIFEST.json",VENDOR/"LICENSE",
            MODEL_SOURCE,REFERENCE/"config.py",REFERENCE/"train.py",REFERENCE/"training/trainer.py",
            REFERENCE/"utils/clustering.py",
            ROOT/"protocols/reference_recipes.md"]


def build_model(device="cpu",settings=None):
    p=PROTOCOL if settings is None else settings
    source_audit()
    spec=importlib.util.spec_from_file_location("_paper1_pinned_pro_t_model28",MODEL_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.StudentTVaDE(p["latent_dim"],p["n_clusters"],p["nu_init"]).to(device)


def setup(seed,device):
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device(device)


def optimizer_for(model,settings=None):
    p=PROTOCOL if settings is None else settings
    return torch.optim.AdamW(model.parameters(),lr=p["learning_rate"],weight_decay=p["weight_decay"],
                             betas=tuple(p["adam_betas"]),eps=p["adam_eps"])


def training_loss(model,x,settings=None):
    p=PROTOCOL if settings is None else settings
    mean,logvar=model.encode(x);z=model.reparameterize(mean,logvar);reconstruction=model.decode(z)
    bce=F.binary_cross_entropy(reconstruction.clamp(p["epsilon"],1-p["epsilon"]),x,reduction="sum")
    prior=model.student_t_elbo_loss(z,mean,logvar,eps=p["epsilon"])
    return {"loss":bce+p["prior_weight"]*prior,"bce_sum":bce,"prior_batch_mean":prior}


def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def _hardware(device):
    answer={"device":str(device),"torch_version":torch.__version__,"cuda_build":torch.version.cuda,
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "peak_process_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    if device.type=="cuda":
        answer.update(gpu=torch.cuda.get_device_name(device),peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                      peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
    return answer


def _atomic_json(path,value):
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _rng_state():
    state=np.random.get_state()
    return {"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python":random.getstate(),"numpy":{"name":state[0],"keys":torch.tensor(state[1].astype(np.int64)),
                "position":state[2],"has_gauss":state[3],"cached_gaussian":state[4]}}


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():raise ValueError("CUDA checkpoint requires CUDA resume")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"]);s=state["numpy"]
    np.random.set_state((s["name"],s["keys"].numpy().astype(np.uint32),s["position"],s["has_gauss"],s["cached_gaussian"]))


def _save_checkpoint(output,model,optimizer,state,generator,campaign,record,*,name="latest_checkpoint.pt"):
    path=output/name;temporary=path.with_name(path.name+".tmp")
    payload={"schema":"paper1-reference-pro-t-checkpoint-v1","campaign":campaign,"campaign_sha256":_digest(campaign),
             "origin_config_sha256":record["config_sha256"],"source_sha256":record["source_sha256"],
             "model":model.state_dict(),"optimizer":optimizer.state_dict(),"training_state":state,
             "permutation_rng":generator.get_state(),"rng":_rng_state()}
    with temporary.open("wb") as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    # A checkpoint and its checksum are two files. Retain the preceding valid
    # pair so an interruption between their atomic replacements is recoverable.
    receipt_path=path.with_suffix(".json")
    if name=="latest_checkpoint.pt" and path.exists() and receipt_path.exists():
        old_receipt=json.loads(receipt_path.read_text())
        if old_receipt["checkpoint_sha256"]==sha256(path):
            for old,target in [(path,output/"previous_checkpoint.pt"),(receipt_path,output/"previous_checkpoint.json")]:
                backup=target.with_name(target.name+".tmp")
                backup.unlink(missing_ok=True);os.link(old,backup);backup.replace(target)
    temporary.replace(path)
    _atomic_json(path.with_suffix(".json"),{"checkpoint_sha256":sha256(path),"campaign_sha256":_digest(campaign),
        "source_sha256":record["source_sha256"],"phase":state["phase"],"completed_updates":state["updates"],
        "completed_epochs":state["epochs_completed"],"overwrite_done":state["overwrite_done"]})
    return path


def _load_checkpoint(path,model,optimizer,generator,campaign,source_sha):
    receipt=json.loads(path.with_suffix(".json").read_text())
    if receipt["checkpoint_sha256"]!=sha256(path):raise ValueError("Resume checkpoint checksum mismatch")
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if sha256(path.parent/"config.json")!=saved["origin_config_sha256"]:
        raise ValueError("Resume origin configuration hash mismatch")
    if saved.get("schema")!="paper1-reference-pro-t-checkpoint-v1" or saved["campaign"]!=campaign or saved["campaign_sha256"]!=_digest(campaign):
        raise ValueError("Resume protocol/data/seed/device/budget mismatch")
    if saved["source_sha256"]!=source_sha or receipt["source_sha256"]!=source_sha:
        raise ValueError("Resume source hash mismatch")
    state=saved["training_state"]
    if (receipt["campaign_sha256"]!=saved["campaign_sha256"] or receipt["phase"]!=state["phase"]
        or receipt["completed_updates"]!=state["updates"] or receipt["completed_epochs"]!=state["epochs_completed"]
        or receipt["overwrite_done"]!=state["overwrite_done"]):
        raise ValueError("Resume receipt/training state mismatch")
    if state["phase"] not in {"training","post_training","post_refit","endpoint"}:
        raise ValueError("Unsupported resume phase")
    if state["epochs_completed"]<campaign["settings"]["overwrite_epoch"] and state["overwrite_done"]:
        raise ValueError("Inconsistent prior-overwrite flag")
    overwrite_epoch=campaign["settings"]["overwrite_epoch"]
    must_have_overwritten=(state["epochs_completed"]>overwrite_epoch or
                          (state["epochs_completed"]==overwrite_epoch and state["phase"]!="post_training"))
    if must_have_overwritten and not state["overwrite_done"]:
        raise ValueError("Missing required prior overwrite")
    if state["overwrite_count"]!=int(state["overwrite_done"]):
        raise ValueError("Invalid number of prior overwrites")
    if not 0<=state["updates"]<=campaign["update_budget"]:
        raise ValueError("Invalid optimizer update count")
    model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
    generator.set_state(saved["permutation_rng"]);_restore_rng(saved["rng"])
    return state


@torch.no_grad()
def embeddings(model,x,device,batch_size=256):
    model.eval()
    return np.concatenate([model.encode(x[i:i+batch_size].to(device))[0].cpu().numpy()
                           for i in range(0,len(x),batch_size)])


def fit_training_gmm(z,settings=None):
    p=PROTOCOL if settings is None else settings
    gmm=GaussianMixture(n_components=p["n_clusters"],**p["gmm"]).fit(z)
    if not all(np.isfinite(a).all() for a in [gmm.means_,gmm.covariances_,gmm.weights_]):
        raise FloatingPointError("Nonfinite training GMM")
    return gmm


def gmm_arrays(gmm):
    return {"means":gmm.means_,"covariances":gmm.covariances_,"weights":gmm.weights_,
            "precisions_cholesky":gmm.precisions_cholesky_}


@torch.no_grad()
def overwrite_prior(model,gmm,epsilon=1e-6):
    """Source's hard overwrite: covariance diagonal becomes t scale; df retained."""
    reference=model.gmm_means
    means=torch.as_tensor(gmm.means_,dtype=reference.dtype,device=reference.device)
    variance=torch.as_tensor(np.diagonal(gmm.covariances_,axis1=1,axis2=2).copy(),dtype=reference.dtype,device=reference.device).clamp_min(epsilon)
    weights=torch.as_tensor(gmm.weights_,dtype=reference.dtype,device=reference.device).clamp_min(epsilon)
    model.gmm_means.copy_(means);model.gmm_logvars.copy_(variance.log());model.logit_pi.copy_(weights.log())


@torch.no_grad()
def native_responsibilities(model,z,device,batch_size=1024):
    model.eval();answer=[]
    for start in range(0,len(z),batch_size):
        value=torch.as_tensor(z[start:start+batch_size],dtype=model.gmm_means.dtype,device=device)
        nu=model.nu_c;variance=model.gmm_logvars.exp()+1e-6
        delta=((value[:,None]-model.gmm_means).square()/variance).sum(-1)
        log_density=(torch.lgamma((nu+model.D)/2)-torch.lgamma(nu/2)-model.D/2*torch.log(nu*math.pi)
                     -.5*variance.log().sum(-1)-(nu+model.D)/2*torch.log1p(delta/nu))
        answer.append((log_density+model.logit_pi.log_softmax(0)).softmax(-1).cpu().numpy())
    return np.concatenate(answer)


def _validate_manifest(path,validation_only,settings):
    manifest=json.loads(path.read_text())
    if manifest.get("dataset")!="mnist" or manifest.get("n_features")!=784 or manifest.get("n_clusters")!=10:
        raise ValueError("Reference Pro-t requires the MNIST28x28/10cluster manifest")
    if not validation_only and manifest.get("sample_counts")!=PROTOCOL["split_counts"]:
        raise ValueError("Final reference fits require the declared54k/6k/10k split")
    if any(manifest["splits"][split]["n_samples"]!=manifest["sample_counts"][split]
           for split in ["train","validation","test"]):
        raise ValueError("Split entries disagree with manifest sample counts")
    if not validation_only and settings!=PROTOCOL:raise ValueError("Final reference protocol is frozen")
    return manifest


def _inputs(path,split):
    arrays=load_inputs(path,split)
    x=arrays["x"]
    if x.shape[1:]!=(1,28,28) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError("MNIST inputs must be finite float32NCHW in[0,1]")
    return torch.from_numpy(x),arrays["ids"]


def _save_predictions(path,probabilities,ids):
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1,atol=2e-5):
        raise FloatingPointError("Invalid prediction probabilities")
    np.savez_compressed(path,sample_ids=ids,predictions=probabilities.argmax(1),responsibilities=probabilities,
                        cluster_labels=np.arange(probabilities.shape[1]))


def train(manifest_path,output,*,seed,device="cuda",validation_only=False,validation_updates=4,
          resume_from=None,settings=None,stop_after_updates=None):
    """Fresh fixed endpoint or verified same-campaign continuation to a new path.

    stop_after_updates is a validation-only interruption hook, not an endpoint
    selector. It is used to test exact checkpoint continuation.
    """
    output,manifest_path=Path(output),Path(manifest_path)
    if output.exists():raise FileExistsError("Use a new attempt directory; existing runs are preserved")
    if seed not in [901,902,903]:raise ValueError("Approved reference seeds are901,902,903")
    if stop_after_updates is not None and not validation_only:raise ValueError("Interruption hook is validation-only")
    if validation_only and not 1<=validation_updates<=100:raise ValueError("Validation allows1-100updates")
    settings=protocol() if settings is None else copy.deepcopy(settings)
    manifest=_validate_manifest(manifest_path,validation_only,settings)
    effective_seed=seed+100000 if validation_only else seed
    dev=setup(effective_seed,device);audit=source_audit()
    train_x,train_ids=_inputs(manifest_path,"train")
    if len(train_x)<settings["batch_size"]:raise ValueError("At least one full training batch required")
    budget=validation_updates if validation_only else settings["epochs"]*math.ceil(len(train_x)/settings["batch_size"])
    inputs_hashes={split:manifest["artifact_sha256"][manifest["splits"][split]["inputs"]] for split in ["train","validation","test"]}
    campaign={"protocol_version":VERSION,"settings":settings,"manifest_sha256":sha256(manifest_path),
              "input_sha256":inputs_hashes,"manifest_seed":seed,"effective_seed":effective_seed,
              "validation_only":validation_only,"update_budget":budget,"device":device,
              "runtime":{"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__,
                         "sklearn":sklearn.__version__,"cuda_build":torch.version.cuda,
                         "gpu_name":torch.cuda.get_device_name(dev) if dev.type=="cuda" else None}}
    config=dict(campaign,manifest_path=str(manifest_path.resolve()),source_audit=audit,
                resume_from=str(Path(resume_from).resolve()) if resume_from else None,
                resume_checkpoint_sha256=sha256(Path(resume_from)) if resume_from else None,
                training_label_access=False,test_inputs_used_during_training=False)
    record=start_run(output,stage="validation" if validation_only else "final_benchmark",config=config,sources=sources())
    model=optimizer=state=generator=None;start=time.perf_counter()
    try:
        model=build_model(dev,settings);optimizer=optimizer_for(model,settings)
        generator=torch.Generator().manual_seed(effective_seed+71)
        state={"phase":"training","epochs_completed":0,"updates":0,"order":torch.empty(0,dtype=torch.long),
               "cursor":0,"overwrite_done":False,"overwrite_count":0,"history":[],"batch_lengths":[],
               "epoch_totals":{"loss":0.,"bce_sum":0.,"prior_batch_mean":0.,"updates":0},
               "training_seconds":0.,"overwrite_gmm":None}
        if resume_from:
            state=_load_checkpoint(Path(resume_from),model,optimizer,generator,campaign,record["source_sha256"])
        if dev.type=="cuda":torch.cuda.reset_peak_memory_stats(dev)
        _save_checkpoint(output,model,optimizer,state,generator,campaign,record)
        while state["updates"]<budget or state["phase"] in {"post_training","post_refit"}:
            if state["phase"]=="training":
                if not len(state["order"]):
                    state["order"]=torch.randperm(len(train_x),generator=generator);state["cursor"]=0
                indices=state["order"][state["cursor"]:state["cursor"]+settings["batch_size"]]
                model.train();optimizer.zero_grad(set_to_none=True);_sync(dev);step=time.perf_counter()
                loss=training_loss(model,train_x[indices].to(dev),settings)
                if not all(torch.isfinite(v) for v in loss.values()):raise FloatingPointError("Nonfinite source-recipe objective")
                loss["loss"].backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip"],error_if_nonfinite=True)
                optimizer.step();_sync(dev)
                if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError("Nonfinite post-update parameters")
                state["training_seconds"]+=time.perf_counter()-step
                state["updates"]+=1;state["cursor"]+=len(indices);state["batch_lengths"].append(len(indices))
                for key in ["loss","bce_sum","prior_batch_mean"]:state["epoch_totals"][key]+=float(loss[key].detach())
                state["epoch_totals"]["updates"]+=1
                if state["cursor"]>=len(train_x):
                    state["epochs_completed"]+=1;state["phase"]="post_training"
                    _save_checkpoint(output,model,optimizer,state,generator,campaign,record)
                if stop_after_updates is not None and state["updates"]==stop_after_updates:
                    _save_checkpoint(output,model,optimizer,state,generator,campaign,record)
                    raise InterruptedError("Controlled validation interruption after a completed update")
            if state["phase"]=="post_training":
                if state["epochs_completed"]==settings["overwrite_epoch"] and not state["overwrite_done"]:
                    _save_checkpoint(output,model,optimizer,state,generator,campaign,record,name="before_overwrite_checkpoint.pt")
                    gmm=fit_training_gmm(embeddings(model,train_x,dev),settings)
                    overwrite_prior(model,gmm,settings["epsilon"])
                    state["overwrite_done"]=True;state["overwrite_count"]+=1
                    state["overwrite_gmm"]={"converged":bool(gmm.converged_),"iterations":int(gmm.n_iter_),
                        "lower_bound":float(gmm.lower_bound_),"train_ids_sha256":_ids_hash(train_ids)}
                    np.savez_compressed(output/"overwrite_epoch_gmm.npz",**gmm_arrays(gmm))
                    state["phase"]="post_refit"
                    _save_checkpoint(output,model,optimizer,state,generator,campaign,record,name="after_overwrite_checkpoint.pt")
                else:state["phase"]="post_refit"
                _save_checkpoint(output,model,optimizer,state,generator,campaign,record)
            if state["phase"]=="post_refit":
                state["history"].append(dict(epoch=state["epochs_completed"],completed_updates=state["updates"],
                    **state["epoch_totals"],nu=model.nu_c.detach().cpu().tolist(),overwrite_done=state["overwrite_done"]))
                state["epoch_totals"]={"loss":0.,"bce_sum":0.,"prior_batch_mean":0.,"updates":0}
                state["order"]=torch.empty(0,dtype=torch.long);state["cursor"]=0;state["phase"]="training"
                _atomic_json(output/"training_history.json",state["history"])
                _save_checkpoint(output,model,optimizer,state,generator,campaign,record)
                print(json.dumps({"epoch":state["epochs_completed"],"updates":state["updates"],"overwrite_done":state["overwrite_done"]}),flush=True)
        if not validation_only and (state["epochs_completed"]!=settings["epochs"] or state["overwrite_count"]!=1):
            raise RuntimeError("Final endpoint/overwrite budget not met")
        state["phase"]="endpoint"
        checkpoint=_save_checkpoint(output,model,optimizer,state,generator,campaign,record,name="endpoint_checkpoint.pt")
        _save_checkpoint(output,model,optimizer,state,generator,campaign,record)
        # Freeze model and fit the reporting GMM BEFORE reading heldout inputs.
        train_z=embeddings(model,train_x,dev);gmm=fit_training_gmm(train_z,settings)
        np.savez_compressed(output/"endpoint_train_gmm.npz",**gmm_arrays(gmm))
        np.savez_compressed(output/"training_ids.npz",sample_ids=train_ids)
        prediction_splits=["train","validation"] if validation_only else ["train","validation","test"]
        for split in prediction_splits:
            if split=="train":z,ids=train_z,train_ids
            else:
                x,ids=_inputs(manifest_path,split);z=embeddings(model,x,dev)
            _save_predictions(output/f"{split}_native_predictions.npz",native_responsibilities(model,z,dev),ids)
            _save_predictions(output/f"{split}_gmm_refit_predictions.npz",gmm.predict_proba(z),ids)
        np.save(output/"batch_lengths.npy",np.asarray(state["batch_lengths"],dtype=np.int32))
        _atomic_json(output/"training_history.json",state["history"])
        finish_run(output,record,status="completed",labels_loaded=False,test_inputs_used_during_training=False,
                   fixed_endpoint=True,publishable_numerical_result=not validation_only,
                   prediction_splits=prediction_splits,test_inputs_loaded=not validation_only,
                   completed_epochs=state["epochs_completed"],optimizer_updates=state["updates"],
                   overwrite_count=state["overwrite_count"],overwrite_gmm=state["overwrite_gmm"],
                   endpoint_gmm={"converged":bool(gmm.converged_),"iterations":int(gmm.n_iter_),"lower_bound":float(gmm.lower_bound_)},
                   checkpoint_sha256=sha256(checkpoint),training_seconds=state["training_seconds"],
                   attempt_elapsed_seconds=time.perf_counter()-start,hardware=_hardware(dev))
        return output
    except BaseException as error:
        # The latest checkpoint is the last validated update/boundary, not a
        # potentially nonfinite failing operation. Failure details remain separate.
        if model is not None and state is not None and isinstance(error,InterruptedError):
            _save_checkpoint(output,model,optimizer,state,generator,campaign,record,name="interrupted_checkpoint.pt")
        finish_run(output,record,status="failed",error=f"{type(error).__name__}: {error}",labels_loaded=False,
                   completed_updates=state["updates"] if state else 0,
                   resumable_checkpoint="latest_checkpoint.pt" if (output/"latest_checkpoint.pt").exists() else None,
                   attempt_elapsed_seconds=time.perf_counter()-start,hardware=_hardware(dev))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("protocol")
    fit=sub.add_parser("train");fit.add_argument("--manifest",type=Path,required=True)
    fit.add_argument("--seed",type=int,required=True,choices=[901,902,903]);fit.add_argument("--output",type=Path,required=True)
    fit.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    fit.add_argument("--validation-only",action="store_true");fit.add_argument("--validation-updates",type=int,default=4)
    fit.add_argument("--resume-from",type=Path)
    args=parser.parse_args()
    if args.command=="protocol":print(json.dumps(protocol(),indent=2))
    else:train(args.manifest,args.output,seed=args.seed,device=args.device,validation_only=args.validation_only,
               validation_updates=args.validation_updates,resume_from=args.resume_from)


if __name__=="__main__":main()
