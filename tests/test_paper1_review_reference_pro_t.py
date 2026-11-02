"""Source-equation and continuation checks; no MNIST labels or old weights."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import multivariate_t
import torch
from torch.nn import functional as F

from experiments.paper1_review import reference_pro_t as ref


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture_manifest(root, n_train=64):
    root.mkdir();rng=np.random.default_rng(937)
    manifest={"schema_version":"paper1-review-dataset-v1","status":"completed","dataset":"mnist",
              "n_features":784,"n_clusters":10,"splits":{},"artifact_sha256":{},"sample_counts":{}}
    for split,count in [("train",n_train),("validation",12),("test",13)]:
        x=rng.random((count,1,28,28),dtype=np.float32);ids=np.array([f"fixture:{split}:{i}" for i in range(count)])
        filename=split+".npz";np.savez_compressed(root/filename,x=x,ids=ids)
        manifest["splits"][split]={"inputs":filename,"labels":"MUST_NOT_EXIST.npz","n_samples":count,
            "ids_sha256":ref._ids_hash(ids),"input_shape":list(x.shape)}
        manifest["artifact_sha256"][filename]=ref.sha256(root/filename);manifest["sample_counts"][split]=count
    path=root/"manifest.json";path.write_text(json.dumps(manifest));return path


def small_settings():
    p=ref.protocol();p["overwrite_epoch"]=1;p["gmm"]["n_init"]=1
    return p


def assert_nested_equal(left,right):
    if torch.is_tensor(left):assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:assert_nested_equal(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_nested_equal(a,b)
    else:assert left==right


def test_pinned_source_and_exact_residual_architecture():
    audit=ref.source_audit();assert len(audit["verified_files"])==6
    assert "MIT License" in (ref.VENDOR/"LICENSE").read_text()
    model=ref.build_model().eval();mu,lv=model.encode(torch.zeros(2,1,28,28))
    assert mu.shape==lv.shape==(2,8)
    assert model.decode(mu).shape==(2,1,28,28)
    assert model.encoder.fc1.in_features==128*7*7 and model.encoder.fc1.out_features==512
    assert len([m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)])==8
    assert [m.p for m in model.modules() if isinstance(m,torch.nn.Dropout2d)]==[.1]*4
    assert model.gmm_means.shape==(10,8)
    torch.testing.assert_close(model.nu_c,torch.full((10,),5.))
    optimizer=ref.optimizer_for(model)
    assert optimizer.defaults["lr"]==.001 and optimizer.defaults["weight_decay"]==.0001


def test_source_objective_bce_is_batch_sum_and_prior_is_batch_mean():
    ref.setup(29,"cpu");model=ref.build_model().eval();x=torch.rand(3,1,28,28)
    torch.manual_seed(21);actual=ref.training_loss(model,x)
    torch.manual_seed(21);mean,lv=model.encode(x);z=model.reparameterize(mean,lv)
    expected_bce=F.binary_cross_entropy(model.decode(z).clamp(1e-6,1-1e-6),x,reduction="sum")
    expected_prior=model.student_t_elbo_loss(z,mean,lv)
    torch.testing.assert_close(actual["loss"],expected_bce+5*expected_prior,rtol=0,atol=0)
    duplicated=model.student_t_elbo_loss(z.repeat(2,1),mean.repeat(2,1),lv.repeat(2,1))
    torch.testing.assert_close(duplicated,expected_prior)
    torch.testing.assert_close(F.binary_cross_entropy(model.decode(z).repeat(2,1,1,1),x.repeat(2,1,1,1),reduction="sum"),2*expected_bce)


def test_source_prior_value_and_gradients_against_distribution_formula():
    ref.setup(5,"cpu");model=ref.build_model().double()
    with torch.no_grad():
        model.gmm_means.normal_();model.gmm_logvars.normal_(0,.3);model.logit_pi.normal_()
    mu=torch.randn(4,8,dtype=torch.double,requires_grad=True)
    lv=torch.randn(4,8,dtype=torch.double,requires_grad=True)
    z=torch.randn(4,8,dtype=torch.double,requires_grad=True)
    actual=model.student_t_elbo_loss(z,mu,lv)
    # Independent Gamma KL implementation in torch.distributions; preserve the
    # source's epsilon in variance but unmodified logvar in Gaussian entropy.
    nu=model.nu_c;scale=model.gmm_logvars.exp()+1e-6
    r=torch.as_tensor(ref.native_responsibilities(model,z.detach().numpy(),torch.device("cpu")),dtype=torch.double)
    distance=((mu[:,None]-model.gmm_means).square()+(lv.exp()+1e-6)[:,None]).div(scale).sum(-1)
    a=(nu+8)/2;b=(nu+distance)/2;q_tau=torch.distributions.Gamma(a,b)
    gamma_kl=torch.distributions.kl_divergence(q_tau,torch.distributions.Gamma(nu/2,nu/2))
    gaussian=.5*(scale.log().sum(-1)-8*(torch.digamma(a)-b.log())-lv.sum(-1)[:,None]-8+(a/b)*distance)
    expected=(r*(gaussian+gamma_kl+(r+1e-6).log()-model.logit_pi.log_softmax(0))).sum(-1).mean()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=[mu,lv,model.gmm_means,model.gmm_logvars,model.logit_pi,model.nu_rho]
    got=torch.autograd.grad(actual,params+[z],allow_unused=True)
    want=torch.autograd.grad(expected,params)
    assert got[-1] is None # sampled responsibilities are explicitly detached
    for a,b in zip(got[:-1],want):
        assert torch.isfinite(a).all();torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-11)


def test_native_predictions_agree_with_scipy_student_density():
    ref.setup(7,"cpu");model=ref.build_model().double()
    with torch.no_grad():
        model.gmm_means.normal_();model.gmm_logvars.normal_(0,.3);model.logit_pi.normal_()
    z=np.random.default_rng(87).normal(size=(6,8))
    actual=ref.native_responsibilities(model,z,torch.device("cpu"))
    means=model.gmm_means.detach().numpy();scales=model.gmm_logvars.detach().exp().numpy()+1e-6
    nu=model.nu_c.detach().numpy();logpi=model.logit_pi.detach().log_softmax(0).numpy()
    scores=np.stack([multivariate_t.logpdf(z,loc=m,shape=np.diag(s),df=n) for m,s,n in zip(means,scales,nu)],axis=1)
    np.testing.assert_allclose(actual,softmax(scores+logpi,axis=1),rtol=3e-13,atol=3e-13)


def test_gmm_overwrite_copies_covariance_diagonal_without_changing_df_or_adam():
    model=ref.build_model();optimizer=ref.optimizer_for(model)
    (model.gmm_means.square().sum()+model.gmm_logvars.sum()+model.logit_pi.sum()+model.nu_rho.sum()).backward()
    optimizer.step();before=copy.deepcopy(optimizer.state_dict());nu=model.nu_c.detach().clone()
    cov=np.tile(np.eye(8),(10,1,1))*3;cov[:,0,1]=cov[:,1,0]=.7
    gmm=SimpleNamespace(means_=np.full((10,8),.25),covariances_=cov,weights_=np.arange(1,11)/55)
    ref.overwrite_prior(model,gmm)
    torch.testing.assert_close(model.gmm_logvars.exp(),torch.full((10,8),3.))
    torch.testing.assert_close(model.gmm_means,torch.full((10,8),.25))
    torch.testing.assert_close(model.logit_pi.softmax(0),torch.tensor(gmm.weights_,dtype=torch.float32))
    torch.testing.assert_close(model.nu_c,nu,rtol=0,atol=0);assert_nested_equal(before,optimizer.state_dict())


@pytest.mark.parametrize("interrupt_after",[1,2])
def test_actual_residual_model_resume_is_bitwise_equal_including_overwrite(tmp_path,monkeypatch,interrupt_after):
    manifest=fixture_manifest(tmp_path/"data");settings=small_settings()
    original=ref._inputs;active_output=[None]
    def inputs(path,split):
        assert split!="test" # validation never consumes official test inputs
        if split!="train":assert (active_output[0]/"endpoint_checkpoint.pt").is_file()
        return original(path,split)
    monkeypatch.setattr(ref,"_inputs",inputs)
    def run(output,**kwargs):
        active_output[0]=output
        return ref.train(manifest,output,seed=901,device="cpu",validation_only=True,validation_updates=4,settings=settings,**kwargs)
    continuous=tmp_path/"continuous";run(continuous)
    interrupted=tmp_path/"interrupted"
    with pytest.raises(InterruptedError):run(interrupted,stop_after_updates=interrupt_after)
    resumed=tmp_path/"resumed";run(resumed,resume_from=interrupted/"latest_checkpoint.pt")
    a=torch.load(continuous/"endpoint_checkpoint.pt",weights_only=True)
    b=torch.load(resumed/"endpoint_checkpoint.pt",weights_only=True)
    for key in ["model","optimizer","rng","permutation_rng"]:assert_nested_equal(a[key],b[key])
    for value in [a,b]:value["training_state"].pop("training_seconds")
    assert_nested_equal(a["training_state"],b["training_state"])
    assert b["training_state"]["overwrite_count"]==1
    assert b["training_state"]["epochs_completed"]==2 and b["training_state"]["updates"]==4
    assert (resumed/"previous_checkpoint.pt").is_file()
    for split in ["train","validation"]:
        for mode in ["native","gmm_refit"]:
            with np.load(continuous/f"{split}_{mode}_predictions.npz") as left,np.load(resumed/f"{split}_{mode}_predictions.npz") as right:
                for key in left.files:np.testing.assert_array_equal(left[key],right[key])
    record=json.loads((resumed/"run.json").read_text())
    assert not record["labels_loaded"] and not record["publishable_numerical_result"]
    assert record["stage"]=="validation" and record["status"]=="completed"
    assert not record["test_inputs_loaded"]
    assert not (resumed/"test_native_predictions.npz").exists()
    if interrupt_after==2:
        # An already completed overwrite must never be performed a second time.
        after=tmp_path/"resume_after_overwrite"
        run(after,resume_from=continuous/"after_overwrite_checkpoint.pt")
        c=torch.load(after/"endpoint_checkpoint.pt",weights_only=True)
        for key in ["model","optimizer","rng","permutation_rng"]:assert_nested_equal(a[key],c[key])
        assert c["training_state"]["overwrite_count"]==1
        assert not (after/"before_overwrite_checkpoint.pt").exists()
    with pytest.raises(FileExistsError):run(resumed)
    # The old attempt remains complete and immutable; wrong campaign is rejected.
    bad_campaign=copy.deepcopy(b["campaign"]);bad_campaign["manifest_seed"]=902
    with pytest.raises(ValueError,match="mismatch"):
        ref._load_checkpoint(resumed/"endpoint_checkpoint.pt",ref.build_model(),None,None,bad_campaign,b["source_sha256"])
    bad=tmp_path/"tampered.pt";bad.write_bytes((resumed/"endpoint_checkpoint.pt").read_bytes()+b"altered")
    bad.with_suffix(".json").write_bytes((resumed/"endpoint_checkpoint.json").read_bytes())
    with pytest.raises(ValueError,match="checksum"):
        ref._load_checkpoint(bad,None,None,None,b["campaign"],b["source_sha256"])
    with pytest.raises(ValueError,match="source hash"):
        ref._load_checkpoint(resumed/"endpoint_checkpoint.pt",None,None,None,b["campaign"],"wrong-source")
    config=resumed/"config.json";config.write_text(config.read_text()+" ")
    with pytest.raises(ValueError,match="configuration hash"):
        ref._load_checkpoint(resumed/"endpoint_checkpoint.pt",None,None,None,b["campaign"],b["source_sha256"])


def test_failed_objective_keeps_valid_initial_checkpoint_and_failure_record(tmp_path,monkeypatch):
    manifest=fixture_manifest(tmp_path/"data",32);out=tmp_path/"failed"
    monkeypatch.setattr(ref,"training_loss",lambda *a,**kw:{"loss":torch.tensor(float("nan"))})
    with pytest.raises(FloatingPointError):
        ref.train(manifest,out,seed=903,device="cpu",validation_only=True,validation_updates=1,settings=small_settings())
    record=json.loads((out/"run.json").read_text());assert record["status"]=="failed"
    checkpoint=torch.load(out/"latest_checkpoint.pt",weights_only=True)
    assert checkpoint["training_state"]["updates"]==0
    assert all(torch.isfinite(p).all() for p in checkpoint["model"].values())
    assert not (out/"endpoint_checkpoint.pt").exists()


def test_final_counts_seed_budget_and_cuda_gates(tmp_path):
    manifest=fixture_manifest(tmp_path/"data",32)
    with pytest.raises(ValueError,match="54k"):
        ref.train(manifest,tmp_path/"not_final",seed=901,device="cpu")
    with pytest.raises(ValueError,match="seeds"):
        ref.train(manifest,tmp_path/"bad_seed",seed=101,device="cpu",validation_only=True)
    with pytest.raises(ValueError,match="1-100"):
        ref.train(manifest,tmp_path/"bad_budget",seed=901,device="cpu",validation_only=True,validation_updates=101)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match="unavailable"):ref.setup(1,"cuda")
