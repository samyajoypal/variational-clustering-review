"""Independent exact Gaussian recognition-mixture control checks."""
import math
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from experiments.paper1_structured_inference.structured import CenteredGaussianSite
from experiments.paper1_structured_inference.gaussian_reference import gaussian_mixture_posterior
D=torch.float64


def inputs():
    return (
        torch.tensor([[-1.,.2],[1.5,-.3]],dtype=D),
        torch.tensor([[.7,1.3],[1.2,.8]],dtype=D),
        torch.tensor([.2,-.4],dtype=D),
        torch.tensor([[-.4,.1],[1.2,-.7]],dtype=D),
        torch.tensor([[.4,1.1],[2.,.7]],dtype=D),
    )


def test_evidence_against_independent_scipy_normal_convolution():
    loc,cov,logits,y,lam=inputs();q=gaussian_mixture_posterior(loc,cov,logits,CenteredGaussianSite(y,lam))
    w=logits.softmax(-1).numpy()
    for b in range(len(y)):
        evidence=[]
        for k in range(len(loc)):
            log_convolution=multivariate_normal.logpdf(y[b].numpy(),mean=loc[k].numpy(),cov=np.diag(cov[k].numpy()+1/lam[b].numpy()))
            evidence.append(log_convolution+.5*np.log(2*np.pi/lam[b].numpy()).sum())
        ln=logsumexp(np.log(w)+evidence)
        np.testing.assert_allclose(q.component_log_evidence[b],evidence,atol=2e-14,rtol=2e-14)
        np.testing.assert_allclose(q.log_normalizer[b],ln,atol=2e-14,rtol=2e-14)
        np.testing.assert_allclose(q.responsibilities[b],np.exp(np.log(w)+evidence-ln),atol=2e-14,rtol=2e-14)
    assert q.conditional_mean.shape==(2,2,1,2)
    assert torch.equal(q.component_node_weights,torch.ones((2,2,1),dtype=D))


def test_linear_elbo_equals_exact_mixture_evidence():
    loc,cov,logits,x,lam=inputs();q=gaussian_mixture_posterior(loc,cov,logits,CenteredGaussianSite(x,lam))
    ell=q.expected_linear_log_likelihood(x,torch.eye(2,dtype=D),1/lam)
    log_evidence=q.log_normalizer-.5*(torch.log(2*math.pi/lam)).sum(-1)
    torch.testing.assert_close(q.beta_objective(ell),log_evidence,atol=2e-14,rtol=2e-14)


def test_marginal_density_ratio_equals_site_over_normalizer():
    loc,cov,logits,y,lam=inputs();q=gaussian_mixture_posterior(loc,cov,logits,CenteredGaussianSite(y,lam))
    points=np.random.default_rng(481).normal(size=(17,2))*2
    logprior=logsumexp(np.stack([logits.log_softmax(-1)[k].item()+multivariate_normal.logpdf(points,mean=loc[k].numpy(),cov=np.diag(cov[k].numpy())) for k in range(2)]),axis=0)
    for b in range(2):
        logpost=logsumexp(np.stack([np.log(q.responsibilities[b,k].item())+multivariate_normal.logpdf(points,mean=q.component_mean[b,k].numpy(),cov=np.diag(q.component_variance[b,k].numpy())) for k in range(2)]),axis=0)
        logsite=-.5*(lam[b].numpy()*(points-y[b].numpy())**2).sum(-1)
        np.testing.assert_allclose(logpost-logprior,logsite-q.log_normalizer[b].item(),atol=3e-14,rtol=3e-14)


def test_all_inputs_have_correct_gradients_and_sampled_api_works():
    values=tuple(t.requires_grad_() for t in inputs())
    def f(loc,cov,logits,y,lam):
        q=gaussian_mixture_posterior(loc,cov,logits,CenteredGaussianSite(y,lam))
        ell=q.expected_linear_log_likelihood(torch.zeros_like(y),torch.eye(2,dtype=D),.8)
        return q.beta_objective(ell,.2)+q.mean.sum(-1)*.1+q.variance.sum(-1)*.03
    assert torch.autograd.gradcheck(f,values,eps=1e-6,atol=2e-6,rtol=2e-5)
    loc,cov,logits,y,lam=values;q=gaussian_mixture_posterior(loc,cov,logits,CenteredGaussianSite(y,lam))
    v=q.expected_log_likelihood_sampled_nodes(lambda z:-z.square().sum(-1),n_samples=8,generator=torch.Generator().manual_seed(178))
    gradients=torch.autograd.grad(v.sum(),values)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g.abs().max()>1e-8 for g in gradients)


def test_zero_site_returns_gaussian_prior_exactly():
    loc,cov,logits,y,lam=inputs();q=gaussian_mixture_posterior(loc,cov,logits,CenteredGaussianSite(y,torch.zeros_like(lam)))
    w=logits.softmax(-1);mean=(w[:,None]*loc).sum(0);var=(w[:,None]*(cov+loc.square())).sum(0)-mean.square()
    torch.testing.assert_close(q.mean,mean.expand_as(y),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(q.variance,var.expand_as(y),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(q.log_normalizer,torch.zeros(2,dtype=D),atol=1e-14,rtol=1e-14)
