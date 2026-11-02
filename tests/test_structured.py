"""Independent mathematical checks for the isolated Paper1 structured family."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t
import torch
from torch import nn

from experiments.paper1_structured_inference.structured import (
    CenteredGaussianSite,
    DiagonalStudentMixture,
    structured_posterior,
)

DTYPE = torch.float64


def make_prior(d=1):
    return DiagonalStudentMixture(
        torch.tensor([[-2.] + [0.] * (d-1), [2.] + [0.] * (d-1)], dtype=DTYPE),
        torch.ones((2,d), dtype=DTYPE),
        torch.tensor([5.,20.], dtype=DTYPE),
        torch.tensor([.2,-.1], dtype=DTYPE),
    )


@pytest.mark.parametrize("sigma", [.5, 2., 10.])
def test_linear_posterior_against_independent_scipy_z_quadrature(sigma):
    prior=make_prior()
    x=torch.tensor([[-4.],[-1.],[0.],[1.],[4.]], dtype=DTYPE)
    post=structured_posterior(prior, CenteredGaussianSite(x, torch.ones_like(x)/sigma**2), nodes=256)
    weights=prior.log_weights.detach().exp().numpy()
    for b,value in enumerate(x[:,0].tolist()):
        evidence=[];first=[];second=[]
        for mu,nu in [(-2.,5.),(2.,20.)]:
            def f(z):return t.pdf(z,df=nu,loc=mu)*np.exp(-.5*((z-value)/sigma)**2)
            ints=[quad(lambda z:f(z)*z**power,-np.inf,np.inf,epsabs=2e-11,epsrel=2e-11,limit=300)[0] for power in range(3)]
            evidence.append(ints[0]);first.append(ints[1]);second.append(ints[2])
        norm=weights@evidence;mean=(weights@first)/norm;variance=(weights@second)/norm-mean**2
        np.testing.assert_allclose(post.log_normalizer[b].detach(),np.log(norm),atol=2e-9,rtol=2e-9)
        np.testing.assert_allclose(post.responsibilities[b].detach(),weights*np.array(evidence)/norm,atol=2e-9,rtol=2e-9)
        np.testing.assert_allclose(post.mean[b,0].detach(),mean,atol=2e-9,rtol=2e-9)
        np.testing.assert_allclose(post.variance[b,0].detach(),variance,atol=2e-8,rtol=2e-9)
    ell=post.expected_linear_log_likelihood(x,torch.ones((1,1),dtype=DTYPE),sigma**2)
    # This site is exactly the likelihood up to its normalizing constant.
    true_log_evidence=post.log_normalizer-.5*math.log(2*math.pi*sigma**2)
    torch.testing.assert_close(post.beta_objective(ell),true_log_evidence,atol=2e-10,rtol=2e-10)


def test_no_site_returns_prior_numerically():
    prior=make_prior(d=3)
    center=torch.tensor([[1.,2.,3.],[-3.,4.,1.]],dtype=DTYPE)
    post=structured_posterior(prior,CenteredGaussianSite(center,torch.zeros_like(center)),nodes=256)
    weights=prior.log_weights.exp()
    expected_mean=(weights[:,None]*prior.loc).sum(0)
    component_variance=prior.scale_diag*prior.df[:,None]/(prior.df[:,None]-2)
    expected_variance=(weights[:,None]*(component_variance+prior.loc.square())).sum(0)-expected_mean.square()
    torch.testing.assert_close(post.prior_quadrature_mass,torch.ones(2,dtype=DTYPE),atol=2e-12,rtol=0)
    torch.testing.assert_close(post.responsibilities,weights.expand(2,2),atol=2e-12,rtol=0)
    torch.testing.assert_close(post.mean,expected_mean.expand(2,3),atol=2e-12,rtol=0)
    torch.testing.assert_close(post.variance,expected_variance.expand(2,3),atol=3e-10,rtol=0)
    torch.testing.assert_close(post.log_normalizer,torch.zeros(2,dtype=DTYPE),atol=2e-12,rtol=0)
    assert torch.count_nonzero(post.expected_log_site)==0


@pytest.mark.parametrize("extra", [1,3,10])
def test_appending_zero_site_coordinates_preserves_evidence_and_observed_marginal(extra):
    y=torch.tensor([[-1.],[1.]],dtype=DTYPE)
    base=structured_posterior(make_prior(),CenteredGaussianSite(y,torch.ones_like(y)*.25),nodes=128)
    center=torch.cat([y,torch.ones((2,extra),dtype=DTYPE)*7],dim=-1)
    precision=torch.cat([torch.ones_like(y)*.25,torch.zeros((2,extra),dtype=DTYPE)],dim=-1)
    extended=structured_posterior(make_prior(1+extra),CenteredGaussianSite(center,precision),nodes=128)
    for name in ["log_normalizer","responsibilities","expected_log_site","integration_weights"]:
        torch.testing.assert_close(getattr(base,name),getattr(extended,name),atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(base.mean,extended.mean[:,:1],atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(base.variance,extended.variance[:,:1],atol=2e-14,rtol=2e-14)
    assert (extended.variance[:,1:]>1).all()  # retained Student nuisance uncertainty


def test_64_128_256_node_convergence_and_mass_diagnostic():
    prior=make_prior()
    y=torch.tensor([[-1.],[1.]],dtype=DTYPE)
    site=CenteredGaussianSite(y,torch.ones_like(y))
    outputs=[structured_posterior(prior,site,nodes=n) for n in [64,128,256]]
    def flat(q):return torch.cat([q.log_normalizer,q.mean.flatten(),q.variance.flatten(),q.responsibilities.flatten()])
    error64=(flat(outputs[0])-flat(outputs[-1])).abs().max()
    error128=(flat(outputs[1])-flat(outputs[-1])).abs().max()
    assert error128 < error64/100
    assert error128 < 1e-7
    # A too-coarse quadrature is observable; do not silently substitute a new prior.
    assert (outputs[0].prior_quadrature_mass-1).abs().max()>1e-4


class _FunctionalCheck(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior=DiagonalStudentMixture(
            torch.tensor([[-.8,.2],[.9,-.3]],dtype=DTYPE),
            torch.tensor([[1.1,.8],[.9,1.2]],dtype=DTYPE),
            torch.tensor([6.,15.],dtype=DTYPE),
            torch.tensor([.2,-.1],dtype=DTYPE),
        )

    def forward(self,center,precision,noise):
        q=structured_posterior(self.prior,CenteredGaussianSite(center,precision),nodes=64,log_tau_bounds=(-10.,4.))
        # Nonlinear likelihood and fixed Gaussian base noise: test all pathways,
        # including changes of normalized integration weights with nu and site.
        ell=q.expected_log_likelihood_mc(lambda z:-.5*(torch.sin(z[...,0])+z[...,1].square()-.3).square(),n_samples=2,base_noise=noise)
        return q.beta_objective(ell,beta=.2)+.1*q.mean.sum(-1)+.03*q.variance.sum(-1)


def test_gradcheck_all_prior_parameters_site_and_fixed_noise_mc():
    module=_FunctionalCheck()
    params=dict(module.named_parameters())
    names=list(params)
    values=tuple(p.detach().clone().requires_grad_() for p in params.values())
    center=torch.tensor([[.1,-.2]],dtype=DTYPE,requires_grad=True)
    precision=torch.tensor([[.4,.7]],dtype=DTYPE,requires_grad=True)
    noise=torch.randn((1,2,64,2,2),dtype=DTYPE,generator=torch.Generator().manual_seed(934))
    def f(*args):
        return torch.func.functional_call(module,dict(zip(names,args[:-2])),(args[-2],args[-1],noise))
    assert torch.autograd.gradcheck(f,(*values,center,precision),eps=1e-6,atol=3e-5,rtol=3e-4)
    out=f(*values,center,precision).sum()
    gradients=torch.autograd.grad(out,(*values,center,precision))
    for g in gradients:
        assert torch.isfinite(g).all()
        assert g.abs().max()>1e-8
    assert "prior.log_df" in names  # explicit nu derivative coverage


def test_linear_expectation_has_decoder_and_site_gradients():
    prior=make_prior(d=2)
    center=torch.tensor([[.2,.1]],dtype=DTYPE,requires_grad=True)
    precision=torch.tensor([[.6,.4]],dtype=DTYPE,requires_grad=True)
    matrix=torch.tensor([[.3,.7],[-.2,.4]],dtype=DTYPE,requires_grad=True)
    x=torch.tensor([[.1,-.3]],dtype=DTYPE)
    noise=torch.tensor([.7,1.2],dtype=DTYPE,requires_grad=True)
    def f(y,lam,A,R):
        q=structured_posterior(prior,CenteredGaussianSite(y,lam),nodes=128)
        return q.beta_objective(q.expected_linear_log_likelihood(x,A,R),.2)
    assert torch.autograd.gradcheck(f,(center,precision,matrix,noise),eps=1e-6,atol=2e-5,rtol=2e-4)


def test_invalid_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        DiagonalStudentMixture(torch.zeros((2,1)),torch.ones((2,1)),torch.tensor([-1.,2.]))
    prior=make_prior()
    with pytest.raises(ValueError):
        structured_posterior(prior,CenteredGaussianSite(torch.ones((1,1),dtype=DTYPE),-torch.ones((1,1),dtype=DTYPE)))
    with pytest.raises(ValueError):
        structured_posterior(prior,CenteredGaussianSite(torch.ones((1,1),dtype=DTYPE),torch.ones((1,1),dtype=DTYPE)),nodes=1)


def test_sampled_nodes_exhaustive_ordered_pairs_match_full_value_and_gradient():
    """Enumerate every Q=2 ordered-pair outcome; no Monte Carlo tolerances.

    The two normal draws use fixed -1,+1 abscissae. For this quadratic likelihood
    their average is the exact conditional Gaussian expectation. Node sampling
    remains independent across slots; its probabilities are detached when taking
    the expectation of the gradient estimator, exactly as sampling requires.
    """
    import itertools
    prior=DiagonalStudentMixture(
        torch.tensor([[-.9],[.8]],dtype=DTYPE),
        torch.tensor([[.7],[1.3]],dtype=DTYPE),
        torch.tensor([5.,9.],dtype=DTYPE),
        torch.tensor([.2,-.3],dtype=DTYPE),
    )
    center=torch.tensor([[.2]],dtype=DTYPE,requires_grad=True)
    precision=torch.tensor([[.4]],dtype=DTYPE,requires_grad=True)
    q=structured_posterior(prior,CenteredGaussianSite(center,precision),nodes=2,log_tau_bounds=(-1.,1.))
    x=torch.tensor([[.4]],dtype=DTYPE)
    exact=q.expected_linear_log_likelihood(x,torch.ones((1,1),dtype=DTYPE),.9).sum()
    noise=torch.tensor([-1.,1.],dtype=DTYPE).reshape(1,1,2,1).expand(1,2,2,1)
    def likelihood(z):return -.5*(math.log(2*math.pi*.9)+(z[...,0]-.4).square()/.9)
    expected=exact.new_zeros(())
    total_probability=0.
    for outcome in itertools.product(range(2),repeat=4):
        indices=torch.tensor(outcome,dtype=torch.long).reshape(1,2,2)
        probability=q.component_node_weights.gather(2,indices).prod().detach()
        value=q.expected_log_likelihood_sampled_nodes(likelihood,n_samples=2,node_indices=indices,base_noise=noise).sum()
        expected=expected+probability*value
        total_probability+=float(probability)
    assert abs(total_probability-1)<1e-12
    torch.testing.assert_close(expected,exact,atol=2e-12,rtol=2e-12)
    parameters=tuple(prior.parameters())+(center,precision)
    expected_gradients=torch.autograd.grad(expected,parameters,retain_graph=True)
    exact_gradients=torch.autograd.grad(exact,parameters)
    for actual,reference in zip(expected_gradients,exact_gradients):
        torch.testing.assert_close(actual,reference,atol=2e-11,rtol=2e-11)


def test_sampled_nodes_random_draw_smoke_and_validation():
    prior=make_prior()
    center=torch.tensor([[.1],[.4]],dtype=DTYPE,requires_grad=True)
    precision=torch.ones_like(center)
    q=structured_posterior(prior,CenteredGaussianSite(center,precision),nodes=64)
    generator=torch.Generator().manual_seed(99)
    value=q.expected_log_likelihood_sampled_nodes(lambda z:-torch.sin(z[...,0]).square(),n_samples=8,generator=generator)
    assert value.shape==(2,)
    gradients=torch.autograd.grad(value.sum(),tuple(prior.parameters())+(center,))
    assert all(torch.isfinite(g).all() for g in gradients)
    with pytest.raises(ValueError):
        q.expected_log_likelihood_sampled_nodes(lambda z:z[...,0],n_samples=1)
