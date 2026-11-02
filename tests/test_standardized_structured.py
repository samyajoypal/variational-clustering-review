"""Checks for parameter-dependent precision integration, including its Jacobian."""
import math
import torch
from torch import nn
from experiments.paper1_structured_inference.structured import DiagonalStudentMixture, CenteredGaussianSite, structured_posterior

D=torch.float64


def prior(df=(5.,20.)):
    return DiagonalStudentMixture(torch.tensor([[-1.,.2],[2.,-.3]],dtype=D),
                                  torch.tensor([[1.,2.],[.8,1.2]],dtype=D),torch.tensor(df,dtype=D))


def test_standardized_matches_independent_fixed_log_grid():
    p=prior(); y=torch.tensor([[-3.,1.],[1.,-2.]],dtype=D)
    site=CenteredGaussianSite(y,torch.tensor([[.5,0.],[4.,.1]],dtype=D))
    fixed=structured_posterior(p,site,nodes=512,log_tau_bounds=(-24,8))
    scaled=structured_posterior(p,site,nodes=256,grid='standardized_log_tau')
    for name in ['log_normalizer','responsibilities','mean','variance','expected_log_site']:
        torch.testing.assert_close(getattr(fixed,name),getattr(scaled,name),atol=2e-9,rtol=2e-9)


def test_standardized_resolves_high_df_prior_concentration():
    for nu in [5.,30.,300.,1e4,1e6]:
        p=prior((nu,nu));site=CenteredGaussianSite(torch.zeros((1,2),dtype=D),torch.zeros((1,2),dtype=D))
        q=structured_posterior(p,site,nodes=128,grid='standardized_log_tau')
        torch.testing.assert_close(q.prior_quadrature_mass,torch.ones(2,dtype=D),atol=2e-8,rtol=0)
        assert q.kl_to_prior.abs().max()<2e-8
    fixed=structured_posterior(p,site,nodes=128,log_tau_bounds=(-16,8))
    assert (fixed.prior_quadrature_mass-1).abs().max()>.1


def test_high_df_limit_matches_gaussian_site_inference():
    p=prior((1e6,1e6)); y=torch.tensor([[.3,-.4]],dtype=D);lam=torch.tensor([[.7,1.2]],dtype=D)
    q=structured_posterior(p,CenteredGaussianSite(y,lam),nodes=128,grid='standardized_log_tau')
    s=p.scale_diag;den=1+lam*s
    cm=(p.loc+lam*s*y)/den;cv=s/den
    lz=-.5*(torch.log(den)+lam*(p.loc-y).square()/den).sum(-1)
    ln=torch.logsumexp(p.log_weights+lz,0);r=(p.log_weights+lz-ln).exp()
    mean=(r[:,None]*cm).sum(0)
    var=(r[:,None]*(cv+cm.square())).sum(0)-mean.square()
    torch.testing.assert_close(q.log_normalizer,ln[None],atol=3e-6,rtol=0)
    torch.testing.assert_close(q.mean,mean[None],atol=3e-6,rtol=0)
    torch.testing.assert_close(q.variance,var[None],atol=3e-6,rtol=0)


class Functional(nn.Module):
    def __init__(self):super().__init__();self.prior=prior((7.,20.))
    def forward(self,y,lam):
        q=structured_posterior(self.prior,CenteredGaussianSite(y,lam),nodes=96,grid='standardized_log_tau')
        ell=q.expected_linear_log_likelihood(torch.tensor([[.5]],dtype=D),torch.tensor([[1.,.3]],dtype=D),.7)
        return q.beta_objective(ell)+q.mean.sum(-1)*.2


def test_gradcheck_includes_moving_nodes_and_jacobian():
    m=Functional();params=dict(m.named_parameters());names=list(params)
    vals=tuple(p.detach().clone().requires_grad_() for p in params.values())
    y=torch.tensor([[.2,-.5]],dtype=D,requires_grad=True)
    lam=torch.tensor([[.7,1.2]],dtype=D,requires_grad=True)
    def call(*inputs):return torch.func.functional_call(m,dict(zip(names,inputs[:-2])),inputs[-2:])
    assert torch.autograd.gradcheck(call,(*vals,y,lam),eps=1e-5,atol=3e-6,rtol=3e-5)
