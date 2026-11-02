"""Exact diagonal Gaussian-mixture prior times a diagonal Gaussian site.

This recognition-mixture control separates conditional mixture capacity from
Student radial-scale integration. It shares StructuredPosterior's objective and
conditional sampling API, with one deterministic precision node per component.
"""
from __future__ import annotations
import torch
from torch import Tensor
try:
    from .structured import CenteredGaussianSite, StructuredPosterior
except ImportError:  # direct-script imports used by the isolated pilot
    from structured import CenteredGaussianSite, StructuredPosterior


def gaussian_mixture_posterior(
    loc: Tensor,
    cov_diag: Tensor,
    logits: Tensor,
    site: CenteredGaussianSite,
) -> StructuredPosterior:
    """Exact q(z)=p_GMM(z)*site(z)/Z, keeping all input tensors differentiable.

    loc/cov_diag are (K,D), logits (K,), and site center/precision (B,D).
    cov_diag is Gaussian covariance, unlike the Student scale parameter.
    """
    y, precision = site.center, site.precision
    if loc.ndim != 2 or cov_diag.shape != loc.shape or logits.shape != loc.shape[:1]:
        raise ValueError("loc/cov_diag must have shape (K,D), logits shape (K,)")
    if y.ndim != 2 or y.shape != precision.shape or y.shape[-1] != loc.shape[-1]:
        raise ValueError("site center/precision must have matching shape (B,D)")
    tensors=(loc,cov_diag,logits,y,precision)
    if any(t.dtype!=loc.dtype or t.device!=loc.device for t in tensors):
        raise ValueError("All tensors must share dtype and device")
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("All tensors must be finite")
    if (cov_diag<=0).any() or (precision<0).any():
        raise ValueError("Covariance must be positive and site precision nonnegative")
    covariance=cov_diag[None,:,:];mu=loc[None,:,:];lam=precision[:,None,:];center=y[:,None,:]
    denominator=1+lam*covariance
    component_mean=(mu+lam*covariance*center)/denominator
    component_variance=covariance/denominator
    component_log_evidence=-.5*(torch.log1p(lam*covariance)+lam*(mu-center).square()/denominator).sum(-1)
    component_logjoint=logits.log_softmax(-1)[None,:]+component_log_evidence
    log_normalizer=component_logjoint.logsumexp(-1)
    responsibilities=(component_logjoint-log_normalizer[:,None]).exp()
    mean=(responsibilities[...,None]*component_mean).sum(1)
    second=(responsibilities[...,None]*(component_variance+component_mean.square())).sum(1)
    variance=second-mean.square()
    conditional_logsite=-.5*(lam*((component_mean-center).square()+component_variance)).sum(-1)
    expected_log_site=(responsibilities*conditional_logsite).sum(-1)
    ones=loc.new_ones((*responsibilities.shape,1))
    return StructuredPosterior(
        log_normalizer=log_normalizer,
        responsibilities=responsibilities,
        component_log_evidence=component_log_evidence,
        conditional_mean=component_mean.unsqueeze(2),
        conditional_variance=component_variance.unsqueeze(2),
        integration_weights=responsibilities.unsqueeze(-1),
        component_node_weights=ones,
        component_log_node_weights=torch.zeros_like(ones),
        mean=mean,
        variance=variance,
        expected_log_site=expected_log_site,
        component_mean=component_mean,
        component_variance=component_variance,
        tau=loc.new_ones(1),
        prior_quadrature_mass=loc.new_ones(loc.shape[0]),
    )
