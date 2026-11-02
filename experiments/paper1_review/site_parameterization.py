"""Candidate site head for a future protocol; frozen development models unchanged.

The learned precision has no positive lower bound:
    u = softplus(raw), lambda = ceiling * u / (1 + u).
Its upper bound is explicit and its inverse allows exact initialization to an
existing positive precision. This standard parameterization is not a novelty
claim. Finite real raw values give mathematically positive precisions that
approach zero as raw tends to minus infinity; a fixed inactive-coordinate mask
provides exact zeros in controlled tests. Floating-point underflow is not used
as the mathematical exact-zero mechanism.

A differentiable nonnegative function attaining zero at an interior point has
zero derivative there. Exact zeros with both smoothness and a nonzero escape
gradient cannot be promised. The mask is supplied structural information, not
a learned gate or evidence that a neural network discovers inactive directions.

At zero precision, the corresponding site center is unidentifiable. Masked
centers are therefore canonically set to zero. Near-zero precision alone does
not prove weak information if the center diverges; prior_site_strength checks
the combined precision/location effect when prior second moments exist.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.paper1_structured_inference.structured import CenteredGaussianSite

DEFAULT_MAX_PRECISION = math.exp(9.)  # preserves the frozen DEV upper ceiling


def _ceiling(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("max_precision must be positive and finite")
    return float(value)


def bounded_softplus_precision(raw: Tensor, *, max_precision: float = DEFAULT_MAX_PRECISION) -> Tensor:
    """No positive floor; no clipping of raw values or exp(raw) overflow."""
    maximum = _ceiling(max_precision)
    if not isinstance(raw, Tensor) or not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError("raw must be a finite floating-point tensor")
    u = F.softplus(raw)
    return maximum * (u / (1 + u))


def raw_from_precision(precision: Tensor, *, max_precision: float = DEFAULT_MAX_PRECISION) -> Tensor:
    """Inverse on 0 < precision < ceiling; zero is represented by a mask only.

    inverse_softplus(u) = u + log(-expm1(-u)) is stable for small and large u.
    """
    maximum = _ceiling(max_precision)
    if not isinstance(precision, Tensor) or not precision.is_floating_point() or not torch.isfinite(precision).all():
        raise ValueError("precision must be a finite floating-point tensor")
    if (precision <= 0).any() or (precision >= maximum).any():
        raise ValueError("initial precision must be strictly between zero and max_precision")
    u = precision / (maximum - precision)
    return u + torch.log(-torch.expm1(-u))


def _mask(mask: Any | None, dimension: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(dimension, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, device=device)
    if mask.shape != (dimension,) or mask.dtype != torch.bool:
        raise ValueError("active_mask must be a Boolean vector of length latent_dim")
    return mask


def gaussian_site_from_raw(center: Tensor, raw_precision: Tensor, *,
                           max_precision: float = DEFAULT_MAX_PRECISION,
                           active_mask: Any | None = None) -> CenteredGaussianSite:
    """Build a site, giving exact zeros and canonical centers to masked axes."""
    if center.ndim != 2 or raw_precision.shape != center.shape or center.dtype != raw_precision.dtype or center.device != raw_precision.device:
        raise ValueError("center/raw_precision must have matching (B,D) shape, dtype and device")
    if not center.is_floating_point() or not torch.isfinite(center).all():
        raise ValueError("center must be finite and floating point")
    mask = _mask(active_mask, center.shape[-1], center.device)
    precision = bounded_softplus_precision(raw_precision, max_precision=max_precision)
    return CenteredGaussianSite(torch.where(mask, center, torch.zeros_like(center)),
                                torch.where(mask, precision, torch.zeros_like(precision)))


class GaussianSitePrecisionHead(nn.Module):
    """Independent precision head for structured inference only.

    Default features are the center with known inactive coordinates zeroed,
    preventing those coordinates from influencing active precision outputs.
    If external features are supplied, the caller owns their nuisance-invariance
    properties. Gaussian/direct-t encoder variance heads need not change.
    """

    def __init__(self, latent_dim: int, *, feature_dim: int | None = None,
                 initial_precision: Any = 10., max_precision: float = DEFAULT_MAX_PRECISION,
                 active_mask: Any | None = None):
        super().__init__()
        feature_dim = latent_dim if feature_dim is None else feature_dim
        if not isinstance(latent_dim, int) or not isinstance(feature_dim, int) or latent_dim < 1 or feature_dim < 1:
            raise ValueError("latent_dim and feature_dim must be positive integers")
        self.max_precision = _ceiling(max_precision)
        self.latent_dim, self.feature_dim = latent_dim, feature_dim
        mask = _mask(active_mask, latent_dim, torch.device("cpu"))
        initial = torch.as_tensor(initial_precision, dtype=torch.get_default_dtype())
        if initial.ndim == 0:
            initial = initial.expand(latent_dim).clone()
        if initial.shape != (latent_dim,) or not torch.isfinite(initial).all():
            raise ValueError("initial_precision must be a finite scalar or vector of length latent_dim")
        if (initial < 0).any() or (initial >= self.max_precision).any() or (initial[mask] <= 0).any():
            raise ValueError("Active initial precisions must be in (0, ceiling); zero is allowed only for masked axes")
        self.raw_layer = nn.Linear(feature_dim, latent_dim)
        nn.init.zeros_(self.raw_layer.weight)
        with torch.no_grad():
            self.raw_layer.bias.zero_()
            self.raw_layer.bias[mask] = raw_from_precision(initial[mask], max_precision=self.max_precision)
        self.register_buffer("active_mask", mask)
        self.register_buffer("initial_precision", torch.where(mask, initial, torch.zeros_like(initial)))

    def forward(self, center: Tensor, *, features: Tensor | None = None) -> CenteredGaussianSite:
        if center.ndim != 2 or center.shape[-1] != self.latent_dim:
            raise ValueError("center must have shape (B, latent_dim)")
        if not center.is_floating_point() or not torch.isfinite(center).all():
            raise ValueError("center must be finite and floating point")
        canonical_center = torch.where(self.active_mask, center, torch.zeros_like(center))
        if features is None:
            features = canonical_center
        if features.shape != (len(center), self.feature_dim):
            raise ValueError("features must have shape (B, feature_dim)")
        return gaussian_site_from_raw(canonical_center, self.raw_layer(features),
                                      max_precision=self.max_precision, active_mask=self.active_mask)

    def configuration(self) -> dict[str, Any]:
        return {
            "parameterization": "ceiling * softplus(raw) / (1 + softplus(raw))",
            "positive_lower_bound": None, "max_precision": self.max_precision,
            "latent_dim": self.latent_dim, "feature_dim": self.feature_dim,
            "initial_precision": self.initial_precision.detach().cpu().tolist(),
            "active_mask": self.active_mask.detach().cpu().tolist(),
            "mask_is_learned": False,
            "zero_precision_center_convention": "zero on fixed inactive axes",
        }


def prior_site_strength(site: CenteredGaussianSite, loc: Tensor, covariance_diag: Tensor,
                        logits: Tensor) -> Tensor:
    """A(x)=E_prior[-log site(z,x)], using actual component COVARIANCES.

    For exact continuous q=prior*site/Z and nonnegative Gaussian-site energy,
    both KL(q||prior) and KL(prior||q) are <= A, by Jensen's inequality.
    This is a sufficient, conservative near-prior diagnostic; finite-quadrature
    outputs do not acquire a certificate from this identity. Student components
    require df>2 and covariance=scale*df/(df-2), not the Student scale itself.
    """
    center, precision = site.center, site.precision
    if center.ndim != 2 or precision.shape != center.shape or loc.ndim != 2 or covariance_diag.shape != loc.shape or center.shape[-1] != loc.shape[-1] or logits.shape != loc.shape[:1]:
        raise ValueError("Expected site (B,D), component loc/covariance (K,D), logits (K,)")
    tensors = (center, precision, loc, covariance_diag, logits)
    if any(t.dtype != center.dtype or t.device != center.device for t in tensors):
        raise ValueError("All tensors must share dtype and device")
    if not all(torch.isfinite(t).all() for t in tensors) or (precision < 0).any() or (covariance_diag < 0).any():
        raise ValueError("Inputs must be finite with nonnegative precision/covariance")
    per_component = 0.5 * (precision[:, None] * (covariance_diag[None] + (loc[None] - center[:, None]).square())).sum(-1)
    return (logits.softmax(0) * per_component).sum(-1)
