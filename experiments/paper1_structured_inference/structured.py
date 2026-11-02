"""Differentiable Student mixture prior multiplied by a diagonal Gaussian site.

The target density is continuous in the scalar precision tau. Fixed Gauss-Legendre
nodes integrate u=log(tau); normalized node contributions approximate integrals,
not a claim that the continuous variational distribution has point masses.
Finite quadrature need not retain an exact ELBO inequality. Check node/bound
convergence, especially for extreme degrees of freedom or inverse-tau moments.

Student ``scale_diag`` contains diagonal SCALE-MATRIX entries, not standard
deviations or covariance. Component covariance is nu/(nu-2)*scale_diag for nu>2.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn


class DiagonalStudentMixture(nn.Module):
    """Trainable mixture parameters; positive values use logarithmic coordinates."""

    def __init__(
        self,
        loc: Tensor,
        scale_diag: Tensor,
        df: Tensor,
        logits: Tensor | None = None,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc)
        if not loc.is_floating_point():
            loc = loc.to(torch.get_default_dtype())
        scale_diag = torch.as_tensor(scale_diag, dtype=loc.dtype, device=loc.device)
        df = torch.as_tensor(df, dtype=loc.dtype, device=loc.device)
        if loc.ndim != 2 or scale_diag.shape != loc.shape or df.shape != loc.shape[:1]:
            raise ValueError("loc/scale_diag must have shape (K,D), and df shape (K,)")
        if not torch.isfinite(loc).all() or not torch.isfinite(scale_diag).all() or not torch.isfinite(df).all():
            raise ValueError("Prior parameters must be finite")
        if (scale_diag <= 0).any() or (df <= 0).any():
            raise ValueError("Scale entries and df must be positive")
        if logits is None:
            logits = loc.new_zeros(loc.shape[0])
        logits = torch.as_tensor(logits, dtype=loc.dtype, device=loc.device)
        if logits.shape != df.shape or not torch.isfinite(logits).all():
            raise ValueError("logits must have shape (K,) and be finite")
        self.loc = nn.Parameter(loc.detach().clone(), requires_grad=trainable)
        self.log_scale_diag = nn.Parameter(scale_diag.detach().clone().log(), requires_grad=trainable)
        self.log_df = nn.Parameter(df.detach().clone().log(), requires_grad=trainable)
        self.logits = nn.Parameter(logits.detach().clone(), requires_grad=trainable)

    @property
    def scale_diag(self) -> Tensor:
        return self.log_scale_diag.exp()

    @property
    def df(self) -> Tensor:
        return self.log_df.exp()

    @property
    def log_weights(self) -> Tensor:
        return self.logits.log_softmax(dim=0)


@dataclass
class CenteredGaussianSite:
    """t(z|x)=exp[-0.5*sum_j precision_j*(z_j-center_j)^2], shapes (B,D).

    Precision may be exactly zero: these coordinates supply no evidence. Use
    softplus outside this class when producing positive precision with a network.
    """

    center: Tensor
    precision: Tensor


@dataclass
class StructuredPosterior:
    """Quadrature approximations to the continuous prior-times-site posterior."""

    log_normalizer: Tensor                 # (B,)
    responsibilities: Tensor                # (B,K)
    component_log_evidence: Tensor          # (B,K)
    conditional_mean: Tensor               # (B,K,Q,D)
    conditional_variance: Tensor           # (B,K,Q,D)
    integration_weights: Tensor            # (B,K,Q), sum_{k,q}=1
    component_node_weights: Tensor         # (B,K,Q), sum_q=1 for each component
    component_log_node_weights: Tensor     # (B,K,Q), stable before exponentiation
    mean: Tensor                           # (B,D)
    variance: Tensor                       # (B,D), marginal diagonal
    expected_log_site: Tensor              # (B,)
    component_mean: Tensor                 # (B,K,D)
    component_variance: Tensor             # (B,K,D)
    tau: Tensor                            # (Q,) fixed grid; (K,Q) standardized grid
    prior_quadrature_mass: Tensor           # (K,), should be close to 1

    @property
    def kl_to_prior(self) -> Tensor:
        """Continuous identity E_q log(site)-logZ, approximately integrated."""
        return self.expected_log_site - self.log_normalizer

    def beta_objective(self, expected_log_likelihood: Tensor, beta: float | Tensor = 1.0) -> Tensor:
        """Per-observation expected log likelihood minus beta times KL to prior."""
        if expected_log_likelihood.shape != self.log_normalizer.shape:
            raise ValueError("expected_log_likelihood must have shape (B,)")
        return expected_log_likelihood - beta * self.kl_to_prior

    def expected_linear_log_likelihood(
        self,
        x: Tensor,
        matrix: Tensor,
        noise_variance: float | Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        """E_q log N(x; matrix @ z + bias, diag(noise_variance)), shape (B,).

        ``matrix`` has shape (P,D); x is (B,P). noise_variance is scalar, (P,)
        or (B,P). The conditional Gaussian expectation is analytic. Gradients
        pass through decoder, observations, site and all mixture parameters.
        """
        if matrix.ndim != 2 or matrix.shape[-1] != self.mean.shape[-1]:
            raise ValueError("matrix must have shape (P,D)")
        if x.shape != (self.mean.shape[0], matrix.shape[0]):
            raise ValueError("x must have shape (B,P)")
        noise = torch.as_tensor(noise_variance, dtype=x.dtype, device=x.device)
        if not torch.isfinite(noise).all() or (noise <= 0).any():
            raise ValueError("noise_variance must be positive and finite")
        noise = torch.broadcast_to(noise, x.shape)
        prediction = torch.einsum("bkqd,pd->bkqp", self.conditional_mean, matrix)
        if bias is not None:
            prediction = prediction + bias
        prediction_variance = torch.einsum("bkqd,pd->bkqp", self.conditional_variance, matrix.square())
        squared_error = (x[:, None, None, :] - prediction).square() + prediction_variance
        conditional = -0.5 * (
            math.log(2 * math.pi) + noise[:, None, None, :].log()
            + squared_error / noise[:, None, None, :]
        ).sum(dim=-1)
        return (self.integration_weights * conditional).sum(dim=(1, 2))

    def expected_log_likelihood_mc(
        self,
        log_likelihood: Callable[[Tensor], Tensor],
        n_samples: int = 1,
        *,
        base_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Conditional Gaussian MC, enumerating components and precision nodes.

        The callable accepts z of shape (B,K,Q,S,D) and returns (B,K,Q,S).
        Supply fixed base_noise with that shape for gradient checks/common random
        numbers. No categorical sampling occurs; gradients include changes in
        normalized integration weights as well as conditional Gaussian samples.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        shape = (*self.conditional_mean.shape[:-1], n_samples, self.mean.shape[-1])
        if base_noise is None:
            base_noise = torch.randn(shape, dtype=self.mean.dtype, device=self.mean.device, generator=generator)
        elif base_noise.shape != shape:
            raise ValueError(f"base_noise must have shape {shape}")
        samples = self.conditional_mean.unsqueeze(-2) + self.conditional_variance.sqrt().unsqueeze(-2) * base_noise
        values = log_likelihood(samples)
        if values.shape != shape[:-1]:
            raise ValueError("log_likelihood must return shape (B,K,Q,S)")
        return (self.integration_weights * values.mean(dim=-1)).sum(dim=(1, 2))

    def expected_log_likelihood_sampled_nodes(
        self,
        log_likelihood: Callable[[Tensor], Tensor],
        n_samples: int = 8,
        *,
        node_indices: Tensor | None = None,
        base_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Enumerate components and sample precision nodes with a score correction.

        The callable accepts (B,K,S,D) and returns (B,K,S). For each component,
        draw S>=2 independent node indices with replacement, then one conditional
        Gaussian sample per node. Conditional normals use pathwise derivatives;
        node weights use a score-function surrogate with a leave-one-out baseline.
        Component responsibility derivatives are retained outside this estimator.

        This yields an unbiased value and gradient estimator of the FINITE
        QUADRATURE expectation, under independent sampling. It does not assert
        that the continuous target posterior is a discrete-node distribution.
        Supplied node_indices/base_noise are for controlled experiments and tests;
        arbitrary dependent draws do not retain the independent-LOO guarantee.
        """
        if n_samples < 2:
            raise ValueError("At least two independent samples are required for the leave-one-out baseline")
        batch, components, nodes = self.component_node_weights.shape
        if node_indices is None:
            node_indices = torch.multinomial(
                self.component_node_weights.detach().reshape(-1, nodes),
                n_samples, replacement=True, generator=generator,
            ).reshape(batch, components, n_samples)
        elif node_indices.shape != (batch, components, n_samples) or node_indices.dtype != torch.long:
            raise ValueError("node_indices must be a LongTensor with shape (B,K,S)")
        if node_indices.device != self.mean.device or (node_indices < 0).any() or (node_indices >= nodes).any():
            raise ValueError("node_indices must be on the posterior device and within the quadrature range")
        expanded = node_indices[..., None].expand(-1, -1, -1, self.mean.shape[-1])
        conditional_mean = self.conditional_mean.gather(2, expanded)
        conditional_variance = self.conditional_variance.gather(2, expanded)
        if base_noise is None:
            base_noise = torch.randn(
                conditional_mean.shape, dtype=self.mean.dtype, device=self.mean.device,
                generator=generator,
            )
        elif base_noise.shape != conditional_mean.shape:
            raise ValueError("base_noise must have shape (B,K,S,D)")
        samples = conditional_mean + conditional_variance.sqrt() * base_noise
        values = log_likelihood(samples)
        if values.shape != (batch, components, n_samples):
            raise ValueError("log_likelihood must return shape (B,K,S)")
        # Use log probabilities before exponentiation: very small node weights
        # can underflow, including nodes manually supplied in deterministic tests.
        log_node_weights = self.component_log_node_weights.gather(2, node_indices)
        detached_values = values.detach()
        baseline = (detached_values.sum(dim=-1, keepdim=True) - detached_values) / (n_samples - 1)
        surrogate = values + (detached_values - baseline) * (log_node_weights - log_node_weights.detach())
        return (self.responsibilities * surrogate.mean(dim=-1)).sum(dim=-1)


@lru_cache(maxsize=32)
def _legendre_numpy(nodes: int, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    if nodes < 2 or not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Use nodes>=2 and finite increasing log_tau_bounds")
    x, w = np.polynomial.legendre.leggauss(nodes)
    return lower + (x + 1) * (upper - lower) / 2, w * (upper - lower) / 2


def structured_posterior(
    prior: DiagonalStudentMixture,
    site: CenteredGaussianSite,
    *,
    nodes: int = 128,
    log_tau_bounds: tuple[float, float] = (-16.0, 8.0),
    grid: str = "log_tau",
    standardized_bounds: tuple[float, float] = (-24.0, 12.0),
) -> StructuredPosterior:
    """Integrate continuous Gamma precision with fixed differentiable quadrature.

    In the default grid, nodes/bounds do not move with nu, so autograd includes the complete
    Gamma-density derivative with respect to nu. Component prior quadrature mass
    is exposed rather than secretly normalized to substitute a different prior.
    Finite bounds can miss inverse-tau moments when nu is close to 2, and large
    nu needs finer nodes. Validate the quantities you use against wider/finer
    integration. Float64 is recommended for confirmation experiments.

    ``grid='standardized_log_tau'`` uses the exact prior log-precision mean
    digamma(nu/2)-log(nu/2) and standard deviation sqrt(trigamma(nu/2)).
    Legendre nodes span ``standardized_bounds`` in these coordinates, and both
    the moving nodes and their Jacobian remain differentiable. This resolves
    shrinking prior width at high nu, but does not certify posterior tail error:
    an extreme recognition site can shift posterior mass outside this interval.
    """
    y, precision = site.center, site.precision
    if y.ndim != 2 or y.shape != precision.shape or y.shape[-1] != prior.loc.shape[-1]:
        raise ValueError("Site center/precision must have matching shape (B,D)")
    if y.dtype != prior.loc.dtype or y.device != prior.loc.device or precision.dtype != y.dtype or precision.device != y.device:
        raise ValueError("Site and prior must share dtype and device")
    if not torch.isfinite(y).all() or not torch.isfinite(precision).all() or (precision < 0).any():
        raise ValueError("Site center must be finite; precision finite and nonnegative")
    shape = prior.df / 2
    rate = shape
    if grid not in {"log_tau", "standardized_log_tau"}:
        raise ValueError("grid must be 'log_tau' or 'standardized_log_tau'")
    bounds = log_tau_bounds if grid == "log_tau" else standardized_bounds
    coordinates_np, w_np = _legendre_numpy(int(nodes), float(bounds[0]), float(bounds[1]))
    coordinates = torch.as_tensor(coordinates_np, dtype=y.dtype, device=y.device)
    log_integration = torch.as_tensor(w_np, dtype=y.dtype, device=y.device).log()
    if grid == "standardized_log_tau":
        log_tau_mean = torch.digamma(shape) - rate.log()
        log_tau_sd = torch.polygamma(1, shape).sqrt()
        u = log_tau_mean[:, None] + log_tau_sd[:, None] * coordinates
        log_integration = log_integration[None, :] + log_tau_sd[:, None].log()
        tau = u.exp()
        tau_by_component = tau
    else:
        u = coordinates[None, :]
        tau = coordinates.exp()
        tau_by_component = tau[None, :]
    # Gamma density times d tau/du: exponent is shape*u, not (shape-1)*u.
    log_gamma_du = shape[:, None] * rate[:, None].log() - torch.lgamma(shape[:, None]) + shape[:, None] * u - rate[:, None] * tau_by_component
    log_base = log_gamma_du + log_integration
    prior_mass = torch.logsumexp(log_base, dim=-1).exp()
    scale = prior.scale_diag[None, :, None, :]
    loc = prior.loc[None, :, None, :]
    y4 = y[:, None, None, :]
    lam = precision[:, None, None, :]
    tau4 = tau_by_component[None, :, :, None]
    denominator = tau4 + lam * scale
    conditional_variance = scale / denominator
    conditional_mean = (tau4 * loc + lam * scale * y4) / denominator
    log_site_integral = -0.5 * (
        torch.log1p(lam * scale / tau4) + lam * (loc - y4).square() * tau4 / denominator
    ).sum(dim=-1)
    log_component = log_base[None, :, :] + log_site_integral
    component_log_evidence = torch.logsumexp(log_component, dim=-1)
    joint = log_component + prior.log_weights[None, :, None]
    log_normalizer = torch.logsumexp(joint.flatten(start_dim=1), dim=-1)
    weights = (joint - log_normalizer[:, None, None]).exp()
    responsibilities = weights.sum(dim=-1)
    component_log_node_weights = log_component - component_log_evidence[:, :, None]
    component_node_weights = component_log_node_weights.exp()
    component_mean = (component_node_weights[..., None] * conditional_mean).sum(dim=2)
    component_second = (component_node_weights[..., None] * (conditional_variance + conditional_mean.square())).sum(dim=2)
    component_variance = component_second - component_mean.square()
    mean = (weights[..., None] * conditional_mean).sum(dim=(1, 2))
    second = (weights[..., None] * (conditional_variance + conditional_mean.square())).sum(dim=(1, 2))
    variance = second - mean.square()
    conditional_log_site = -0.5 * (lam * ((conditional_mean - y4).square() + conditional_variance)).sum(dim=-1)
    expected_log_site = (weights * conditional_log_site).sum(dim=(1, 2))
    return StructuredPosterior(
        log_normalizer=log_normalizer,
        responsibilities=responsibilities,
        component_log_evidence=component_log_evidence,
        conditional_mean=conditional_mean,
        conditional_variance=conditional_variance,
        integration_weights=weights,
        component_node_weights=component_node_weights,
        component_log_node_weights=component_log_node_weights,
        mean=mean,
        variance=variance,
        expected_log_site=expected_log_site,
        component_mean=component_mean,
        component_variance=component_variance,
        tau=tau,
        prior_quadrature_mass=prior_mass,
    )
