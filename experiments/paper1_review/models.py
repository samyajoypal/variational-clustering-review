"""Matched-backbone objective adapters, NOT native-protocol reproductions.

All probabilistic models use a feature-summed Gaussian observation likelihood
with fixed variance and batch-mean loss. DEC uses target KL alone; IDEC uses
feature-mean reconstruction MSE + gamma * target KL, matching its loss units.

Source audit (source files are read, never imported/executed):
* VaDE eca52b4b57e32a68f578de7a54eda5d39907b0fe / VaDE.py: sampled Gaussian
  assignments plus analytic Gaussian and categorical terms. Its legacy theta^D
  assignment bug is NOT copied: mixture weights enter once, outside the D sum.
* IDEC 4a661c211f082022249758ab3312f7db21e2aa99 / DEC.py, IDEC.py: Student
  assignment kernel, detached full-dataset sharpened targets, MSE + .1 KL.
* Pro-t 11966cd0a512dfeaeae636aebf7df80832e985b4 / src/tvae/reference/model28.py:
  sampled-z responsibilities detached; full Gaussian/Gamma/categorical bound.
  This adapter uses exact-positive parameters and stable logs rather than adding
  epsilon to variances while retaining the unmodified log-variance entropy.
  Shared MLP, Gaussian likelihood, reductions, bounded df and initialization
  differ from its native CNN/BCE training recipe.

Structured inference reuses the separately tested continuous-quadrature
primitives. Finite quadrature is an approximation, not a certified exact ELBO.
DEV df bounds are configurable safety controls, not a final benchmark choice.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from experiments.paper1_structured_inference.structured import (
    CenteredGaussianSite, DiagonalStudentMixture, StructuredPosterior, structured_posterior,
)
from experiments.paper1_structured_inference.gaussian_reference import gaussian_mixture_posterior
from experiments.paper1_review.site_parameterization import DEFAULT_MAX_PRECISION, GaussianSitePrecisionHead

METHODS = (
    "gaussian_q_gaussian_prior", "gaussian_q_student_prior", "student_q_student_prior",
    "structured_q_gaussian_prior", "structured_q_student_prior", "vae_gmm",
    "vade", "dec", "idec", "pro_t_vae",
)
SOURCE_REFERENCES = {
    "vade": "https://github.com/slim1017/VaDE/blob/eca52b4b57e32a68f578de7a54eda5d39907b0fe/VaDE.py",
    "dec_idec": "https://github.com/XifengGuo/IDEC/tree/4a661c211f082022249758ab3312f7db21e2aa99",
    "pro_t_vae": "https://github.com/xuzhikangnba/torch-tvae/blob/11966cd0a512dfeaeae636aebf7df80832e985b4/src/tvae/reference/model28.py",
}


def component_log_density(z: Tensor, prior: DiagonalStudentMixture, *, gaussian: bool) -> Tensor:
    """Log component densities (...,K); weights deliberately excluded."""
    d = z.shape[-1]
    delta = ((z.unsqueeze(-2) - prior.loc).square() / prior.scale_diag).sum(-1)
    if gaussian:
        # Do not evaluate a dormant Student branch: 0 * NaN can poison gradients.
        return -0.5 * (d * math.log(2 * math.pi) + prior.log_scale_diag.sum(-1) + delta)
    nu = prior.df
    return (torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2)
            - d / 2 * torch.log(nu * math.pi) - 0.5 * prior.log_scale_diag.sum(-1)
            - (nu + d) / 2 * torch.log1p(delta / nu))


def mixture_log_responsibilities(z: Tensor, prior: DiagonalStudentMixture, *, gaussian: bool) -> Tensor:
    return (component_log_density(z, prior, gaussian=gaussian) + prior.log_weights).log_softmax(-1)


def diagonal_gaussian_component_kl(mean: Tensor, log_variance: Tensor, prior: DiagonalStudentMixture) -> Tensor:
    """Exact KL(q Gaussian || each Gaussian component), shape (B,K)."""
    delta_bar = ((mean[:, None] - prior.loc).square() + log_variance.exp()[:, None]) / prior.scale_diag
    return 0.5 * (prior.log_scale_diag.sum(-1) - log_variance.sum(-1)[:, None]
                  - mean.shape[-1] + delta_bar.sum(-1))


def gaussian_categorical_regularizer(mean: Tensor, log_variance: Tensor, z: Tensor,
                                    prior: DiagonalStudentMixture) -> dict[str, Tensor]:
    """VaDE-style sampled-z categorical q; all pathwise gradients retained."""
    log_r = mixture_log_responsibilities(z, prior, gaussian=True)
    r = log_r.exp()
    kl_components = diagonal_gaussian_component_kl(mean, log_variance, prior)
    categorical = (r * (log_r - prior.log_weights)).sum(-1)
    gaussian = (r * kl_components).sum(-1)
    return {"regularizer": gaussian + categorical, "gaussian": gaussian,
            "categorical": categorical, "responsibilities": r}


def pro_t_augmented_regularizer(mean: Tensor, log_variance: Tensor, z: Tensor,
                                prior: DiagonalStudentMixture, *, softassign_detach: bool = True) -> dict[str, Tensor]:
    """Full mean-field Gaussian/Gamma/categorical objective from pinned Pro-t.

    q(u|c,x) uses the encoder-expected Mahalanobis distance. The source's
    sampled-z category update is detached by default; this is not the exact
    coordinate-optimal category update of the encoder-averaged bound.
    """
    log_r = mixture_log_responsibilities(z, prior, gaussian=False)
    if softassign_detach:
        log_r = log_r.detach()
    r = log_r.exp()
    d = mean.shape[-1]
    delta_bar = (((mean[:, None] - prior.loc).square() + log_variance.exp()[:, None]) / prior.scale_diag).sum(-1)
    a0 = prior.df[None] / 2
    b0 = a0
    a = a0 + d / 2
    b = b0 + delta_bar / 2
    eu = a / b
    elog_u = torch.digamma(a) - b.log()
    gaussian_components = 0.5 * (
        prior.log_scale_diag.sum(-1) - log_variance.sum(-1)[:, None] - d
        - d * elog_u + eu * delta_bar)
    gamma_components = ((a - a0) * torch.digamma(a) + torch.lgamma(a0) - torch.lgamma(a)
                        + a0 * (b.log() - b0.log()) + a * (b0 - b) / b)
    gaussian = (r * gaussian_components).sum(-1)
    gamma = (r * gamma_components).sum(-1)
    # log_r stays finite even when exp(log_r) underflows to zero.
    categorical = (r * (log_r - prior.log_weights)).sum(-1)
    return {"regularizer": gaussian + gamma + categorical, "gaussian": gaussian,
            "gamma": gamma, "categorical": categorical, "responsibilities": r,
            "gamma_shape": a.expand_as(b), "gamma_rate": b}


def multivariate_student_log_prob(z: Tensor, mean: Tensor, scale_diag: Tensor, df: Tensor) -> Tensor:
    """Shared-radial-precision multivariate t; NOT independent univariate t's."""
    d = z.shape[-1]
    delta = ((z - mean).square() / scale_diag).sum(-1)
    return (torch.lgamma((df + d) / 2) - torch.lgamma(df / 2)
            - d / 2 * torch.log(df * math.pi) - 0.5 * scale_diag.log().sum(-1)
            - (df + d) / 2 * torch.log1p(delta / df))


def dec_log_assignments(z: Tensor, centers: Tensor, alpha: float = 1.0) -> Tensor:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("DEC alpha must be positive and finite")
    distance = (z[:, None] - centers).square().sum(-1)
    return (-0.5 * (alpha + 1) * torch.log1p(distance / alpha)).log_softmax(-1)


def _validate_probabilities(q: Tensor, name: str) -> None:
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0 or not q.is_floating_point():
        raise ValueError(f"{name} must be a nonempty floating-point N-by-K matrix")
    if not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(-1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=0):
        raise ValueError(f"{name} must contain finite nonnegative unit-sum rows")


def dec_target_distribution(responsibilities: Tensor) -> Tensor:
    """Detached DEC target from FULL-DATA responsibilities, not a minibatch.

    An empty component remains zero; no 0/0 or differentiation into the target.
    The runner must refresh this global target on its predeclared schedule.
    """
    _validate_probabilities(responsibilities, "responsibilities")
    q = responsibilities.detach()
    weight = q.square() / q.sum(0).clamp_min(torch.finfo(q.dtype).tiny)
    return weight / weight.sum(-1, keepdim=True)


def dec_target_kl(target: Tensor, log_assignments: Tensor) -> Tensor:
    _validate_probabilities(target, "target")
    if target.shape != log_assignments.shape:
        raise ValueError("target and assignments must have the same shape")
    p = target.detach()
    return (p * (p.clamp_min(torch.finfo(p.dtype).tiny).log() - log_assignments)).sum(-1)


def _mlp(widths: Sequence[int], *, last_activation: str = "identity") -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.ReLU())
    if last_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif last_activation != "identity":
        raise ValueError("decoder_activation must be sigmoid or identity")
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    """Shared initialization module; encoder/decoder state keys match all methods."""

    def __init__(self, input_dim: int, latent_dim: int = 10,
                 hidden_dims: Sequence[int] = (500, 500, 2000), decoder_activation: str = "sigmoid"):
        super().__init__()
        if any(not isinstance(x, int) or x <= 0 for x in (input_dim, latent_dim, *hidden_dims)):
            raise ValueError("Network dimensions must be positive integers")
        self.encoder = _mlp((input_dim, *hidden_dims, latent_dim))
        self.decoder = _mlp((latent_dim, *reversed(hidden_dims), input_dim), last_activation=decoder_activation)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


class MatchedClusteringModel(MLPAutoencoder):
    def __init__(self, method: str, input_dim: int, n_clusters: int, *, latent_dim: int = 10,
                 hidden_dims: Sequence[int] = (500, 500, 2000), observation_variance: float = 0.05,
                 decoder_activation: str = "sigmoid", n_samples: int = 8, prediction_samples: int = 64,
                 nodes: int = 64, prediction_nodes: int = 128, standardized_bounds: tuple[float, float] = (-24., 12.),
                 beta: float = 1., idec_gamma: float = 0.1, dec_alpha: float = 1., df_init: float = 8.,
                 df_bounds: tuple[float, float] = (3., 30.), posterior_df_bounds: tuple[float, float] = (3., 100.),
                 initial_posterior_variance: float = 0.1, log_variance_bounds: tuple[float, float] = (-9., 2.),
                 structured_site_mode: str = "legacy", structured_site_max_precision: float = DEFAULT_MAX_PRECISION,
                 structured_site_initial_precision: float = 10.):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if not all(isinstance(value, int) for value in (n_samples, prediction_samples, nodes, prediction_nodes)) or n_samples < 2 or prediction_samples < 1 or nodes < 2 or prediction_nodes < 2:
            raise ValueError("Require n_samples>=2 (LOO estimator), prediction_samples>=1, quadrature nodes>=2")
        if not math.isfinite(observation_variance) or observation_variance <= 0 or not math.isfinite(initial_posterior_variance) or initial_posterior_variance <= 0:
            raise ValueError("Observation/posterior variances must be positive and finite")
        if not all(math.isfinite(v) for v in (*df_bounds, *posterior_df_bounds, df_init, beta, idec_gamma, dec_alpha)):
            raise ValueError("Model scalar settings must be finite")
        if not 2 < df_bounds[0] <= df_init <= df_bounds[1] or not 2 < posterior_df_bounds[0] <= df_init <= posterior_df_bounds[1]:
            raise ValueError("df bounds must contain df_init and have lower bounds greater than two")
        if beta < 0 or idec_gamma < 0 or dec_alpha <= 0:
            raise ValueError("beta/gamma must be nonnegative and DEC alpha positive")
        if len(log_variance_bounds) != 2 or not all(math.isfinite(v) for v in log_variance_bounds) or log_variance_bounds[0] >= log_variance_bounds[1]:
            raise ValueError("log_variance_bounds must be finite and increasing")
        if len(standardized_bounds) != 2 or not all(math.isfinite(v) for v in standardized_bounds) or standardized_bounds[0] >= standardized_bounds[1]:
            raise ValueError("standardized_bounds must be finite and increasing")
        if structured_site_mode not in {"legacy", "bounded_softplus"}:
            raise ValueError("structured_site_mode must be legacy or bounded_softplus")
        super().__init__(input_dim, latent_dim, hidden_dims, decoder_activation)
        self.method, self.input_dim, self.latent_dim, self.n_clusters = method, input_dim, latent_dim, n_clusters
        self.observation_variance, self.n_samples, self.prediction_samples = observation_variance, n_samples, prediction_samples
        self.nodes, self.prediction_nodes, self.standardized_bounds = nodes, prediction_nodes, standardized_bounds
        self.beta, self.idec_gamma, self.dec_alpha = beta, idec_gamma, dec_alpha
        self.df_bounds, self.posterior_df_bounds, self.log_variance_bounds = df_bounds, posterior_df_bounds, log_variance_bounds
        self.is_structured = method.startswith("structured_q_")
        self.structured_site_mode = structured_site_mode
        self.is_gaussian_prior = method in {"gaussian_q_gaussian_prior", "structured_q_gaussian_prior", "vade"}
        self.is_dec = method in {"dec", "idec"}
        self.variance_head = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(self.variance_head.bias, math.log(initial_posterior_variance))
        self.site_head = None
        if self.is_structured and structured_site_mode == "bounded_softplus":
            self.site_head = GaussianSitePrecisionHead(latent_dim,
                initial_precision=structured_site_initial_precision, max_precision=structured_site_max_precision)
            self.variance_head.requires_grad_(False)
        self.q_df_head = nn.Linear(latent_dim, 1) if method == "student_q_student_prior" else None
        if self.q_df_head is not None:
            nn.init.zeros_(self.q_df_head.weight)
            nn.init.constant_(self.q_df_head.bias, math.log(df_init))
        self.prior = None
        self.cluster_centers = None
        if self.is_dec:
            self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))
            self.variance_head.requires_grad_(False)
            if method == "dec":
                self.decoder.requires_grad_(False)
        elif method == "vae_gmm":
            self.register_buffer("gmm_means", torch.zeros(n_clusters, latent_dim))
            self.register_buffer("gmm_log_covariance", torch.zeros(n_clusters, latent_dim))
            self.register_buffer("gmm_log_weights", torch.full((n_clusters,), -math.log(n_clusters)))
            self.register_buffer("gmm_fitted", torch.tensor(False))
        else:
            scale = 1. if self.is_gaussian_prior else (df_init - 2) / df_init
            self.prior = DiagonalStudentMixture(torch.zeros(n_clusters, latent_dim), torch.full((n_clusters, latent_dim), scale), torch.full((n_clusters,), df_init))
            if self.is_gaussian_prior:
                self.prior.log_df.requires_grad_(False)
        self.objective_metadata = {
            "method": method, "native_reproduction": False,
            "input_dim": input_dim, "latent_dim": latent_dim, "n_clusters": n_clusters,
            "hidden_dims": list(hidden_dims), "decoder_activation": decoder_activation,
            "observation_model": "feature-summed fixed-variance Gaussian" if not self.is_dec else "DEC target KL; IDEC adds feature-mean MSE",
            "observation_variance": observation_variance, "beta": beta, "idec_gamma": idec_gamma,
            "prior_df_bounds": list(df_bounds), "posterior_df_bounds": list(posterior_df_bounds),
            "prior_df_init": df_init, "scale_initialization": "Student component covariance matched to supplied GMM covariance",
            "prior_scale_diagonal_bounds": [1e-4, 100.], "log_variance_bounds": list(log_variance_bounds),
            "initial_posterior_variance": initial_posterior_variance,
            "structured_site_mode": structured_site_mode if self.is_structured else None,
            "structured_site": (self.site_head.configuration() if self.site_head is not None else {
                "parameterization": "exp(-clamp(log_variance, lower, upper))",
                "positive_lower_bound": math.exp(-log_variance_bounds[1]),
                "max_precision": math.exp(-log_variance_bounds[0]),
                "initial_precision": math.exp(-min(max(math.log(initial_posterior_variance), log_variance_bounds[0]), log_variance_bounds[1])),
            } if self.is_structured else None),
            "n_samples": n_samples, "prediction_samples": prediction_samples,
            "quad_grid": "standardized_log_tau", "nodes": nodes, "prediction_nodes": prediction_nodes,
            "standardized_bounds": list(standardized_bounds),
            "pro_t_softassign_detach": method == "pro_t_vae",
            "vade_assignment": "differentiable sampled-z Gaussian responsibility; mixture weight enters once",
            "prediction_policy": "native assignments; G/Student encoders average posterior-sample responsibilities; structured integrates components; VAE+GMM uses post-fit GMM on encoder means",
            "source_references": SOURCE_REFERENCES,
        }

    @torch.no_grad()
    def set_cluster_parameters(self, loc: Any, covariance_diag: Any, weights: Any) -> None:
        """Initialize mixture/centers; VAE+GMM must call again after VAE fitting."""
        reference = next(self.encoder.parameters())
        loc, cov, weights = [torch.as_tensor(value, dtype=reference.dtype, device=reference.device) for value in (loc, covariance_diag, weights)]
        if loc.shape != (self.n_clusters, self.latent_dim) or cov.shape != loc.shape or weights.shape != (self.n_clusters,):
            raise ValueError("Expected loc/covariance (K,D), weights (K,)")
        if not all(torch.isfinite(t).all() for t in (loc, cov, weights)) or (cov <= 0).any() or (weights <= 0).any():
            raise ValueError("Initialization requires finite locations, positive covariance and positive weights")
        log_weights = weights.log().log_softmax(0)
        if self.is_dec:
            self.cluster_centers.copy_(loc)
        elif self.method == "vae_gmm":
            self.gmm_means.copy_(loc)
            self.gmm_log_covariance.copy_(cov.log())
            self.gmm_log_weights.copy_(log_weights)
            self.gmm_fitted.fill_(True)
        else:
            scale = cov if self.is_gaussian_prior else cov * (self.prior.df[:, None] - 2) / self.prior.df[:, None]
            self.prior.loc.copy_(loc)
            self.prior.log_scale_diag.copy_(scale.log())
            self.prior.logits.copy_(log_weights)

    @torch.no_grad()
    def constrain_parameters(self) -> None:
        if self.prior is not None:
            if not self.is_gaussian_prior:
                self.prior.log_df.clamp_(math.log(self.df_bounds[0]), math.log(self.df_bounds[1]))
            self.prior.log_scale_diag.clamp_(math.log(1e-4), math.log(100.))

    def _encoded(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.encoder(x)
        return mean, self.variance_head(mean).clamp(*self.log_variance_bounds)

    def _samples(self, mean: Tensor, log_variance: Tensor, count: int) -> tuple[Tensor, Tensor]:
        eps = torch.randn((len(mean), count, self.latent_dim), dtype=mean.dtype, device=mean.device)
        if self.q_df_head is not None:
            nu = self.q_df_head(mean).clamp(math.log(self.posterior_df_bounds[0]), math.log(self.posterior_df_bounds[1])).exp()
            scale = log_variance.exp() * (nu - 2) / nu
            precision = torch.distributions.Gamma(nu / 2, nu / 2).rsample((count,)).permute(1, 0, 2)
            z = mean[:, None] + scale.sqrt()[:, None] * eps / precision.sqrt()
            log_q = multivariate_student_log_prob(z, mean[:, None], scale[:, None], nu)
        else:
            z = mean[:, None] + (0.5 * log_variance).exp()[:, None] * eps
            log_q = -0.5 * (self.latent_dim * math.log(2 * math.pi) + log_variance.sum(-1)[:, None] + eps.square().sum(-1))
        return z, log_q

    def make_site(self, mean: Tensor, log_variance: Tensor) -> CenteredGaussianSite:
        """Single site-construction path for training and quadrature diagnostics.

        The legacy input is the already bounded encoder log variance. In the
        optional no-floor mode log_variance is unused and its head is frozen.
        Finite learned raw values approach zero; no inactive axes are assumed.
        """
        if not self.is_structured:
            raise ValueError("Gaussian sites apply only to structured methods")
        if self.site_head is not None:
            return self.site_head(mean)
        return CenteredGaussianSite(mean, (-log_variance).exp())

    def _structured(self, mean: Tensor, log_variance: Tensor, *, prediction: bool = False) -> StructuredPosterior:
        site = self.make_site(mean, log_variance)
        if self.is_gaussian_prior:
            return gaussian_mixture_posterior(self.prior.loc, self.prior.scale_diag, self.prior.logits, site)
        return structured_posterior(self.prior, site, nodes=self.prediction_nodes if prediction else self.nodes,
                                    grid="standardized_log_tau", standardized_bounds=self.standardized_bounds)

    def _log_likelihood(self, x: Tensor, recon: Tensor) -> Tensor:
        return -0.5 * (math.log(2 * math.pi * self.observation_variance) + (x - recon).square() / self.observation_variance).sum(-1)

    def training_loss(self, x: Tensor, target: Tensor | None = None) -> dict[str, Tensor]:
        mean, log_variance = self._encoded(x)
        mass_error = x.new_zeros(())
        if self.is_dec:
            if target is None:
                raise ValueError("DEC/IDEC require a detached target refreshed from the full training dataset")
            regularizer = dec_target_kl(target, dec_log_assignments(mean, self.cluster_centers, self.dec_alpha)).mean()
            reconstruction = (self.decoder(mean) - x).square().mean() if self.method == "idec" else x.new_zeros(())
            loss = regularizer if self.method == "dec" else reconstruction + self.idec_gamma * regularizer
        elif self.is_structured:
            q = self._structured(mean, log_variance)
            def likelihood(z: Tensor) -> Tensor:
                return self._log_likelihood(x[:, None, None], self.decoder(z))
            reconstruction = -q.expected_log_likelihood_sampled_nodes(likelihood, n_samples=self.n_samples).mean()
            regularizer = q.kl_to_prior.mean()
            mass_error = (q.prior_quadrature_mass - 1).abs().max()
            loss = reconstruction + self.beta * regularizer
        else:
            z, log_q = self._samples(mean, log_variance, self.n_samples)
            reconstruction = -self._log_likelihood(x[:, None], self.decoder(z)).mean()
            if self.method == "vae_gmm":
                # A previously fitted cluster head becomes stale when VAE training resumes.
                self.gmm_fitted.fill_(False)
                regularizer = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(-1).mean()
            elif self.method == "vade":
                regularizer = gaussian_categorical_regularizer(mean, log_variance, z[:, 0], self.prior)["regularizer"].mean()
            elif self.method == "pro_t_vae":
                regularizer = pro_t_augmented_regularizer(mean, log_variance, z[:, 0], self.prior, softassign_detach=True)["regularizer"].mean()
            else:
                log_prior = torch.logsumexp(component_log_density(z, self.prior, gaussian=self.is_gaussian_prior) + self.prior.log_weights, -1)
                regularizer = (log_q - log_prior).mean()
            loss = reconstruction + self.beta * regularizer
        return {"loss": loss, "objective": -loss, "reconstruction": reconstruction,
                "regularizer": regularizer, "prior_mass_error": mass_error}

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        mean, log_variance = self._encoded(x)
        if self.is_dec:
            return dec_log_assignments(mean, self.cluster_centers, self.dec_alpha).exp()
        if self.method == "vae_gmm":
            if not bool(self.gmm_fitted):
                raise RuntimeError("Fit and set the evaluation GMM on final training encoder means before VAE+GMM prediction")
            delta = ((mean[:, None] - self.gmm_means).square() / self.gmm_log_covariance.exp()).sum(-1)
            log_density = -0.5 * (self.latent_dim * math.log(2 * math.pi) + self.gmm_log_covariance.sum(-1) + delta)
            return (log_density + self.gmm_log_weights).softmax(-1)
        if self.is_structured:
            return self._structured(mean, log_variance, prediction=True).responsibilities
        z, _ = self._samples(mean, log_variance, self.prediction_samples)
        return mixture_log_responsibilities(z, self.prior, gaussian=self.is_gaussian_prior).exp().mean(1)

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        mean, log_variance = self._encoded(x)
        if self.is_structured:
            mean = self._structured(mean, log_variance, prediction=True).mean
        return self.decoder(mean)

    @torch.no_grad()
    def reconstruction_diagnostics(self, x: Tensor) -> dict[str, Tensor]:
        reconstruction = self.reconstruct(x)
        return {"mse_at_posterior_mean": (reconstruction - x).square().mean(),
                "gaussian_nll_at_posterior_mean": -self._log_likelihood(x, reconstruction).mean()}


def create_model(method: str, input_dim: int, n_clusters: int, **kwargs: Any) -> MatchedClusteringModel:
    """Factory; canonical method names are listed in METHODS."""
    return MatchedClusteringModel(method, input_dim, n_clusters, **kwargs)
