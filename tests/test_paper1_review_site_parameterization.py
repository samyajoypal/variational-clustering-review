"""Candidate no-floor sites: transforms, continuous reference and ignored axes."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paper1_review.site_parameterization import (
    DEFAULT_MAX_PRECISION, GaussianSitePrecisionHead, bounded_softplus_precision,
    gaussian_site_from_raw, prior_site_strength, raw_from_precision,
)
from experiments.paper1_structured_inference.structured import (
    CenteredGaussianSite, DiagonalStudentMixture, structured_posterior,
)


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_prior(extra_dimensions=0):
    loc = torch.tensor([[-1.], [2.]], dtype=torch.float64)
    scale = torch.tensor([[0.7], [1.3]], dtype=torch.float64)
    if extra_dimensions:
        loc = torch.cat([loc, torch.tensor([[0.4], [-0.8]], dtype=torch.float64).expand(-1, extra_dimensions)], 1)
        scale = torch.cat([scale, torch.tensor([[0.6], [1.7]], dtype=torch.float64).expand(-1, extra_dimensions)], 1)
    return DiagonalStudentMixture(loc, scale, torch.tensor([5., 12.], dtype=torch.float64),
                                 torch.tensor([0.35, 0.65], dtype=torch.float64).log())


def integrate(prior, site):
    return structured_posterior(prior, site, nodes=384, log_tau_bounds=(-18., 6.))


def test_inverse_and_gradients_without_positive_floor():
    precision = torch.tensor([1e-12, 1e-6, .1, 10., 100., .9 * DEFAULT_MAX_PRECISION], dtype=torch.float64)
    torch.testing.assert_close(bounded_softplus_precision(raw_from_precision(precision)), precision, rtol=2e-14, atol=0)
    raw = torch.tensor([-30., -3., 2., 10.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(bounded_softplus_precision, (raw,))
    values = bounded_softplus_precision(raw)
    values.sum().backward()
    assert 0 < values[0] < 1e-8
    assert torch.isfinite(raw.grad).all() and (raw.grad > 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_raw_finite_without_exp_overflow(dtype):
    raw = torch.tensor([-1000., -100., 0., 100., 1e20], dtype=dtype, requires_grad=True)
    values = bounded_softplus_precision(raw)
    values.sum().backward()
    assert torch.isfinite(values).all() and torch.isfinite(raw.grad).all()
    assert (values >= 0).all() and (values <= DEFAULT_MAX_PRECISION).all()
    # Underflow can produce numerical zero, but is not the exact-zero mechanism.
    assert values[1] < 1e-35


def test_head_initializes_existing_precision_and_mask_is_explicit():
    head = GaussianSitePrecisionHead(3, initial_precision=[10., .8, 0.], active_mask=[True, True, False])
    center = torch.tensor([[.2, -.3, 50.], [.7, 1., -80.]], requires_grad=True)
    site = head(center)
    torch.testing.assert_close(site.precision, torch.tensor([[10., .8, 0.]]).expand(2, -1))
    torch.testing.assert_close(site.center[:, 2], torch.zeros(2))
    site.precision.sum().backward()
    assert torch.isfinite(head.raw_layer.bias.grad).all()
    assert head.raw_layer.bias.grad[2] == 0
    assert center.grad[:, 2].eq(0).all()
    metadata = json.loads(json.dumps(head.configuration()))
    assert metadata["positive_lower_bound"] is None and metadata["mask_is_learned"] is False


def test_masked_centers_do_not_leak_into_active_precision_features():
    head = GaussianSitePrecisionHead(2, active_mask=[True, False]).double()
    with torch.no_grad():
        head.raw_layer.weight.fill_(.2)
    first = head(torch.tensor([[.3, -100.]], dtype=torch.float64))
    second = head(torch.tensor([[.3, 100.]], dtype=torch.float64))
    torch.testing.assert_close(first.center, second.center, atol=0, rtol=0)
    torch.testing.assert_close(first.precision, second.precision, atol=0, rtol=0)


def test_exact_masked_no_site_recovers_continuous_prior_moments():
    prior = make_prior(2)
    site = gaussian_site_from_raw(torch.ones(2, 3, dtype=torch.float64),
                                  torch.zeros(2, 3, dtype=torch.float64), active_mask=[False] * 3)
    q = integrate(prior, site)
    weights = prior.log_weights.exp()
    mean = (weights[:, None] * prior.loc).sum(0)
    covariance = prior.scale_diag * prior.df[:, None] / (prior.df[:, None] - 2)
    variance = (weights[:, None] * (covariance + prior.loc.square())).sum(0) - mean.square()
    torch.testing.assert_close(q.log_normalizer, torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(q.responsibilities, weights.expand(2, -1), rtol=0, atol=1e-12)
    torch.testing.assert_close(q.mean, mean.expand(2, -1), rtol=0, atol=1e-12)
    torch.testing.assert_close(q.variance, variance.expand(2, -1), rtol=0, atol=2e-10)
    torch.testing.assert_close(q.kl_to_prior, torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-12)


@pytest.mark.parametrize("extra_dimensions", [7, 31])
def test_exact_zero_nuisance_evidence_assignments_and_df_gradient_invariance(extra_dimensions):
    base = make_prior()
    appended = make_prior(extra_dimensions)
    center = torch.tensor([[-.3], [1.2]], dtype=torch.float64)
    precision = torch.full_like(center, .8)
    raw = raw_from_precision(precision)
    site = gaussian_site_from_raw(center, raw)
    nuisance_center = torch.full((2, extra_dimensions), 20., dtype=torch.float64)
    nuisance_raw = torch.full_like(nuisance_center, 3.)
    enlarged = gaussian_site_from_raw(torch.cat([center, nuisance_center], 1),
                                     torch.cat([raw, nuisance_raw], 1),
                                     active_mask=[True] + [False] * extra_dimensions)
    q0, q1 = integrate(base, site), integrate(appended, enlarged)
    for name in ["log_normalizer", "responsibilities", "kl_to_prior"]:
        torch.testing.assert_close(getattr(q0, name), getattr(q1, name), rtol=0, atol=2e-12)
    torch.testing.assert_close(q0.mean, q1.mean[:, :1], rtol=0, atol=2e-12)
    torch.testing.assert_close(q0.variance, q1.variance[:, :1], rtol=0, atol=2e-12)
    grad0 = torch.autograd.grad(q0.log_normalizer.sum(), base.log_df)[0]
    grad1 = torch.autograd.grad(q1.log_normalizer.sum(), appended.log_df)[0]
    torch.testing.assert_close(grad0, grad1, rtol=0, atol=2e-12)


def test_signal_site_matches_independent_direct_student_density_integration():
    prior = make_prior()
    center = torch.tensor([[-.3], [1.2]], dtype=torch.float64)
    site = gaussian_site_from_raw(center, raw_from_precision(torch.full_like(center, .8)))
    q = integrate(prior, site)
    for row, x in enumerate(center[:, 0].tolist()):
        evidence, first, second = [], [], []
        for mu, scale, nu in [(-1., .7, 5.), (2., 1.3, 12.)]:
            def density(z):
                return t.pdf(z, df=nu, loc=mu, scale=math.sqrt(scale)) * math.exp(-.4 * (z - x) ** 2)
            moments = [quad(lambda z: z ** power * density(z), -np.inf, np.inf,
                            epsabs=2e-12, epsrel=2e-12)[0] for power in range(3)]
            evidence.append(moments[0]); first.append(moments[1]); second.append(moments[2])
        weights = np.array([.35, .65])
        normalizer = weights @ evidence
        mean = weights @ first / normalizer
        variance = weights @ second / normalizer - mean ** 2
        np.testing.assert_allclose(q.log_normalizer[row].detach(), math.log(normalizer), atol=2e-11)
        np.testing.assert_allclose(q.responsibilities[row].detach(), weights * evidence / normalizer, atol=2e-11)
        np.testing.assert_allclose(q.mean[row, 0].detach(), mean, atol=2e-11)
        np.testing.assert_allclose(q.variance[row, 0].detach(), variance, atol=2e-11)


def test_learned_near_zero_sites_approach_prior_and_strength_bounds_kl():
    prior = make_prior()
    covariance = prior.scale_diag * prior.df[:, None] / (prior.df[:, None] - 2)
    prior_mean = (prior.log_weights.exp()[:, None] * prior.loc).sum(0)
    kls, mean_errors = [], []
    for precision in [1e-2, 1e-4, 1e-6]:
        center = torch.tensor([[.4]], dtype=torch.float64)
        raw = raw_from_precision(torch.full_like(center, precision)).requires_grad_()
        site = gaussian_site_from_raw(center, raw)
        q = integrate(prior, site)
        strength = prior_site_strength(site, prior.loc, covariance, prior.logits)
        assert -1e-12 <= q.kl_to_prior.item() <= strength.item() + 1e-12
        q.kl_to_prior.sum().backward()
        assert torch.isfinite(raw.grad).all()
        kls.append(q.kl_to_prior.item())
        mean_errors.append((q.mean - prior_mean).abs().item())
    assert kls[0] > kls[1] > kls[2] and kls[-1] < 1e-10
    assert mean_errors[0] > mean_errors[1] > mean_errors[2] and mean_errors[-1] < 1e-5


def test_small_precision_alone_is_not_weak_information_when_center_diverges():
    prior = make_prior()
    covariance = prior.scale_diag * prior.df[:, None] / (prior.df[:, None] - 2)
    small = CenteredGaussianSite(torch.zeros(1, 1, dtype=torch.float64), torch.full((1, 1), 1e-6, dtype=torch.float64))
    huge_center = CenteredGaussianSite(torch.full((1, 1), 1e6, dtype=torch.float64), small.precision)
    assert prior_site_strength(small, prior.loc, covariance, prior.logits).item() < 1e-5
    assert prior_site_strength(huge_center, prior.loc, covariance, prior.logits).item() > 1e5


@pytest.mark.parametrize("precision", [0., -1., DEFAULT_MAX_PRECISION, float("nan")])
def test_inverse_rejects_values_without_finite_representations(precision):
    with pytest.raises(ValueError):
        raw_from_precision(torch.tensor([precision]))


@pytest.mark.parametrize("mask", [[1, 0], [True], [[True, False]]])
def test_mask_requires_explicit_boolean_vector(mask):
    with pytest.raises(ValueError, match="Boolean vector"):
        GaussianSitePrecisionHead(2, active_mask=mask)
