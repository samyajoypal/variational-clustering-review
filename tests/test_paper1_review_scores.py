from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal, multivariate_t

from experiments.paper1_review.provenance import start_run, finish_run, sha256
from experiments.paper1_review.score_assignment_diagnostic import component_values, mixture_values


@pytest.mark.parametrize("d", [2, 10, 50])
@pytest.mark.parametrize("matching", ["covariance", "scale"])
def test_log_density_and_derivatives_against_scipy(d, matching):
    x = np.zeros((4, d)); x[:, 0] = [0., .5, 2., 100.]
    mu = np.zeros(d)
    nu, h = 5., 1e-4
    scale = lambda n: (n - 2) / n if matching == "covariance" else 1.
    reference = lambda z, n: multivariate_t.logpdf(z, loc=mu, shape=np.eye(d)*scale(n), df=n)
    lp, score, ds = component_values(x, mu, nu, matching)
    np.testing.assert_allclose(lp, reference(x, nu), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ds, (reference(x, nu+h)-reference(x, nu-h))/(2*h), rtol=1e-7, atol=1e-7)
    xp, xm = x.copy(), x.copy(); xp[:, 0] += h; xm[:, 0] -= h
    np.testing.assert_allclose(score[:, 0], (reference(xp, nu)-reference(xm, nu))/(2*h), rtol=1e-7, atol=1e-7)


def test_gaussian_limit_and_score():
    x = np.array([[0., 0.], [1., 3.]])
    lp, score, ds = component_values(x, np.zeros(2), np.inf, "covariance")
    np.testing.assert_allclose(lp, multivariate_normal.logpdf(x, mean=np.zeros(2), cov=np.eye(2)))
    np.testing.assert_allclose(score, -x)
    assert ds is None


def test_mixture_score_matches_density_derivative_and_weights():
    x = np.array([[0., 1.], [2., -.5], [30., 3.]])
    means = np.array([[-1., 0.], [1., 0.]])
    args = (means, [3., 30.], [.2, .8], "covariance")
    lp, score, r, entropy, _ = mixture_values(x, *args)
    np.testing.assert_allclose(r.sum(1), 1., atol=1e-14)
    assert np.all(entropy >= 0)
    for j in range(2):
        delta = np.zeros_like(x); delta[:, j] = 1e-4
        numerical = (mixture_values(x+delta, *args)[0]-mixture_values(x-delta, *args)[0])/(2e-4)
        np.testing.assert_allclose(score[:, j], numerical, atol=1e-8, rtol=1e-7)


def test_small_location_score_does_not_imply_uncertain_assignment():
    x = np.array([[1e6, 0.]])
    _, score, r, entropy, _ = mixture_values(x, np.array([[-1., 0.], [1., 0.]]), [3., 30.], [.5, .5], "covariance")
    assert np.linalg.norm(score) < 1e-4
    assert r[0, 0] > 1-1e-10
    assert entropy[0] < 1e-8


@pytest.mark.parametrize("weights", [[.5, .5], [.2, .8]])
def test_gaussian_tail_weights_do_not_lose_normalization(weights):
    # At this orthogonal point both log densities are near -5e11.
    # Directly subtracting logsumexp from those values loses the normalizer.
    _, _, r, entropy, _ = mixture_values(np.array([[0., 1e6]]), np.array([[-1., 0.], [1., 0.]]), [np.inf, np.inf], weights, "covariance")
    np.testing.assert_allclose(r, [weights], atol=1e-14)
    np.testing.assert_allclose(entropy, [-np.sum(np.array(weights)*np.log(weights))], atol=1e-14)


def test_provenance_preserves_snapshot_and_refuses_overwrite(tmp_path):
    source = Path(__file__).resolve()
    run = tmp_path / "fresh"
    metadata = start_run(run, stage="deterministic_simulation", config={"test": True}, sources=[source])
    assert metadata["fresh_run"] and not metadata["legacy_outputs_used"]
    assert metadata["source_file_sha256"]["tests/test_paper1_review_scores.py"] == sha256(source)
    assert len(metadata["source_sha256"]) == 64
    finish_run(run, metadata)
    with pytest.raises(FileExistsError):
        start_run(run, stage="deterministic_simulation", config={}, sources=[source])
