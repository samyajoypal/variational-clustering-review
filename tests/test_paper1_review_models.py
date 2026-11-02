"""Equation and integration checks for matched objective adapters, no datasets."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_t
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paper1_review.models import (
    METHODS, MLPAutoencoder, component_log_density, create_model,
    dec_log_assignments, dec_target_distribution, dec_target_kl,
    diagonal_gaussian_component_kl, gaussian_categorical_regularizer,
    mixture_log_responsibilities, multivariate_student_log_prob,
    pro_t_augmented_regularizer,
)
from experiments.paper1_structured_inference.structured import DiagonalStudentMixture


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prior(d=2, *, dtype=torch.float64):
    return DiagonalStudentMixture(torch.tensor([[-0.6] * d, [0.8] * d], dtype=dtype),
                                  torch.tensor([[0.7] * d, [1.3] * d], dtype=dtype),
                                  torch.tensor([3.5, 18.], dtype=dtype),
                                  torch.tensor([0.3, 0.7], dtype=dtype).log())


def small_model(method, **kwargs):
    model = create_model(method, 4, 2, latent_dim=2, hidden_dims=(8,), n_samples=4,
                         prediction_samples=8, nodes=64, prediction_nodes=64, **kwargs)
    model.set_cluster_parameters([[-0.5, 0.2], [0.7, -0.3]], [[0.7, 1.2], [1.1, 0.8]], [0.35, 0.65])
    return model


@pytest.mark.parametrize("d", [1, 4, 10])
def test_vade_mixture_weight_counted_once_not_theta_to_dimension(d):
    p = DiagonalStudentMixture(torch.zeros(2, d, dtype=torch.float64), torch.ones(2, d, dtype=torch.float64),
                              torch.full((2,), 8., dtype=torch.float64), torch.tensor([0.2, 0.8], dtype=torch.float64).log())
    z = torch.ones(3, d, dtype=torch.float64)
    responsibilities = mixture_log_responsibilities(z, p, gaussian=True).exp()
    torch.testing.assert_close(responsibilities, torch.tensor([[0.2, 0.8]] * 3, dtype=torch.float64))
    mean = torch.zeros_like(z)
    reg = gaussian_categorical_regularizer(mean, torch.zeros_like(mean), z, p)
    torch.testing.assert_close(reg["regularizer"], torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)


def test_gaussian_categorical_objective_matches_independent_distribution_kls():
    p = prior()
    mean = torch.tensor([[0.2, -0.1], [1., 0.7]], dtype=torch.float64)
    lv = torch.tensor([[-1., -0.4], [0.2, -0.6]], dtype=torch.float64)
    z = mean + 0.3
    result = gaussian_categorical_regularizer(mean, lv, z, p)
    q = torch.distributions.Normal(mean[:, None], (lv / 2).exp()[:, None])
    components = torch.distributions.Normal(p.loc, p.scale_diag.sqrt())
    expected_component_kl = torch.distributions.kl_divergence(q, components).sum(-1)
    torch.testing.assert_close(diagonal_gaussian_component_kl(mean, lv, p), expected_component_kl)
    categorical = torch.distributions.kl_divergence(torch.distributions.Categorical(probs=result["responsibilities"]), torch.distributions.Categorical(logits=p.logits))
    expected = (result["responsibilities"] * expected_component_kl).sum(-1) + categorical
    torch.testing.assert_close(result["regularizer"], expected)


def test_vade_categorical_bound_is_distinct_from_collapsed_gaussian_mixture_kl():
    p = DiagonalStudentMixture(torch.tensor([[-2.], [2.]], dtype=torch.float64), torch.ones(2, 1, dtype=torch.float64), torch.full((2,), 8., dtype=torch.float64))
    mean = torch.zeros(1, 1, dtype=torch.float64)
    augmented = gaussian_categorical_regularizer(mean, mean, mean, p)["regularizer"].item()
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    z = torch.tensor(np.sqrt(2) * nodes, dtype=torch.float64)[:, None]
    log_q = torch.distributions.Normal(0., 1.).log_prob(z).sum(-1)
    log_p = torch.logsumexp(component_log_density(z, p, gaussian=True) + p.log_weights, -1)
    collapsed = float((torch.tensor(weights / np.sqrt(np.pi)) * (log_q - log_p)).sum().detach())
    assert augmented == pytest.approx(2.)
    assert 0 < collapsed < augmented - 0.1


def test_pro_t_full_analytic_bound_matches_gamma_kl_and_log_t_identity():
    p = prior()
    mean = torch.tensor([[0.2, -0.1], [0.8, 0.6]], dtype=torch.float64)
    lv = torch.tensor([[-1., -0.5], [-0.7, 0.3]], dtype=torch.float64)
    result = pro_t_augmented_regularizer(mean, lv, mean + 0.4, p)
    a, b = result["gamma_shape"], result["gamma_rate"]
    kl_gamma = torch.distributions.kl_divergence(torch.distributions.Gamma(a, b), torch.distributions.Gamma(p.df[None] / 2, p.df[None] / 2))
    torch.testing.assert_close(result["gamma"], (result["responsibilities"] * kl_gamma).sum(-1))
    d = mean.shape[1]
    delta_bar = ((lv.exp()[:, None] + (mean[:, None] - p.loc).square()) / p.scale_diag).sum(-1)
    nu = p.df
    log_t_at_expected_delta = (torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2) - d / 2 * torch.log(nu * math.pi)
                               - p.log_scale_diag.sum(-1) / 2 - (nu + d) / 2 * torch.log1p(delta_bar / nu))
    expected_log_q = -0.5 * (d * (math.log(2 * math.pi) + 1) + lv.sum(-1))
    expected_bound = (result["responsibilities"] * (expected_log_q[:, None] - log_t_at_expected_delta)).sum(-1) + result["categorical"]
    torch.testing.assert_close(result["regularizer"], expected_bound, atol=2e-14, rtol=1e-13)
    assert not result["responsibilities"].requires_grad


def test_pro_t_detaches_sampled_assignments_but_preserves_other_gradients_and_zero_tails():
    p = prior()
    with torch.no_grad():
        p.logits.copy_(torch.tensor([0., -1000.], dtype=torch.float64))
    mean = torch.tensor([[0.1, 0.2]], dtype=torch.float64, requires_grad=True)
    lv = torch.tensor([[-1., -0.7]], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([[2., -1.]], dtype=torch.float64, requires_grad=True)
    result = pro_t_augmented_regularizer(mean, lv, z, p)
    assert result["responsibilities"][0, 1] == 0
    result["regularizer"].sum().backward()
    assert z.grad is None
    for tensor in [mean, lv, p.loc, p.log_scale_diag, p.log_df, p.logits]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_gaussian_density_does_not_evaluate_unused_student_df_branch():
    p = prior()
    with torch.no_grad():
        p.log_df.fill_(float("nan"))
    z = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    value = component_log_density(z, p, gaussian=True).sum()
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(z.grad).all()
    assert p.log_df.grad is None


def test_direct_student_density_matches_multivariate_scipy_not_product_marginals():
    mean = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    scale = torch.tensor([[0.7, 1.4]], dtype=torch.float64)
    df = torch.tensor([5.], dtype=torch.float64)
    z = torch.tensor([[0.1, 0.8], [1.2, -0.4]], dtype=torch.float64)
    actual = multivariate_student_log_prob(z, mean, scale, df)
    expected = multivariate_t.logpdf(z.numpy(), loc=mean[0].numpy(), shape=np.diag(scale[0].numpy()), df=5)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-13)


def test_dec_target_normalization_global_frequencies_and_detachment_with_empty_column():
    q = torch.tensor([[0.8, 0.2, 0.], [0.4, 0.6, 0.], [0.1, 0.9, 0.]], dtype=torch.float64, requires_grad=True)
    target = dec_target_distribution(q)
    independent = q.detach().numpy()[:, :2] ** 2 / q.detach().numpy()[:, :2].sum(0)
    independent /= independent.sum(1, keepdims=True)
    np.testing.assert_allclose(target[:, :2].numpy(), independent)
    assert not target.requires_grad
    assert torch.equal(target[:, 2], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(target.sum(1), torch.ones(3, dtype=torch.float64))
    # Full-data frequency correction differs from an independently normalized minibatch.
    assert not torch.allclose(target[:1, :2], dec_target_distribution(q[:1])[:, :2])


def test_dec_training_never_differentiates_the_external_target():
    z = torch.tensor([[0.1], [0.8]], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[0.], [1.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64, requires_grad=True)
    loss = dec_target_kl(target, dec_log_assignments(z, centers)).mean()
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(z.grad).all() and torch.isfinite(centers.grad).all()


def test_idec_uses_feature_mean_mse_and_gamma_point_one():
    losses = []
    for dimension in [2, 4]:
        model = create_model("idec", dimension, 2, latent_dim=1, hidden_dims=(), decoder_activation="identity")
        with torch.no_grad():
            for parameter in model.encoder.parameters():
                parameter.zero_()
            for parameter in model.decoder.parameters():
                parameter.zero_()
        model.set_cluster_parameters([[0.], [1.]], [[1.], [1.]], [0.5, 0.5])
        x = torch.tensor([[1., 2.] * (dimension // 2)])
        target = torch.tensor([[0.8, 0.2]])
        output = model.training_loss(x, target)
        assert output["reconstruction"].item() == pytest.approx(2.5)
        assert output["loss"].item() == pytest.approx(2.5 + 0.1 * output["regularizer"].item())
        losses.append(output["loss"].item())
    assert losses[0] == pytest.approx(losses[1])


@pytest.mark.parametrize("method", METHODS)
def test_all_ten_adapters_backward_predict_and_shared_ae_state_keys(method):
    torch.manual_seed(7101)
    model = small_model(method)
    ae = MLPAutoencoder(4, 2, (8,))
    model.encoder.load_state_dict(ae.encoder.state_dict())
    model.decoder.load_state_dict(ae.decoder.state_dict())
    x = torch.rand(5, 4)
    target = dec_target_distribution(model.predict(x)) if model.is_dec else None
    output = model.training_loss(x, target)
    assert set(output) == {"loss", "objective", "reconstruction", "regularizer", "prior_mass_error"}
    assert all(torch.isfinite(value) and value.ndim == 0 for value in output.values())
    output["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if model.prior is not None:
        for p in [model.prior.loc, model.prior.log_scale_diag, model.prior.logits]:
            assert p.grad is not None and torch.isfinite(p.grad).all()
        if not model.is_gaussian_prior:
            assert model.prior.log_df.grad is not None and torch.isfinite(model.prior.log_df.grad).all()
    model.constrain_parameters()
    if method == "vae_gmm":
        with pytest.raises(RuntimeError, match="Fit and set"):
            model.predict(x)
        model.set_cluster_parameters([[-0.5, 0.2], [0.7, -0.3]], [[0.7, 1.2], [1.1, 0.8]], [0.35, 0.65])
    responsibilities = model.predict(x)
    assert responsibilities.shape == (5, 2) and torch.isfinite(responsibilities).all()
    torch.testing.assert_close(responsibilities.sum(-1), torch.ones(5), atol=2e-5, rtol=0)
    diagnostics = model.reconstruction_diagnostics(x)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert model.objective_metadata["native_reproduction"] is False


def test_default_architecture_and_covariance_matched_student_initialization():
    ae = MLPAutoencoder(4)
    assert [layer.out_features for layer in ae.encoder if isinstance(layer, nn.Linear)] == [500, 500, 2000, 10]
    assert [layer.out_features for layer in ae.decoder if isinstance(layer, nn.Linear)] == [2000, 500, 500, 4]
    model = small_model("gaussian_q_student_prior")
    covariance = model.prior.scale_diag * model.prior.df[:, None] / (model.prior.df[:, None] - 2)
    torch.testing.assert_close(covariance, torch.tensor([[0.7, 1.2], [1.1, 0.8]]))
    with torch.no_grad():
        model.prior.log_df.copy_(torch.tensor([0., math.log(100.)]))
        model.prior.log_scale_diag.fill_(-100.)
    model.constrain_parameters()
    torch.testing.assert_close(model.prior.df, torch.tensor([3., 30.]))
    torch.testing.assert_close(model.prior.scale_diag, torch.full((2, 2), 1e-4))


def test_vae_gmm_training_uses_exact_standard_normal_kl_and_frozen_postfit_head():
    model = small_model("vae_gmm")
    x = torch.rand(3, 4)
    mean, lv = model._encoded(x)
    expected = 0.5 * (mean.square() + lv.exp() - 1 - lv).sum(-1).mean()
    torch.testing.assert_close(model.training_loss(x)["regularizer"], expected)
    assert model.prior is None
    assert all(not tensor.requires_grad for name, tensor in model.named_buffers() if name.startswith("gmm_"))


@pytest.mark.parametrize("kwargs", [{"n_samples": 1}, {"n_samples": 2.5}, {"observation_variance": 0}, {"df_bounds": (1., 30.)}, {"standardized_bounds": (1., -1.)}])
def test_invalid_safety_configuration_fails_before_training(kwargs):
    with pytest.raises(ValueError):
        create_model("structured_q_student_prior", 4, 2, hidden_dims=(8,), **kwargs)


def test_dec_missing_target_and_bad_probability_target_fail_explicitly():
    model = small_model("dec")
    with pytest.raises(ValueError, match="full training dataset"):
        model.training_loss(torch.rand(2, 4))
    with pytest.raises(ValueError, match="unit-sum"):
        dec_target_distribution(torch.tensor([[0., 0.]]))


@pytest.mark.parametrize("method", ["structured_q_gaussian_prior", "structured_q_student_prior"])
def test_optional_no_floor_site_uses_public_constructor_and_trains_separate_head(method):
    torch.manual_seed(19)
    model = small_model(method, structured_site_mode="bounded_softplus")
    x = torch.rand(4, 4)
    mean, lv = model._encoded(x)
    site = model.make_site(mean, lv)
    torch.testing.assert_close(site.precision, torch.full_like(mean, 10.))
    assert model.site_head is not None
    assert not any(p.requires_grad for p in model.variance_head.parameters())
    # Variance-head outputs cannot change sites in this mode.
    torch.testing.assert_close(model.make_site(mean, lv + 20).precision, site.precision)
    calls = []
    hook = model.site_head.register_forward_hook(lambda *args: calls.append(1))
    loss = model.training_loss(x)["loss"]
    loss.backward()
    model.predict(x)
    model.reconstruction_diagnostics(x)
    hook.remove()
    assert len(calls) == 3
    assert all(p.grad is None for p in model.variance_head.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.site_head.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
    assert model.objective_metadata["structured_site_mode"] == "bounded_softplus"
    assert model.objective_metadata["structured_site"]["positive_lower_bound"] is None
    assert model.objective_metadata["structured_site"]["initial_precision"] == [10., 10.]


@pytest.mark.parametrize("method", ["structured_q_gaussian_prior", "structured_q_student_prior"])
def test_optional_site_can_approach_prior_below_legacy_floor_with_finite_gradients(method):
    model = small_model(method, structured_site_mode="bounded_softplus").double()
    # A near-zero continuous KL is resolved using refined quadrature, not a
    # tolerance chosen to hide 64-node normalization error (~2.3e-7 here).
    model.nodes = 256
    with torch.no_grad():
        model.site_head.raw_layer.bias.fill_(-30.)
    mean = torch.tensor([[.2, -.1]], dtype=torch.float64, requires_grad=True)
    lv = torch.full_like(mean, 2.)
    site = model.make_site(mean, lv)
    assert (site.precision < 1e-8).all()
    q = model._structured(mean, lv)
    torch.testing.assert_close(q.responsibilities, model.prior.log_weights.exp()[None], atol=2e-7, rtol=0)
    assert abs(q.kl_to_prior.item()) < 1e-10
    q.log_normalizer.sum().backward()
    assert torch.isfinite(mean.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.site_head.parameters())


def test_legacy_site_default_retains_original_transform_and_state_layout():
    model = small_model("structured_q_student_prior")
    assert model.site_head is None
    assert all(p.requires_grad for p in model.variance_head.parameters())
    mean, lv = model._encoded(torch.rand(2, 4))
    site = model.make_site(mean, lv)
    torch.testing.assert_close(site.precision, (-lv).exp(), rtol=0, atol=0)
    assert not any(key.startswith("site_head.") for key in model.state_dict())
    assert model.objective_metadata["structured_site_mode"] == "legacy"
    assert model.objective_metadata["structured_site"]["positive_lower_bound"] == math.exp(-2.)


@pytest.mark.parametrize("method", [m for m in METHODS if not m.startswith("structured_q_")])
def test_optional_site_mode_does_not_change_other_methods(method):
    torch.manual_seed(912)
    legacy = small_model(method)
    torch.manual_seed(912)
    selected = small_model(method, structured_site_mode="bounded_softplus")
    assert selected.site_head is None
    assert selected.objective_metadata["structured_site_mode"] is None
    assert list(legacy.state_dict()) == list(selected.state_dict())
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, selected.state_dict()[key], rtol=0, atol=0)
    assert [p.requires_grad for p in legacy.parameters()] == [p.requires_grad for p in selected.parameters()]
    with pytest.raises(ValueError, match="only to structured"):
        selected.make_site(torch.zeros(1, 2), torch.zeros(1, 2))


def test_invalid_site_mode_rejected():
    with pytest.raises(ValueError, match="structured_site_mode"):
        small_model("structured_q_student_prior", structured_site_mode="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; no CPU fallback for this device test")
@pytest.mark.parametrize("method", ["structured_q_gaussian_prior", "structured_q_student_prior"])
def test_cuda_optional_no_floor_site_backward_and_public_diagnostic_path(method):
    model = small_model(method, structured_site_mode="bounded_softplus").cuda()
    x = torch.rand(3, 4, device="cuda")
    mean, lv = model._encoded(x)
    site = model.make_site(mean, lv)
    assert site.center.is_cuda and site.precision.is_cuda
    torch.testing.assert_close(site.precision, torch.full_like(mean, 10.))
    output = model.training_loss(x)
    output["loss"].backward()
    for parameter in model.site_head.parameters():
        assert parameter.grad is not None and parameter.grad.is_cuda and torch.isfinite(parameter.grad).all()
    assert all(parameter.grad is None for parameter in model.variance_head.parameters())
    responsibilities = model.predict(x)
    assert responsibilities.is_cuda and torch.isfinite(responsibilities).all()
    torch.testing.assert_close(responsibilities.sum(-1), torch.ones(3, device="cuda"), rtol=0, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; no CPU fallback for this device test")
@pytest.mark.parametrize("method", ["student_q_student_prior", "structured_q_student_prior", "pro_t_vae"])
def test_cuda_draws_and_structured_integration_stay_on_selected_device(method):
    model = small_model(method).cuda()
    x = torch.rand(3, 4, device="cuda")
    output = model.training_loss(x)
    assert output["loss"].device.type == "cuda"
    output["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.predict(x).device.type == "cuda"
