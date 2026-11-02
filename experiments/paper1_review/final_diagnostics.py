"""Label-free, no-update objective/gradient diagnostics for the matched panel."""
from __future__ import annotations
import math
import torch
from .models import (component_log_density, gaussian_categorical_regularizer,
                     pro_t_augmented_regularizer, dec_log_assignments, dec_target_kl)
from .site_parameterization import prior_site_strength


def objective_terms(model, x, target=None):
    """Reconstruction, prior cross entropy and negative entropy on one graph.

    Structured/VaDE/Pro-t splits use their respective augmented joint variables;
    they must not be called marginal-z entropy. DEC has no generative split.
    The MC draw here is diagnostic only, not a replacement training update.
    """
    mean, lv = model._encoded(x)
    d = mean.shape[-1]
    if model.is_dec:
        if target is None:
            raise ValueError("DEC diagnostic needs its full-pool target rows")
        cluster = dec_target_kl(target, dec_log_assignments(mean, model.cluster_centers, model.dec_alpha)).mean()
        terms = {"target_kl": cluster * (model.idec_gamma if model.method == "idec" else 1.)}
        if model.method == "idec":
            terms["reconstruction"] = (model.decoder(mean) - x).square().mean()
        return terms, "non-generative target objective; prior/entropy split not applicable"
    if model.is_structured:
        q = model._structured(mean, lv)
        def likelihood(z):
            return model._log_likelihood(x[:, None, None], model.decoder(z))
        reconstruction = -q.expected_log_likelihood_sampled_nodes(likelihood, n_samples=model.n_samples).mean()
        prior = model.prior
        conditional_delta = ((q.conditional_mean - prior.loc[None, :, None]).square() + q.conditional_variance) / prior.scale_diag[None, :, None]
        tau = q.tau[None].expand(len(prior.df), -1) if q.tau.ndim == 1 else q.tau
        log_normal = -.5 * (d * math.log(2 * math.pi) + prior.log_scale_diag.sum(-1)[:, None]
                             - d * tau.log() + tau[None] * conditional_delta.sum(-1))
        if model.is_gaussian_prior:
            log_gamma = torch.zeros_like(tau)
        else:
            a = prior.df[:, None] / 2
            log_gamma = a * a.log() - torch.lgamma(a) + (a - 1) * tau.log() - a * tau
        prior_cross_entropy = -(q.integration_weights * (prior.log_weights[None, :, None] + log_gamma[None] + log_normal)).sum((1, 2)).mean()
        negative_entropy = q.kl_to_prior.mean() - prior_cross_entropy
        scope = "continuous augmented (c,tau,z) entropy integrated approximately" if not model.is_gaussian_prior else "augmented (c,z) entropy"
    else:
        z, log_q = model._samples(mean, lv, model.n_samples)
        reconstruction = -model._log_likelihood(x[:, None], model.decoder(z)).mean()
        if model.method == "vae_gmm":
            prior_cross_entropy = .5 * (mean.square() + lv.exp() + math.log(2 * math.pi)).sum(-1).mean()
            negative_entropy = -.5 * (1 + math.log(2 * math.pi) + lv).sum(-1).mean()
            scope = "Gaussian marginal-z entropy"
        elif model.method in {"vade", "pro_t_vae"}:
            prior = model.prior
            gaussian_entropy = -.5 * (d * (1 + math.log(2 * math.pi)) + lv.sum(-1))
            if model.method == "vade":
                result = gaussian_categorical_regularizer(mean, lv, z[:, 0], prior)
                r = result["responsibilities"]
                cat_negative_entropy = (r * r.clamp_min(torch.finfo(r.dtype).tiny).log()).sum(-1)
                negative_entropy = (gaussian_entropy + cat_negative_entropy).mean()
                scope = "augmented (c,z) entropy, sampled differentiable categorical rule"
            else:
                result = pro_t_augmented_regularizer(mean, lv, z[:, 0], prior, softassign_detach=True)
                r = result["responsibilities"]
                a, b = result["gamma_shape"], result["gamma_rate"]
                gamma_negative_entropy = (a - 1) * torch.digamma(a) - a - torch.lgamma(a) + b.log()
                cat_negative_entropy = r.clamp_min(torch.finfo(r.dtype).tiny).log()
                negative_entropy = (gaussian_entropy + (r * (cat_negative_entropy + gamma_negative_entropy)).sum(-1)).mean()
                scope = "augmented (c,u,z) entropy; categorical assignments detached as in training surrogate"
            prior_cross_entropy = result["regularizer"].mean() - negative_entropy
        else:
            log_p = torch.logsumexp(component_log_density(z, model.prior, gaussian=model.is_gaussian_prior) + model.prior.log_weights, -1)
            prior_cross_entropy = -log_p.mean()
            negative_entropy = log_q.mean()
            scope = "Monte Carlo marginal-z entropy"
    return {"reconstruction": reconstruction, "prior_cross_entropy": model.beta * prior_cross_entropy,
            "negative_entropy": model.beta * negative_entropy}, scope


def gradient_diagnostic(model, x, *, seed, target=None):
    """Autograd component norms; forked RNG and autograd.grad preserve training."""
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        terms, scope = objective_terms(model, x, target)
        results = {}
        for name, term in terms.items():
            gradients = torch.autograd.grad(term, [p for _, p in parameters], retain_graph=True, allow_unused=True)
            grouped = {}
            for (parameter_name, _), gradient in zip(parameters, gradients):
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise ArithmeticError("Nonfinite diagnostic gradient")
                group = parameter_name.split(".")[0]
                squared = float(gradient.detach().double().square().sum())
                grouped[group] = grouped.get(group, 0.) + squared
            results[name] = {"value": float(term.detach()), "preclip_norm": math.sqrt(sum(grouped.values())),
                             "group_norms": {k: math.sqrt(v) for k, v in grouped.items()}}
    return {"scope": scope, "components": results, "diagnostic_seed": seed, "optimizer_update": False,
            "component_units": "actual objective coefficients including beta/IDEC gamma",
            "sum_objective": sum(v["value"] for v in results.values())}


@torch.no_grad()
def parameter_diagnostics(model, batch):
    mean, lv = model._encoded(batch)
    result = {"encoder_mean_abs_max": float(mean.abs().max()),
              "recognition_log_variance_min": float(lv.min()), "recognition_log_variance_max": float(lv.max())}
    if model.prior is not None:
        p = model.prior
        result.update(component_weights=p.log_weights.exp().cpu().tolist(), locations=p.loc.cpu().tolist(),
                      scale_diagonal=p.scale_diag.cpu().tolist(),
                      scale_condition_ratio=(p.scale_diag.max(-1).values / p.scale_diag.min(-1).values).cpu().tolist())
        if not model.is_gaussian_prior:
            result.update(prior_df=p.df.cpu().tolist(),
                          df_lower_hits=int((p.df <= model.df_bounds[0] * (1 + 1e-6)).sum()),
                          df_upper_hits=int((p.df >= model.df_bounds[1] * (1 - 1e-6)).sum()))
    if model.is_structured:
        site = model.make_site(mean, lv)
        covariance = model.prior.scale_diag
        if not model.is_gaussian_prior:
            covariance = covariance * model.prior.df[:, None] / (model.prior.df[:, None] - 2)
        strength = prior_site_strength(site, model.prior.loc, covariance, model.prior.logits)
        result.update(site_precision_min=float(site.precision.min()), site_precision_max=float(site.precision.max()),
                      site_precision_below_1e_minus6_count=int((site.precision < 1e-6).sum()),
                      prior_expected_site_energy_mean=float(strength.mean()), prior_expected_site_energy_max=float(strength.max()),
                      unused_variance_head=True)
    if model.q_df_head is not None:
        qdf = model.q_df_head(mean).clamp(math.log(model.posterior_df_bounds[0]), math.log(model.posterior_df_bounds[1])).exp()
        result.update(recognition_df_min=float(qdf.min()), recognition_df_max=float(qdf.max()))
    return result
