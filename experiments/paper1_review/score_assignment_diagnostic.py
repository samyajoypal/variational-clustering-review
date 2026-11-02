"""Fresh S1: fixed-density scores and assignments, not a trained-VAE benchmark.

The Student scale matrix is s*I. Covariance matching uses s=(nu-2)/nu;
scale matching uses s=1. The df derivative follows that declared convention.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper1_review_mpl")
import numpy as np
from scipy.special import digamma, gammaln, logsumexp

from experiments.paper1_review.provenance import ROOT, finish_run, start_run, write_json


def component_values(x: np.ndarray, mean: np.ndarray, nu: float, matching: str):
    """Return log density, latent score vector, and df score (None for Gaussian)."""
    x, mean = np.asarray(x, dtype=float), np.asarray(mean, dtype=float)
    if x.ndim != 2 or mean.shape != (x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(mean).all():
        raise ValueError("Finite [N,d] observations and [d] mean required")
    if matching not in {"covariance", "scale"}:
        raise ValueError("Unknown matching convention")
    diff = x - mean
    d = x.shape[1]
    r2 = np.sum(diff * diff, axis=1)
    if np.isposinf(nu):
        return -0.5 * (d * math.log(2 * math.pi) + r2), -diff, None
    if not math.isfinite(nu) or nu <= 2:
        raise ValueError("This diagnostic requires finite nu>2 or Gaussian infinity")
    scale = (nu - 2) / nu if matching == "covariance" else 1.0
    ds = 2 / nu**2 if matching == "covariance" else 0.0
    u = nu * scale
    du = scale + nu * ds
    lp = (gammaln((nu + d) / 2) - gammaln(nu / 2)
          - d / 2 * math.log(nu * math.pi * scale)
          - (nu + d) / 2 * np.log1p(r2 / u))
    score = -(nu + d) * diff / (u + r2)[:, None]
    df_score = (0.5 * (digamma((nu + d) / 2) - digamma(nu / 2))
                - d / (2 * nu) - d / 2 * ds / scale
                - 0.5 * np.log1p(r2 / u)
                + (nu + d) / 2 * r2 * du / (u * (u + r2)))
    return lp, score, df_score


def mixture_values(x, means, dfs, weights, matching):
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(dfs) or len(means) != len(dfs) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Positive normalized weights and matching component counts required")
    values = [component_values(x, m, nu, matching) for m, nu in zip(means, dfs)]
    component_lp = np.stack([v[0] for v in values], axis=1)
    # Normalize before restoring a potentially enormous negative log-density.
    # Subtracting lp near -1e12 loses the small normalizer and breaks row sums.
    maximum = component_lp.max(axis=1, keepdims=True)
    centered = component_lp - maximum + np.log(weights)
    normalizer = logsumexp(centered, axis=1, keepdims=True)
    lp = (maximum + normalizer)[:, 0]
    lr = centered - normalizer
    resp = np.exp(lr)
    score = np.sum(np.stack([v[1] for v in values], axis=1) * resp[:, :, None], axis=1)
    entropy = -np.sum(resp * lr, axis=1)
    return lp, score, resp, entropy, values


def label(nu):
    return "gaussian" if math.isinf(nu) else f"t{int(nu)}"


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path):
    config = {
        "module": "S1", "model": "fixed isotropic one/two-component densities",
        "dimensions": [2, 10, 50], "dfs": [3, 5, 15, 30, "gaussian"],
        "matching": ["covariance", "scale"], "weights": [[0.5, 0.5], [0.2, 0.8]],
        "single_radii": {"zero": True, "log10_min": -3, "log10_max": 6, "count": 201},
        "mixture_radii": {"central_linear": [-3, 3, 121], "signed_log10_tails": [0.5, 6, 100]},
        "mixture_means": "-e1,+e1", "trajectories": ["axis_between_centers", "orthogonal_axis"],
        "pair_dfs": [[3, 3], [5, 5], [15, 15], [30, 30], ["gaussian", "gaussian"], [3, 30], [5, 30], [3, "gaussian"]],
        "randomness": "none", "legacy_results_used": False,
        "interpretation": "Density derivatives only; no neural or fitted-estimator robustness guarantee",
        "formula_source": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html",
    }
    metadata = start_run(output, stage="deterministic_simulation", config=config, sources=[Path(__file__)])
    try:
        dfs = [3., 5., 15., 30., np.inf]
        radii = np.r_[0., np.logspace(-3, 6, 201)]
        single_rows = []
        for d in config["dimensions"]:
            x = np.zeros((len(radii), d)); x[:, 0] = radii
            for matching in config["matching"]:
                for nu in dfs:
                    lp, score, ds = component_values(x, np.zeros(d), nu, matching)
                    for i, r in enumerate(radii):
                        single_rows.append(dict(dimension=d, matching=matching, family=label(nu), radius=r,
                                                log_density=lp[i], latent_score_norm=np.linalg.norm(score[i]),
                                                location_score_norm=np.linalg.norm(score[i]),
                                                df_score=None if ds is None else ds[i]))
        write_rows(output / "single_component.csv", single_rows)
        tail = np.logspace(0.5, 6, 100)
        positions = np.unique(np.r_[-tail, np.linspace(-3, 3, 121), tail])
        pairs = [(n, n) for n in dfs] + [(3., 30.), (5., 30.), (3., np.inf)]
        mixture_rows = []
        for d in config["dimensions"]:
            means = np.zeros((2, d)); means[:, 0] = [-1., 1.]
            for matching in config["matching"]:
                for nus in pairs:
                    for weights in config["weights"]:
                        for axis, trajectory in enumerate(config["trajectories"]):
                            x = np.zeros((len(positions), d)); x[:, axis] = positions
                            lp, score, resp, entropy, values = mixture_values(x, means, nus, weights, matching)
                            for i, r in enumerate(positions):
                                mixture_rows.append(dict(dimension=d, matching=matching, family0=label(nus[0]), family1=label(nus[1]),
                                    weight0=weights[0], trajectory=trajectory, coordinate=r, log_density=lp[i],
                                    latent_score_norm=np.linalg.norm(score[i]), responsibility0=resp[i, 0],
                                    responsibility1=resp[i, 1], entropy=entropy[i],
                                    df_score0=None if values[0][2] is None else resp[i, 0] * values[0][2][i],
                                    df_score1=None if values[1][2] is None else resp[i, 1] * values[1][2][i]))
        write_rows(output / "mixture.csv", mixture_rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
        for nu in [3., 5., 30., np.inf]:
            rr = [v for v in single_rows if v["dimension"] == 10 and v["matching"] == "covariance" and v["family"] == label(nu) and v["radius"] > 0]
            axes[0].loglog([v["radius"] for v in rr], [v["latent_score_norm"] for v in rr], label=label(nu))
            if not math.isinf(nu):
                axes[2].semilogx([v["radius"] for v in rr], [v["df_score"] for v in rr], label=label(nu))
        rr = [v for v in mixture_rows if v["dimension"] == 10 and v["matching"] == "covariance" and v["family0"] == "t3" and v["family1"] == "t30" and v["weight0"] == .5 and v["trajectory"] == "axis_between_centers" and v["coordinate"] > 0]
        axes[1].semilogx([v["coordinate"] for v in rr], [v["responsibility0"] for v in rr], color="C0")
        axes[0].set(xlabel="Radius", ylabel="Latent score norm", title="Bounded t location scores")
        axes[1].set(xlabel="Coordinate along centers", ylabel="Responsibility of t3 component", ylim=(-.02, 1.02), title="Tail allocation can be confident")
        axes[2].set(xlabel="Radius", ylabel="Derivative with respect to df", title="A different parameter score")
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(); axes[2].legend()
        fig.suptitle("Fixed densities, d=10, unit covariance — no neural robustness guarantee")
        fig.savefig(output / "score_assignment_diagnostic.png", dpi=180)
        fig.savefig(output / "score_assignment_diagnostic.pdf")
        plt.close(fig)
        checks = {"single_rows": len(single_rows), "mixture_rows": len(mixture_rows),
                  "finite_density_and_scores": all(np.isfinite(r["log_density"]) and np.isfinite(r["latent_score_norm"]) for r in single_rows + mixture_rows),
                  "maximum_responsibility_sum_error": max(abs(r["responsibility0"] + r["responsibility1"] - 1) for r in mixture_rows)}
        if not checks["finite_density_and_scores"] or checks["maximum_responsibility_sum_error"] > 1e-8:
            raise ArithmeticError(checks)
        write_json(output / "checks.json", checks)
        finish_run(output, metadata, checks=checks)
        return checks
    except Exception as error:
        finish_run(output, metadata, status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper1_review_20261027/s1_fixed_density_v2")
    args = parser.parse_args()
    print(run(args.output))
