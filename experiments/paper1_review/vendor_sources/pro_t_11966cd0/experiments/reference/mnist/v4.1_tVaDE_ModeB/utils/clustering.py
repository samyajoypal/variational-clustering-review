#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM clustering and evaluation utilities.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def refit_gmm(Z_train: np.ndarray, num_clusters: int, covariance_type: str = "full", n_init: int = 10):
    """
    Fit a sklearn GMM to latent embeddings.

    Args:
        Z_train: Latent embeddings [N, D]
        num_clusters: Number of clusters
        covariance_type: Type of covariance matrix
        n_init: Number of initializations

    Returns:
        Fitted GaussianMixture model
    """
    gmm = GaussianMixture(
        n_components=num_clusters,
        covariance_type=covariance_type,
        random_state=0,
        n_init=n_init
    )
    gmm.fit(Z_train)
    return gmm


def hungarian_accuracy(true_labels: np.ndarray, pred_clusters: np.ndarray, n_classes: int = 10) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm.

    Args:
        true_labels: Ground truth labels
        pred_clusters: Predicted cluster assignments
        n_classes: Number of classes

    Returns:
        Clustering accuracy
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(len(true_labels)):
        cm[int(true_labels[i]), int(pred_clusters[i])] += 1.0
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / max(1, len(true_labels))
    return float(acc)


def eval_refit_metrics(gmm: GaussianMixture, Z: np.ndarray, Y: np.ndarray, num_clusters: int):
    """
    Evaluate clustering metrics on embeddings.

    Args:
        gmm: Fitted GMM model
        Z: Latent embeddings
        Y: True labels
        num_clusters: Number of clusters

    Returns:
        acc, nmi, ari metrics
    """
    pred = gmm.predict(Z)
    acc = hungarian_accuracy(Y, pred, n_classes=num_clusters)
    nmi = normalized_mutual_info_score(Y, pred)
    ari = adjusted_rand_score(Y, pred)
    return acc, nmi, ari


@torch.no_grad()
def collect_embeddings_and_images(model, loader, device, return_sample_ids=False):
    """
    Collect mu embeddings, true labels, images from a data loader.

    Args:
        model: VAE model
        loader: Data loader (must have shuffle=False if return_sample_ids=True)
        device: Computing device
        return_sample_ids: If True, additionally return per-sample identifier
            list aligned with iteration order. Pulls from
            loader.dataset.sample_ids (for direct Datasets) or from
            loader.dataset.dataset.sample_ids[loader.dataset.indices] (for
            torch.utils.data.Subset). Falls back to "#i" indices.

    Returns:
        Z: [N,D] numpy embeddings
        Y: [N] numpy labels
        X: [N,1,H,W] torch CPU images
        (sample_ids: list[str], length N — only if return_sample_ids=True)
    """
    model.eval()
    Z_list, Y_list, X_list = [], [], []
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        Z_list.append(mu.detach().cpu().numpy())
        Y_list.append(y.numpy())
        X_list.append(x.detach().cpu())
    Z = np.vstack(Z_list)
    Y = np.hstack(Y_list)
    X = torch.cat(X_list, dim=0)
    if not return_sample_ids:
        return Z, Y, X

    ds = loader.dataset
    sample_ids = None
    # Subset(orig_ds, indices=...): need to map back to orig_ds.sample_ids
    if hasattr(ds, "indices") and hasattr(ds, "dataset") \
            and hasattr(ds.dataset, "sample_ids"):
        sample_ids = [str(ds.dataset.sample_ids[int(i)]) for i in ds.indices]
    # Direct dataset with sample_ids attribute
    elif hasattr(ds, "sample_ids"):
        sample_ids = [str(s) for s in ds.sample_ids]
    if sample_ids is None or len(sample_ids) != int(X.shape[0]):
        sample_ids = [f"#{i}" for i in range(int(X.shape[0]))]
    return Z, Y, X, sample_ids


def sharpen_probs(P: np.ndarray, gamma_power: float):
    """
    Sharpen probability distributions.

    Args:
        P: Probability matrix [N, K]
        gamma_power: Power to raise probabilities to

    Returns:
        Sharpened probabilities
    """
    if gamma_power == 1.0:
        return P
    P2 = np.power(P, gamma_power)
    P2 = P2 / (P2.sum(axis=1, keepdims=True) + 1e-12)
    return P2


@torch.no_grad()
def overwrite_vae_prior_from_refit_gmm(vae, gmm, eps: float = 1e-6, ema_alpha: float = 1.0):
    """
    Force (or EMA) overwrite VAE's trainable diagonal-cov GMM prior
    using sklearn refit GMM (means_, covariances_, weights_).

    Args:
        vae: StaticGMMVAE model
        gmm: Fitted sklearn GMM
        eps: Small constant for numerical stability
        ema_alpha: 1.0 = hard overwrite, < 1.0 = smooth update
    """
    device = next(vae.parameters()).device

    # means: [K,D]
    means = torch.tensor(gmm.means_, dtype=torch.float32, device=device)

    # cov -> diag var: [K,D]
    if gmm.covariance_type == "full":
        cov = gmm.covariances_  # [K,D,D]
        var = np.stack([np.diag(c) for c in cov], axis=0)  # [K,D]
    elif gmm.covariance_type == "diag":
        var = gmm.covariances_  # [K,D]
    elif gmm.covariance_type == "spherical":
        # [K,] -> [K,D]
        var = np.repeat(gmm.covariances_[:, None], vae.D, axis=1)
    else:  # "tied" etc. (not recommended here)
        # tied: [D,D]
        v = np.diag(gmm.covariances_)
        var = np.repeat(v[None, :], vae.K, axis=0)

    var = torch.tensor(var, dtype=torch.float32, device=device).clamp_min(eps)
    logvars = torch.log(var)

    # weights -> logit_pi
    w = torch.tensor(gmm.weights_, dtype=torch.float32, device=device).clamp_min(eps)
    logit_pi = torch.log(w)

    # EMA / overwrite
    a = float(ema_alpha)
    vae.gmm_means.copy_((1 - a) * vae.gmm_means + a * means)
    vae.gmm_logvars.copy_((1 - a) * vae.gmm_logvars + a * logvars)
    vae.logit_pi.copy_((1 - a) * vae.logit_pi + a * logit_pi)


@torch.no_grad()
def eval_prior_student_t_metrics(vae, Z: np.ndarray, Y: np.ndarray, num_clusters: int, device):
    """
    Evaluate clustering using VAE's own Student-t prior parameters.
    Computes gamma_c = softmax(log pi_c + log T_c(z)) and uses argmax for cluster assignment.

    Args:
        vae: StudentTVaDE model
        Z: [N, D] numpy embeddings
        Y: [N] numpy labels
        num_clusters: Number of clusters
        device: Computing device

    Returns:
        acc, nmi, ari
    """
    vae.eval()
    z_t = torch.tensor(Z, dtype=torch.float32, device=device)

    nu = vae.nu_c  # (K,)
    D = vae.D
    var = torch.exp(vae.gmm_logvars) + 1e-6  # (K, D)
    precision = 1.0 / var  # (K, D)

    diff = z_t[:, None, :] - vae.gmm_means[None, :, :]  # (N, K, D)
    delta2 = torch.sum(diff ** 2 * precision[None, :, :], dim=-1)  # (N, K)

    half_nu_d = (nu + D) / 2.0
    log_t = (
        torch.lgamma(half_nu_d) - torch.lgamma(nu / 2.0)
        - (D / 2.0) * torch.log(nu * math.pi)
        + 0.5 * torch.sum(torch.log(precision), dim=-1)
        - half_nu_d * torch.log(1.0 + delta2 / nu[None, :])
    )

    log_pi = F.log_softmax(vae.logit_pi, dim=0)[None, :]
    log_gamma = log_pi + log_t  # (N, K)
    pred = log_gamma.argmax(dim=1).cpu().numpy()

    acc = hungarian_accuracy(Y, pred, n_classes=num_clusters)
    nmi = normalized_mutual_info_score(Y, pred)
    ari = adjusted_rand_score(Y, pred)
    return acc, nmi, ari


# ============================================================
# Label-free clustering quality + Student-t stage predictor
# (added for compute-efficiency / OASIS dummy-label support)
# ============================================================

import math as _math
from sklearn.metrics import silhouette_score as _silhouette_score
from sklearn.metrics import davies_bouldin_score as _davies_bouldin_score


def predict_refit(gmm: GaussianMixture, Z: np.ndarray) -> np.ndarray:
    """Cluster assignment from a fitted sklearn GMM. Returns (N,) int array."""
    return gmm.predict(Z).astype(int)


@torch.no_grad()
def predict_stage_gaussian(vae, Z: np.ndarray, device) -> np.ndarray:
    """
    Stage-level assignment under a Gaussian mixture prior
    (uses vae.gmm_means, vae.gmm_logvars, vae.logit_pi).
    """
    vae.eval()
    z_t = torch.tensor(Z, dtype=torch.float32, device=device)
    var = torch.exp(vae.gmm_logvars) + 1e-6
    diff = z_t[:, None, :] - vae.gmm_means[None, :, :]
    logp = -0.5 * torch.sum(
        diff * diff / var[None, :, :] + torch.log(var[None, :, :]), dim=-1
    )
    log_pi = torch.log_softmax(vae.logit_pi, dim=0)[None, :]
    return (log_pi + logp).argmax(dim=1).cpu().numpy().astype(int)


@torch.no_grad()
def predict_stage_student_t(vae, Z: np.ndarray, device) -> np.ndarray:
    """
    Stage-level assignment under a Student-t mixture prior.
    Requires vae.nu_c (degrees of freedom) in addition to the Gaussian params.
    """
    vae.eval()
    z_t = torch.tensor(Z, dtype=torch.float32, device=device)
    nu = vae.nu_c
    D = vae.D
    var = torch.exp(vae.gmm_logvars) + 1e-6
    prec = 1.0 / var
    diff = z_t[:, None, :] - vae.gmm_means[None, :, :]
    delta2 = torch.sum(diff ** 2 * prec[None, :, :], dim=-1)
    half_nu_d = (nu + D) / 2.0
    log_t = (
        torch.lgamma(half_nu_d) - torch.lgamma(nu / 2.0)
        - (D / 2.0) * torch.log(nu * _math.pi)
        + 0.5 * torch.sum(torch.log(prec), dim=-1)
        - half_nu_d * torch.log(1.0 + delta2 / nu[None, :])
    )
    log_pi = F.log_softmax(vae.logit_pi, dim=0)[None, :]
    return (log_pi + log_t).argmax(dim=1).cpu().numpy().astype(int)


def predict_stage(vae, Z: np.ndarray, device) -> np.ndarray:
    """
    Dispatch: use Student-t assignment if the model exposes nu_c,
    otherwise fall back to Gaussian mixture assignment.
    """
    if hasattr(vae, "nu_c"):
        return predict_stage_student_t(vae, Z, device)
    return predict_stage_gaussian(vae, Z, device)


def cluster_size_entropy(pred: np.ndarray, num_clusters: int) -> float:
    """
    Normalised cluster-size entropy H / log K  in [0, 1].
    1.0 = uniform distribution; 0.0 = fully collapsed into a single cluster.
    """
    pred = np.asarray(pred).astype(int)
    if pred.size == 0 or num_clusters <= 1:
        return 0.0
    counts = np.bincount(pred, minlength=num_clusters).astype(np.float64)
    p = counts / counts.sum()
    p_nz = p[p > 0]
    ent = float(-np.sum(p_nz * np.log(p_nz)))
    return round(ent / _math.log(num_clusters), 4)


def safe_silhouette(Z: np.ndarray, pred: np.ndarray):
    if len(np.unique(pred)) < 2:
        return None
    try:
        return round(float(_silhouette_score(Z, pred)), 4)
    except Exception:
        return None


def safe_davies_bouldin(Z: np.ndarray, pred: np.ndarray):
    if len(np.unique(pred)) < 2:
        return None
    try:
        return round(float(_davies_bouldin_score(Z, pred)), 4)
    except Exception:
        return None


def compute_label_free_metrics(Z: np.ndarray, c_stage: np.ndarray, c_refit: np.ndarray,
                               num_clusters: int, prefix: str = "") -> dict:
    """
    Bundle: stage/refit silhouette + DB + size entropy (label-free).
    """
    out = {}
    out[f"{prefix}stage_size_entropy"] = cluster_size_entropy(c_stage, num_clusters)
    out[f"{prefix}refit_size_entropy"] = cluster_size_entropy(c_refit, num_clusters)
    out[f"{prefix}stage_silhouette"] = safe_silhouette(Z, c_stage)
    out[f"{prefix}refit_silhouette"] = safe_silhouette(Z, c_refit)
    out[f"{prefix}stage_davies_bouldin"] = safe_davies_bouldin(Z, c_stage)
    out[f"{prefix}refit_davies_bouldin"] = safe_davies_bouldin(Z, c_refit)
    return out
