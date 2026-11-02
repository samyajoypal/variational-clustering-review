#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-VaDE: VAE with Student-t Mixture Model prior (Mode B only).

Uses the principled 3-KL closed-form ELBO decomposition:
  sum_c gamma_c * [KL_gauss(c) + KL_gamma(c) + KL_cat(c)]

All three KLs are closed-form via the Gamma-Gaussian decomposition.

Encoder, Decoder, ResidualBlock are UNCHANGED.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Encoder / Decoder / ResidualBlock — UNCHANGED
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + identity)


class Encoder(nn.Module):
    """Convolutional encoder for VAE."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(1, 32, dropout_rate=0.1)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.res2 = ResidualBlock(64, 64, dropout_rate=0.1)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Convolutional decoder for VAE."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128 * 7 * 7)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.res1 = ResidualBlock(64, 64, dropout_rate=0.1)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.res2 = ResidualBlock(32, 32, dropout_rate=0.1)
        self.conv_out = nn.Conv2d(32, 1, 3, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        return self.act(self.conv_out(x))


# ============================================================
# t-VaDE Model (Mode B only)
# ============================================================

class StudentTVaDE(nn.Module):
    """
    VAE with trainable Student-t Mixture Model prior (Mode B).

    Uses the principled 3-KL closed-form ELBO via student_t_elbo_loss().

    All GMM parameters (gmm_means, gmm_logvars, logit_pi) are retained
    and used identically. nu_c adds one scalar per cluster.
    """

    def __init__(self, latent_dim: int, num_clusters: int, nu_init: float = 5.0):
        """
        Args:
            latent_dim: dimension of latent space
            num_clusters: number of mixture components
            nu_init: initial degrees of freedom (higher = closer to Gaussian)
        """
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # GMM parameters — same as original
        self.gmm_means = nn.Parameter(torch.randn(num_clusters, latent_dim) * 0.05)
        self.gmm_logvars = nn.Parameter(torch.zeros(num_clusters, latent_dim))
        self.logit_pi = nn.Parameter(torch.zeros(num_clusters))

        # NEW: degrees of freedom per cluster
        # nu_c = softplus(nu_rho) to ensure nu_c > 0
        # Initialise so that softplus(nu_rho) ≈ nu_init
        nu_rho_init = math.log(math.exp(nu_init) - 1.0)  # inverse softplus
        self.nu_rho = nn.Parameter(torch.full((num_clusters,), nu_rho_init))

        self.K = num_clusters
        self.D = latent_dim

    @property
    def nu_c(self):
        """Degrees of freedom, guaranteed positive via softplus."""
        return F.softplus(self.nu_rho)  # (K,)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Principled 3-KL closed-form ELBO
    #
    # This replaces BOTH loss_kl AND gmm_loss with the proper decomposition:
    #   L = Recon - sum_c gamma_c * [KL_gauss_c + KL_gamma_c + KL_cat_c]
    #
    # Use this by calling student_t_elbo_loss(z, mu_phi, logvar_phi) in the trainer
    # and REMOVING the separate loss_kl term.
    # ------------------------------------------------------------------

    def student_t_elbo_loss(self, z, mu_phi, logvar_phi, eps: float = 1e-6):
        """
        Principled closed-form ELBO for Student-t mixture prior.

        Replaces BOTH loss_kl and gmm_loss. The trainer should use:
            loss = w_recon * loss_recon + w_kl * vae.student_t_elbo_loss(z, mu, logvar)

        Derivation follows the PDF: we decompose the t-prior as
            p(z|c) = int N(z|mu_c, (u*Lambda_c)^{-1}) * Gamma(u|nu/2, nu/2) du
        and the ELBO becomes:
            sum_c gamma_c * [KL_gauss(c) + KL_gamma(c) + KL_cat(c)]

        All three KLs are closed-form.

        Args:
            z:            (B, D) reparameterised samples
            mu_phi:       (B, D) encoder mean
            logvar_phi:   (B, D) encoder log-variance
            eps: numerical stability

        Returns:
            scalar loss (mean over batch) — this is the NEGATIVE ELBO KL terms,
            so it should be ADDED to the reconstruction loss.
        """
        B, D = z.shape
        nu = self.nu_c                                        # (K,)

        # Cluster prior precision (diagonal): Lambda_c = 1/var_c
        var_c = torch.exp(self.gmm_logvars) + eps              # (K, D)
        precision_c = 1.0 / var_c                              # (K, D)

        # Encoder variance
        var_phi = torch.exp(logvar_phi) + eps                  # (B, D)

        # ---- Compute soft responsibilities gamma_c (detached) ----
        with torch.no_grad():
            # Use the Student-t density for gamma (not Gaussian)
            diff_g = z[:, None, :] - self.gmm_means[None, :, :]
            delta2_g = torch.sum(diff_g ** 2 * precision_c[None, :, :], dim=-1)  # (B, K)
            half_nu_d = (nu + D) / 2.0
            log_t_g = (
                torch.lgamma(half_nu_d) - torch.lgamma(nu / 2.0)
                - (D / 2.0) * torch.log(nu * math.pi)
                + 0.5 * torch.sum(torch.log(precision_c), dim=-1)
                - half_nu_d * torch.log(1.0 + delta2_g / nu[None, :])
            )
            log_pi = F.log_softmax(self.logit_pi, dim=0)[None, :]
            gamma = F.softmax(log_pi + log_t_g, dim=-1)        # (B, K)

        # ---- Expected Mahalanobis distance under q_phi ----
        # E_{q_phi}[(z-mu_c)^T Lambda_c (z-mu_c)]
        #   = sum_j Lambda_{c,j} * (sigma_phi_j^2 + (mu_phi_j - mu_c_j)^2)
        diff = mu_phi[:, None, :] - self.gmm_means[None, :, :]  # (B, K, D)
        delta2 = torch.sum(
            precision_c[None, :, :] * (var_phi[:, None, :] + diff ** 2),
            dim=-1
        )  # (B, K)

        # ---- Gamma posterior parameters (exact, by conjugacy) ----
        alpha_t = (nu[None, :] + D) / 2.0                     # (1, K) -> broadcasts to (B, K)
        beta_t = (nu[None, :] + delta2) / 2.0                  # (B, K)

        # Sufficient statistics of q(u|z,c) = Gamma(alpha_t, beta_t)
        E_u = alpha_t / beta_t                                  # (B, K)
        E_log_u = torch.digamma(alpha_t) - torch.log(beta_t)   # (B, K)

        # ---- Term (I): Expected Gaussian KL ----
        # KL[N(mu_phi, diag(var_phi)) || N(mu_c, (u*Lambda_c)^{-1})]
        # averaged over q(u|z,c), which replaces u -> E[u], log u -> E[log u]
        #
        # = -0.5 * [d + d*E[log u] + log|Lambda_c| + sum_j log(var_phi_j) - E[u]*delta2]

        log_det_precision = torch.sum(torch.log(precision_c), dim=-1)  # (K,)
        log_det_var_phi = torch.sum(logvar_phi, dim=-1)                # (B,)

        kl_gauss = -0.5 * (
            D
            + D * E_log_u                                       # (B, K)
            + log_det_precision[None, :]                        # (1, K)
            + log_det_var_phi[:, None]                          # (B, 1)
            - E_u * delta2                                      # (B, K)
        )  # (B, K)

        # ---- Term (II): Gamma KL ----
        # KL[Gamma(alpha_t, beta_t) || Gamma(alpha_0, beta_0)]
        # where alpha_0 = beta_0 = nu_c/2

        alpha_0 = nu[None, :] / 2.0                            # (1, K)
        beta_0 = nu[None, :] / 2.0                             # (1, K)

        kl_gamma = (
            (alpha_t - alpha_0) * torch.digamma(alpha_t)
            + torch.lgamma(alpha_0) - torch.lgamma(alpha_t)
            + alpha_0 * (torch.log(beta_t) - torch.log(beta_0))
            + alpha_t * (beta_0 - beta_t) / beta_t
        )  # (B, K)

        # ---- Term (III): Categorical KL ----
        # gamma_c * (log gamma_c - log pi_c)
        log_pi_full = F.log_softmax(self.logit_pi, dim=0)[None, :]  # (1, K)
        kl_cat = torch.log(gamma + eps) - log_pi_full                # (B, K)

        # ---- Assemble ----
        # Total KL = sum_c gamma_c * (KL_gauss_c + KL_gamma_c + KL_cat_c)
        total_kl = torch.sum(
            gamma * (kl_gauss + kl_gamma + kl_cat),
            dim=-1
        )  # (B,)

        return total_kl.mean()

    def pi_balance_loss(self):
        """Encourage uniform mixture weights."""
        pi = F.softmax(self.logit_pi, dim=0)
        pi_u = torch.full_like(pi, 1.0 / pi.numel())
        return F.mse_loss(pi, pi_u)
