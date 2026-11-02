#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for t-VaDE Mode B (Principled 3-KL ELBO).
"""


class CFG:
    """Main configuration class for training hyperparameters."""

    # Paths
    results_dir = "results_v4.1_tVaDE_ModeB"

    # Random seed
    seed = 42

    # Training settings
    batch_size = 32
    latent_dim = 8
    num_clusters = 10
    epoch_num = 100

    lr = 1e-3
    weight_decay = 1e-4

    # Data subset: 256*50 per class (~128k samples)
    num_samples_per_class = 256 * 50

    # ---- t-VaDE specific ----
    nu_init = 5.0               # initial degrees of freedom (5-10 is a good range)
                                # lower = heavier tails = more robust to outliers
                                # higher = closer to Gaussian

    # Loss weights
    w_recon = 1.0
    w_elbo = 5.0               # fixed weight for 3-KL ELBO term
    w_pi_balance = 0.0

    eps = 1e-6

    # Refit interval: overwrite VAE prior every N epochs
    refit_interval = 20

    # Refit GMM settings (unchanged — still uses sklearn GMM for evaluation)
    refit_covariance_type = "full"
    refit_n_init = 10

    # Scheme-A atlas settings (unchanged)
    atlas_save_every = 1
    sample_topn_per_cluster = 64
    sample_nrow = 8
    conf_thresh = 0.90
    gamma_power = 2.0
    max_per_cluster_for_atlas = 5000
    atlas_robust = "median"
    save_decoded_prior_means_every = 1
