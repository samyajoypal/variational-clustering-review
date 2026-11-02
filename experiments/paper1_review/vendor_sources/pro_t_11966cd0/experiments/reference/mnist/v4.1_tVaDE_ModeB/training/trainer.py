#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training logic for t-VaDE Mode B (Principled 3-KL ELBO).

Uses the principled 3-KL decomposition. Replaces BOTH loss_kl and gmm_loss.
    loss = w_recon * L_recon(sum) + w_elbo * student_t_elbo_loss(z, mu, logvar)(mean)
"""

import os
import torch
import torch.nn.functional as F

from utils import (
    ensure_dir, write_csv,
    refit_gmm, eval_refit_metrics,
    collect_embeddings_and_images,
    sharpen_probs,
    overwrite_vae_prior_from_refit_gmm,
    save_grid,
    save_refit_gmm_means_atlas,
    save_refit_gmm_uncertainty_atlas,
    save_cluster_sample_grids,
    build_cluster_atlas_A,
    eval_prior_student_t_metrics,
    save_student_t_means_decoded,
    save_latent_tsne,
    predict_refit, predict_stage, compute_label_free_metrics,
    EpochProfiler,
)


def train_epoch(vae, train_loader, optimizer, cfg, device):
    """
    Train t-VaDE for one epoch (Mode B: Principled 3-KL ELBO).
    Loss = w_recon * L_recon(sum) + w_elbo * L_elbo(mean).
    Returns dict of total and component losses.
    """
    vae.train()
    total_loss = 0.0
    total_recon = 0.0
    total_elbo = 0.0

    prof = EpochProfiler(device=device)
    prof.start_epoch()

    for x, _ in train_loader:
        x = x.to(device)
        prof.step_begin()
        optimizer.zero_grad(set_to_none=True)

        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        xhat = vae.decode(z)

        loss_recon = F.binary_cross_entropy(xhat.clamp(1e-6, 1 - 1e-6), x, reduction="sum")
        loss_elbo = vae.student_t_elbo_loss(z, mu, logvar, eps=cfg.eps)
        loss = cfg.w_recon * loss_recon + cfg.w_elbo * loss_elbo

        if cfg.w_pi_balance > 0.0:
            loss = loss + cfg.w_pi_balance * vae.pi_balance_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
        optimizer.step()
        prof.step_end()

        total_loss += float(loss.detach().cpu())
        total_recon += float(loss_recon.detach().cpu())
        total_elbo += float(loss_elbo.detach().cpu())

    stats = prof.end_epoch()
    return {
        "total": total_loss, "recon": total_recon, "elbo": total_elbo,
        "profile": stats,
    }


@torch.no_grad()
def compute_test_loss(vae, test_loader, cfg, device):
    """Compute loss on test set (no gradient)."""
    vae.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_elbo = 0.0

    for x, _ in test_loader:
        x = x.to(device)
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        xhat = vae.decode(z)

        loss_recon = F.binary_cross_entropy(xhat.clamp(1e-6, 1 - 1e-6), x, reduction="sum")
        loss_elbo = vae.student_t_elbo_loss(z, mu, logvar, eps=cfg.eps)
        loss = cfg.w_recon * loss_recon + cfg.w_elbo * loss_elbo

        total_loss += float(loss.cpu())
        total_recon += float(loss_recon.cpu())
        total_elbo += float(loss_elbo.cpu())

    return {"total": total_loss, "recon": total_recon, "elbo": total_elbo}


def evaluate_and_save(vae, train_loader_eval, test_loader_eval, gmm, cfg, epoch, device,
                      best_test_acc, best_test_stage_acc,
                      test_recon=None, best_test_recon=None):
    """
    Evaluate model with both refit GMM and prior Student-t metrics.
    Save visualizations including Student-t means decoded and t-SNE.

    `best_vae.pth` is saved on **min test reconstruction loss** (with the
    first 25 % of `cfg.epoch_num` excluded as warm-up). Reconstruction loss
    is universally meaningful: on OASIS dummy labels the legacy
    `test_refit_accuracy` criterion picks epoch 1 lucky permutations and
    saves an untrained model. Caller passes per-sample `test_recon` (i.e.
    `test_losses["recon"] / N_test`) and the running `best_test_recon`
    tracker. If either is None, the function falls back to the legacy
    refit-accuracy criterion (so MNIST/FMNIST scripts that have not been
    upgraded keep working unchanged).
    """
    Z_train, Y_train, X_train_cpu, train_sample_ids = collect_embeddings_and_images(
        vae, train_loader_eval, device, return_sample_ids=True)
    Z_test, Y_test, _ = collect_embeddings_and_images(vae, test_loader_eval, device)

    train_acc, train_nmi, train_ari = eval_refit_metrics(gmm, Z_train, Y_train, cfg.num_clusters)
    test_acc, test_nmi, test_ari = eval_refit_metrics(gmm, Z_test, Y_test, cfg.num_clusters)

    stage_train_acc, stage_train_nmi, stage_train_ari = eval_prior_student_t_metrics(
        vae, Z_train, Y_train, cfg.num_clusters, device
    )
    stage_test_acc, stage_test_nmi, stage_test_ari = eval_prior_student_t_metrics(
        vae, Z_test, Y_test, cfg.num_clusters, device
    )

    # --- Label-free cluster quality (silhouette / DB / size entropy) ---
    c_refit_train = predict_refit(gmm, Z_train)
    c_refit_test  = predict_refit(gmm, Z_test)
    c_stage_train = predict_stage(vae, Z_train, device)
    c_stage_test  = predict_stage(vae, Z_test, device)
    lf_train = compute_label_free_metrics(
        Z_train, c_stage_train, c_refit_train, cfg.num_clusters, prefix="train_"
    )
    lf_test = compute_label_free_metrics(
        Z_test, c_stage_test, c_refit_test, cfg.num_clusters, prefix="test_"
    )

    save_refit_gmm_means_atlas(
        vae=vae, gmm=gmm,
        out_dir=os.path.join(cfg.results_dir, "atlas_refit_means"),
        epoch=epoch, nrow=cfg.num_clusters
    )
    save_refit_gmm_uncertainty_atlas(
        vae=vae, gmm=gmm,
        out_dir=os.path.join(cfg.results_dir, "atlas_refit_uncertainty"),
        epoch=epoch, nrow=cfg.num_clusters
    )

    save_student_t_means_decoded(
        vae=vae,
        out_dir=os.path.join(cfg.results_dir, "st_means_decoded"),
        epoch=epoch, nrow=cfg.num_clusters
    )

    P_train = gmm.predict_proba(Z_train)
    P_train = sharpen_probs(P_train, cfg.gamma_power)

    stats = save_cluster_sample_grids(
        X_cpu=X_train_cpu, Y_true=Y_train, prob=P_train,
        out_dir=os.path.join(cfg.results_dir, "cluster_samples"),
        epoch=epoch, topn_per_cluster=cfg.sample_topn_per_cluster,
        conf_thresh=cfg.conf_thresh, nrow=cfg.sample_nrow,
        sample_ids=train_sample_ids,
        label_every=getattr(cfg, "cluster_label_every", 25)
    )
    purity_csv = os.path.join(cfg.results_dir, f"cluster_purity_epoch_{epoch:03d}.csv")
    write_csv(stats, purity_csv)

    if epoch % cfg.atlas_save_every == 0:
        build_cluster_atlas_A(
            X_cpu=X_train_cpu, prob=P_train,
            out_dir=os.path.join(cfg.results_dir, "atlas_A"),
            epoch=epoch, conf_thresh=cfg.conf_thresh,
            gamma_power=1.0, max_per_cluster=cfg.max_per_cluster_for_atlas,
            robust=cfg.atlas_robust
        )

    # Track best refit accuracy for CSV reporting only (no longer triggers
    # checkpoint save — that's now driven by recon loss below).
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    if stage_test_acc > best_test_stage_acc:
        best_test_stage_acc = stage_test_acc

    # NEW: best_vae.pth saved on min test_recon, after a 25 % warm-up window.
    # Falls back to the legacy `test_acc > best_test_acc` rule if caller did
    # not pass test_recon (preserves backward-compat).
    if test_recon is not None and best_test_recon is not None:
        # Total epochs may live as `epoch_num` (v1.* / v4.0 / v4.1) or as a
        # phase1+phase2 split (v4.1a). Combine both robustly.
        _total_eps = (getattr(cfg, "epoch_num", 0)
                       or (getattr(cfg, "phase1_epochs", 0)
                           + getattr(cfg, "phase2_epochs", 0))
                       or 100)
        warmup_threshold = int(0.25 * _total_eps)
        if epoch > warmup_threshold and test_recon < best_test_recon:
            best_test_recon = test_recon
            torch.save(vae.state_dict(),
                       os.path.join(cfg.results_dir, "best_vae.pth"))
            save_latent_tsne(
                Z=Z_train, Y=Y_train, gmm=gmm,
                out_path=os.path.join(cfg.results_dir,
                                      f"tsne_best_epoch_{epoch:03d}_train.png"),
                title=f"Train t-SNE (Best by recon, Ep {epoch}, "
                      f"recon={test_recon:.4f})"
            )
            save_latent_tsne(
                Z=Z_test, Y=Y_test, gmm=gmm,
                out_path=os.path.join(cfg.results_dir,
                                      f"tsne_best_epoch_{epoch:03d}_test.png"),
                title=f"Test t-SNE (Best by recon, Ep {epoch}, "
                      f"recon={test_recon:.4f})"
            )
    else:
        # Legacy fallback (kept verbatim for compatibility).
        if test_acc >= best_test_acc:   # use >= because best_test_acc was just updated
            torch.save(vae.state_dict(),
                       os.path.join(cfg.results_dir, "best_vae.pth"))
            save_latent_tsne(
                Z=Z_train, Y=Y_train, gmm=gmm,
                out_path=os.path.join(cfg.results_dir,
                                      f"tsne_best_epoch_{epoch:03d}_train.png"),
                title=f"Train t-SNE (Best Ep {epoch}, Acc={test_acc:.3f})"
            )
            save_latent_tsne(
                Z=Z_test, Y=Y_test, gmm=gmm,
                out_path=os.path.join(cfg.results_dir,
                                      f"tsne_best_epoch_{epoch:03d}_test.png"),
                title=f"Test t-SNE (Best Ep {epoch}, Acc={test_acc:.3f})"
            )

    R = lambda x: round(float(x), 3)
    row = {
        "epoch": epoch,
        "train_refit_accuracy": R(train_acc),
        "train_refit_nmi": R(train_nmi),
        "train_refit_ari": R(train_ari),
        "test_refit_accuracy": R(test_acc),
        "test_refit_nmi": R(test_nmi),
        "test_refit_ari": R(test_ari),
        "train_stage_accuracy": R(stage_train_acc),
        "train_stage_nmi": R(stage_train_nmi),
        "train_stage_ari": R(stage_train_ari),
        "test_stage_accuracy": R(stage_test_acc),
        "test_stage_nmi": R(stage_test_nmi),
        "test_stage_ari": R(stage_test_ari),
        "best_test_refit_accuracy_so_far": R(best_test_acc),
        "best_test_stage_accuracy_so_far": R(best_test_stage_acc),
        "conf_thresh": R(cfg.conf_thresh),
        "gamma_power": R(cfg.gamma_power),
        "atlas_robust": str(cfg.atlas_robust),
        # Label-free clustering metrics (silhouette / DB / size entropy)
        **lf_train,
        **lf_test,
    }

    return row, best_test_acc, best_test_stage_acc, best_test_recon, purity_csv
