#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for t-VaDE Mode B (Principled 3-KL ELBO).

Supports dynamic w_elbo scheduling (linear warm-up from w_elbo_start to w_elbo_end).
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CFG
from models import StudentTVaDE
from data import create_balanced_mnist_subset, create_data_loaders
from utils import (
    set_seed, ensure_dir, write_csv, refit_gmm, overwrite_vae_prior_from_refit_gmm,
    count_params, compute_flops, write_meta_json,
)
from training import train_epoch, compute_test_loss, evaluate_and_save


def main():
    cfg = CFG()
    set_seed(cfg.seed)

    ensure_dir(cfg.results_dir)
    ensure_dir(os.path.join(cfg.results_dir, "cluster_samples"))
    ensure_dir(os.path.join(cfg.results_dir, "atlas_A"))
    ensure_dir(os.path.join(cfg.results_dir, "atlas_refit_means"))
    ensure_dir(os.path.join(cfg.results_dir, "st_means_decoded"))

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading and preparing data...")
    subset_data = create_balanced_mnist_subset(cfg.num_samples_per_class)
    train_loader, test_loader, train_loader_eval, test_loader_eval = create_data_loaders(
        subset_data, batch_size=cfg.batch_size, train_split=0.8
    )

    vae = StudentTVaDE(
        cfg.latent_dim, cfg.num_clusters, nu_init=cfg.nu_init
    ).to(device)

    opt_name = "AdamW"
    optimizer = optim.AdamW(vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- One-shot compute profile ---
    probe_x, _ = subset_data[0]
    input_shape = tuple(probe_x.shape)
    params_total = count_params(vae)
    try:
        flops_per_sample = compute_flops(vae, input_shape, device=device)
    except Exception as e:
        print(f"[warn] FLOPs computation failed: {e}")
        flops_per_sample = None
    write_meta_json(
        os.path.join(cfg.results_dir, "meta.json"),
        algo="v4.1_tVaDE_ModeB",
        input_shape=list(input_shape),
        latent_dim=cfg.latent_dim,
        num_clusters=cfg.num_clusters,
        batch_size=cfg.batch_size,
        epoch_num=cfg.epoch_num,
        lr=cfg.lr,
        optimizer=opt_name,
        params_total=params_total,
        flops_per_sample=flops_per_sample,
        nu_init=cfg.nu_init,
        w_elbo=cfg.w_elbo,
    )
    print(f"Model params: {params_total:,}   FLOPs/sample: {flops_per_sample}")

    best_test_acc = 0.0
    best_test_stage_acc = 0.0
    best_test_recon = float("inf")    # used for best_vae.pth (recon-min criterion)
    metrics_rows = []
    total_train_time_sec = 0.0

    # Metrics CSV with descriptive suffix
    metrics_suffix = f"{datetime.now().strftime('%Y%m%d')}_{opt_name}_lr{cfg.lr}_bs{cfg.batch_size}"
    metrics_csv_path = os.path.join(cfg.results_dir, f"metrics_{metrics_suffix}.csv")

    print(f"\nStarting t-VaDE training for {cfg.epoch_num} epochs...")
    print(f"  Mode: Principled 3-KL ELBO, refit at epoch 20")
    print(f"  w_elbo: {cfg.w_elbo} (fixed), nu_init: {cfg.nu_init}, lr: {cfg.lr}, batch_size: {cfg.batch_size}, optimizer: {opt_name}")
    print("=" * 80)

    for epoch in range(cfg.epoch_num):
        # 1) Train
        train_losses = train_epoch(vae, train_loader, optimizer, cfg, device)

        # 1b) Compute test loss
        test_losses = compute_test_loss(vae, test_loader, cfg, device)

        # 2) Refit sklearn GMM (for evaluation)
        from utils.clustering import collect_embeddings_and_images
        Z_train, Y_train, X_train_cpu = collect_embeddings_and_images(vae, train_loader_eval, device)

        gmm = refit_gmm(
            Z_train, cfg.num_clusters,
            covariance_type=cfg.refit_covariance_type,
            n_init=cfg.refit_n_init
        )

        # Overwrite VAE prior from refit GMM at epoch 20 only
        # (nu_c is NOT overwritten — it keeps learning on its own)
        if (epoch + 1) == 20:
            overwrite_vae_prior_from_refit_gmm(
                vae=vae, gmm=gmm, eps=cfg.eps, ema_alpha=1.0
            )
            print(f"  -> Overwrote VAE prior (means, logvars, pi) from refit GMM at epoch {epoch + 1}")
            print(f"     nu_c values: {vae.nu_c.detach().cpu().numpy().round(2)}")

        # 3) Evaluate and save (best_vae.pth uses recon-min criterion w/ warmup)
        N_test = len(test_loader.dataset)
        test_recon_per_sample = test_losses["recon"] / N_test
        row, best_test_acc, best_test_stage_acc, best_test_recon, purity_csv = evaluate_and_save(
            vae, train_loader_eval, test_loader_eval, gmm,
            cfg, epoch + 1, device, best_test_acc, best_test_stage_acc,
            test_recon=test_recon_per_sample, best_test_recon=best_test_recon,
        )

        R = lambda x: round(float(x), 3)
        N_train = len(train_loader.dataset)
        N_test = len(test_loader.dataset)
        row["train_loss"] = R(train_losses["total"] / N_train)
        row["train_loss_recon"] = R(train_losses["recon"] / N_train)
        row["train_loss_elbo"] = R(train_losses["elbo"] / N_train)
        row["test_loss"] = R(test_losses["total"] / N_test)
        row["test_loss_recon"] = R(test_losses["recon"] / N_test)
        row["test_loss_elbo"] = R(test_losses["elbo"] / N_test)

        nu_vals = vae.nu_c.detach().cpu().numpy()
        row["nu_mean"] = R(nu_vals.mean())
        row["nu_min"] = R(nu_vals.min())
        row["nu_max"] = R(nu_vals.max())

        # --- compute-efficiency profile ---
        prof = train_losses.get("profile", {})
        row["epoch_time_sec"] = prof.get("epoch_time_sec")
        row["step_time_ms_mean"] = prof.get("step_time_ms_mean")
        row["step_time_ms_std"] = prof.get("step_time_ms_std")
        row["peak_gpu_mem_mb"] = prof.get("peak_gpu_mem_mb")
        total_train_time_sec += float(prof.get("epoch_time_sec") or 0.0)
        row["total_train_time_sec_so_far"] = round(total_train_time_sec, 3)

        row = {
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "train_loss_recon": row["train_loss_recon"],
            "train_loss_elbo": row["train_loss_elbo"],
            "test_loss": row["test_loss"],
            "test_loss_recon": row["test_loss_recon"],
            "test_loss_elbo": row["test_loss_elbo"],
            "nu_mean": row["nu_mean"],
            "nu_min": row["nu_min"],
            "nu_max": row["nu_max"],
            **{k: v for k, v in row.items() if k not in [
                "epoch", "train_loss", "train_loss_recon", "train_loss_elbo",
                "test_loss", "test_loss_recon", "test_loss_elbo",
                "nu_mean", "nu_min", "nu_max"
            ]}
        }

        metrics_rows.append(row)
        write_csv(metrics_rows, metrics_csv_path)

        # Print epoch summary
        print(
            f"Epoch {epoch+1:03d}/{cfg.epoch_num} | "
            f"Loss: {row['train_loss']:.3f} | "
            f"Refit: {row['test_refit_accuracy']:.3f} | "
            f"Stage: {row['test_stage_accuracy']:.3f} | "
            f"NMI: {row['test_refit_nmi']:.3f} | "
            f"nu: [{row['nu_min']:.1f}, {row['nu_max']:.1f}] | "
            f"Best(R/S): {best_test_acc:.3f}/{best_test_stage_acc:.3f}"
        )

        # Print prior parameters (3 decimal places)
        pi_vals = torch.softmax(vae.logit_pi, dim=0).detach().cpu().numpy()
        means_norm = np.linalg.norm(vae.gmm_means.detach().cpu().numpy(), axis=1)
        logvar_mean = vae.gmm_logvars.detach().cpu().numpy().mean(axis=1)
        print(f"  Prior: nu={np.array2string(nu_vals, precision=3, separator=', ')}")
        print(f"         pi={np.array2string(pi_vals, precision=3, separator=', ')}")
        print(f"         |mu|={np.array2string(means_norm, precision=3, separator=', ')}")
        print(f"         logvar_mean={np.array2string(logvar_mean, precision=3, separator=', ')}")

    print("=" * 80)
    print(f"Training complete! Results saved in: {cfg.results_dir}")
    print(f"Best test refit accuracy: {best_test_acc:.3f}")
    print(f"Best test stage accuracy: {best_test_stage_acc:.3f}")
    print(f"Final nu_c: {vae.nu_c.detach().cpu().numpy().round(2)}")
    print(f"Total training time: {total_train_time_sec:.2f} sec")

    # Persist final total training time into meta.json
    import json
    meta_path = os.path.join(cfg.results_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["total_train_time_sec"] = round(total_train_time_sec, 3)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Plot training curves
    from plot_training_curves import plot_training_curves
    plot_training_curves(metrics_csv_path, out_dir=cfg.results_dir)


if __name__ == "__main__":
    main()
