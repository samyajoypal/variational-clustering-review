"""Fixed matched-panel candidate; source-native and release gates are separate."""
from __future__ import annotations
import hashlib
import json
import math
from experiments.paper1_review.workload import DATASETS, METHODS, SEEDS

PROTOCOL_VERSION = "paper1-air-matched-final-v1"
CONTAMINATION_CONDITIONS = ("gaussian_training_sd0.3_ratio0.1", "semantic_training_ratio0.1")
CONTAMINATION_METHODS = ("vae_gmm", "pro_t_matched", "student_q_student_prior",
                         "structured_q_gaussian_prior", "structured_q_student_prior")
INFERENCE_CONDITIONS = ("gaussian_sd0.1", "gaussian_sd0.2", "gaussian_sd0.3", "dropout0.25", "dropout0.50", "dropout0.75")
ALIASES = {"vade_matched": "vade", "dec_matched": "dec", "idec_matched": "idec", "pro_t_matched": "pro_t_vae"}
PROTOCOL = {
    "protocol_version": PROTOCOL_VERSION,
    "release_status": "implementation_candidate_requires_reference_and_resource_gates",
    "datasets": DATASETS, "methods": METHODS, "seeds": SEEDS,
    "native_reproduction": False, "source_native_fits": "separate 12 fits; not implemented by this runner",
    "architecture": {"hidden_dims": [500, 500, 2000], "latent_dim": 10,
                     "decoder": {"images": "sigmoid", "reuters10k": "identity"}},
    "batch_size": 256, "pretrain_epochs": 100, "training_epochs": 200,
    "budget": "epochs * ceil(clean optimization count / batch_size) updates; same budget for appended contamination",
    "sampler": "seeded permutations of declared training pool; final short batch retained; restart permutation after each pass",
    "initialization": "fresh AE and train-only diagonal GMM shared across methods within dataset/condition/seed",
    "ae": {"optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
           "weight_decay": 0., "loss": "feature-mean MSE", "gradient_clip": 20.},
    "train": {"optimizer": "Adam", "lr": .0003, "betas": [.9, .999], "eps": 1e-8,
              "weight_decay": 0., "gradient_clip": 20., "lr_schedule": "constant",
              "selection": "fixed final update; no labels, early stopping or best-validation checkpoint"},
    "gmm": {"covariance_type": "diag", "n_init": 5, "reg_covar": .001, "max_iter": 500,
            "tol": .001, "selection": "highest train likelihood among fixed starts; must converge"},
    "objective": {"observation_variance": .05, "beta": 1., "idec_gamma": .1,
                  "reconstruction": "feature-summed Gaussian for generative methods; IDEC feature-mean MSE",
                  "prior_df_bounds": [3., 30.], "recognition_df_bounds": [3., 100.], "df_init": 8.,
                  "scale_bounds": [.0001, 100.], "log_variance_bounds": [-9., 2.],
                  "bounded_df_interpretation": "finite bounded Student family; boundary hits reported, not Gaussian infinity"},
    "structured": {"site_mode": "bounded_softplus", "initial_precision": 10., "max_precision": math.exp(9),
                   "training_nodes": 128, "prediction_nodes": 128, "bounds": [-24., 12.],
                   "reference_nodes": 512, "reference_bounds": [-32., 16.],
                   "refined_reference_nodes": 1024, "refined_reference_bounds": [-40., 20.],
                   "log_normalizer_atol": .001, "responsibility_atol": .0001, "kl_atol": .001,
                   "reference_tolerance_factor": .1,
                   "claim": "attained-parameter numerical refinement, not certified continuous ELBO"},
    "training_samples": 8, "prediction_samples": 64,
    "dec_target_refresh": "every ceil(clean train count/batch size) optimizer updates, from full declared training pool",
    "diagnostic_epochs": [0, 1, 10, 25, 50, 100, 150, 200],
    "checkpoint_epochs": [0, 50, 100, 150, 200],
    "rolling_checkpoint_every_epochs": 10,
    "common_gradient_batch_size": 32,
    "gradient_diagnostics": "separate fixed-seed batch; reconstruction, prior cross entropy and negative joint entropy where defined; no optimizer step",
    "diagnostic_rng": "forked RNG; cannot perturb optimization sample stream",
    "labels_loaded": False, "evaluation": "separate prediction phase; no evaluation labels are loaded",
    "inference_corruptions": {"conditions": list(INFERENCE_CONDITIONS), "datasets": DATASETS[:3],
                             "paired_generation": "seed plus SHA256(dataset,condition); generate in sorted-ID order, restore input order",
                             "noise": "Gaussian clipped to [0,1]; dropout sets pixels to zero; no retraining"},
    "failure_policy": "preserve failed attempt and checkpoint; never drop seeds or adapt budgets to scores",
    "calibration_basis": "fresh A30 synthetic timing plus label-blind v1/v2 engineering; 100/200 chooses approved ceilings, not tuned superiority",
    "known_behavior": "all 633 updates clipped at norm 20 in each of four v2 structured development runs; record component/preclip norms, do not silently retune",
    "release_gates": ["12 separate source-native/reference ports and checks", "full-size Reuters identity-decoder numerical/resource validation",
                      "whole-study resource/storage budget and hardware policy", "attained quadrature checks on clean/corrupted inputs"],
}


def protocol_digest(protocol=PROTOCOL):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def update_budget(clean_count: int, phase: str, protocol=PROTOCOL) -> int:
    if clean_count < 1 or phase not in {"pretrain", "training"}:
        raise ValueError("Positive clean count and pretrain/training phase required")
    return math.ceil(clean_count / protocol["batch_size"]) * protocol[f"{phase}_epochs"]


def validate_run(dataset, method, seed, condition="clean"):
    if dataset not in DATASETS or seed not in SEEDS or method not in METHODS:
        raise ValueError("Outside the fixed matched dataset/method/seed allocation")
    if condition != "clean" and (condition not in CONTAMINATION_CONDITIONS or dataset not in DATASETS[:2] or method not in CONTAMINATION_METHODS):
        raise ValueError("Outside the approved contamination panel")
