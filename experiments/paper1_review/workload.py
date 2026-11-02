"""Materialize the approved workload without submitting or running any fit."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

from experiments.paper1_review.provenance import ROOT, write_json

METHODS = ["vade_matched", "dec_matched", "idec_matched", "vae_gmm", "pro_t_matched",
           "gaussian_q_gaussian_prior", "gaussian_q_student_prior", "student_q_student_prior",
           "structured_q_gaussian_prior", "structured_q_student_prior"]
DATASETS = ["mnist", "fashion_mnist", "usps", "reuters10k"]
SEEDS = [101, 102, 103, 104, 105]


def plan():
    fits = []
    for dataset, method, seed in product(DATASETS, METHODS, SEEDS):
        fits.append(dict(run_id=f"clean/{dataset}/{method}/seed{seed}", track="matched_clean",
                         dataset=dataset, method=method, seed=seed, status="planned"))
    contamination_methods = ["vae_gmm", "pro_t_matched", "student_q_student_prior",
                             "structured_q_gaussian_prior", "structured_q_student_prior"]
    for dataset, method, condition, seed in product(DATASETS[:2], contamination_methods,
                                                   ["gaussian_training_sd0.3_ratio0.1", "semantic_training_ratio0.1"], SEEDS):
        fits.append(dict(run_id=f"contaminated/{dataset}/{method}/{condition}/seed{seed}",
                         track="matched_training_contamination", dataset=dataset, method=method,
                         condition=condition, seed=seed, status="planned",
                         paired_clean_run=f"clean/{dataset}/{method}/seed{seed}"))
    for method, seed in product(["vade", "dec", "idec", "pro_t"], [901, 902, 903]):
        fits.append(dict(run_id=f"native_validation/mnist/{method}/seed{seed}", track="native_validation",
                         dataset="mnist", method=method, seed=seed, status="planned"))
    evaluations = []
    conditions = ["gaussian_sd0.1", "gaussian_sd0.2", "gaussian_sd0.3", "dropout0.25", "dropout0.50", "dropout0.75"]
    for dataset, method, seed, condition in product(DATASETS[:3], METHODS, SEEDS, conditions):
        evaluations.append(dict(run_id=f"corruption/{dataset}/{method}/seed{seed}/{condition}",
                                trained_run=f"clean/{dataset}/{method}/seed{seed}", condition=condition, status="planned"))
    simulations = []
    tail_settings = [("t5", sd, m) for sd, m in product([.5, 2., 10.], [0, 20])]
    tail_settings += [("gaussian", 2., m) for m in [0, 20]]
    for setting, seed, family in product(tail_settings, range(1101, 1106),
                                         ["gaussian_q_student_prior", "student_q_student_prior", "structured_q_student_prior"]):
        simulations.append(dict(module="S2_primary", truth=setting[0], noise_sd=setting[1], nuisance_dimension=setting[2],
                                seed=seed, family=family, n_fit=512, n_heldout=4096, starts=2, status="planned"))
    for sd, seed, family in product([.5, 2.], range(1101, 1106),
                                    ["independent_gaussian_gamma", "coupled_gaussian_gamma", "full_covariance_gaussian"]):
        simulations.append(dict(module="S2_anchor", truth="t5", noise_sd=sd, nuisance_dimension=5,
                                seed=seed, family=family, n_fit=512, n_heldout=4096, starts=2, status="planned"))
    allocation = [("t3_t30", sd, m) for sd, m in product([.5, 2.], [0, 100])]
    allocation += [(truth, .5, m) for truth, m in product(["t5_t5", "gaussian_gaussian"], [0, 100])]
    for setting, seed, family in product(allocation, range(1201, 1206), ["gaussian", "direct_student"]):
        simulations.append(dict(module="S3", truth=setting[0], noise_sd=setting[1], nuisance_dimension=setting[2],
                                seed=seed, family=family, n_fit=2000, starts=2, status="planned"))
    result = {
        "protocol_version": "paper1-review-20261027-v1",
        "status": "approved_design_pending_implementation_and_resource_freeze",
            "reuse_previous_empirical_outputs": False,
        "counts": {"real_data_fits": len(fits), "real_data_by_track": dict(Counter(x["track"] for x in fits)),
                   "inference_corruption_evaluations": len(evaluations), "optimized_simulation_configurations": len(simulations),
                   "simulation_by_module": dict(Counter(x["module"] for x in simulations)),
                   "optimization_starts_per_configuration": 2},
        "primary_metric": "ari",
        "mandatory_companion_metrics": ["hungarian_acc", "nmi_arithmetic", "ami_arithmetic"],
        "secondary_metrics": ["mapped_macro_f1", "mapped_balanced_accuracy", "homogeneity", "completeness", "v_measure", "per_class_recall", "occupancy"],
        "unresolved_before_final_launch": ["faithful comparator ports", "all dataset/split provenance", "method likelihood and loss definitions",
                                           "training budget and GPU concurrency", "pretraining/initialization protocol", "quadrature refinement on attained parameters"],
        "real_data_fits": fits, "corruption_evaluations": evaluations, "optimized_simulations": simulations,
        "simulation_note": "S1 deterministic curves and exact numerical references are separate; optimized configurations are not a count of all optimizer calls.",
    }
    assert len(fits) == 312 and len(evaluations) == 900 and len(simulations) == 230
    assert len(set(x["run_id"] for x in fits)) == len(fits)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/workload_manifest.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous manifest; use a new version path")
    specification = plan()
    write_json(args.output, specification)
    print(specification["counts"])
