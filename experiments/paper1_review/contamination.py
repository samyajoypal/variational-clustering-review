"""Versioned, label-blind appended contaminants for the approved Paper 1 panel.

Inputs contain only x/ids. Origin information is saved separately for evaluation.
No test inputs, labels, fitted features, or model weights are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .data import _ids_hash, file_hash, load_inputs
from .provenance import timestamp, write_json

PROTOCOL = "paper1-review-training-contamination-v1"
CONDITIONS = ("gaussian_training_sd0.3_ratio0.1", "semantic_training_ratio0.1")


def prepare(clean_manifest: Path, output: Path, *, condition: str, seed: int,
            source_manifest: Path | None = None) -> Path:
    if condition not in CONDITIONS or seed not in range(101, 106):
        raise ValueError("Use a planned contamination condition and seed 101–105")
    clean_manifest = Path(clean_manifest).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing data views; choose a new output directory")
    clean = json.loads(clean_manifest.read_text())
    if clean["dataset"] not in {"mnist", "fashion_mnist"}:
        raise ValueError("Contamination is approved only for MNIST and Fashion-MNIST")
    pools = {split: load_inputs(clean_manifest, split) for split in ("train", "validation")}
    if set(pools["train"]["ids"]) & set(pools["validation"]["ids"]):
        raise ValueError("Clean split IDs overlap")
    if condition.startswith("semantic"):
        if source_manifest is None:
            raise ValueError("Semantic contamination requires the opposite dataset's manifest")
        source_manifest = Path(source_manifest).resolve()
        source = json.loads(source_manifest.read_text())
        opposite = {"mnist": "fashion_mnist", "fashion_mnist": "mnist"}[clean["dataset"]]
        if source["dataset"] != opposite:
            raise ValueError("Semantic contaminants must come from the other approved dataset")
        eligible = load_inputs(source_manifest, "train")
    else:
        if source_manifest is not None:
            raise ValueError("Gaussian contamination uses the clean training pool")
        source_manifest, source, eligible = clean_manifest, clean, pools["train"]
    clean_count = sum(len(v["x"]) for v in pools.values())
    count = clean_count // 10
    if count < 1 or count > len(eligible["x"]):
        raise ValueError("Insufficient source pool for the approved 10% ratio")
    for pool in [*pools.values(), eligible]:
        if (pool["x"].dtype != np.float32 or not np.isfinite(pool["x"]).all()
                or pool["x"].min() < 0 or pool["x"].max() > 1
                or pool["x"].shape[1:] != pools["train"]["x"].shape[1:]):
            raise ValueError("Expected aligned finite image inputs in [0,1]")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261028, 1]))
    selected = rng.choice(len(eligible["x"]), size=count, replace=False)
    corrupt = eligible["x"][selected].copy()
    if condition.startswith("gaussian"):
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 20261028, 2]))
        corrupt = np.clip(corrupt + noise_rng.normal(0, .3, corrupt.shape), 0, 1).astype(np.float32)
    source_ids = eligible["ids"][selected]
    # Opaque IDs avoid exposing the origin to a training consumer.
    ids = np.asarray(["aug:" + hashlib.sha256(
        f"{PROTOCOL}|{clean['dataset']}|{condition}|{seed}|{sid}".encode()).hexdigest()
        for sid in source_ids])
    is_validation = np.asarray([int(hashlib.sha256(
        ("partition|" + sid).encode()).hexdigest()[:16], 16) < (2**64 // 10) for sid in ids])
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "inputs").mkdir(); (output / "evaluation").mkdir(); (output / "source").mkdir()
        for name in ("contamination.py", "data.py", "provenance.py"):
            shutil.copyfile(Path(__file__).with_name(name), output / "source" / name)
        splits, counts = {}, {}
        for split in ("train", "validation"):
            mask = is_validation if split == "validation" else ~is_validation
            original = pools[split]
            all_ids = np.concatenate([original["ids"], ids[mask]])
            x = np.concatenate([original["x"], corrupt[mask]])
            if len(np.unique(all_ids)) != len(all_ids):
                raise ValueError("Duplicate augmented observation IDs")
            # Shuffle the pool so position does not disclose the appended origin.
            order_rng = np.random.default_rng(np.random.SeedSequence([seed, 20261028, 3 if split == "train" else 4]))
            order = order_rng.permutation(len(x))
            relative = f"inputs/{split}.npz"
            np.savez_compressed(output / relative, x=x[order], ids=all_ids[order])
            origin = np.concatenate([np.zeros(len(original["x"]), dtype=np.bool_), np.ones(mask.sum(), dtype=np.bool_)])
            parents = np.concatenate([original["ids"], source_ids[mask]])
            np.savez_compressed(output / f"evaluation/{split}_origin.npz", ids=all_ids[order],
                                is_contaminant=origin[order], source_ids=parents[order])
            splits[split] = dict(inputs=relative, n_samples=len(x), input_shape=list(x.shape), ids_sha256=_ids_hash(all_ids[order]))
            counts[split] = dict(clean=len(original["x"]), contaminants=int(mask.sum()),
                                 contamination_fraction=float(mask.sum() / len(x)))
        manifest = dict(schema_version="paper1-review-training-view-v1", protocol_version=PROTOCOL,
                        status="completed", created_utc=timestamp(), dataset=clean["dataset"], condition=condition,
                        seed=seed, clean_manifest_sha256=file_hash(clean_manifest),
                        clean_manifest=str(clean_manifest), source_manifest=str(source_manifest),
                        source_manifest_sha256=file_hash(source_manifest), source_dataset=source["dataset"],
                        source_split="train", clean_train_count=len(pools["train"]["x"]),
                        n_features=clean["n_features"], n_clusters=clean["n_clusters"], splits=splits,
                        realized_counts=counts, appended_to_clean_ratio=count/clean_count,
                        labels_loaded=False, test_inputs_loaded=False, model_weights_loaded=False,
                        partition_rule="SHA256 of opaque augmentation ID, threshold 0.1; original clean partitions retained",
                        sampling="source training split only, without replacement; independent NumPy seed streams",
                        validation_note="Gaussian validation contaminants can be noisy copies of clean training images; validation is for diagnostics only, never checkpoint selection or reported test accuracy.",
                        environment={"numpy": np.__version__},
                        artifact_sha256={str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()})
        write_json(output / "manifest.json", manifest)
        return output / "manifest.json"
    except Exception as error:
        write_json(output / "FAILED.json", {"status": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.manifest, args.output, condition=args.condition, seed=args.seed, source_manifest=args.source_manifest))


if __name__ == "__main__":
    main()
