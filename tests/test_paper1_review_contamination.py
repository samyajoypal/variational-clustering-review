import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paper1_review.contamination import CONDITIONS, prepare
from experiments.paper1_review.data import _ids_hash, file_hash, load_inputs


def fixture_dataset(root, name):
    root = root / name
    root.mkdir()
    splits = {}
    for split, count in [("train", 900), ("validation", 100)]:
        ids = np.asarray([f"{name}:{split}:{i}" for i in range(count)])
        x = np.random.default_rng(count).uniform(size=(count, 1, 2, 2)).astype(np.float32)
        relative = f"{split}.npz"
        np.savez(root / relative, x=x, ids=ids)
        splits[split] = dict(inputs=relative, n_samples=count, input_shape=list(x.shape), ids_sha256=_ids_hash(ids))
    value = dict(schema_version="paper1-review-dataset-v1", status="completed", dataset=name,
                 n_features=4, n_clusters=10, splits=splits,
                 artifact_sha256={p.name: file_hash(p) for p in root.glob("*.npz")})
    (root / "manifest.json").write_text(json.dumps(value))
    return root / "manifest.json"


@pytest.mark.parametrize("condition", CONDITIONS)
def test_fresh_paired_contamination_preserves_splits_and_separates_origin(tmp_path, condition):
    clean = fixture_dataset(tmp_path, "mnist")
    source = fixture_dataset(tmp_path, "fashion_mnist") if condition.startswith("semantic") else None
    path = prepare(clean, tmp_path / "one", condition=condition, seed=101, source_manifest=source)
    duplicate = prepare(clean, tmp_path / "two", condition=condition, seed=101, source_manifest=source)
    manifest = json.loads(path.read_text())
    assert manifest["clean_train_count"] == 900
    assert manifest["appended_to_clean_ratio"] == .1
    assert not manifest["labels_loaded"] and not manifest["test_inputs_loaded"]
    train_ids, validation_ids, corrupt_parents = set(), set(), []
    for split in ("train", "validation"):
        entry = manifest["splits"][split]
        with np.load(path.parent / entry["inputs"]) as inputs, np.load(path.parent / f"evaluation/{split}_origin.npz") as origin:
            assert set(inputs.files) == {"x", "ids"}
            assert inputs["x"].dtype == np.float32
            assert np.all((inputs["x"] >= 0) & (inputs["x"] <= 1))
            np.testing.assert_array_equal(inputs["ids"], origin["ids"])
            original = load_inputs(clean, split)
            lookup = {sid: i for i, sid in enumerate(inputs["ids"])}
            for sid, x in zip(original["ids"], original["x"]):
                np.testing.assert_array_equal(inputs["x"][lookup[sid]], x)
                assert not origin["is_contaminant"][lookup[sid]]
            mask = origin["is_contaminant"]
            corrupt_parents.extend(origin["source_ids"][mask].tolist())
            assert mask.sum() == manifest["realized_counts"][split]["contaminants"]
            (train_ids if split == "train" else validation_ids).update(inputs["ids"])
            with np.load(duplicate.parent / entry["inputs"]) as again:
                np.testing.assert_array_equal(inputs["x"], again["x"])
                np.testing.assert_array_equal(inputs["ids"], again["ids"])
    assert not train_ids & validation_ids
    assert len(corrupt_parents) == len(set(corrupt_parents)) == 100
    assert set(corrupt_parents) <= set(load_inputs(source or clean, "train")["ids"])
    assert 0 < manifest["realized_counts"]["validation"]["contaminants"] < 100
    for name, digest in manifest["artifact_sha256"].items():
        assert file_hash(path.parent / name) == digest
    with pytest.raises(FileExistsError):
        prepare(clean, path.parent, condition=condition, seed=101, source_manifest=source)


def test_reject_wrong_source_and_tampered_data(tmp_path):
    clean = fixture_dataset(tmp_path, "mnist")
    with pytest.raises(ValueError, match="other approved"):
        prepare(clean, tmp_path / "wrong", condition=CONDITIONS[1], seed=101, source_manifest=clean)
    with (clean.parent / "train.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(clean, tmp_path / "bad", condition=CONDITIONS[0], seed=101)


def test_conditions_and_seed_scope_are_fixed(tmp_path):
    with pytest.raises(ValueError, match="planned"):
        prepare(Path("absent"), tmp_path / "bad", condition=CONDITIONS[0], seed=106)
