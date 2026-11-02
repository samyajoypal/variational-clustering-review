from __future__ import annotations

import bz2
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.paper1_review import data


def write_idx(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        stream.write(bytes([0, 0, 8, values.ndim]))
        stream.write(struct.pack(">" + "I" * values.ndim, *values.shape))
        stream.write(values.astype(np.uint8).tobytes())


def image_fixture(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    folder = raw / "mnist"
    labels = np.tile(np.arange(10, dtype=np.uint8), 2)
    write_idx(folder / "train-images-idx3-ubyte.gz", np.arange(20 * 28 * 28, dtype=np.uint8).reshape(20, 28, 28))
    write_idx(folder / "train-labels-idx1-ubyte.gz", labels)
    write_idx(folder / "t10k-images-idx3-ubyte.gz", np.zeros((10, 28, 28), dtype=np.uint8))
    write_idx(folder / "t10k-labels-idx1-ubyte.gz", np.arange(10, dtype=np.uint8))
    original = data.resources("mnist")
    fixture_resources = [data.Resource(r.filename, r.urls, data.file_hash(folder / r.filename, "md5"))
                         for r in original]
    monkeypatch.setattr(data, "resources", lambda _: fixture_resources)
    monkeypatch.setitem(data.SPECS, "mnist", {"train": 20, "test": 10, "shape": (1, 28, 28), "n_clusters": 10})
    return raw


def test_image_artifacts_preserve_official_test_and_separate_labels(tmp_path, monkeypatch):
    raw = image_fixture(tmp_path, monkeypatch)
    path = data.prepare_dataset("mnist", tmp_path / "prepared", raw)
    manifest = json.loads(path.read_text())
    assert manifest["sample_counts"] == {"train": 18, "validation": 2, "test": 10}
    assert manifest["n_features"] == 784 and manifest["n_clusters"] == 10
    sets = []
    for split in data.SPLITS:
        inputs = data.load_inputs(path, split)
        labels = data.load_evaluation_labels(path, split)
        assert set(inputs) == {"x", "ids"}
        assert set(labels) == {"labels", "ids"}
        np.testing.assert_array_equal(inputs["ids"], labels["ids"])
        assert inputs["x"].dtype == np.float32
        assert inputs["x"].shape[1:] == (1, 28, 28)
        assert inputs["x"].min() >= 0 and inputs["x"].max() <= 1
        sets.append(set(inputs["ids"]))
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert data.load_inputs(path, "test")["ids"].tolist() == [f"mnist:official_test:{i}" for i in range(10)]
    assert all(r["integrity"] == "published_md5_verified" for r in manifest["raw_sources"])
    with pytest.raises(FileExistsError):
        data.prepare_dataset("mnist", tmp_path / "prepared", raw)


def test_input_reader_detects_tampering(tmp_path, monkeypatch):
    raw = image_fixture(tmp_path, monkeypatch)
    manifest = data.prepare_dataset("mnist", tmp_path / "prepared", raw)
    with (manifest.parent / "inputs/train.npz").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_inputs(manifest, "train")


def test_inventory_does_not_create_files_and_corrupt_cache_is_not_overwritten(tmp_path, monkeypatch):
    empty = tmp_path / "absent"
    assert not data.inventory_dataset("usps", empty)["ready"]
    assert not empty.exists()
    raw = image_fixture(tmp_path, monkeypatch)
    target = raw / "mnist/train-labels-idx1-ubyte.gz"
    target.write_bytes(b"bad bytes")
    before = target.read_bytes()
    with pytest.raises(ValueError, match="MD5 mismatch"):
        data.download_dataset("mnist", raw)
    assert target.read_bytes() == before


def test_idx_checks_header_shape_and_payload(tmp_path):
    path = tmp_path / "labels.gz"
    write_idx(path, np.arange(10, dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        data.read_idx(path, (9,))
    with gzip.open(path, "wb") as stream:
        stream.write(b"\x00\x00\x08\x01" + struct.pack(">I", 10) + b"\x01\x02")
    with pytest.raises(ValueError, match="payload"):
        data.read_idx(path, (10,))
    with gzip.open(path, "wb") as stream:
        stream.write(b"\x00\x00\x0b\x01" + struct.pack(">I", 10))
    with pytest.raises(ValueError, match="header"):
        data.read_idx(path, (10,))


@pytest.mark.parametrize("labels", [np.array([0., 1.]), np.array([[0, 1]]),
                                   np.array([0, 2]), np.array([-1, 1])])
def test_bad_labels_rejected(labels):
    with pytest.raises(ValueError):
        data.validate_labels(labels, 2, 2)


def test_sparse_benchmark_class_domain_is_rejected():
    with pytest.raises(ValueError, match="sparse"):
        data.validate_labels(np.array([0, 0, 2, 2]), 4, 3)


def test_usps_preserves_float_scaling_and_rejects_missing_coordinates(tmp_path):
    path = tmp_path / "usps.bz2"
    values = np.linspace(-1, 1, 256)
    row = "10 " + " ".join(f"{i+1}:{x:.8f}" for i, x in enumerate(values)) + "\n"
    with bz2.open(path, "wt") as stream:
        stream.write(row)
    x, y = data.read_usps(path, 1)
    np.testing.assert_allclose(x.ravel(), (values + 1) / 2, atol=1e-7)
    assert y.tolist() == [9] and x.shape == (1, 1, 16, 16)
    with bz2.open(path, "wt") as stream:
        stream.write(" ".join(row.split()[:-1]) + "\n")
    with pytest.raises(ValueError, match="width"):
        data.read_usps(path, 1)


def test_usps_rejects_duplicate_coordinate_and_nonfinite_value(tmp_path):
    path = tmp_path / "usps.bz2"
    fields = [f"{i}:0" for i in range(1, 257)]
    fields[-1] = "255:0"
    with bz2.open(path, "wt") as stream:
        stream.write("1 " + " ".join(fields) + "\n")
    with pytest.raises(ValueError, match="coordinates"):
        data.read_usps(path, 1)
    fields[-1] = "256:nan"
    with bz2.open(path, "wt") as stream:
        stream.write("1 " + " ".join(fields) + "\n")
    with pytest.raises(ValueError, match="range"):
        data.read_usps(path, 1)


def test_splits_do_not_depend_on_labels_and_have_agreed_sizes():
    split = data.split_indices(10000, seed=1235, test_fraction=.2)
    assert {key: len(value) for key, value in split.items()} == {"train": 7200, "validation": 800, "test": 2000}
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(10000))
    for key, value in split.items():
        np.testing.assert_array_equal(value, data.split_indices(10000, seed=1235, test_fraction=.2)[key])


def reuters_fixture(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "reuters10k"
    raw.mkdir(parents=True)
    # Forty eligible documents, plus two documents excluded by root-label rules.
    selected = np.asarray(range(1, 41))[np.random.RandomState(1234).permutation(40)[:20]]
    indices = data.split_indices(20, seed=1235, test_fraction=.2)
    held_out = set(selected[np.concatenate([indices["test"], indices["validation"]])].tolist())
    qrels = []
    cats = list(data.ROOT_CATEGORIES)
    for did in range(1, 41):
        qrels.append(f"{cats[(did-1)%4]} {did} 1\n")
    qrels += ["CCAT 41 1\n", "GCAT 41 1\n", "C11 42 1\n"]
    (raw / "rcv1-v2.topics.qrels").write_text("".join(qrels))
    groups = np.array_split(np.arange(1, 43), 5)
    for filename, group in zip(data.TOKEN_FILES, groups):
        with gzip.open(raw / filename, "wt") as stream:
            for did in group:
                text = "alpha alpha beta gamma delta epsilon"
                if did in held_out:
                    text += " testonly"
                stream.write(f".I {did}\n.W\n{text}\n")
    fixture_resources = [data.Resource(f, ("https://fixture.invalid/" + f,), allow_plain=True)
                         for f in data.TOKEN_FILES + ["rcv1-v2.topics.qrels.gz"]]
    monkeypatch.setattr(data, "resources", lambda _: fixture_resources)
    monkeypatch.setitem(data.SPECS, "reuters10k", {"selected": 20, "eligible": 40, "documents": 42,
                                                 "n_features": 5, "n_clusters": 4})
    return raw.parent, selected


def test_reuters_selects_before_training_only_tfidf_and_preserves_ids(tmp_path, monkeypatch):
    raw, selected = reuters_fixture(tmp_path, monkeypatch)
    manifest_path = data.prepare_dataset("reuters10k", tmp_path / "prepared", raw)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sample_counts"] == {"train": 15, "validation": 1, "test": 4}
    assert json.loads((manifest_path.parent / "transforms/selected_document_ids.json").read_text()) == selected.tolist()
    vocab = json.loads((manifest_path.parent / "transforms/vocabulary.json").read_text())
    assert "testonly" not in vocab  # Held-out-only term never enters fit vocabulary.
    np.testing.assert_allclose(np.load(manifest_path.parent / "transforms/idf.npy"), 1.0)
    for split in data.SPLITS:
        inputs = data.load_inputs(manifest_path, split)
        assert all(int(x.split(":")[1]) in selected for x in inputs["ids"])
        np.testing.assert_allclose(np.linalg.norm(inputs["x"], axis=1), np.sqrt(5), rtol=1e-6)
    assert manifest["transform"]["fit_ids_sha256"] == manifest["splits"]["train"]["ids_sha256"]
    assert manifest["historical_transductive_reproduction"] is False


def test_rcv1_flushes_final_document_and_checks_headers(tmp_path):
    path = tmp_path / "tokens.dat"
    path.write_text(".I 11\n.W\nalpha\n.I 12\n.W\nbeta")
    assert list(data.iter_rcv1_documents([path])) == [(11, "alpha"), (12, "beta")]
    assert list(data.iter_rcv1_documents([path], keep_text=False)) == [(11, ""), (12, "")]
    path.write_text(".I 11\nalpha\n")
    with pytest.raises(ValueError, match="Unexpected"):
        list(data.iter_rcv1_documents([path]))


def test_reuters_duplicate_ids_fail_before_vectorization(tmp_path, monkeypatch):
    raw, _ = reuters_fixture(tmp_path, monkeypatch)
    path = raw / "reuters10k" / data.TOKEN_FILES[-1]
    with gzip.open(path, "at") as stream:
        stream.write(".I 1\n.W\nduplicate\n")
    with pytest.raises(ValueError, match="Duplicate RCV1"):
        data.prepare_dataset("reuters10k", tmp_path / "prepared", raw)


def test_root_filter_uses_only_four_roots_and_rejects_duplicate_annotation(tmp_path):
    path = tmp_path / "qrels"
    path.write_text("CCAT 1 1\nC11 1 1\nGCAT 2 1\nMCAT 2 1\nECAT 3 1\n")
    assert data.read_root_labels(path) == {1: 0, 3: 3}
    path.write_text("CCAT 1 1\nCCAT 1 1\n")
    with pytest.raises(ValueError, match="Duplicate root"):
        data.read_root_labels(path)


def test_atomic_download_retries_and_never_replaces_existing_target(tmp_path, monkeypatch):
    import io

    class Response(io.BytesIO):
        headers = {"Content-Length": "7"}

        def geturl(self):
            return "https://fixture.invalid/data"

    calls = []

    def fake_open(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary")
        return Response(b"payload")

    monkeypatch.setattr(data, "urlopen", fake_open)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    resource = data.Resource("raw.gz", ("https://fixture.invalid/data",), expected_size=7)
    target = tmp_path / resource.filename
    data._download(resource, target, retries=2)
    assert target.read_bytes() == b"payload" and len(calls) == 2
    assert not list(tmp_path.glob("*.part-*"))
    receipt = json.loads((tmp_path / "raw.gz.download.json").read_text())
    assert receipt["sha256"] == data.file_hash(target)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since"):
        data._raw_record(target, resource)


def test_failed_checksum_download_leaves_no_published_file(tmp_path, monkeypatch):
    import io

    class Response(io.BytesIO):
        headers = {}

        def geturl(self):
            return "https://fixture.invalid/data"

    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: Response(b"bad"))
    resource = data.Resource("raw.gz", ("https://fixture.invalid/data",), md5="0" * 32)
    with pytest.raises(RuntimeError, match="MD5"):
        data._download(resource, tmp_path / "raw.gz", retries=1)
    assert not (tmp_path / "raw.gz").exists()
    assert not list(tmp_path.glob("*.part-*"))
