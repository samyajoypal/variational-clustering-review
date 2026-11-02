"""Fresh, inductive Paper 1 datasets with physically separate evaluation labels.

This module reads original IDX, LIBSVM, and RCV1 token files, never model weights
or previously fitted features. Downloading is opt-in. See DATA_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen

import numpy as np


PROTOCOL_VERSION = "paper1-review-inductive-data-v1"
SPLITS = ("train", "validation", "test")
ROOT_CATEGORIES = {"CCAT": 0, "GCAT": 1, "MCAT": 2, "ECAT": 3}
TOKEN_FILES = [f"lyrl2004_tokens_test_pt{i}.dat.gz" for i in range(4)] + [
    "lyrl2004_tokens_train.dat.gz"
]
SPECS = {
    "mnist": {"train": 60000, "test": 10000, "shape": (1, 28, 28), "n_clusters": 10},
    "fashion_mnist": {"train": 60000, "test": 10000, "shape": (1, 28, 28), "n_clusters": 10},
    "usps": {"train": 7291, "test": 2007, "shape": (1, 16, 16), "n_clusters": 10},
    "reuters10k": {"selected": 10000, "eligible": 685071, "documents": 804414,
                   "n_features": 2000, "n_clusters": 4},
}
SOURCE_RECORDS = {
    "mnist": "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py",
    "fashion_mnist": "https://github.com/zalandoresearch/fashion-mnist",
    "usps": "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/usps.py",
    "reuters10k": "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm",
}


@dataclass(frozen=True)
class Resource:
    filename: str
    urls: tuple[str, ...]
    md5: str | None = None
    expected_size: int | None = None
    allow_plain: bool = False


def canonical_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    key = {"fashionmnist": "fashion_mnist", "reuters_10k": "reuters10k",
           "reuters": "reuters10k"}.get(key, key)
    if key not in SPECS:
        raise ValueError(f"Unknown dataset {name!r}; choose {list(SPECS)}")
    return key


def resources(name: str) -> list[Resource]:
    name = canonical_name(name)
    if name in {"mnist", "fashion_mnist"}:
        files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        if name == "mnist":
            digests = ["f68b3c2dcbeaaa9fbdd348bbdeb94873", "d53e105ee54ea40749a09fcbcd1e9432",
                       "9fb629c4189551a2d022fa330f9573f3", "ec29112dd5afa0611ce80d1b7f02629c"]
            bases = ("https://ossci-datasets.s3.amazonaws.com/mnist/",)
        else:
            digests = ["8d4fb7e6c68d591d4c3dfef9ec88bf0d", "25c81989df183df01b3e8a0aad5dffbe",
                       "bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310"]
            bases = ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
                     "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")
        return [Resource(f, tuple(b + f for b in bases), md5=d) for f, d in zip(files, digests)]
    if name == "usps":
        base = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
        return [Resource("usps.bz2", (base + "usps.bz2",), "ec16c51db3855ca6c91edd34d0e9b197"),
                Resource("usps.t.bz2", (base + "usps.t.bz2",), "8ea070ee2aca1ac39742fdd1ef5ed118")]
    # Original publisher-associated archive: HTTPS refused and newer JMLR paths
    # returned 404 during the 2026-10-27 availability check. No fitted features.
    base = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/"
    sizes = [44734992, 45595102, 44507510, 42052117, 5108963]
    return [Resource(f, (base + "a12-token-files/" + f,), expected_size=s, allow_plain=True)
            for f, s in zip(TOKEN_FILES, sizes)] + [
        Resource("rcv1-v2.topics.qrels.gz", (base + "a08-topic-qrels/rcv1-v2.topics.qrels.gz",),
                 expected_size=7272130, allow_plain=True)
    ]


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(ids.tolist(), ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _directories(raw_root: Path, name: str) -> list[Path]:
    legacy = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST", "usps": "USPS",
              "reuters10k": "reuters"}[name]
    return list(dict.fromkeys([raw_root / name, raw_root / name / "raw",
                              raw_root / legacy / "raw", raw_root / legacy,
                              raw_root / "raw", raw_root]))


def _find_resource(raw_root: Path, name: str, resource: Resource) -> Path | None:
    filenames = [resource.filename]
    if resource.allow_plain:
        filenames.append(resource.filename.removesuffix(".gz"))
    for directory in _directories(raw_root, name):
        for filename in filenames:
            path = directory / filename
            if path.is_file():
                return path
    return None


def _raw_record(path: Path, resource: Resource) -> dict:
    compressed = path.name == resource.filename
    digest = file_hash(path)
    if compressed and resource.md5 and file_hash(path, "md5") != resource.md5:
        raise ValueError(f"MD5 mismatch for {path}; existing raw files are never overwritten")
    if compressed and resource.expected_size is not None and path.stat().st_size != resource.expected_size:
        raise ValueError(f"Unexpected byte count for {path}; existing raw files are never overwritten")
    receipt = path.with_name(path.name + ".download.json")
    receipt_value = json.loads(receipt.read_text()) if receipt.exists() else None
    if receipt_value and receipt_value.get("sha256") != digest:
        raise ValueError(f"Raw file changed since its download receipt: {path}")
    return {"file_id": resource.filename, "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest, "expected_md5": resource.md5 if compressed else None,
            "canonical_urls": list(resource.urls), "download_receipt": receipt_value,
            "integrity": "published_md5_verified" if compressed and resource.md5 else
            "sha256_recorded_not_independently_authenticated"}


def inventory_dataset(name: str, raw_root: str | Path) -> dict:
    """Read-only cache inventory; no directory creation, network, or unpickling."""
    name, raw_root = canonical_name(name), Path(raw_root)
    entries = []
    for resource in resources(name):
        path = _find_resource(raw_root, name, resource)
        if path is None:
            entries.append({"file_id": resource.filename, "status": "missing",
                            "canonical_urls": list(resource.urls)})
        else:
            try:
                entries.append(dict(_raw_record(path, resource), status="verified_present"))
            except ValueError as error:
                entries.append({"file_id": resource.filename, "path": str(path),
                                "status": "invalid", "error": str(error)})
    return {"dataset": name, "raw_root": str(raw_root), "resources": entries,
            "ready": all(item["status"] == "verified_present" for item in entries)}


def _download(resource: Resource, target: Path, retries: int = 3) -> None:
    """Download to a temporary file, validate, then publish without overwriting."""
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(retries):
        for url in resource.urls:
            fd, temporary = tempfile.mkstemp(prefix=target.name + ".part-", dir=target.parent)
            part = Path(temporary)
            try:
                with os.fdopen(fd, "wb") as output:
                    request = Request(url, headers={"User-Agent": "Paper1-review-data/1.0"})
                    with urlopen(request, timeout=60) as response:
                        final_url = response.geturl()
                        declared_size = response.headers.get("Content-Length")
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if declared_size and part.stat().st_size != int(declared_size):
                    raise ValueError("Incomplete HTTP response")
                if resource.md5 and file_hash(part, "md5") != resource.md5:
                    raise ValueError("Downloaded archive MD5 mismatch")
                if resource.expected_size is not None and part.stat().st_size != resource.expected_size:
                    raise ValueError("Downloaded archive byte count mismatch")
                # Same-directory link is atomic and fails if another writer won.
                os.link(part, target)
                _write_json(target.with_name(target.name + ".download.json"), {
                    "source_url": url, "resolved_url": final_url,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": file_hash(target), "bytes": target.stat().st_size})
                return
            except FileExistsError:
                _raw_record(target, resource)
                return
            except Exception as error:
                errors.append(f"{url}: {type(error).__name__}: {error}")
            finally:
                part.unlink(missing_ok=True)
        if attempt + 1 < retries:
            time.sleep(min(attempt + 1, 3))
    raise RuntimeError(f"Could not download {resource.filename}: " + "; ".join(errors))


def download_dataset(name: str, raw_root: str | Path) -> dict:
    name, raw_root = canonical_name(name), Path(raw_root)
    for resource in resources(name):
        path = _find_resource(raw_root, name, resource)
        if path is not None:
            _raw_record(path, resource)  # Corrupt caches fail; never silently replaced.
        else:
            _download(resource, raw_root / name / resource.filename)
    return inventory_dataset(name, raw_root)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="strict")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="strict")
    return path.open("rt", encoding="utf-8", errors="strict")


def read_idx(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        header = stream.read(4)
        if len(header) != 4 or header[:3] != b"\x00\x00\x08" or header[3] != len(expected_shape):
            raise ValueError(f"Invalid uint8 IDX header: {path}")
        sizes = stream.read(4 * header[3])
        if len(sizes) != 4 * header[3]:
            raise ValueError(f"Truncated IDX dimensions: {path}")
        shape = struct.unpack(">" + "I" * header[3], sizes)
        if shape != expected_shape:
            raise ValueError(f"IDX shape {shape} != expected {expected_shape}: {path}")
        payload = stream.read(math.prod(shape) + 1)
        if len(payload) != math.prod(shape):
            raise ValueError(f"IDX payload length mismatch: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


def validate_labels(labels: np.ndarray, count: int, n_clusters: int, *, require_all: bool = True) -> None:
    if labels.ndim != 1 or len(labels) != count or labels.dtype.kind not in "iu":
        raise ValueError("Evaluation labels must be a one-dimensional integer vector aligned with inputs")
    if count == 0 or np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError("Evaluation labels outside the predeclared class domain")
    if require_all and not np.array_equal(np.unique(labels), np.arange(n_clusters)):
        raise ValueError("Incomplete/sparse class domain in the original benchmark labels")


def read_usps(path: Path, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse every coordinate explicitly; reject missing, duplicate or bad indices."""
    images = np.empty((expected_count, 256), dtype=np.float32)
    labels = np.empty(expected_count, dtype=np.int64)
    count = 0
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or count >= expected_count or len(fields) != 257:
                raise ValueError(f"USPS row count/width mismatch at {path}:{line_number}")
            try:
                label = int(fields[0])
                values = [field.split(":") for field in fields[1:]]
                indices = np.asarray([int(v[0]) for v in values])
                row = np.asarray([float(v[1]) for v in values], dtype=np.float64)
            except (ValueError, IndexError) as error:
                raise ValueError(f"Malformed USPS row at {path}:{line_number}") from error
            if not np.array_equal(indices, np.arange(1, 257)):
                raise ValueError(f"USPS coordinates must be exactly 1..256: {path}:{line_number}")
            if not 1 <= label <= 10 or not np.all(np.isfinite(row)) or np.any(np.abs(row) > 1):
                raise ValueError(f"USPS label/pixel outside its documented range: {path}:{line_number}")
            images[count] = (row + 1) / 2  # Preserve floating precision, no uint8 quantization.
            labels[count] = label - 1
            count += 1
    if count != expected_count:
        raise ValueError(f"USPS has {count} rows; expected {expected_count}")
    return images.reshape(-1, 1, 16, 16), labels


def read_root_labels(path: Path) -> dict[int, int]:
    roots: dict[int, set[str]] = {}
    with _open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Malformed RCV1 qrels at line {number}")
            try:
                document_id, relevance = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise ValueError(f"Non-integer RCV1 qrels at line {number}") from error
            if document_id <= 0 or relevance != 1:
                raise ValueError(f"Invalid RCV1 qrels ID/relevance at line {number}")
            if fields[0] in ROOT_CATEGORIES:
                values = roots.setdefault(document_id, set())
                if fields[0] in values:
                    raise ValueError(f"Duplicate root annotation for RCV1 document {document_id}")
                values.add(fields[0])
    eligible = {did: ROOT_CATEGORIES[next(iter(cats))] for did, cats in roots.items() if len(cats) == 1}
    if not eligible:
        raise ValueError("No single-root RCV1 documents in qrels")
    return eligible


def iter_rcv1_documents(paths: list[Path], *, keep_text: bool = True) -> Iterator[tuple[int, str]]:
    """Stream token files in declared order; flush each file's final document."""
    for path in paths:
        document_id = None
        words_started = False
        has_words = False
        parts: list[str] = []
        with _open_text(path) as stream:
            for number, line in enumerate(stream, 1):
                stripped = line.strip()
                if stripped.startswith(".I "):
                    if document_id is not None:
                        if not words_started or not has_words:
                            raise ValueError(f"Empty/malformed RCV1 document {document_id}")
                        yield document_id, " ".join(parts)
                    try:
                        fields = stripped.split()
                        if len(fields) != 2:
                            raise ValueError
                        document_id = int(fields[1])
                        if document_id <= 0:
                            raise ValueError
                    except ValueError as error:
                        raise ValueError(f"Malformed RCV1 document ID: {path}:{number}") from error
                    words_started, has_words, parts = False, False, []
                elif stripped == ".W":
                    if document_id is None or words_started:
                        raise ValueError(f"Misplaced RCV1 .W: {path}:{number}")
                    words_started = True
                elif stripped:
                    if document_id is None or not words_started or stripped.startswith("."):
                        raise ValueError(f"Unexpected RCV1 token/header: {path}:{number}")
                    has_words = True
                    if keep_text:
                        parts.append(stripped)
        if document_id is not None:
            if not words_started or not has_words:
                raise ValueError(f"Empty/malformed RCV1 final document {document_id}")
            yield document_id, " ".join(parts)


def split_indices(count: int, *, seed: int, validation_fraction: float = .1,
                  test_fraction: float = 0) -> dict[str, np.ndarray]:
    """Uniform, label-independent indices. Validation fraction applies after test."""
    if count < 2 or not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Invalid split size/fractions")
    order = np.random.RandomState(seed).permutation(count)
    n_test = int(math.floor(count * test_fraction))
    n_validation = int(math.floor((count - n_test) * validation_fraction))
    if n_validation == 0 or count - n_test - n_validation == 0 or (test_fraction and n_test == 0):
        raise ValueError("Split fractions yield an empty required split")
    return {"test": order[:n_test], "validation": order[n_test:n_test + n_validation],
            "train": order[n_test + n_validation:]}


def _reuters_arrays(paths: dict[str, Path], *, seed: int, validation_fraction: float):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    spec = SPECS["reuters10k"]
    labels_by_id = read_root_labels(paths["rcv1-v2.topics.qrels.gz"])
    token_paths = [paths[f] for f in TOKEN_FILES]
    ordered_ids, seen = [], set()
    for did, _ in iter_rcv1_documents(token_paths, keep_text=False):
        if did in seen:
            raise ValueError(f"Duplicate RCV1 document ID {did}")
        seen.add(did)
        if did in labels_by_id:
            ordered_ids.append(did)
    if len(seen) != spec["documents"] or len(ordered_ids) != spec["eligible"]:
        raise ValueError(f"RCV1 corpus counts differ: {len(seen)} documents, {len(ordered_ids)} eligible")
    if set(labels_by_id) != set(ordered_ids):
        raise ValueError("Single-root qrels and token document IDs do not match")
    del seen
    selection = np.random.RandomState(seed).permutation(len(ordered_ids))[:spec["selected"]]
    selected = np.asarray(ordered_ids, dtype=np.int64)[selection]
    selected_set = set(selected.tolist())
    texts = {}
    for did, text in iter_rcv1_documents(token_paths):
        if did in selected_set:
            texts[did] = text
    if set(texts) != selected_set:
        raise ValueError("Selected RCV1 text missing in second pass")
    documents = [texts[int(did)] for did in selected]
    y = np.asarray([labels_by_id[int(did)] for did in selected], dtype=np.int64)
    validate_labels(y, spec["selected"], spec["n_clusters"])
    ids = np.asarray([f"rcv1:{did}" for did in selected])
    indices = split_indices(len(selected), seed=seed + 1, validation_fraction=validation_fraction,
                            test_fraction=.2)
    vectorizer = CountVectorizer(dtype=np.float64, max_features=spec["n_features"],
                                 lowercase=True, token_pattern=r"(?u)\b\w\w+\b", analyzer="word",
                                 stop_words=None, ngram_range=(1, 1), min_df=1, max_df=1.0, binary=False)
    counts = vectorizer.fit_transform([documents[i] for i in indices["train"]])
    if counts.shape[1] != spec["n_features"]:
        raise ValueError(f"Training vocabulary has {counts.shape[1]} terms, expected {spec['n_features']}")
    transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    transformer.fit(counts)
    arrays = {}
    for split in SPLITS:
        idx = indices[split]
        matrix = counts if split == "train" else vectorizer.transform([documents[i] for i in idx])
        x = transformer.transform(matrix).toarray().astype(np.float32)
        x *= np.float32(math.sqrt(spec["n_features"]))
        arrays[split] = (x, y[idx], ids[idx])
    artifacts = {"vocabulary.json": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
                 "selected_document_ids.json": selected.tolist()}
    transform = {"kind": "training_only_tfidf", "fit_split": "train", "fit_ids_sha256": _ids_hash(ids[indices["train"]]),
                 "vocabulary_file": "transforms/vocabulary.json", "idf_file": "transforms/idf.npy",
                 "lowercase": True, "token_pattern": r"(?u)\b\w\w+\b", "max_features": spec["n_features"],
                 "norm": "l2", "sublinear_tf": True, "smooth_idf": True,
                 "feature_multiplier": math.sqrt(spec["n_features"]),
                 "subset_selection_seed": seed, "split_seed": seed + 1,
                 "eligible_corpus_count": len(ordered_ids), "document_count": spec["documents"],
                 "zero_feature_rows": {s: int(np.sum(np.count_nonzero(a[0], axis=1) == 0)) for s, a in arrays.items()}}
    return arrays, transform, artifacts, transformer.idf_


def _image_arrays(name: str, paths: dict[str, Path], *, seed: int, validation_fraction: float):
    spec = SPECS[name]
    if name == "usps":
        x_train, y_train = read_usps(paths["usps.bz2"], spec["train"])
        x_test, y_test = read_usps(paths["usps.t.bz2"], spec["test"])
        normalization = "(raw_float_pixel+1)/2; no quantization, resizing or learned transform"
    else:
        height, width = spec["shape"][1:]
        x_train = read_idx(paths["train-images-idx3-ubyte.gz"], (spec["train"], height, width))
        x_test = read_idx(paths["t10k-images-idx3-ubyte.gz"], (spec["test"], height, width))
        y_train = read_idx(paths["train-labels-idx1-ubyte.gz"], (spec["train"],)).astype(np.int64)
        y_test = read_idx(paths["t10k-labels-idx1-ubyte.gz"], (spec["test"],)).astype(np.int64)
        x_train = x_train[:, None].astype(np.float32) / 255
        x_test = x_test[:, None].astype(np.float32) / 255
        normalization = "uint8/255; no binarization, resizing or learned transform"
    validate_labels(y_train, spec["train"], spec["n_clusters"])
    validate_labels(y_test, spec["test"], spec["n_clusters"])
    train_ids = np.asarray([f"{name}:official_train:{i}" for i in range(spec["train"])])
    test_ids = np.asarray([f"{name}:official_test:{i}" for i in range(spec["test"])])
    indices = split_indices(len(x_train), seed=seed, validation_fraction=validation_fraction)
    arrays = {s: (x_train[indices[s]], y_train[indices[s]], train_ids[indices[s]])
              for s in ("train", "validation")}
    arrays["test"] = (x_test, y_test, test_ids)
    return arrays, {"kind": "fixed_pixel_scaling", "normalization": normalization,
                    "split_seed": seed, "official_test_untouched": True}, {}, None


def prepare_dataset(name: str, output_root: str | Path, raw_root: str | Path, *,
                    download: bool = False, seed: int = 1234, validation_fraction: float = .1) -> Path:
    """Create a new dataset directory and return its completed manifest path.

    Inputs are float32 (N,C,H,W for images, N,D for text); labels live only in
    evaluation/*.npz. Existing output directories and corrupt raw caches fail.
    """
    name = canonical_name(name)
    output_root, raw_root = Path(output_root), Path(raw_root)
    target = output_root / name
    if target.exists():
        raise FileExistsError(f"Dataset output already exists: {target}")
    if not isinstance(seed, int) or not 0 <= seed < 2**32 - 1:
        raise ValueError("Seed must be an integer in [0, 2**32-2]")
    if download:
        download_dataset(name, raw_root)
    paths, raw_records = {}, []
    for resource in resources(name):
        path = _find_resource(raw_root, name, resource)
        if path is None:
            raise FileNotFoundError(f"Missing raw {resource.filename} for {name}; use --download")
        paths[resource.filename] = path
        raw_records.append(_raw_record(path, resource))
    print(f"Preparing {name} from verified raw files", file=sys.stderr, flush=True)
    if name == "reuters10k":
        arrays, transform, extras, idf = _reuters_arrays(paths, seed=seed, validation_fraction=validation_fraction)
    else:
        arrays, transform, extras, idf = _image_arrays(name, paths, seed=seed, validation_fraction=validation_fraction)
    all_ids = np.concatenate([a[2] for a in arrays.values()])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Sample IDs overlap between splits")
    for x, labels, ids in arrays.values():
        if x.dtype != np.float32 or not np.all(np.isfinite(x)) or len(x) != len(ids):
            raise ValueError("Nonfinite, misaligned or incorrect-dtype inputs")
        validate_labels(labels, len(x), SPECS[name]["n_clusters"], require_all=False)
    target.mkdir(parents=True, exist_ok=False)
    try:
        (target / "inputs").mkdir()
        (target / "evaluation").mkdir()
        split_records = {}
        for split, (x, labels, ids) in arrays.items():
            inputs = f"inputs/{split}.npz"
            evaluation = f"evaluation/{split}_labels.npz"
            np.savez_compressed(target / inputs, x=x, ids=ids)
            np.savez_compressed(target / evaluation, labels=labels, ids=ids)
            split_records[split] = {"inputs": inputs, "labels": evaluation, "n_samples": len(x),
                                    "input_shape": list(x.shape), "ids_sha256": _ids_hash(ids)}
        if extras:
            (target / "transforms").mkdir()
            for filename, value in extras.items():
                _write_json(target / "transforms" / filename, value)
            np.save(target / "transforms" / "idf.npy", idf, allow_pickle=False)
        versions = {"numpy": np.__version__, "python": platform.python_version()}
        for package in ("scikit-learn", "scipy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        manifest = {"schema_version": "paper1-review-dataset-v1", "protocol_version": PROTOCOL_VERSION,
                    "dataset": name, "status": "completed", "fresh_preparation": True,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_record": SOURCE_RECORDS[name], "source_checked_on": "2026-10-27",
                    "source_sha256": file_hash(Path(__file__)), "raw_sources": raw_records,
                    "n_features": int(np.prod(arrays["train"][0].shape[1:])),
                    "n_clusters": SPECS[name]["n_clusters"], "n_clusters_source": "predeclared_benchmark",
                    "sample_counts": {s: len(arrays[s][0]) for s in SPLITS}, "splits": split_records,
                    "split_selection": "label-independent RandomState permutation; no stratification or retries",
                    "validation_fraction_of_training_pool": validation_fraction,
                    "labels_role": "benchmark construction/integrity and evaluation only; never model fitting or selection",
                    "historical_transductive_reproduction": False, "legacy_learned_features_used": False,
                    "transform": transform, "environment": versions,
                    "artifact_sha256": {str(p.relative_to(target)): file_hash(p)
                                         for p in sorted(target.rglob("*")) if p.is_file()}}
        _write_json(target / "manifest.json", manifest)
    except Exception as error:
        _write_json(target / "FAILED.json", {"status": "failed", "error": repr(error)})
        raise
    return target / "manifest.json"


def _load_split(manifest_path: str | Path, split: str, *, evaluation: bool) -> dict[str, np.ndarray]:
    path = Path(manifest_path)
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "paper1-review-dataset-v1" or manifest.get("status") != "completed":
        raise ValueError("A completed Paper 1 dataset manifest is required")
    entry = manifest["splits"][split]
    relative = entry["labels" if evaluation else "inputs"]
    artifact = (path.parent / relative).resolve()
    artifact.relative_to(path.parent.resolve())
    if file_hash(artifact) != manifest["artifact_sha256"][relative]:
        raise ValueError(f"Dataset artifact hash mismatch: {relative}")
    with np.load(artifact, allow_pickle=False) as data:
        required = {"labels", "ids"} if evaluation else {"x", "ids"}
        if set(data.files) != required:
            raise ValueError("Unexpected keys in data artifact; inputs must never contain labels")
        result = {key: data[key] for key in data.files}
    if len(result["ids"]) != entry["n_samples"] or _ids_hash(result["ids"]) != entry["ids_sha256"]:
        raise ValueError("Split IDs do not match manifest")
    if evaluation:
        validate_labels(result["labels"], entry["n_samples"], manifest["n_clusters"], require_all=False)
    elif result["x"].dtype != np.float32 or list(result["x"].shape) != entry["input_shape"]:
        raise ValueError("Input dtype/shape does not match manifest")
    return result


def load_inputs(manifest_path: str | Path, split: str) -> dict[str, np.ndarray]:
    """Training-safe reader returning only {'x', 'ids'} with hash verification."""
    return _load_split(manifest_path, split, evaluation=False)


def load_evaluation_labels(manifest_path: str | Path, split: str) -> dict[str, np.ndarray]:
    """Evaluation-only reader; do not call from optimization or model selection."""
    return _load_split(manifest_path, split, evaluation=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(SPECS))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--mode", choices=["prepare", "inventory", "download"], default="prepare")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.mode == "inventory":
        print(json.dumps(inventory_dataset(args.dataset, args.raw_root), indent=2))
    elif args.mode == "download":
        print(json.dumps(download_dataset(args.dataset, args.raw_root), indent=2))
    else:
        if args.output_root is None:
            parser.error("--output-root is required for preparation")
        path = prepare_dataset(args.dataset, args.output_root, args.raw_root,
                               download=args.download, seed=args.seed)
        print(path)


if __name__ == "__main__":
    main()
