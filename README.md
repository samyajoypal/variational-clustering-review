# Variational clustering: priors, inference and robustness

Research code and derived results for **Priors, Approximate Inference and Robustness in Variational Deep Clustering**, by Samyajoy Pal.

The package studies distributional choices, approximate-inference limitations and robustness in variational clustering. It includes mathematical checks, known-model simulations, source-recipe adaptations and a four-dataset neural benchmark.

The matched panel uses MNIST, Fashion-MNIST, USPS and Reuters-10k, ten implementations, five seeds, 300 fitted endpoints and 1,200 test prediction panels. ARI is primary; all nine specified clustering metrics and the complete paired analyses are retained. Findings are conditional on the documented implementations, partitions and budgets.

## Regenerate the numerical report

From the repository root with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python scripts/verify_public_package.py
python scripts/reproduce_report.py --output generated/matched_report
```

The output directory must be new. The generator verifies the completed analysis and every five-seed group, then produces numerical CSV tables and 18 figures. It does not download data, read labels or fit models.

## Contents

| Location | Contents |
|---|---|
| `experiments/` | Models, inference calculations, data preparation, simulations, metrics and integrity checks |
| `tests/` | Mathematical, metric, data-integrity and checkpoint-continuation tests |
| `results/` | Complete seed-level scores, contingencies, diagnostics, numerical tables and figures |
| `protocols/` | Concise experimental specifications |
| `literature/` | Scholarly bibliographic metadata |
| `provenance/` | Public checksums, original-to-export bindings, upstream pins and data receipts |

[Reproduction guide](REPRODUCIBILITY.md) · [Data](DATA.md) · [Result index](RESULTS.md) · [Validation](VALIDATION.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Run tests

```bash
python -m pip install -e '.[test]'
python scripts/fetch_reference_sources.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The fetcher retrieves only explicitly selected upstream source files at pinned commits and verifies their hashes. Other upstream files, historical weights and datasets are not downloaded. CUDA-only checks are skipped without CUDA. The selected MIT Pro-t source and its license are included; other source prerequisites remain ignored local files.

Raw data, prepared feature/label arrays and fitted checkpoints are not distributed. Report regeneration works directly from the included results; fresh fitting requires original data and compute. See `CITATION.cff` for citation metadata.
