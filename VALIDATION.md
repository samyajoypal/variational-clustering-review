# Validation

The public source tree was tested independently of the private workspace imports
on 2 November 2026. The complete suite passed **393 tests**, with **12 CUDA-only
tests skipped** because this validation host had no accessible CUDA device.
An independent comparison also checked 104,896 CSV numeric fields and seed
values against the completed analysis, with exact agreement. Every publishable
file was scanned as bytes; all 18 figure PDFs were additionally checked through
extracted text and metadata. No private infrastructure identities were found.

Tests cover distributional calculations, quadrature references, model objectives,
data/label separation, paired metrics, artifact tampering, checkpoint continuation
and endpoint analysis. They use synthetic fixtures and do not rerun the full
GPU campaign.

Using only the portable completed analysis, the report generator reproduced all
six numerical tables and all 18 PNG figures without changing any numerical values. Figure PDF timestamps
can vary between renderings; the plotted pixels and source values were unchanged.
The simulation and separate reference plots also regenerated from the included
scientific summaries.

The upstream source fetcher was checked in a fresh isolated directory: it fetched
139 explicitly selected source files at exact commits and verified all 144
required files including the already supplied provenance manifests. Full source
archives, historical weights and datasets are not required by that fetcher.

The public package verifier checks every distributed checksum, all completed
analysis artifacts, complete five-seed groups, report artifact hashes and the
diagnostic-to-analysis binding. It confirms 40 initializations, 300 fits and
1,200 prediction panels with no incomplete groups. It does not repeat the bulk
prediction/checkpoint audit. The original audit and final analysis anchors are
retained separately from the public-derivative checksums.

All four preparation-manifest digests match the original campaign bindings.
Each of the 20 original training-view manifest digests was checked against a
completed-fit configuration whose bytes match the independent audit's saved
configuration digest. The public receipts contain no feature or label arrays.

Run the checks locally using the commands in the README. This validation record
describes the delivered source and derived outputs; it does not promise identical
training results on other software versions or hardware.
