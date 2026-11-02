# Third-party sources and data

The repository's MIT license covers the original implementation, analysis code
and accompanying reproducibility material. It does not replace the licenses or
terms of cited publications, benchmark datasets or upstream software.

The selected `torch-tvae` files under
`experiments/paper1_review/vendor_sources/pro_t_11966cd0/` are reproduced under
their included MIT license, copyright 2026 torch-tvae contributors, at commit
`11966cd0a512dfeaeae636aebf7df80832e985b4`. Only its pinned `model28.py` is executed
by the separate reference-recipe adapter. The original source notice is retained.

DEC, IDEC and VaDE upstream source trees are **not distributed** here. Their
recorded file digests and repository commits are included for verification.
`scripts/fetch_reference_sources.py` downloads exactly the selected files from
the recorded commits into ignored local directories and checks each
digest. The source trees remain governed by their upstream terms; no new license
is asserted for repositories without an explicit license in the selected source.
The DEC selection includes the upstream DEC and Caffe license files. These
historical files serve source inspection, not a dependency on Caffe or Keras.

All upstream URLs, commits, archive digests and selected file digests are listed
in [provenance/upstream_sources.json](provenance/upstream_sources.json). The
PyTorch reference adapters disclose departures from the original recipes in
their `PROTOCOL` records and in the reference protocol. A source pin is not a
claim to reproduce a published leaderboard score.

No raw benchmark data, prepared feature/label arrays or trained weights are
distributed. Dataset sources and preparation are described in
[DATA.md](DATA.md). Readers acquire datasets from their original providers under
the applicable terms. The bibliography contains scholarly metadata, not copies of publication PDFs.
