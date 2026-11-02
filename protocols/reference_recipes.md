# Reference-recipe adaptations

The separate MNIST reference panel uses seeds 901–903, the inductive 54,000/6,000/10,000 split and fresh initialization. It is distinct from the matched five-seed panel. Test labels are not available to optimization or checkpoint selection.

| Method | Pinned revision | Recipe and declared adaptation |
|---|---|---|
| DEC | `e551e01f90a3d67d2ef9c90e968c8faf5d8f857d` | PyTorch implementation of stacked pretraining and Caffe-style momentum; 50,000 updates per layer plus 100,000 joint updates; target refresh every 160; label-free assignment stopping or 20,000-update cap |
| IDEC | `4a661c211f082022249758ab3312f7db21e2aa99` | Shared fresh DEC initialization; feature-mean MSE plus 0.1 KL; Adam with explicit epsilon; target refresh every 140; label-free stopping or 20,000-update cap |
| VaDE | `eca52b4b57e32a68f578de7a54eda5d39907b0fe` | 100 pretraining epochs and 3,000 training epochs; sampled native assignments; each mixture weight enters once |
| Pro-t | `11966cd0a512dfeaeae636aebf7df80832e985b4` | Pinned residual architecture; 100 fresh epochs; epoch 20 training-only mixture overwrite; fixed endpoint; native and GMM-refit assignments separate |

Full architecture, optimization, loss reductions, covariance/scale conventions, stopping and prediction definitions are machine-readable in each reference module's `PROTOCOL` or `protocol()` output. Upstream source revisions, selected file hashes and licenses are in `provenance/upstream_sources.json` and `THIRD_PARTY_NOTICES.md`.

Reference checks cover source formulas and optimizer behavior, fresh initialization, resume integrity, endpoint completion and prediction validity. These adaptations change framework, splitting and/or endpoint selection relative to historical experiments; no published-score reproduction is asserted. No downloaded pretrained model is used.
