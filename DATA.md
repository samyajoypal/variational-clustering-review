# Data and partitions

No raw data, prepared feature/label arrays or weights are included. The exact
preparation implementation is `experiments/paper1_review/data.py`. Public
preparation receipts under `provenance/data/` record original source URLs, source
digests, split counts, identifier hashes and transform digests. The 20 training-view receipts in
`provenance/training_views/` preserve realized clean/contaminant counts and
source/split digests; each is bound to an independently audited final-fit
configuration through `provenance/training_views.json`.

| Dataset | Source | Train / validation / test | Transformation |
|---|---|---|---|
| MNIST | [Original dataset](https://yann.lecun.com/exdb/mnist/), obtained from the [OSSCI IDX mirror](https://ossci-datasets.s3.amazonaws.com/mnist/) | 54,000 / 6,000 / 10,000 | Official test set; pixels divided by 255 |
| Fashion-MNIST | [Authors' repository](https://github.com/zalandoresearch/fashion-mnist) | 54,000 / 6,000 / 10,000 | Official test set; pixels divided by 255 |
| USPS | [LIBSVM archive](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps) | 6,562 / 729 / 2,007 | Official test set; values mapped from [-1,1] to [0,1]; no resizing |
| Reuters-10k | [RCV1-v2 archive description](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) | 7,200 / 800 / 2,000 | Four root categories; single-root documents; 2,000 features; train-only vocabulary and TF–IDF |

Image validation IDs use a label-independent 10% split of the training pool with
`RandomState(1234)`. Reuters selects 10,000 documents with seed 1234 and splits them
with seed 1235. Its vocabulary and TF–IDF transform fit only the 7,200 optimization
documents. This inductive benchmark differs from historical transductive
DEC/VaDE datasets. Root-topic labels construct the Reuters benchmark; labels
remain inaccessible to optimization and checkpoint selection.

```bash
python -m experiments.paper1_review.data --dataset mnist \
  --raw-root raw_data --output-root prepared_data --download --seed 1234
```

Repeat for `fashion_mnist`, `usps` and `reuters10k`. Outputs must be new. Check the
original providers' access/usage terms before acquisition. The image archives
are checked against their declared MD5 values and retained SHA256 receipts.
For the historical Reuters HTTP token archives, published compressed sizes and
newly retained SHA256 values are distinguished: a newly computed digest is not
independent authentication of the first download.

`contamination.py` creates the declared training views for MNIST/Fashion-MNIST;
it appends noisy copies or opposite-dataset training observations and preserves
the clean partitions. A training-view manifest records its origin and input
hashes. Prediction corruptions are generated using the same sorted-ID
randomization across methods. See the data-preparation and contamination modules
for contamination ratios and corruption strengths.

Repreparation can produce a new manifest hash because timestamps and environment
metadata change. Compare data, split-ID and transform digests for scientific
identity; never replace an old manifest hash with a new one in a claimed original
run. Public data receipts are documentary records, not runnable array manifests.
