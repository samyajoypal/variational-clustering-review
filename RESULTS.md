# Result index

The following folders contain completed evidence for the review:

| Folder | Contents |
|---|---|
| [Matched analysis](results/paper1_review_20261031/matched_analysis_complete_20261031_v1/) | Evaluation JSON with all seed rows and contingencies, five CSV score/contrast panels, inventory and training-origin allocation |
| [Matched report](results/paper1_review_20261031/matched_report_complete_v1/) | Six CSV tables and 18 figures in PDF/PNG |
| [Matched diagnostics](results/paper1_review_20261031/matched_diagnostics_complete_20261031_v1/) | 300 fitted endpoints, 60 endpoint groups and 480 trajectory groups |
| [Known-model simulations](results/paper1_review_20261028/scientific_analysis_complete_v2/) | Complete S2/S3 scientific summary |
| [Reference adaptations](results/paper1_review_20261029/reference_analysis_complete_v2/) |12 separate MNIST fits; native and Pro-t GMM-refit prediction panels |

The complete [paired companion-metric table](results/paper1_review_20261031/matched_report_complete_v1/tables/paired_all_metrics.csv)
is retained with all other numerical outputs. No metrics
were removed to favor a method. ARI is primary; ACC, arithmetic NMI/AMI,
homogeneity, completeness, V-measure, mapped macro F1 and mapped balanced
accuracy remain available. Arithmetic NMI and V-measure are equal by definition.

Paired intervals are descriptive, unadjusted 95% t intervals over five seeds,
not multiplicity-corrected significance tests. Scores are fractions. There is no
cross-dataset pooled leaderboard. `Delta` is condition minus clean;
`Gamma = Delta_A - Delta_B`; positive Gamma need not mean a higher corrupted
absolute score. Frozen-map metrics apply to inference corruption of the same
checkpoint, not arbitrary component labels from retrained models.

The exported numeric values are unchanged from the completed scientific
analysis. Operational paths and receipt bindings are portable derivatives,
documented in [provenance/export_bindings.json](provenance/export_bindings.json).
All publication files have their own public SHA256 manifest.
