COMPLETE MATCHED REPORT

ARI is primary; all prespecified companion metrics are retained. Method order is fixed, with no winner highlighting or metric selection.
Only the five seeds 101--105 together define an aggregate. Missing groups are pending; available seeds are never averaged as a substitute.
Absolute-score tables show mean (sample SD); seed-point plots show every seed and its mean. Paired intervals are evaluator-supplied descriptive, unadjusted 95% t intervals (df=4), not significance tests.
Delta = condition minus clean. Plotted degradation = clean minus condition = -Delta. D_condition = A_condition - B_condition. Gamma = Delta_A - Delta_B; it is not D_condition.
Inference-corruption maps are independently remapped unless explicitly labelled frozen-map. Frozen-map deltas apply only to the same checkpoint, not retrained models.
Scores use unit fractions, not percentages. No intervals are clipped. No cross-dataset average is computed.
V-measure equals arithmetic NMI; mapped balanced accuracy is mean class recall. These are not independent corroborating outcomes.
This layer reads only anchored analysis JSON, never labels, inputs, predictions, or checkpoints. It does not repeat the upstream artifact audit.
An inventory-only ledger describes the artifacts visible to that analysis snapshot, not live scheduler state or necessarily the whole HPC campaign.
ARI is primary. ACC, arithmetic NMI and AMI are mandatory companions; no metric or seed selection.
Five seeds are descriptive replications. Paired Student-t intervals use df=4, are unadjusted descriptive uncertainty, and are not significance tests or population guarantees. No p-values are computed.
No cross-dataset pooling. Incomplete five-seed groups and paired contrasts have no aggregate estimate.
Delta is condition minus clean; Gamma is delta(A)-delta(B), positive favoring A's change, not necessarily its condition score.
Inference corruption reuses one checkpoint and reports both remapped and clean-map-frozen scores. Retrained model cluster IDs are never aligned by numeric ID.
Mapped F1/recall use the displayed ACC-optimal map. Responsibility entropy is not calibrated semantic uncertainty; no class log-loss/Brier score.
V-measure and arithmetic NMI coincide; balanced accuracy is mean class recall. These are not independent corroborating outcomes.
This is the matched protocol, separate from the twelve source-recipe reference fits. VAE+GMM is its registered endpoint GMM, not a retrospectively chosen refit of every model.
Fixed update budgets need not imply equal runtime or examples seen under appended contamination. Costs/exposure are reported separately; no cost-matched superiority claim.
Attained quadrature checks cover saved diagnostic observations, not a certified continuous ELBO or uniform integration error bound.
Finite df bounds [3,30] on Student priors are not a test of the Gaussian limit. Boundary/gradient diagnostics retain their original scope.
Finite nonnegative responsibilities within the existing 1e-5 native/float64 row-mass tolerance are normalized in float64 for diagnostics; original files and hard assignments remain unchanged. Per-artifact roundoff and adjustment sizes are retained in evidence.

Tables are full-precision long-form numeric CSV, with seed values and descriptive intervals.
Method key:
VaDE: VaDE (matched) (vade_matched)
DEC: DEC (matched) (dec_matched)
IDEC: IDEC (matched) (idec_matched)
VAE+GMM: VAE + GMM (vae_gmm)
Pro-t: Pro-t (matched) (pro_t_matched)
Gq / G: Gaussian q / Gaussian prior (gaussian_q_gaussian_prior)
Gq / t: Gaussian q / Student prior (gaussian_q_student_prior)
tq / t: Student q / Student prior (student_q_student_prior)
Site / G: Site q / Gaussian prior (structured_q_gaussian_prior)
Site / t: Site q / Student prior (structured_q_student_prior)