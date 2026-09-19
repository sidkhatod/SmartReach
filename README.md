# SmartReach

## Causal Uplift Modeling on the Criteo Incrementality Dataset

Estimating heterogeneous treatment effects of ad exposure — with every
methodological choice justified by a diagnostic finding in the data, not picked
because it's the textbook default. Built on the Criteo AI Lab Uplift Modeling
Dataset (Diemert, Betlei, Renaudin, Amini — AdKDD/KDD 2018), a genuine randomized
incrementality trial: 13,979,592 users, 12 anonymized features (f0-f11), a
`treatment` flag, and `visit`/`conversion`/`exposure` outcomes.

## Results summary

| Stage | Finding |
|---|---|
| Randomization audit | Confirmed. Max \|Cohen's d\| = 0.047 across all 12 features — negligible by convention despite every feature being "significant" at n=14M (textbook p-value/effect-size divergence at scale) |
| ATE (conversion) | +0.1152% absolute, +59.45% relative (95% CI [0.1085%, 0.1219%]), z=28.52, p≈7.3×10⁻¹⁷⁹ |
| ATE (visit, cross-validation) | +1.0342% absolute, +27.07% relative (95% CI [1.0056%, 1.0629%]) — same direction, confirms methodology |
| Power analysis | MDE at 80% power = 4.76% relative — test was overpowered even for trivial effects |
| Heterogeneity | Confirmed, holdout-validated (not just exploratory). f9: ~25x uplift multiplier between cohorts (0.020% vs. 0.495% ATE). f8: monotonic 0.52%→0.029%→0.007% gradient. All non-overlapping CIs |
| Meta-learner diagnosis | S-learner dilutes treatment signal (0.37% gain share, rank 11/13) → T-learner shows 3x worse control-model overfitting (0.0104 vs. 0.0036 train-val AUC gap) → X-learner cuts CATE variance 95.2% (3.17×10⁻⁴ → 1.52×10⁻⁵) while preserving the full heterogeneity range |
| Policy simulation | Targeting top-20% by predicted CATE: 4x more efficient than random (2,199.5 vs. 548.1 incremental conversions), net lift +1,651.4 (95% CI [1,415.6, 1,959.4]) on a clean, untouched 2.8M-row test set |
| Compliance / CACE | Only 3.61% of assigned-treatment users were actually exposed (control exposure = exactly 0%, i.e. one-sided noncompliance). 2SLS CACE = 2.3416% absolute (≈20x the diluted ITT), first-stage F=133,463 |

## Why this project is structured the way it is

Most public uplift-modeling projects skip straight to fitting a meta-learner and
reporting a Qini curve. This one is structured so every modeling decision responds
to something *found in the data*, not a default:

- Randomization wasn't assumed — it was tested (with multiple-comparisons correction).
- Heterogeneity wasn't assumed before modeling — evidence for it was established
  first, then validated on a held-out slice to rule out winner's-curse inflation
  from the feature-selection step.
- X-learner wasn't picked because it's "better" — S-learner's dilution and
  T-learner's overfitting gap were diagnosed empirically first, and those specific
  failures motivated the next step.
- A degenerate-leaf-node artifact (T-learner predictions hitting the ±100%
  mathematical boundary) was caught, root-caused to overfit leaves on a rare-event
  target in the data-starved control arm, and fixed (`min_child_samples` increase)
  rather than reported as real heterogeneity.
- The dataset's `exposure` field (distinct from `treatment`) was used to run a
  proper instrumental-variables analysis instead of stopping at the diluted ITT.
- Every headline number is reported with a confidence interval, not as a bare point estimate.

## Full methodology, phase by phase

### Phase 0 — Data integrity
13,979,592 rows, 0 nulls, base rates (visit 4.70%, conversion 0.29%, treatment
ratio 85.0%) consistent with the dataset's documented characteristics. Features
f0-f11 show heavy skew (notably f4, f7, f9, f10) — informed the choice of
tree-based models over distance-based methods downstream.

### Phase 1 — Randomization audit
Per-feature normality assessed at scale (Shapiro-Wilk is oversensitive at n=14M
and was rejected in favor of Welch's t-test + Cohen's d for practical effect size).
Cohen's d used the **group-size-weighted pooled standard deviation** (critical
given the 85/15 treatment/control imbalance — an unweighted average would let the
small control group's variance dominate). Benjamini-Hochberg FDR correction
applied across the 12 simultaneous tests. Every feature reached statistical
significance (p≈0) purely from sample size; every feature's effect size was
negligible (max |d|=0.047, over 4x below Cohen's own "small effect" threshold of
0.2). **Verdict: randomization holds, no covariate adjustment needed for
identification.**

### Phase 2 — Hypothesis testing
Two-proportion z-test on conversion: pooled SE for the test statistic (correct
under the null), unpooled SE for the confidence interval (correct for estimating
the true difference without assuming the null). Absolute lift 0.1152%, relative
lift 59.45%, both directionally confirmed by the visit outcome (27.07% relative
lift) as a cross-validation check. Power analysis confirmed the test was
overpowered even to detect trivial effects (MDE=4.76% relative) — a null result
would have been strong evidence of a true null, not insufficient power.

### Phase 3 — Heterogeneity evidence
Within-arm point-biserial correlation (treatment vs. control, computed
separately) surfaced f3, f9, f8 as the top divergent features. Quantile binning
found: f3 collapsed to a single bin (point-mass dominated — a false lead despite
its correlation-divergence rank); f9 split into two cohorts with a ~32x raw
multiplier; f8 showed a clean 3-bin monotonic gradient. Because these features
were *selected* for their divergence out of 12 candidates, the raw multipliers
were treated as inflated by winner's-curse/selection bias — **re-validated on an
untouched 20% holdout**, where the effects survived at slightly attenuated but
still large magnitudes (f9: ~25x, f8: 0.52%→0.029%→0.007%, all CIs non-overlapping,
smallest per-bin N > 82,000). A follow-up histogram check confirmed f9's split
point (16.226) sits on a massive point-mass spike (>8M rows) — f9 is a
missing-data/default-value sentinel, not a continuous gradient; the finding is
reported as "value present vs. sentinel," not a smooth dose-response relationship.
**Verdict: heterogeneity is real, holdout-confirmed, and justifies individual-level
modeling.**

### Phase 4 — Meta-learner comparison (diagnosis-driven)
All three learners share the same LightGBM base configuration, isolating the
causal-architecture choice from the algorithm choice.

- **S-learner**: treatment feature ranked 11th of 13 by gain, 0.37% gain share —
  empirical, not textbook, evidence of dilution.
- **T-learner**: control-arm model (trained on the data-starved 15% split) showed
  a train-val AUC gap of 0.0104 vs. 0.0036 for the treatment-arm model — ~3x worse
  overfitting, the concrete justification for moving to X-learner. Successfully
  recovered the f8/f9 heterogeneity patterns from Phase 3, with predictions
  slightly regularized toward the global mean vs. raw bin ATEs, as expected.
  Also surfaced a **degenerate-leaf artifact**: CATE predictions hitting the
  mathematical ±100% boundary from leaves predicting probability exactly 0 or 1 —
  root-caused to overfitting on the rare-event (0.2% base rate), data-starved
  control arm, fixed by raising `min_child_samples`.
- **X-learner**: cross-imputed effects + propensity weighting. Cut CATE prediction
  variance across refits by 95.2% (3.17×10⁻⁴ → 1.52×10⁻⁵) relative to T-learner.
  Verified this was genuine stabilization, not signal collapse: post-fix CATE
  percentiles are tight and plausible relative to the 0.2% baseline (only 0.39% of
  ~2.8M validation predictions exceeded ±5 percentage points), and the f8/f9
  heterogeneity patterns were fully preserved (f9 sentinel: 0.023% vs. 0.380%
  present, ~16x; f8: 0.392%→0.028%→0.018%, still clean monotonic).

### Phase 5 — Complier Average Causal Effect (CACE)
The dataset separates `treatment` (assignment) from `exposure` (actual ad
delivery) — a real-world non-compliance setting. Compliance rate: 3.61%
(P(exposure=1|treatment=1)); **control-arm exposure was exactly 0%**, confirming
one-sided noncompliance (no contamination, no defiers by construction) — which
means the IV estimate here is more precisely an **Average Treatment Effect on the
Treated (ATET)**, a stronger identification claim than a generic LATE. Estimated
via 2SLS (`treatment` instrumenting `exposure`, f0-f11 as exogenous covariates):
first-stage F=133,463 (far above the conventional weak-instrument threshold of
10), CACE = 2.3416% absolute lift — roughly 20x the diluted ITT (0.1124% on the
IV analysis sample). A no-covariate robustness specification (2.79%, 95% CI
[2.16%, 3.42%]) comfortably contains both the covariate-adjusted 2SLS estimate and
the simple Wald estimator (3.12%), indicating the differences across estimators
reflect standard estimator variance, not instrument instability or a covariate
artifact — an informal consistency check, not a formal specification test.

**ITT vs. CACE, in plain terms**: ITT measures the effect of *assigning* someone
to the campaign, diluted across everyone assigned regardless of whether the ad
ever reached them. CACE/ATET measures the effect *on people the campaign actually
reached* — the number that speaks to whether the ad creative and targeting work,
as distinct from the separate question of why exposure/reach was so low.

### Phase 6 — Evaluation (Qini/AUUC)
Run entirely on `df_test_clean`, a 2.8M-row split carved out before any feature
selection, model selection, or hyperparameter tuning touched it (a separate,
already-used `df_val` handled all exploratory/tuning work in Phases 3-4). Qini
curve shape: steep climb through the first ~20-25% of the population (the
"persuadables," matching the f8/f9 cohorts from Phase 3), then a near-total
flattening through the 25-90% range (targeting this block is close to pure
budget waste), with a slight dip in the final decile suggestive of possible
negative response ("sleeping dogs") — **this tail effect was not formally
hypothesis-tested and is reported as a visual observation only, not a confirmed
finding.**

### Phase 7 — Policy simulation
Targeting the top 20% by predicted CATE: 2,199.5 realized incremental conversions
vs. 548.1 for random 20% targeting — a 4x efficiency multiplier, bootstrap 95% CI
on the net lift [1,415.6, 1,959.4] (500-1000 resamples), confirming the result
holds even at the CI's lower bound. Total 100%-rollout potential ≈2,750
incremental conversions, meaning the top-20% policy captures roughly 80% of the
campaign's total achievable value from one-fifth of the spend.

## Known limitations (stated plainly, not hidden)

- The "sleeping dogs" tail effect (Phase 6) is visual, not statistically tested —
  would need a dedicated bottom-decile hypothesis test to confirm.
- R-learner (a planned stretch goal) was not built — X-learner was sufficient to
  demonstrate the architecture-selection methodology, and time was prioritized on
  the CACE analysis instead.
- Full robustness checks (alternate random seeds/folds for the S/T/X ranking,
  broader hyperparameter sensitivity beyond the `min_child_samples` fix) were
  partially covered by the cross-fold variance analysis in Phase 4 but not run as
  a separate dedicated pass.
- CACE/ATET describes the effect among the 3.61% who were actually exposed — it
  is not automatically the effect that would be observed if the exposure
  mechanism were fixed and compliance rose; that's a different, unanswered
  question.

## Project structure

```
uplift-modeling/
├── README.md              <- you are here
├── PLAN.md                <- original phase-by-phase analysis plan
├── requirements.txt
├── data/
│   ├── raw/                 <- full Criteo CSV (not committed — see below)
│   └── processed/           <- sample.parquet (2M-row stratified dev subsample)
├── src/                     <- data loading + one module per analysis phase
├── notebooks/                <- one per phase
└── outputs/
    ├── figures/               <- Qini curves, bin-effect plots
    └── tables/
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Full dataset (13,979,592 rows, v2.1 release):
- GitHub mirror (confirmed working, no auth): `https://github.com/jszymon/uplift_sklearn_data/releases/download/Criteo/criteo-research-uplift-v2.1.csv.gz`
- Original host: `https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`
- Kaggle mirror: `arashnic/uplift-modeling`

Citation:
```
@inproceedings{Diemert2018,
  author={Diemert Eustache and Betlei Artem and Renaudin, Christophe and Massih-Reza, Amini},
  title={A Large Scale Benchmark for Uplift Modeling},
  booktitle={Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom, August 20, 2018},
  publisher={ACM},
  year={2018}
}
```
