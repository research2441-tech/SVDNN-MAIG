# SVDNN-MAIG

**Hybrid Support Vectorized Decision Neural Networks with Mean Absolute Information Gain for
Interpretable Diabetes Risk Stratification in Post-Pandemic Health Systems**

M. Rabiyathul Fathima and A. M. J. Md Zubair Rahman, Al Ameen Engineering College (Autonomous), Erode, India.
Manuscript under review at *Scientific Reports*.

This repository contains everything needed to reproduce the study from scratch:

- the synthetic-cohort generator, including the outcome-label construction;
- the MAIG feature-ranking operator;
- the SVDNN model and every comparator and ablation;
- the leakage-free nested evaluation protocol;
- the statistical analysis;
- the explanation-validation and counterfactual-plausibility analyses;
- validation on real patient data;
- the script that builds every table and figure.

One command regenerates all reported numbers, and they all come from a single results store.

> **Interpretation.** The main experiments use a statistically simulated cohort. They are
> proof-of-concept evidence about the method, not evidence of clinical diagnostic performance. The
> same pipeline is also run on real patient records (PIMA, and optionally CDC BRFSS 2015). Those
> datasets have no post-COVID variables, so they test whether the method transfers to real clinical
> data, not the post-COVID findings.

---

## Repository contents

| File | Purpose |
|---|---|
| `config.yaml` | Every parameter: generator, MAIG, evaluation protocol, search spaces, seeds, run modes |
| `run_experiments.py` | Single entry point; runs all stages and resumes after interruption |
| `data_generator.py` | Structural causal model of the post-COVID cohort, label generation, missingness, case-enriched sampling, data dictionary, checksum |
| `DATA_GENERATION.md` | Complete written specification of the data-generation procedure and its assumptions |
| `maig.py` | MAIG (Eqs. 1–5) with redundancy-aware selection; reference rankers: IG, best-split IG, MI, ReliefF, mRMR, ANOVA-F |
| `svdnn.py` | Full SVDNN formulation (Eqs. 6–13), shared neural training loop, ablation variants |
| `baselines.py` | LR, RF, XGBoost, RBF-SVM, MLP, LSTM-RNN, CNN-LSTM; model factory and default settings |
| `evaluation.py` | Nested repeated cross-validation engine: fold-confined preprocessing, selection, balancing, tuning and thresholding |
| `metrics_stats.py` | Discrimination, calibration, decision curves, corrected resampled t-test, Wilcoxon, Holm, DeLong, bootstrap CIs, effect sizes |
| `explain.py` | Gradient saliency, cumulative attribution, integrated gradients, deletion-fidelity test, constrained counterfactuals, plausibility checks |
| `leakage_audit.py` | Label-circularity and information-leakage audit |
| `maig_robustness.py` | Controlled MAIG robustness study under class imbalance, label noise, correlated predictors and measurement noise |
| `external_validation.py` | Real-patient validation (PIMA Indians Diabetes; CDC BRFSS 2015 if provided) |
| `make_report.py` | Builds every table (CSV + Markdown), figure (PNG + PDF) and `results_summary.json` |
| `utils.py` | Configuration loading, seeding, I/O |
| `test_pipeline.py` | Unit tests for the properties claimed in the manuscript |
| `requirements.txt` | Python dependencies |

## Installation

```bash
git clone https://github.com/<user>/SVDNN-MAIG.git
cd SVDNN-MAIG
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q test_pipeline.py                              # about 30 s
```

A CPU is sufficient. No GPU is required.

## Reproducing the study

```bash
python run_experiments.py --mode quick     # smoke test: small cohort, 3 folds, about 10 minutes
python run_experiments.py --mode full      # full study as reported in the manuscript
```

Individual stages can be run with `--stages`, for example `--stages main,report`. Every completed
outer fold is saved immediately, so re-running the same command after an interruption continues
where it stopped.

| Stage | What it produces |
|---|---|
| `data` | `outputs/full/data/cohort.csv` (13,600 × 52 predictors + label), `data_dictionary.csv`, `cohort_latent_audit.csv`, `generator_summary.json` (incl. SHA-256) |
| `main` | 8 models × 25 outer folds, 8 ablations, 6 feature rankers, explanations, counterfactuals |
| `leakage` | `leakage/leakage_audit.json` |
| `robustness` | `robustness/maig_robustness_raw.csv` |
| `external` | real-data results under `external_pima/` (and `external_brfss/`) |
| `report` | `report/tables/*.csv|md`, `report/figures/*.png|pdf`, `report/results_summary.json` |

**Run time (full mode).** The main stage dominates. It runs 8 models with 12 tuning trials each in
every one of 25 outer folds, plus the ablations, ranking comparisons and explanations. Outer folds
run in parallel (`n_workers`, by default half of the CPU cores), and the results do not depend on
the number of workers. One outer fold takes roughly 1 to 1.5 CPU-hours, mostly for tuning the
neural models. Plan for an overnight run on a typical 8-core desktop. To shorten a run, lower
`tuning.n_trials`; the manuscript must then report the value used.

**Optional BRFSS 2015 validation.** Download `diabetes_binary_health_indicators_BRFSS2015.csv`
(UCI Machine Learning Repository, dataset 891), set `external.brfss_csv` in `config.yaml` to its
path, and run `--stages external,report`.

## Evaluation protocol (identical for every model)

Repeated stratified cross-validation: **5 repeats × 5 folds = 25 held-out evaluations per model**.
Each participant receives one held-out prediction per repeat. Within every outer fold:

1. The outer-training data are split 80/20 into inner-train and inner-validation. The test fold is
   not used until the final step.
2. Plausibility filtering, iterative Bayesian-ridge imputation, one-hot encoding and z-scoring are
   fitted on inner-train only.
3. MAIG scores and selection are fitted on inner-train only. γ is chosen on inner-validation with a
   fixed reference model. All models receive the same feature set.
4. Class balancing (Gaussian-noise bootstrap oversampling, Eq. 17) is applied to inner-train only.
   Probabilities are mapped back to the training prevalence by a prior-shift correction.
5. Every model gets the same Optuna TPE budget, optimising inner-validation AUROC. Neural models
   early-stop on the same split.
6. The decision threshold is set on inner-validation (Youden J) by the same rule for every model.
7. The tuned model is evaluated once on the outer test fold.

LSTM-RNN and CNN-LSTM do not impose an order on tabular columns. They read the eight variables
measured at the three clinical visits (T0, T1, T2) as a genuine length-3 sequence, and static
variables are concatenated to the recurrent state. On datasets without repeated visits they are
reported as not applicable.

## Method summary

**MAIG** (`maig.py`). For each feature, entropy reductions are averaged over k = 19 quantile
thresholds, using both partitions of each split and prior-balanced class weights:

$$\mathrm{MAIG}(f_j)=\frac{1}{k}\sum_{t=1}^{k}\sum_{s\in\{-,+\}}\rho_t^{s}\,\bigl|H_w(Y)-H_w(Y\mid S_t^{s})\bigr|,\qquad \hat G(f_j)=\frac{\mathrm{MAIG}(f_j)}{\sum_i \mathrm{MAIG}(f_i)}$$

Properties, each checked in the tests:

- bounded in [0, 1];
- invariant to monotone transformations;
- invariant to prevalence under prior balancing;
- lower variance than single best-split information gain.

Features are selected greedily by Ĝ(f_j)·(1 − η·max|ρ| with already selected features), so that
collinear variables are suppressed explicitly.

**SVDNN** (`svdnn.py`). Margin layer ψ_k(x) = w_kᵀx + b_k (k = 1…m). Each unit is a soft-margin
linear SVM with its own hinge loss and margin regulariser (Eqs. 7–8), and the units are kept
mutually orthogonal. The decision network operates on [ψ(x); x] with L ReLU layers and a sigmoid
output. The objective is

$$\mathcal L=\beta\mathcal L_{CE}+\alpha\frac{1}{m}\sum_k\Bigl[\tfrac{\lVert w_k\rVert^2}{2Cn}+\tfrac1n\sum_i\max(0,1-\tilde y_i\psi_k(x_i))\Bigr]+\mu\frac{\lVert\bar W\bar W^\top-I\rVert_F^2}{m(m-1)}+\lambda\frac{1}{np}\sum_i\lVert\nabla_x s(x_i)\rVert_2^2$$

optimised with Adam, plateau learning-rate decay and early stopping. The ablations remove MAIG, the
margin layer, the margin supervision, orthogonality, the skip connection, the gradient penalty or
the decision network. They also compare SVDNN with an MLP that has a hinge loss on its output.

## Response to review: where each point is addressed

| Comment | Evidence produced by this repository |
|---|---|
| Editor: data generation, labels, leakage | `DATA_GENERATION.md`, `data_generator.py`, `leakage_audit.py`, `table_leakage_audit`, `fig_leakage_audit` |
| Editor: inconsistencies across text, tables, figures | All numbers are generated from one results store by `make_report.py`; `results_summary.json` contains ready-to-use sentences |
| Editor: code availability | This repository; see the Code Availability statement below |
| R1.1 Synthetic data and clinical validity | Proof-of-concept framing; real-data validation (`external_validation.py`, `table_external_validation`) |
| R1.2 Data-generation details | `DATA_GENERATION.md`, `table2_data_dictionary`, `generator_summary.json` |
| R1.3 Real-world validation | PIMA (automatic) and BRFSS 2015 (optional) under the identical protocol |
| R1.4 Label circularity | T3 outcome never released, prevalent cases excluded, oracle ceilings, positive control (`leakage_audit.py`) |
| R1.5–6 SVDNN novelty and full formulation | `svdnn.py` docstring (complete equations, every symbol defined); ablations incl. `mlp_output_hinge` (`table_ablation`) |
| R1.7 MAIG theory and comparison | `maig.py` docstring (definition, properties), unit tests, `table_feature_ranking_methods` (IG, MI, ReliefF, mRMR, ANOVA) |
| R1.8 Robustness to imbalance, noise, correlation | `maig_robustness.py`, `table_maig_robustness`, `table_maig_robustness_tests`, `fig_maig_robustness` |
| R1.9 Fold-nested selection | `evaluation.py` (all fitting inside the training fold); selection-bias demonstration in `leakage_audit.py` |
| R1.10 Folds vs runs | 5 repeats × 5 folds = 25 evaluations per model (`results_summary.json → evaluation_protocol`) |
| R1.11 Statistical significance | Corrected resampled t-test, Wilcoxon, Holm, Cohen d_z, Cliff δ, CIs, DeLong, Friedman (`table_pairwise_statistics`, `table_delong`) |
| R1.12 F1 improvement arithmetic | Absolute (pp) and relative (%) reported separately (`results_summary.json → improvements`) |
| R1.13 False-negative reduction | Baseline and proposed FNR, absolute and relative reduction with bootstrap CIs (`table_false_negative_rates`); sensitivity at fixed specificity |
| R1.14 Fair comparison | Identical folds, preprocessing, features, balancing, tuning budget and threshold rule (`evaluation.py`) |
| R1.15 LSTM/CNN-LSTM on tabular data | Three-visit sequence design (`baselines.py`); not applied without repeated visits |
| R1.16 Interpretability claims | Attributions treated as post-hoc and validated: deletion fidelity, cross-fold stability (`table_explanation_fidelity`, `fig_deletion_curves`) |
| R1.17 Causality | Causal role of every variable known from the generator; rankings shown to include non-causal correlates (`table4_top_features`, `fig7_feature_importance`) |
| R1.18 Counterfactual plausibility | Actionable-only, bounded, coupled counterfactuals vs unconstrained Eq. 20; LOF, Mahalanobis and range checks (`table_counterfactual_plausibility`) |
| R1.19 Class distribution | Case-enriched sample (prevalence 0.324) vs source incidence; PPV/NPV at population incidence (`table_calibration`) |
| R1.20 Variance claims | SD with bootstrap CI, Brown–Forsythe tests, fold-level distribution plots (`table5_fold_variability`, `fig_fold_distributions`) |
| R1.21 Calibration | Brier, calibration intercept and slope, ECE, reliability curves, decision curves (sample and population-weighted) |
| R1.22 Reproducibility | Generator, parameters, seeds, preprocessing, hyper-parameter spaces and traces, evaluation scripts, tests |
| R2 Labels, consistency, ablation | As above (R1.2, R1.4, Editor, R1.5–7) |

## Code Availability (text for the manuscript)

> **Code Availability.** All custom code used in this study is publicly available at
> https://github.com/&lt;user&gt;/SVDNN-MAIG without access restrictions. This includes the
> synthetic-cohort generator with all generation parameters and random seeds, the implementations
> of MAIG and SVDNN, all comparator models, the preprocessing and nested evaluation pipeline,
> hyper-parameter search spaces, statistical analyses and the scripts that produce every table and
> figure. The released version will be archived in a DOI-minting repository (Zenodo) upon acceptance.

## Data Availability (text for the manuscript)

> **Data Availability.** The synthetic cohort analysed in this study is fully regenerated by the
> released code (`python run_experiments.py --mode full --stages data`). Its SHA-256 checksum is
> recorded in `generator_summary.json`, so identical regeneration can be verified. The PIMA Indians
> Diabetes Database and the CDC BRFSS 2015 Diabetes Health Indicators dataset are publicly available
> from the UCI Machine Learning Repository.

## Contact

M. Rabiyathul Fathima: mrabiyathulfathima@gmail.com
