# Synthetic cohort: complete data-generation procedure

This document specifies every step used to create the synthetic post-COVID cohort analysed in the
manuscript. The implementation is `data_generator.py`, and all numeric settings are in `config.yaml`.
Running `python data_generator.py --mode full` regenerates the released file and prints its SHA-256
checksum, which is also written to `outputs/full/data/generator_summary.json`.

> **Scope.** The cohort is a statistical simulation. Its marginal distributions are anchored to
> published UK Biobank and NHANES summary statistics, and its dependence structure follows
> established physiology. It contains no records of real individuals. Results obtained on it are
> proof-of-concept evidence about the *method*. They are not evidence of clinical diagnostic
> performance. Real-patient validation is reported separately (`external_validation.py`).

## 1. Design overview

| Item | Value |
|---|---|
| Source population simulated | 160,000 individuals with a SARS-CoV-2 infection (`data.n_source`) |
| Time points released as predictors | T0 = pre-infection baseline, T1 = 3 months, T2 = 12 months after infection |
| Outcome time point (never released as predictor) | T3 = T2 + 12 months |
| Outcome | Incident (new-onset) type 2 diabetes at T3 by ADA criteria: HbA1c ≥ 6.5 % or FPG ≥ 126 mg/dL |
| Exclusion | Prevalent diabetes: ADA criteria met at T0, T1 or T2 |
| Sampling into the study | Case-enriched: 4,400 incident cases + 9,200 non-cases drawn without replacement |
| Sample prevalence | 0.324 (neither balanced nor the population incidence) |
| Population incidence | Reported by the generator (≈ 0.075 over 12 months in the eligible source population) |
| Released predictors | 52 (28 static + 8 variables × 3 visits) |
| Random seed | `seed: 2026` (single master seed; NumPy PCG64 generator) |

Case enrichment mirrors a nested case-control design and gives enough events for stable estimation.
Each participant carries a sampling weight (`cohort_latent_audit.csv`). The weights allow
discrimination, calibration and decision-curve analyses to be re-weighted to the source-population
incidence.

## 2. Structural causal model

Notation: `N` is an independent standard normal draw, `z_age = (age − 52.3)/12.1`,
`z_ses = (ses − 3.22)/1.18`, `expit(u) = 1/(1 + e^(−u))`, and `t ∈ {T0, T1, T2}` with times
0, 0.25 and 1.0 years after infection. Latent quantities (G, A, D, I, R, IR, B, Y) are written to
`cohort_latent_audit.csv` for auditing only. They are never available to any model.

**G1 Demographics and genetic liability**
- age ~ truncated normal(52.3, 12.1²) on [18, 85]; sex: male ~ Bernoulli(0.47)
- ethnicity ~ Categorical(White 0.62, South Asian 0.10, Black 0.11, East Asian 0.07, Mixed/Other 0.10)
- SES index = 1 + Binomial(6, 0.37) (mean 3.2, SD 1.2)
- latent genetic liability G ~ N(0, 1); measured polygenic score = 0.75 G + 0.66 N
- family history ~ Bernoulli(expit(−1.3 + 0.9 G))

**G2 Behaviour**
- diet quality = clip(5.4 + 0.6 z_ses + 1.9 N, 1, 10)
- smoking (never/former/current) = ordinal probit on (−0.4 z_ses + N) with shares 0.55/0.30/0.15
- alcohol (units/week) ~ Gamma(1.2, 6) × (1 + 0.3 male)
- pandemic lifestyle disruption D ~ Beta(2, 3)
- screen time = 5.3 × Weibull(2.4) × (0.85 + 0.4 D)

**G3 Time-varying activity and sleep**
- log PA_T0 = ln 18 − 0.15 z_age + 0.12 z_ses + 0.5 N (MET-h/week)
- PA_T1 = PA_T0 (1 − 0.55 D) e^(0.15N); PA_T2 = PA_T0 (1 − 0.25 D) e^(0.15N)
- sleep_T0 = 6.6 + 0.2 z_ses + 1.4 N; sleep_T1 = sleep_T0 − 1.6 D + 0.6 N; sleep_T2 = sleep_T0 − 0.7 D + 0.6 N (clipped 1–10)

**G4 Adiposity**
- A = 0.30 G + 0.15 z_age − 0.20 z_ses − 0.25 z(diet) − 0.25 z(log PA_T0) + 0.10 male + 0.85 N, standardised to z_A
- BMI_T0 = 27.8 + 4.4 z_A; BMI_T1 = BMI_T0 + 1.0 D + 0.6 N; BMI_T2 = BMI_T0 + 0.7 D + 0.15 (screen − 4.7) + 0.8 N

**G5 Vaccination, infection severity, treatment**
- vaccination doses (0–3) = ordinal probit on (0.4 z_age + 0.3 z_ses + N), shares 0.12/0.13/0.35/0.40
- baseline inflammation I0 = 0.35 z_A + 0.25 [current smoker] + 0.10 z_age + 0.85 N
- severity (mild/moderate/severe/critical) = ordinal probit on (0.45 z_age + 0.35 z_A + 0.25 I0 − 0.30 (doses − 2) + N), shares 0.55/0.28/0.12/0.05
- hospitalised ~ Bernoulli(0.01/0.12/0.80/1.00 by severity); steroid = hospitalised × Bernoulli(0.6)
- viral clearance (days) ~ Gamma(6, 1.75) × (0.85 + 0.12 severity)

**G6 Post-viral inflammation (manuscript Eq. 15)**
- surge amplitude S = (0.4 + 0.7 severity) e^(0.3N); patient-specific decay rate δ = 3 × Beta(2, 4) per year
- I_t = I0 + 1[t > 0] S e^(−δ t) + 0.2 N; residual inflammation at T2: R = S e^(−δ)
- CRP_t = exp(ln 1.5 + 0.55 I_t + 0.35 N); IL-6_t = exp(ln 5.6 + 0.32 I_t + 0.25 N); ferritin_t = exp(ln 120 + 0.28 I_t + 0.30 male + 0.35 N)
- D-dimer = exp(ln 0.35 + 0.30 severity + 0.20 I_T1 + 0.4 N); NLR = exp(ln 2.0 + 0.15 I_T1 + 0.10 severity + 0.3 N)
- SpO2 = clip(96.9 − 0.8 severity − 0.25 z_A + 1.1 N, 85, 100)
- fatigue = clip(2.2 + 0.8 severity + 1.0 R + 0.35 (6.6 − sleep_T2) + 1.8 N, 0, 10); long-COVID symptoms ~ Poisson(exp(−0.5 + 0.30 severity + 0.45 R))

**G7 Insulin resistance (IR) and beta-cell reserve (B)**, per visit t, with patient random effects u_IR ~ N(0, 0.45²), u_B ~ N(0, 0.55²)
- IR_t = 0.55 z(BMI_t) + 0.20 G + e_IR(ethnicity) + 0.10 z_age − 0.22 z(log PA_t) − 0.10 z(sleep_t) + 0.20 (I_t − I0) + 0.10 I_t + u_IR + 0.1 N
- B_t = −0.30 G − 0.20 z_age − e_B(ethnicity) − injury_t + u_B + 0.1 N, with injury_T0 = 0, injury_T1 = 0.15 severity + 0.25 steroid, injury_T2 = 0.10 severity + 0.30 R
- ethnicity offsets e_IR = (0, 0.45, 0.25, 0.15, 0.10) and e_B = (0, 0.35, 0.20, 0.30, 0.10)

**G8 Dysglycaemia index** Y_t = IR_t − B_t

**G9 Glycaemic measurements at T0–T2**
- HbA1c_t = 5.40 + 0.33 Y_t + o_i + 0.15 N, where o_i = 0.12 N + 0.10 [Black] is a persistent non-glycaemic component (erythrocyte lifespan, iron status)
- FPG_t = (93 + 9.0 Y_t) e^(0.07N); fasting insulin = exp(ln 9.5 + 0.40 IR_T2 + 0.30 N)

**G10 Cardiometabolic correlates** (descendants of IR/adiposity with **no path into the outcome**)
- triglycerides = exp(ln 125 + 0.22 IR_T2 + 0.012 alcohol + 0.30 N); HDL = 54 − 5 IR_T2 − 7 male + 0.2 alcohol + 10 N
- LDL = 128 + 3 z_A + 30 N; SBP = 127 + 6 z_age + 3.5 z_A + 4 male + 13 N; DBP = 0.45 SBP + 22 + 7 N
- waist = 93 + 2.4 (BMI_T2 − 27.8) + 7 male + 4.5 N; self-reported activity days = clip(round(3.4 + 1.3 z(log PA_T2) + 1.5 N), 0, 7)

**G11 Progression and outcome (label generation)**

```
decompensation = expit(1.6 (IR_T2 − B_T2 − 1))
Y_T3 = Y_T2 − 1.00
       + 1.80 R × decompensation                 (residual inflammation × beta-cell failure)
       + 0.60 z(D) × expit(1.5 z(BMI_T2))        (lifestyle disruption × adiposity)
       + 0.60 steroid × decompensation
       + 0.15 (Y_T2 − Y_T0)                       (trajectory)
       + 1.50 N                                    (unexplained 12-month progression)
HbA1c_T3 = 5.40 + 0.33 Y_T3 + o_i + 0.15 N
FPG_T3   = (93 + 9.0 Y_T3) e^(0.07N)
incident_t2dm = 1[HbA1c_T3 ≥ 6.5  or  FPG_T3 ≥ 126]
```

## 3. Why the label is not circular

1. The label is defined from **T3** laboratory values, which are never released. Released HbA1c and
   FPG are earlier, noisy measurements of the latent process.
2. People who met ADA criteria at any released visit are excluded. No released value lies in the
   diagnostic range, apart from deliberately injected unit-recording errors, which the plausibility
   filter removes.
3. Between T2 and T3 the label depends on an independent progression term (SD 1.5) and on fresh
   measurement error. Even a model given the *true latent T2 state* cannot predict it perfectly
   (oracle ceiling in `leakage_audit.py`).
4. Interactions in G11 (inflammation × decompensation, disruption × adiposity) make the outcome a
   non-linear function of the latent state rather than a threshold on any released variable.
5. `leakage_audit.py` quantifies all of this. It reports the best single-variable AUROC, the
   prediabetes-rule benchmark, latent-oracle ceilings, a positive control in which the label *is*
   circular (AUROC ≈ 1), a permuted-label run of the full pipeline (AUROC ≈ 0.5), and a
   demonstration of selection bias when feature selection is not nested.

## 4. EHR artefacts and missing data

| Variable(s) | Mechanism | Rate |
|---|---|---|
| IL-6 T0 | MCAR (rarely measured pre-pandemic) | 45 % |
| IL-6 T1, T2 | MAR on hospitalisation | 5 % hospitalised, 30 % otherwise |
| D-dimer | MAR on hospitalisation | 3 % hospitalised, 40 % otherwise |
| Ferritin T0–T2 / CRP T0–T2 | MCAR | 15 % / 6 % |
| Fasting insulin | MAR on SES | 18 % if SES ≤ 2, 10 % otherwise |
| Sleep T1 / physical activity T1 | MCAR | 10 % / 8 % |
| BMI, HbA1c, FPG (all visits), lipids, BP, waist | MCAR | 1 % |
| FPG recorded in mmol/L; HbA1c recorded in mmol/mol | unit-recording error | 0.2 % per visit |

Missingness depends only on fully observed variables (MAR/MCAR). Within each training fold, values
outside the physiological ranges in `config.yaml` (`plausible_ranges`) are set to missing. All
missing values are then imputed by iterative Bayesian-ridge imputation fitted on that training fold.

## 5. Causal role of every released variable

`data_dictionary.csv` labels each variable with its role in the model above. The labels are used to
test whether feature rankings or attributions can be read causally.

- **glycaemic_marker**: HbA1c, FPG (T0–T2), fasting insulin. These measure the outcome process
  itself, so they are highly predictive but are not intervention targets.
- **upstream_determinant**: on a causal path into glycaemic deterioration (age, sex, ethnicity, SES,
  genetics, family history, diet, activity, sleep, screen time, smoking, BMI, waist,
  CRP/IL-6/ferritin, severity, hospitalisation, steroids, vaccination).
- **non_causal_correlate**: shares causes with the outcome but has no effect on it
  (triglycerides, HDL, LDL, blood pressure, D-dimer, NLR, SpO2, fatigue, viral clearance,
  long-COVID symptoms).
- **independent**: alcohol intake (it affects lipids only).

## 6. Assumptions and limitations

- Coefficients encode published directions of effect (adiposity, inactivity, poor sleep and
  inflammation raise insulin resistance; severe infection and steroid exposure transiently impair
  beta-cell reserve). Their magnitudes were calibrated so that the marginal statistics and the
  12-month incidence are plausible. They are not estimates from individual-level data.
- Effects are homogeneous across subgroups except for the stated ethnicity offsets. Real
  heterogeneity is likely larger.
- Missingness is MAR by construction. Real EHR data may be missing not at random.
- Because the generator is known, models may partly learn its functional form. Performance on this
  cohort therefore cannot be transferred to clinical practice without external validation.
