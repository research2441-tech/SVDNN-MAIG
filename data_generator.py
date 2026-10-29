"""
Synthetic post-COVID incident type 2 diabetes cohort.

The cohort is produced by an explicit structural causal model (SCM) whose
parameters are anchored to published UK Biobank / NHANES summary statistics.
The design addresses three concerns raised during review:

1. Reproducibility  - every distribution, coefficient, correlation path,
   missing-data mechanism and sampling step is defined in this file and in
   config.yaml; the run writes a data dictionary, realised summary statistics
   and a SHA-256 checksum of the released CSV.
2. Label circularity - the outcome is NEW-ONSET diabetes at follow-up T3,
   ascertained by ADA criteria applied to T3 laboratory values. T3 values are
   never released as predictors. Participants who already met ADA criteria at
   T0, T1 or T2 are excluded (prevalent disease), so no predictor contains a
   diagnostic-range value and the label is not a deterministic function of
   any released variable (see leakage_audit.py).
3. Prevalence - the study sample is a case-enriched draw (4,400 cases and
   9,200 controls) from a source population whose incidence is reported, so
   metrics can be re-weighted to the population prevalence.

Time points: T0 = pre-infection baseline, T1 = 3 months post-infection,
T2 = 12 months post-infection (all released); T3 = T2 + 12 months (outcome only).

Run:  python data_generator.py --mode full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from utils import ensure_dir, environment_info, get_logger, load_config, save_json, sha256_file

ETHNICITIES = ["White", "South_Asian", "Black", "East_Asian", "Mixed_Other"]
ETH_PROBS = [0.62, 0.10, 0.11, 0.07, 0.10]
ETH_IR_OFFSET = np.array([0.00, 0.45, 0.25, 0.15, 0.10])     # insulin-resistance shift
ETH_BETA_OFFSET = np.array([0.00, 0.35, 0.20, 0.30, 0.10])   # beta-cell reserve deficit
P_DRIFT = -1.00     # intercept of the 12-month change in the dysglycaemia index (sets source incidence ~7.5%)
P_NOISE = 1.50      # SD of unexplained 12-month progression (irreducible label noise)

DYNAMIC_VARS = ["bmi", "hba1c", "fpg", "crp", "il6", "ferritin", "physical_activity", "sleep_quality"]
TIMEPOINTS = ["t0", "t1", "t2"]
DYNAMIC_COLUMNS = [f"{v}_{t}" for v in DYNAMIC_VARS for t in TIMEPOINTS]
STATIC_COLUMNS = [
    "age", "sex_male", "ethnicity", "ses_index", "family_history", "genetic_risk_score",
    "waist_cm", "sbp", "dbp", "triglycerides", "hdl_cholesterol", "ldl_cholesterol",
    "fasting_insulin", "d_dimer", "nlr", "diet_quality", "screen_time", "smoking_status",
    "alcohol_units", "covid_severity", "hospitalised", "vaccination_doses", "steroid_treatment",
    "oxygen_saturation", "fatigue_score", "viral_clearance_days", "long_covid_symptoms",
    "self_reported_activity_days",
]
FEATURE_COLUMNS = STATIC_COLUMNS + DYNAMIC_COLUMNS        # 28 + 24 = 52 released variables
CATEGORICAL_COLUMNS = ["ethnicity"]
LABEL = "incident_t2dm"

# ---------------------------------------------------------------------------
# Data dictionary: domain, unit, generating mechanism and causal role.
# causal role (with respect to the outcome in the SCM):
#   upstream_determinant : the variable (or a direct measurement of it) lies on
#                          a causal path into glycaemic deterioration
#   glycaemic_marker     : measurement of the glycaemic / insulin-resistance
#                          process itself (predictive, not an intervention target)
#   non_causal_correlate : shares causes with the outcome but has no path into it
#   independent          : no path to or from the outcome process
# ---------------------------------------------------------------------------
DICTIONARY = {
    "age": ("demographic", "years", "truncated normal (52.3, 12.1) on [18, 85]", "upstream_determinant"),
    "sex_male": ("demographic", "binary", "Bernoulli(0.47)", "upstream_determinant"),
    "ethnicity": ("demographic", "category", "categorical (0.62/0.10/0.11/0.07/0.10)", "upstream_determinant"),
    "ses_index": ("demographic", "1-7", "1 + Binomial(6, 0.37)", "upstream_determinant"),
    "family_history": ("demographic", "binary", "Bernoulli(expit(-1.3 + 0.9 G))", "upstream_determinant"),
    "genetic_risk_score": ("demographic", "z", "0.75 G + 0.66 e", "upstream_determinant"),
    "waist_cm": ("clinical", "cm", "93 + 2.4 (BMI_T2 - 27.8) + 7 male + N(0, 4.5^2)", "upstream_determinant"),
    "sbp": ("clinical", "mmHg", "127 + 6 z_age + 3.5 z_A + 4 male + N(0, 13^2)", "non_causal_correlate"),
    "dbp": ("clinical", "mmHg", "0.45 SBP + 22 + N(0, 7^2)", "non_causal_correlate"),
    "triglycerides": ("clinical", "mg/dL", "log-normal, log-mean ln125 + 0.22 IR_T2 + 0.012 alcohol", "non_causal_correlate"),
    "hdl_cholesterol": ("clinical", "mg/dL", "54 - 5 IR_T2 - 7 male + 0.2 alcohol + N(0, 10^2)", "non_causal_correlate"),
    "ldl_cholesterol": ("clinical", "mg/dL", "128 + 3 z_A + N(0, 30^2)", "non_causal_correlate"),
    "fasting_insulin": ("clinical", "uU/mL", "log-normal, log-mean ln9.5 + 0.40 IR_T2", "glycaemic_marker"),
    "d_dimer": ("inflammatory", "mg/L FEU", "log-normal, log-mean ln0.35 + 0.30 severity + 0.20 I_T1", "non_causal_correlate"),
    "nlr": ("inflammatory", "ratio", "log-normal, log-mean ln2.0 + 0.15 I_T1 + 0.10 severity", "non_causal_correlate"),
    "diet_quality": ("behavioural", "1-10", "5.4 + 0.6 z_ses + N(0, 1.9^2), clipped", "upstream_determinant"),
    "screen_time": ("behavioural", "h/day", "Weibull(k = 2.4, scale 5.3) x (0.85 + 0.4 D)", "upstream_determinant"),
    "smoking_status": ("behavioural", "0/1/2", "ordinal probit: never / former / current", "upstream_determinant"),
    "alcohol_units": ("behavioural", "units/week", "Gamma(1.2, 6) x (1 + 0.3 male)", "independent"),
    "covid_severity": ("covid", "0-3", "ordinal probit on age, adiposity, inflammation, vaccination", "upstream_determinant"),
    "hospitalised": ("covid", "binary", "Bernoulli(p_severity) with p = 0.01/0.12/0.80/1.0", "upstream_determinant"),
    "vaccination_doses": ("covid", "0-3", "ordinal probit on age and SES", "upstream_determinant"),
    "steroid_treatment": ("covid", "binary", "hospitalised x Bernoulli(0.6)", "upstream_determinant"),
    "oxygen_saturation": ("covid", "%", "96.9 - 0.8 severity - 0.25 z_A + N(0, 1.1^2), clipped [85, 100]", "non_causal_correlate"),
    "fatigue_score": ("covid", "0-10", "2.2 + 0.8 severity + 1.0 R + 0.35 (6.6 - sleep_T2) + N(0, 1.8^2)", "non_causal_correlate"),
    "viral_clearance_days": ("covid", "days", "Gamma(6, 1.75) x (0.85 + 0.12 severity)", "non_causal_correlate"),
    "long_covid_symptoms": ("covid", "count", "Poisson(exp(-0.5 + 0.30 severity + 0.45 R))", "non_causal_correlate"),
    "self_reported_activity_days": ("behavioural", "days/week", "round(3.4 + 1.3 z_PA_T2 + N(0, 1.5^2)) in [0, 7]", "upstream_determinant"),
}
for _t in TIMEPOINTS:
    DICTIONARY[f"bmi_{_t}"] = ("clinical", "kg/m2", "27.8 + 4.4 z_A + lockdown gain (Eq. G4)", "upstream_determinant")
    DICTIONARY[f"hba1c_{_t}"] = ("clinical", "%", "5.40 + 0.33 Y_t + o_i + N(0, 0.15^2) (Eq. G9)", "glycaemic_marker")
    DICTIONARY[f"fpg_{_t}"] = ("clinical", "mg/dL", "(93 + 9.0 Y_t) exp(N(0, 0.07^2)) (Eq. G9)", "glycaemic_marker")
    DICTIONARY[f"crp_{_t}"] = ("inflammatory", "mg/L", "log-normal, log-mean ln1.5 + 0.55 I_t (Eq. G6)", "upstream_determinant")
    DICTIONARY[f"il6_{_t}"] = ("inflammatory", "pg/mL", "log-normal, log-mean ln5.6 + 0.32 I_t (Eq. G6)", "upstream_determinant")
    DICTIONARY[f"ferritin_{_t}"] = ("inflammatory", "ng/mL", "log-normal, log-mean ln120 + 0.28 I_t + 0.30 male", "upstream_determinant")
    DICTIONARY[f"physical_activity_{_t}"] = ("behavioural", "MET-h/week", "log-normal with pandemic disruption D (Eq. G3)", "upstream_determinant")
    DICTIONARY[f"sleep_quality_{_t}"] = ("behavioural", "1-10", "6.6 + 0.2 z_ses + N(0, 1.4^2) - disruption (Eq. G3)", "upstream_determinant")


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _ordinal(liability: np.ndarray, probs: list[float]) -> np.ndarray:
    """Ordinal variable by thresholding a liability at its empirical quantiles."""
    cuts = np.quantile(liability, np.cumsum(probs)[:-1])
    return np.searchsorted(cuts, liability, side="right")


def simulate_source_population(n: int, cfg: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the full source population. Returns (observed variables, latent variables)."""
    t2 = cfg["data"]["t_years"][2]
    N = lambda: rng.standard_normal(n)  # noqa: E731

    # --- (G1) demographics and genetic liability ---------------------------------
    a, b = (18 - 52.3) / 12.1, (85 - 52.3) / 12.1
    age = stats.truncnorm.rvs(a, b, loc=52.3, scale=12.1, size=n, random_state=rng)
    male = (rng.random(n) < 0.47).astype(int)
    eth = rng.choice(len(ETHNICITIES), size=n, p=ETH_PROBS)
    ses = 1 + rng.binomial(6, 0.37, size=n)
    G = N()                                                     # latent genetic liability
    grs = 0.75 * G + np.sqrt(1 - 0.75**2) * N()                 # measured polygenic score
    famhx = (rng.random(n) < expit(-1.3 + 0.9 * G)).astype(int)
    z_age, z_ses = (age - 52.3) / 12.1, (ses - 3.22) / 1.18

    # --- (G2) behaviour --------------------------------------------------------------
    diet = np.clip(5.4 + 0.6 * z_ses + 1.9 * N(), 1, 10)
    smoking = _ordinal(-0.4 * z_ses + N(), [0.55, 0.30, 0.15])
    alcohol = rng.gamma(1.2, 6.0, size=n) * (1 + 0.3 * male)
    D = rng.beta(2, 3, size=n)                                  # pandemic lifestyle disruption
    screen = 5.3 * rng.weibull(2.4, size=n) * (0.85 + 0.4 * D)

    # --- (G3) time-varying activity and sleep ----------------------------------------
    log_pa0 = np.log(18) - 0.15 * z_age + 0.12 * z_ses + 0.5 * N()
    pa = np.stack([
        np.exp(log_pa0),
        np.exp(log_pa0 + np.log(1 - 0.55 * D) + 0.15 * N()),
        np.exp(log_pa0 + np.log(1 - 0.25 * D) + 0.15 * N()),
    ], axis=1)
    sleep0 = 6.6 + 0.2 * z_ses + 1.4 * N()
    sleep = np.clip(np.stack([sleep0, sleep0 - 1.6 * D + 0.6 * N(), sleep0 - 0.7 * D + 0.6 * N()], axis=1), 1, 10)
    z_logpa = (np.log(pa) - np.log(pa[:, 0]).mean()) / np.log(pa[:, 0]).std()
    z_sleep = (sleep - 6.6) / 1.45

    # --- (G4) adiposity ---------------------------------------------------------------
    A = 0.30 * G + 0.15 * z_age - 0.20 * z_ses - 0.25 * _z(diet) - 0.25 * z_logpa[:, 0] + 0.10 * male + 0.85 * N()
    z_A = _z(A)
    bmi0 = np.clip(27.8 + 4.4 * z_A, 16, 60)
    bmi = np.stack([
        bmi0,
        bmi0 + 1.0 * D + 0.6 * N(),
        bmi0 + 0.7 * D + 0.15 * (screen - 4.7) + 0.8 * N(),
    ], axis=1)
    z_bmi = (bmi - 27.8) / 4.4

    # --- (G5) vaccination, infection severity and treatment --------------------------
    vacc = _ordinal(0.4 * z_age + 0.3 * z_ses + N(), [0.12, 0.13, 0.35, 0.40])
    I0 = 0.35 * z_A + 0.25 * (smoking == 2) + 0.10 * z_age + 0.85 * N()
    sev = _ordinal(0.45 * z_age + 0.35 * z_A + 0.25 * I0 - 0.30 * (vacc - 2) + N(), [0.55, 0.28, 0.12, 0.05])
    hosp = (rng.random(n) < np.array([0.01, 0.12, 0.80, 1.0])[sev]).astype(int)
    steroid = (hosp * (rng.random(n) < 0.6)).astype(int)
    clearance = rng.gamma(6.0, 1.75, size=n) * (0.85 + 0.12 * sev)

    # --- (G6) post-viral inflammation with patient-specific exponential decay ------
    surge = (0.4 + 0.7 * sev) * np.exp(0.3 * N())             # post-viral surge amplitude
    delta = 3.0 * rng.beta(2, 4, size=n)                      # decay rate per year (mean 1.0)
    times = np.array(cfg["data"]["t_years"])
    infl = I0[:, None] + (times[None, :] > 0) * surge[:, None] * np.exp(-delta[:, None] * times[None, :]) \
        + 0.2 * rng.standard_normal((n, 3))
    R = surge * np.exp(-delta * t2)                           # residual inflammation at T2
    crp = np.exp(np.log(1.5) + 0.55 * infl + 0.35 * rng.standard_normal((n, 3)))
    il6 = np.exp(np.log(5.6) + 0.32 * infl + 0.25 * rng.standard_normal((n, 3)))
    ferritin = np.exp(np.log(120) + 0.28 * infl + 0.30 * male[:, None] + 0.35 * rng.standard_normal((n, 3)))
    d_dimer = np.exp(np.log(0.35) + 0.30 * sev + 0.20 * infl[:, 1] + 0.4 * N())
    nlr = np.exp(np.log(2.0) + 0.15 * infl[:, 1] + 0.10 * sev + 0.3 * N())
    o2 = np.clip(96.9 - 0.8 * sev - 0.25 * z_A + 1.1 * N(), 85, 100)
    fatigue = np.clip(2.2 + 0.8 * sev + 1.0 * R + 0.35 * (6.6 - sleep[:, 2]) + 1.8 * N(), 0, 10)
    long_covid = rng.poisson(np.exp(-0.5 + 0.30 * sev + 0.45 * R))

    # --- (G7) insulin resistance and beta-cell reserve (latent, per time point) -----
    u_ir, u_b = 0.45 * N(), 0.55 * N()                        # patient random effects
    IR = (0.55 * z_bmi + 0.20 * G[:, None] + ETH_IR_OFFSET[eth][:, None] + 0.10 * z_age[:, None]
          - 0.22 * z_logpa - 0.10 * z_sleep + 0.20 * (infl - I0[:, None]) + 0.10 * infl
          + u_ir[:, None] + 0.10 * rng.standard_normal((n, 3)))
    injury = np.stack([np.zeros(n), 0.15 * sev + 0.25 * steroid, 0.10 * sev + 0.30 * R], axis=1)
    B = (-0.30 * G - 0.20 * z_age - ETH_BETA_OFFSET[eth])[:, None] - injury + u_b[:, None] \
        + 0.10 * rng.standard_normal((n, 3))
    Y = IR - B                                                # (G8) dysglycaemia index

    # --- (G9) glycaemic measurements at T0-T2 ----------------------------------------
    # HbA1c carries a persistent non-glycaemic component (erythrocyte lifespan, iron status,
    # ethnicity-related glycation differences) in addition to analytical error.
    g_offset = 0.12 * N() + 0.10 * (eth == ETHNICITIES.index("Black"))
    hba1c = 5.40 + 0.33 * Y + g_offset[:, None] + 0.15 * rng.standard_normal((n, 3))
    fpg = (93 + 9.0 * Y) * np.exp(0.07 * rng.standard_normal((n, 3)))
    insulin = np.exp(np.log(9.5) + 0.40 * IR[:, 2] + 0.30 * N())

    # --- (G10) cardiometabolic correlates (no path into the outcome) ------------------
    tg = np.exp(np.log(125) + 0.22 * IR[:, 2] + 0.012 * alcohol + 0.30 * N())
    hdl = 54 - 5 * IR[:, 2] - 7 * male + 0.2 * alcohol + 10 * N()
    ldl = 128 + 3 * z_A + 30 * N()
    sbp = 127 + 6 * z_age + 3.5 * z_A + 4 * male + 13 * N()
    dbp = 0.45 * sbp + 22 + 7 * N()
    waist = 93 + 2.4 * (bmi[:, 2] - 27.8) + 7 * male + 4.5 * N()
    self_act = np.clip(np.round(3.4 + 1.3 * z_logpa[:, 2] + 1.5 * N()), 0, 7)

    # --- (G11) progression to T3 and outcome ascertainment ---------------------------
    # Progression is non-linear: residual inflammation accelerates deterioration mainly when
    # insulin resistance exceeds beta-cell reserve (decompensation), and lifestyle disruption
    # acts mainly in people with excess adiposity.
    decompensation = expit(1.6 * (IR[:, 2] - B[:, 2] - 1.0))
    Y3 = (Y[:, 2] + P_DRIFT
          + 1.80 * R * decompensation
          + 0.60 * ((D - 0.4) / 0.2) * expit(1.5 * z_bmi[:, 2])
          + 0.60 * steroid * decompensation
          + 0.15 * (Y[:, 2] - Y[:, 0])
          + P_NOISE * N())
    hba1c3 = 5.40 + 0.33 * Y3 + g_offset + 0.15 * N()
    fpg3 = (93 + 9.0 * Y3) * np.exp(0.07 * N())
    thr_a, thr_f = cfg["data"]["ada_hba1c"], cfg["data"]["ada_fpg"]
    outcome = ((hba1c3 >= thr_a) | (fpg3 >= thr_f)).astype(int)
    prevalent = ((hba1c >= thr_a) | (fpg >= thr_f)).any(axis=1)

    obs = {
        "age": age, "sex_male": male, "ethnicity": np.array(ETHNICITIES)[eth], "ses_index": ses,
        "family_history": famhx, "genetic_risk_score": grs, "waist_cm": waist, "sbp": sbp, "dbp": dbp,
        "triglycerides": tg, "hdl_cholesterol": hdl, "ldl_cholesterol": ldl, "fasting_insulin": insulin,
        "d_dimer": d_dimer, "nlr": nlr, "diet_quality": diet, "screen_time": screen,
        "smoking_status": smoking, "alcohol_units": alcohol, "covid_severity": sev, "hospitalised": hosp,
        "vaccination_doses": vacc, "steroid_treatment": steroid, "oxygen_saturation": o2,
        "fatigue_score": fatigue, "viral_clearance_days": clearance, "long_covid_symptoms": long_covid,
        "self_reported_activity_days": self_act,
    }
    for arr, name in [(bmi, "bmi"), (hba1c, "hba1c"), (fpg, "fpg"), (crp, "crp"), (il6, "il6"),
                      (ferritin, "ferritin"), (pa, "physical_activity"), (sleep, "sleep_quality")]:
        for j, t in enumerate(TIMEPOINTS):
            obs[f"{name}_{t}"] = arr[:, j]
    observed = pd.DataFrame(obs)[FEATURE_COLUMNS]
    observed[LABEL] = outcome

    latent = pd.DataFrame({
        "G": G, "A": z_A, "D": D, "I0": I0, "surge": surge, "delta": delta, "R_t2": R,
        "IR_t0": IR[:, 0], "IR_t2": IR[:, 2], "B_t0": B[:, 0], "B_t2": B[:, 2],
        "Y_t0": Y[:, 0], "Y_t1": Y[:, 1], "Y_t2": Y[:, 2], "Y_t3": Y3,
        "hba1c_t3": hba1c3, "fpg_t3": fpg3, "prevalent": prevalent.astype(int),
    })
    return observed, latent


def inject_ehr_artefacts(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """MAR / MCAR missingness and unit-recording errors. Only observed covariates drive missingness."""
    df = df.copy()
    n = len(df)
    hosp = df["hospitalised"].to_numpy() == 1
    low_ses = df["ses_index"].to_numpy() <= 2

    def drop(col: str, p: np.ndarray | float) -> None:
        mask = rng.random(n) < p
        df.loc[mask, col] = np.nan

    # MAR mechanisms (probabilities depend only on fully observed variables)
    drop("il6_t0", 0.45)
    drop("il6_t1", np.where(hosp, 0.05, 0.30))
    drop("il6_t2", np.where(hosp, 0.05, 0.30))
    drop("d_dimer", np.where(hosp, 0.03, 0.40))
    for t in TIMEPOINTS:
        drop(f"ferritin_{t}", 0.15)
        drop(f"crp_{t}", 0.06)
    drop("fasting_insulin", np.where(low_ses, 0.18, 0.10))
    drop("sleep_quality_t1", 0.10)
    drop("physical_activity_t1", 0.08)
    # MCAR on routine measurements
    mcar_cols = [c for c in df.columns if c.split("_t")[0] in ("bmi", "hba1c", "fpg")] + \
                ["triglycerides", "hdl_cholesterol", "ldl_cholesterol", "sbp", "dbp", "waist_cm"]
    for c in mcar_cols:
        drop(c, cfg["data"]["mcar_rate"])
    # unit-recording errors: FPG stored in mmol/L, HbA1c stored in IFCC mmol/mol
    err = cfg["data"]["ehr_unit_error_rate"]
    for t in TIMEPOINTS:
        m = rng.random(n) < err
        df.loc[m, f"fpg_{t}"] = df.loc[m, f"fpg_{t}"] / 18.016
        m = rng.random(n) < err
        df.loc[m, f"hba1c_{t}"] = (df.loc[m, f"hba1c_{t}"] - 2.152) / 0.09148
    return df


def generate_cohort(cfg: dict) -> dict:
    """Generate, sample, and write the released cohort plus audit files."""
    log = get_logger()
    rng = np.random.default_rng(cfg["seed"])
    out = ensure_dir(Path(cfg["output_dir"]) / "data")
    dcfg = cfg["data"]

    observed, latent = simulate_source_population(dcfg["n_source"], cfg, rng)
    eligible = latent["prevalent"].to_numpy() == 0
    obs_e, lat_e = observed[eligible].reset_index(drop=True), latent[eligible].reset_index(drop=True)
    y_e = obs_e[LABEL].to_numpy()
    n_case_src, n_ctrl_src = int(y_e.sum()), int((1 - y_e).sum())
    if n_case_src < dcfg["n_cases"] or n_ctrl_src < dcfg["n_controls"]:
        raise RuntimeError(f"Source population too small: {n_case_src} cases / {n_ctrl_src} controls available.")

    # case-enriched sampling without replacement
    case_idx = rng.choice(np.flatnonzero(y_e == 1), dcfg["n_cases"], replace=False)
    ctrl_idx = rng.choice(np.flatnonzero(y_e == 0), dcfg["n_controls"], replace=False)
    idx = rng.permutation(np.concatenate([case_idx, ctrl_idx]))
    sample = obs_e.iloc[idx].reset_index(drop=True)
    lat_sample = lat_e.iloc[idx].reset_index(drop=True)
    weights = np.where(sample[LABEL] == 1, n_case_src / dcfg["n_cases"], n_ctrl_src / dcfg["n_controls"])

    released = inject_ehr_artefacts(sample, cfg, rng)
    released.insert(0, "patient_id", [f"P{i:06d}" for i in range(len(released))])
    lat_sample.insert(0, "patient_id", released["patient_id"])
    lat_sample["sampling_weight"] = weights
    lat_sample[LABEL] = released[LABEL]

    # rounding to fixed precision makes the CSV byte-identical across platforms
    num_cols = released.select_dtypes(include="number").columns
    released[num_cols] = released[num_cols].round(4)
    cohort_path = out / "cohort.csv"
    released.to_csv(cohort_path, index=False)
    lat_sample.round(6).to_csv(out / "cohort_latent_audit.csv", index=False)

    dictionary = build_dictionary(released)
    dictionary.to_csv(out / "data_dictionary.csv", index=False)

    source_incidence = n_case_src / (n_case_src + n_ctrl_src)
    summary = {
        "n_source": dcfg["n_source"],
        "n_prevalent_excluded": int((~eligible).sum()),
        "n_eligible": int(eligible.sum()),
        "source_incidence": source_incidence,
        "n_cases_sampled": dcfg["n_cases"],
        "n_controls_sampled": dcfg["n_controls"],
        "sample_prevalence": dcfg["n_cases"] / (dcfg["n_cases"] + dcfg["n_controls"]),
        "n_features": len(FEATURE_COLUMNS),
        "overall_missing_fraction": float(released[FEATURE_COLUMNS].isna().mean().mean()),
        "cohort_sha256": sha256_file(cohort_path),
        "seed": cfg["seed"],
        "environment": environment_info(),
    }
    save_json(summary, out / "generator_summary.json")
    log.info("Cohort written: %d participants (%d cases, prevalence %.3f); source incidence %.3f; sha256 %s",
             len(released), dcfg["n_cases"], summary["sample_prevalence"], source_incidence,
             summary["cohort_sha256"][:16])
    return summary


def build_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLUMNS:
        domain, unit, mech, role = DICTIONARY[col]
        s = df[col]
        num = pd.to_numeric(s, errors="coerce") if col not in CATEGORICAL_COLUMNS else None
        timepoint = col.rsplit("_", 1)[1].upper() if col in DYNAMIC_COLUMNS else "static"
        rows.append({
            "variable": col, "domain": domain, "unit": unit, "timepoint": timepoint,
            "generating_mechanism": mech, "causal_role": role,
            "missing_pct": round(100 * s.isna().mean(), 2),
            "mean": None if num is None else round(float(num.mean()), 3),
            "sd": None if num is None else round(float(num.std()), 3),
            "p05": None if num is None else round(float(num.quantile(0.05)), 3),
            "p95": None if num is None else round(float(num.quantile(0.95)), 3),
        })
    return pd.DataFrame(rows)


def load_cohort(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load the released cohort (generate it first if absent)."""
    path = Path(cfg["output_dir"]) / "data" / "cohort.csv"
    if not path.exists():
        generate_cohort(cfg)
    df = pd.read_csv(path)
    from utils import load_json
    summary = load_json(Path(cfg["output_dir"]) / "data" / "generator_summary.json")
    return df[FEATURE_COLUMNS], df[LABEL].to_numpy().astype(int), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="full", choices=["quick", "full"])
    args = parser.parse_args()
    generate_cohort(load_config(args.config, args.mode))


if __name__ == "__main__":
    main()
