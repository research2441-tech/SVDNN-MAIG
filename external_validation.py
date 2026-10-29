"""
Validation on real patient-level data (Reviewer 1 comment 3).

The synthetic cohort cannot establish clinical validity. The complete pipeline (MAIG
selection inside each fold, identical preprocessing, tuning budget and evaluation
protocol) is therefore re-run on real, publicly available patient records:

  1. PIMA Indians Diabetes Database (n = 768, 268 cases; National Institute of Diabetes
     and Digestive and Kidney Diseases). Downloaded automatically. Physiologically
     impossible zeros in glucose, blood pressure, skin thickness, insulin and BMI are
     treated as missing, as is standard for this dataset.
  2. CDC BRFSS 2015 Diabetes Health Indicators (n = 253,680; UCI ML Repository id 891,
     file diabetes_binary_health_indicators_BRFSS2015.csv). Optional: set
     external.brfss_csv in config.yaml to the downloaded file. A stratified random
     subsample (external.brfss_subsample) is used to bound run time.

Neither dataset contains post-COVID variables or repeated measurements; these analyses
test whether the METHOD transfers to real clinical data, not the post-COVID findings.
Temporal baselines are reported as not applicable (no visit sequence exists).
"""
from __future__ import annotations

import copy
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import run_nested_cv
from utils import ensure_dir, get_logger

PIMA_COLUMNS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi",
                "diabetes_pedigree", "age", "outcome"]


def load_pima(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    cache = ensure_dir(Path(cfg["output_dir"]) / "external") / "pima.csv"
    if not cache.exists():
        urllib.request.urlretrieve(cfg["external"]["pima_url"], cache)
    df = pd.read_csv(cache, header=None, names=PIMA_COLUMNS)
    for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df.loc[df[c] == 0, c] = np.nan
    return df.drop(columns="outcome"), df["outcome"].to_numpy().astype(int)


def load_brfss(cfg: dict) -> tuple[pd.DataFrame, np.ndarray] | None:
    path = cfg["external"].get("brfss_csv")
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    target = "Diabetes_binary" if "Diabetes_binary" in df.columns else "Diabetes_012"
    y = (df[target] > 0).astype(int).to_numpy()
    X = df.drop(columns=[target])
    n = min(cfg["external"]["brfss_subsample"], len(y))
    rng = np.random.default_rng(cfg["seed"])
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), int(round(n * (y == c).mean())), replace=False)
                          for c in (0, 1)])
    idx = rng.permutation(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def run_external(cfg: dict) -> None:
    log = get_logger()
    ecfg = copy.deepcopy(cfg)
    ecfg["plausible_ranges"] = {}
    datasets = {"external_pima": load_pima(cfg)}
    brfss = load_brfss(cfg)
    if brfss is not None:
        datasets["external_brfss"] = brfss
    else:
        log.info("BRFSS 2015 not configured (external.brfss_csv); running PIMA only.")
    for name, (X, y) in datasets.items():
        log.info("%s: n = %d, prevalence = %.3f, %d predictors", name, len(y), y.mean(), X.shape[1])
        run_nested_cv(X, y, ecfg, name, cfg["external"]["models"], categorical=[],
                      ablations=["svdnn_no_maig", "svdnn_no_margin"])
