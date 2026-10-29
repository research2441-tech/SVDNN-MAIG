"""
Label-circularity and information-leakage audit (Reviewer 1 comments 4 and 9; Editor).

A. How is the label produced and can the predictors reproduce it?
   A1  single-variable AUROC of every released predictor (no variable determines the label)
   A2  ADA-rule check: the diagnostic rule applied to T2 values flags nobody (prevalent cases
       were excluded); the prediabetes rule (HbA1c >= 5.7 % or FPG >= 100 mg/dL) is reported
       as the trivial clinical benchmark
   A3  latent oracle ceilings: a model given the true latent T2 state (unavailable in practice)
       bounds the achievable AUROC; a model given the latent T3 state (which defines the label)
       shows that the label is determined only after the prediction time
   A4  positive control: a deliberately circular label built from released T2 predictors is
       learned almost perfectly, which is what leakage would look like
B. Does the evaluation pipeline leak?
   B1  permuted-label test of the complete nested pipeline: AUROC must be ~0.5
   B2  selection-bias demonstration: with pure-noise predictors and permuted labels,
       feature selection on the full data before CV yields optimistic AUROC, whereas
       selection nested inside each training fold (as used throughout) does not
   B3  duplicate-record check between released rows
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_generator import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, load_cohort
from evaluation import fold_metric_table, load_results, run_nested_cv
from maig import maig_scores
from utils import ensure_dir, get_logger, save_json


def _cv_auc(model, X, y, seed, k=5):
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def _numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["ethnicity"] = X["ethnicity"].astype("category").cat.codes
    return X


def run_leakage_audit(cfg: dict) -> dict:
    log = get_logger()
    out = ensure_dir(Path(cfg["output_dir"]) / "leakage")
    seed = cfg["seed"]
    X, y, summary = load_cohort(cfg)
    latent = pd.read_csv(Path(cfg["output_dir"]) / "data" / "cohort_latent_audit.csv")
    Xn = _numeric(X)
    res: dict = {}

    # A1 single-variable discrimination (descriptive; missing values median-filled)
    single = {}
    for c in FEATURE_COLUMNS:
        v = Xn[c].fillna(Xn[c].median())
        a = roc_auc_score(y, v)
        single[c] = float(max(a, 1 - a))
    res["single_feature_auc"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))
    res["max_single_feature_auc"] = max(single.values())

    # A2 rule-based checks on released T2 laboratory values
    a1c, fpg = Xn["hba1c_t2"], Xn["fpg_t2"]
    diag = ((a1c >= cfg["data"]["ada_hba1c"]) | (fpg >= cfg["data"]["ada_fpg"])).to_numpy()
    pre = ((a1c >= 5.7) | (fpg >= 100)).to_numpy()
    res["ada_diagnostic_rule_t2"] = {"flagged_fraction": float(diag.mean()),
                                     "note": "values >= 6.5 % / 126 mg/dL appear only through EHR unit errors"}
    res["prediabetes_rule_t2"] = {"sensitivity": float(pre[y == 1].mean()), "specificity": float(1 - pre[y == 0].mean()),
                                  "accuracy": float((pre == y).mean())}

    # A3 oracle ceilings and observed-feature reference
    hgb = HistGradientBoostingClassifier(random_state=seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    lat_t2 = latent[["Y_t0", "Y_t1", "Y_t2", "IR_t2", "B_t2", "R_t2", "D", "A"]]
    res["oracle_latent_t2_auc"] = _cv_auc(hgb, lat_t2, y, seed)
    res["oracle_latent_t3_auc"] = _cv_auc(hgb, latent[["Y_t3"]], y, seed)
    res["observed_features_hgb_auc"] = _cv_auc(hgb, Xn, y, seed)
    res["observed_features_lr_auc"] = _cv_auc(lr, Xn.fillna(Xn.median()), y, seed)

    # A4 positive control: circular label = deterministic function of released predictors
    t2 = Xn[["hba1c_t2", "fpg_t2", "bmi_t2"]].fillna(Xn[["hba1c_t2", "fpg_t2", "bmi_t2"]].median())
    score = (t2["hba1c_t2"] - 5.4) / 0.4 + (t2["fpg_t2"] - 95) / 12 + 0.3 * (t2["bmi_t2"] - 28) / 4.4
    y_circ = (score >= np.quantile(score, 1 - y.mean())).astype(int).to_numpy()
    res["circular_label_positive_control"] = {
        "hgb_auc": _cv_auc(hgb, Xn, y_circ, seed),
        "lr_auc": _cv_auc(lr, Xn.fillna(Xn.median()), y_circ, seed),
        "note": "labels defined from HbA1c_T2, FPG_T2 and BMI_T2; shows what label leakage would produce",
    }

    # B1 permuted-label test of the full nested pipeline
    pcfg = copy.deepcopy(cfg)
    pcfg["cv"]["outer_repeats"] = 1
    pcfg["tuning"]["n_trials"] = 0
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(y)
    perm_dir = run_nested_cv(X, y_perm, pcfg, "leakage_permuted", ["logistic_regression", "svdnn"],
                             CATEGORICAL_COLUMNS)
    recs, _ = load_results(perm_dir)
    fm = fold_metric_table(recs)
    res["permuted_label_nested_auc"] = fm.groupby("model")["roc_auc"].agg(["mean", "std"]).to_dict("index")

    # B2 selection bias: 500 noise predictors, permuted labels, top-10 MAIG features
    n = min(1000, len(y))
    idx = rng.choice(len(y), n, replace=False)
    Z = rng.standard_normal((n, 500))
    yz = rng.permutation(y[idx])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    top_full = np.argsort(-maig_scores(Z, yz))[:10]
    p_naive = cross_val_predict(LogisticRegression(max_iter=2000), Z[:, top_full], yz, cv=cv, method="predict_proba")[:, 1]
    p_nested = np.zeros(n)
    for tr, te in cv.split(Z, yz):
        top = np.argsort(-maig_scores(Z[tr], yz[tr]))[:10]
        p_nested[te] = LogisticRegression(max_iter=2000).fit(Z[tr][:, top], yz[tr]).predict_proba(Z[te][:, top])[:, 1]
    res["selection_bias_demo"] = {"selection_before_cv_auc": float(roc_auc_score(yz, p_naive)),
                                  "selection_nested_in_cv_auc": float(roc_auc_score(yz, p_nested)),
                                  "n": n, "n_noise_features": 500}

    # B3 duplicates
    res["duplicate_rows"] = int(X.duplicated().sum())
    res["source_incidence"] = summary["source_incidence"]
    save_json(res, out / "leakage_audit.json")
    pd.Series(res["single_feature_auc"], name="auroc").to_csv(out / "single_feature_auc.csv")
    log.info("Leakage audit: max single-feature AUC %.3f | oracle(T2) %.3f | observed HGB %.3f | circular control %.3f | "
             "permuted nested %s | naive vs nested selection %.3f / %.3f",
             res["max_single_feature_auc"], res["oracle_latent_t2_auc"], res["observed_features_hgb_auc"],
             res["circular_label_positive_control"]["hgb_auc"],
             {k: round(v["mean"], 3) for k, v in res["permuted_label_nested_auc"].items()},
             res["selection_bias_demo"]["selection_before_cv_auc"], res["selection_bias_demo"]["selection_nested_in_cv_auc"])
    return res
