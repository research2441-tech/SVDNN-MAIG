"""
Controlled robustness study of MAIG against conventional feature-ranking methods
(Reviewer 1 comments 7 and 8).

Data with a KNOWN ground truth are simulated so that ranking quality can be measured:
  * n_informative predictors with decreasing true effect sizes (half linear, half
    threshold-shaped effects),
  * n_redundant near-copies of informative predictors (|rho| ~ 0.9),
  * n_noise predictors independent of the outcome.
One factor is varied at a time around a base condition
(prevalence 0.30, label noise 0.05, inter-predictor correlation 0, feature noise 0.5):
  prevalence (class imbalance), label-flip noise, correlation among predictors and
  measurement noise on the predictors.

Metrics (mean over replicates, with 95% CI):
  relevance_auc   AUROC of the score for separating informative+redundant from noise features
  precision_at_k  share of truly informative (non-redundant) features in the top-k, k = n_informative
                  (uses the method's own ordering; MAIG uses its redundancy-aware greedy order)
  spearman_truth  rank correlation between scores of informative features and true effect sizes
  stability       Jaccard overlap of the top-k sets obtained on two random halves of the data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from maig import (MAIGSelector, anova_f, best_split_information_gain, information_gain, maig_scores,
                  mrmr_order, mutual_information, relieff)
from utils import derive_seed, ensure_dir, get_logger, save_json

METHODS = ["maig", "maig_unbalanced", "maig_one_sided", "information_gain", "best_split_ig",
           "mutual_information", "relieff", "mrmr", "anova_f"]


def simulate(n, n_inf, n_red, n_noise, prevalence, label_noise, rho, feat_noise, rng):
    cov = np.full((n_inf, n_inf), rho) + (1 - rho) * np.eye(n_inf)
    Z = rng.multivariate_normal(np.zeros(n_inf), cov, size=n)
    beta = np.linspace(1.0, 0.25, n_inf)
    effects = np.empty_like(Z)
    half = n_inf // 2
    effects[:, :half] = Z[:, :half]
    step = (Z[:, half:] > 0.5).astype(float)
    effects[:, half:] = (step - step.mean(0)) / step.std(0)
    eta = effects @ beta + rng.logistic(size=n)
    y = (eta >= np.quantile(eta, 1 - prevalence)).astype(int)
    flip = rng.random(n) < label_noise
    y[flip] = 1 - y[flip]
    X_inf = Z + feat_noise * rng.standard_normal(Z.shape)
    src = np.arange(n_red) % n_inf
    X_red = 0.9 * Z[:, src] + np.sqrt(1 - 0.81) * rng.standard_normal((n, n_red)) + feat_noise * rng.standard_normal((n, n_red))
    X_noise = rng.standard_normal((n, n_noise))
    X = np.hstack([X_inf, X_red, X_noise])
    role = np.array(["informative"] * n_inf + ["redundant"] * n_red + ["noise"] * n_noise)
    return X, y, role, beta


def method_scores_and_order(method, X, y, seed, n_inf):
    """Return (score per feature or None, ordering)."""
    if method == "maig":
        sel = MAIGSelector(balanced=True, redundancy_eta=0.5, gamma=0.0, min_features=X.shape[1]).fit(X, y)
        return sel.scores_, sel.order_
    if method == "maig_unbalanced":
        s = maig_scores(X, y, balanced=False)
    elif method == "maig_one_sided":
        s = maig_scores(X, y, balanced=False, variant="one_sided")
    elif method == "information_gain":
        s = information_gain(X, y)
    elif method == "best_split_ig":
        s = best_split_information_gain(X, y)
    elif method == "mutual_information":
        s = mutual_information(X, y, seed)
    elif method == "relieff":
        s = relieff(X, y, seed=seed)
    elif method == "anova_f":
        s = anova_f(X, y)
    elif method == "mrmr":
        order = mrmr_order(X, y, k=X.shape[1], seed=seed)
        s = np.empty(X.shape[1])
        s[order] = np.linspace(1, 0, X.shape[1])          # rank-derived score
        return s, order
    else:
        raise ValueError(method)
    return s, np.argsort(-s, kind="mergesort")


def evaluate_method(method, X, y, role, beta, seed, n_inf):
    s, order = method_scores_and_order(method, X, y, seed, n_inf)
    relevant = role != "noise"
    from sklearn.metrics import roc_auc_score

    rel_auc = roc_auc_score(relevant, s)
    prec = float(np.mean(role[order[:n_inf]] == "informative"))
    inf_idx = np.flatnonzero(role == "informative")
    sp = stats.spearmanr(s[inf_idx], beta).correlation
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(y))
    a, b = perm[: len(y) // 2], perm[len(y) // 2:]
    _, oa = method_scores_and_order(method, X[a], y[a], seed, n_inf)
    _, ob = method_scores_and_order(method, X[b], y[b], seed, n_inf)
    sa, sb = set(oa[:n_inf].tolist()), set(ob[:n_inf].tolist())
    return {"relevance_auc": rel_auc, "precision_at_k": prec, "spearman_truth": float(np.nan_to_num(sp)),
            "stability": len(sa & sb) / len(sa | sb)}


def run_robustness(cfg: dict) -> pd.DataFrame:
    log = get_logger()
    rc = cfg["robustness"]
    out = ensure_dir(Path(cfg["output_dir"]) / "robustness")
    path = out / "maig_robustness_raw.csv"
    if path.exists():
        log.info("Robustness results exist: %s", path)
        return pd.read_csv(path)
    base = {"prevalence": 0.30, "label_noise": 0.05, "correlation": 0.0, "feature_noise": 0.5}
    sweeps = {"prevalence": rc["prevalence"], "label_noise": rc["label_noise"],
              "correlation": rc["correlation"], "feature_noise": rc["feature_noise"]}
    rows = []
    for factor, levels in sweeps.items():
        for level in levels:
            cond = {**base, factor: level}
            for rep in range(rc["replicates"]):
                seed = derive_seed(cfg["seed"], "robust", factor, level, rep) % (2**31)
                rng = np.random.default_rng(seed)
                X, y, role, beta = simulate(rc["n_samples"], rc["n_informative"], rc["n_redundant"], rc["n_noise"],
                                            cond["prevalence"], cond["label_noise"], cond["correlation"],
                                            cond["feature_noise"], rng)
                for m in METHODS:
                    rows.append({"factor": factor, "level": level, "replicate": rep, "method": m,
                                 **evaluate_method(m, X, y, role, beta, seed, rc["n_informative"])})
            log.info("robustness: %s = %s done", factor, level)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    summ = df.groupby(["factor", "level", "method"]).agg(["mean", "std"]).drop(columns="replicate")
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out / "maig_robustness_summary.csv", index=False)
    save_json({"base_condition": base, "sweeps": sweeps, "replicates": rc["replicates"]}, out / "design.json")
    return df
