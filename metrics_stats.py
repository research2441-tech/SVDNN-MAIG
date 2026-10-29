"""
Discrimination, calibration, clinical-utility metrics and statistical tests.

Statistical protocol
--------------------
* Paired model comparisons use the J = R x K held-out outer folds (R repeats of K-fold CV).
  Because training sets overlap across folds, the naive paired t-test is anti-conservative;
  the corrected resampled t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004) is used:
      t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)),   df = J - 1.
  The Wilcoxon signed-rank test is reported as a non-parametric check; p-values are
  Holm-adjusted across all comparisons against the proposed model.
* Effect sizes: mean difference with corrected 95% CI, Cohen's d_z and Cliff's delta.
* AUROC differences on pooled out-of-fold predictions: DeLong test (Sun & Xu, 2014).
* Error-rate reductions (FNR): absolute and relative change with patient-level paired
  bootstrap CIs.
* Variance claims: Brown-Forsythe test on fold-level metrics and bootstrap CI of the SD.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, roc_auc_score, roc_curve)

EPS = 1e-7


# ---------------------------------------------------------------------------------
# Probability utilities
# ---------------------------------------------------------------------------------
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def prior_shift(p, pi_from: float, pi_to: float):
    """Bayes prior-shift correction of probabilities (Saerens et al., 2002)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    a = p * pi_to / pi_from
    b = (1 - p) * (1 - pi_to) / (1 - pi_from)
    return a / (a + b)


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(min(thr[j], 1.0))


# ---------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------
def calibration_intercept_slope(y, p, sample_weight=None) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope."""
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    def nll_int(a):
        q = 1 / (1 + np.exp(-(a + lp)))
        return -np.sum(w * (y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS)))

    intercept = optimize.minimize_scalar(nll_int, bounds=(-10, 10), method="bounded").x

    def nll_slope(theta):
        q = 1 / (1 + np.exp(-(theta[0] + theta[1] * lp)))
        return -np.sum(w * (y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS)))

    slope = optimize.minimize(nll_slope, x0=[0.0, 1.0], method="BFGS").x[1]
    return float(intercept), float(slope)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_curve_quantile(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    obs, pred, cnt = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            obs.append(y[m].mean())
            pred.append(p[m].mean())
            cnt.append(int(m.sum()))
    return {"predicted": pred, "observed": obs, "count": cnt}


# ---------------------------------------------------------------------------------
# Decision-curve analysis
# ---------------------------------------------------------------------------------
def net_benefit(y, p, thresholds, sample_weight=None) -> dict:
    y = np.asarray(y)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    n = w.sum()
    prev = (w * y).sum() / n
    nb_model, nb_all = [], []
    for pt in thresholds:
        pos = p >= pt
        tp = (w * (pos & (y == 1))).sum() / n
        fp = (w * (pos & (y == 0))).sum() / n
        odds = pt / (1 - pt)
        nb_model.append(tp - fp * odds)
        nb_all.append(prev - (1 - prev) * odds)
    return {"threshold": list(map(float, thresholds)), "model": nb_model, "treat_all": nb_all,
            "treat_none": [0.0] * len(thresholds)}


# ---------------------------------------------------------------------------------
# Metric suite
# ---------------------------------------------------------------------------------
def compute_metrics(y, p, threshold: float, sample_weight=None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (sens + spec),
        "precision": prec,
        "recall": sens,
        "specificity": spec,
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "fnr": fn / max(tp + fn, 1),
        "fpr": fp / max(tn + fp, 1),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": expected_calibration_error(y, p),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": float(threshold),
    }


def sensitivity_at_specificity(y, p, target_spec: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= target_spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def population_ppv_npv(sens: float, spec: float, prevalence: float) -> tuple[float, float]:
    ppv = sens * prevalence / (sens * prevalence + (1 - spec) * (1 - prevalence) + EPS)
    npv = spec * (1 - prevalence) / (spec * (1 - prevalence) + (1 - sens) * prevalence + EPS)
    return float(ppv), float(npv)


# ---------------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------------
def corrected_resampled_ttest(a, b, n_train: float, n_test: float, alpha: float = 0.05) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    J = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else 0.0
    mean = d.mean()
    if se == 0:
        return {"mean_diff": mean, "t": np.inf if mean != 0 else 0.0, "p": 0.0 if mean != 0 else 1.0,
                "ci_low": mean, "ci_high": mean, "df": J - 1}
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    h = stats.t.ppf(1 - alpha / 2, df=J - 1) * se
    return {"mean_diff": mean, "t": t, "p": p, "ci_low": mean - h, "ci_high": mean + h, "df": J - 1}


def wilcoxon_p(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def cohens_dz(a, b) -> float:
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") * np.sign(d.mean())


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm_adjust(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def brown_forsythe(*groups) -> float:
    return float(stats.levene(*groups, center="median").pvalue)


def bootstrap_sd_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    sds = [rng.choice(x, len(x), replace=True).std(ddof=1) for _ in range(n_boot)]
    return float(np.quantile(sds, alpha / 2)), float(np.quantile(sds, 1 - alpha / 2))


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y, p1, p2) -> dict:
    """Paired DeLong test for two correlated AUROCs (fast algorithm of Sun & Xu, 2014)."""
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([p1, p2])
    aucs, v01, v10 = [], [], []
    for s in preds:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        auc = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[pos] - tx) / n)
        v10.append(1 - (tz[neg] - ty) / m)
    sx, sy = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def paired_bootstrap_rate_reduction(y, yhat_base, yhat_new, n_boot: int = 2000, seed: int = 0,
                                    rate: str = "fnr") -> dict:
    """Absolute and relative reduction of FNR (or FPR) with patient-level bootstrap CIs.

    yhat_* may be 2-D (patients x repeats); patients are resampled with all repeats."""
    y = np.asarray(y)
    yb = np.asarray(yhat_base).reshape(len(y), -1)
    yn = np.asarray(yhat_new).reshape(len(y), -1)
    target = 1 if rate == "fnr" else 0

    def r(yh, yy):
        m = yy == target
        return np.mean([(yh[m, j] != target).mean() for j in range(yh.shape[1])])

    base, new = r(yb, y), r(yn, y)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == target)
    other = np.flatnonzero(y != target)
    abs_d, rel_d = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(other, len(other))])
        b, nn_ = r(yb[idx], y[idx]), r(yn[idx], y[idx])
        abs_d.append(b - nn_)
        rel_d.append((b - nn_) / b if b > 0 else np.nan)
    q = lambda a, lo: float(np.nanquantile(a, lo))  # noqa: E731
    return {
        f"{rate}_baseline": base, f"{rate}_proposed": new,
        "absolute_reduction": base - new, "absolute_ci": [q(abs_d, 0.025), q(abs_d, 0.975)],
        "relative_reduction": (base - new) / base if base > 0 else np.nan,
        "relative_ci": [q(rel_d, 0.025), q(rel_d, 0.975)],
    }


def improvement(new: float, base: float) -> dict:
    """Absolute (percentage-point) and relative (%) improvement, reported separately."""
    return {"absolute_pp": 100 * (new - base), "relative_pct": 100 * (new - base) / base if base else np.nan}
