"""
Explanations for SVDNN and their validation.

Scope statement: the attributions below are explanations of the MODEL, computed after
training (gradient saliency, Eq. 11; cumulative attribution, Eq. 19; integrated
gradients). They are post-hoc explanations and are therefore validated here rather than
assumed to be faithful:
  * fidelity    - deletion test: masking the top-ranked features must reduce AUROC faster
                  than masking random features (AOPC, Samek et al., 2017);
  * stability   - rank agreement of global attributions across the 25 outer folds;
  * ground truth- in the synthetic cohort the causal role of every variable is known
                  (data_dictionary.csv), so the report shows how often highly attributed
                  variables are non-causal correlates. Attribution is predictive, not causal.

Counterfactuals: Eq. 20 (unconstrained minimum-norm class flip) is compared with a
constrained version that only changes actionable lifestyle variables at T2, respects
direction, maximum feasible 12-month change and physiological bounds, and couples waist
circumference to BMI. Plausibility of every counterfactual is assessed with a
local-outlier-factor novelty detector and a Mahalanobis test fitted on training data.
Counterfactuals describe the model's decision boundary only; they are not treatment
recommendations.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

# reference intervals used for the patient-level explanation map (Fig. 8)
REFERENCE_RANGES = {
    "hba1c": (None, 5.7, "ADA normoglycaemia < 5.7 %"),
    "fpg": (None, 100.0, "ADA normal fasting glucose < 100 mg/dL"),
    "bmi": (18.5, 25.0, "WHO normal weight 18.5-24.9 kg/m2"),
    "crp": (None, 3.0, "AHA low cardiovascular risk < 3 mg/L"),
    "il6": (None, 7.0, "laboratory upper reference ~7 pg/mL"),
    "ferritin": (30.0, 300.0, "adult reference 30-300 ng/mL"),
    "triglycerides": (None, 150.0, "NCEP ATP III < 150 mg/dL"),
    "hdl_cholesterol": (40.0, None, "NCEP ATP III > 40 mg/dL"),
    "sbp": (None, 130.0, "ACC/AHA < 130 mmHg"),
    "waist_cm": (None, 102.0, "IDF/ATP III < 102 cm (men)"),
    "fasting_insulin": (None, 25.0, "laboratory upper reference ~25 uU/mL"),
    "oxygen_saturation": (95.0, None, "resting SpO2 >= 95 %"),
    "physical_activity": (8.3, None, "WHO 150 min/week moderate activity ~ 8.3 MET-h"),
    "sleep_quality": (6.0, None, "self-rated adequate sleep"),
}


def _base(name: str) -> str:
    import re

    m = re.match(r"^(.*)_t[0-9]$", name)
    return m.group(1) if m else name


# ---------------------------------------------------------------------------------
# Attributions
# ---------------------------------------------------------------------------------
def gradient_saliency(model, X: np.ndarray, signed: bool = False) -> np.ndarray:
    """S_ij = d p_hat_i / d x_ij (Eq. 11); absolute value unless signed=True."""
    x = torch.as_tensor(X, dtype=torch.float32).requires_grad_(True)
    p = torch.sigmoid(model.logit_tensor(x))
    g = torch.autograd.grad(p.sum(), x)[0].numpy()
    return g if signed else np.abs(g)


def integrated_gradients(model, X: np.ndarray, steps: int = 32) -> np.ndarray:
    """Integrated gradients w.r.t. the all-zero baseline (= training mean after z-scoring)."""
    x = torch.as_tensor(X, dtype=torch.float32)
    total = torch.zeros_like(x)
    for a in (np.arange(steps) + 0.5) / steps:
        xa = (a * x).requires_grad_(True)
        p = torch.sigmoid(model.logit_tensor(xa))
        total += torch.autograd.grad(p.sum(), xa)[0]
    return (x * total / steps).detach().numpy()


def cumulative_feature_attribution(saliency: np.ndarray) -> np.ndarray:
    """CFA_j = sum_i |d p_hat_i / d x_ij| (Eq. 19), normalised to sum to one."""
    cfa = np.abs(saliency).sum(0)
    return cfa / cfa.sum() if cfa.sum() > 0 else cfa


def deletion_aopc(model, X: np.ndarray, y: np.ndarray, order: np.ndarray, max_k: int | None = None) -> dict:
    """AUROC after masking the top-k features (k = 0..K) and the area over the perturbation curve."""
    max_k = len(order) if max_k is None else min(max_k, len(order))
    base = roc_auc_score(y, model.predict_score(X))
    aucs = [base]
    Xm = X.copy()
    for k in range(max_k):
        Xm[:, order[k]] = 0.0
        aucs.append(roc_auc_score(y, model.predict_score(Xm)))
    aucs = np.array(aucs)
    return {"auc_curve": aucs.tolist(), "aopc": float(np.mean(base - aucs[1:]))}


def explanation_fidelity(model, X, y, saliency_order, ig_order, maig_order, n_random: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    K = min(10, X.shape[1])
    out = {
        "saliency": deletion_aopc(model, X, y, saliency_order, K),
        "integrated_gradients": deletion_aopc(model, X, y, ig_order, K),
        "maig": deletion_aopc(model, X, y, maig_order, K),
    }
    rand = [deletion_aopc(model, X, y, rng.permutation(X.shape[1]), K) for _ in range(n_random)]
    out["random"] = {"auc_curve": np.mean([r["auc_curve"] for r in rand], axis=0).tolist(),
                     "aopc": float(np.mean([r["aopc"] for r in rand])),
                     "aopc_sd": float(np.std([r["aopc"] for r in rand]))}
    return out


# ---------------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------------
def _bounds(names, X_raw, mean, scale, cfg_explain, constrained: bool):
    """Per-sample lower/upper bounds of the perturbation in z-score units."""
    n, p = X_raw.shape
    if not constrained:
        return np.full((n, p), -np.inf), np.full((n, p), np.inf), np.ones(p, bool)
    lo, hi = np.zeros((n, p)), np.zeros((n, p))
    allowed = np.zeros(p, bool)
    for var, rule in cfg_explain["actionable"].items():
        if var not in names:
            continue
        j = names.index(var)
        allowed[j] = True
        v = X_raw[:, j]
        if "max_rel_change" in rule:
            step = np.abs(v) * rule["max_rel_change"]
        else:
            step = np.full(n, rule["max_abs_change"])
        if rule["direction"] == "decrease":
            floor = rule.get("floor", -np.inf)
            lo[:, j] = np.minimum(0, np.maximum(-step, floor - v)) / scale[j]
        else:
            ceil = rule.get("ceiling", np.inf)
            hi[:, j] = np.maximum(0, np.minimum(step, ceil - v)) / scale[j]
    return lo, hi, allowed


def counterfactuals(model, X: np.ndarray, names: list[str], mean: np.ndarray, scale: np.ndarray,
                    threshold_logit: float, cfg_explain: dict, constrained: bool, steps: int, lr: float) -> dict:
    """Projected-gradient counterfactual search for patients predicted positive."""
    X_raw = X * scale + mean
    lo, hi, allowed = _bounds(names, X_raw, mean, scale, cfg_explain, constrained)
    lo_t, hi_t = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
    mask = torch.as_tensor(allowed, dtype=torch.float32)
    x0 = torch.as_tensor(X, dtype=torch.float32)
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    coupled = [(names.index(c), names.index(r["driver"]), r["slope"])
               for c, r in cfg_explain.get("coupled", {}).items()
               if constrained and c in names and r["driver"] in names]
    target = threshold_logit - 0.10

    def apply(d):
        xc = x0 + d * mask
        for c, drv, slope in coupled:
            xc = xc.clone()
            xc[:, c] = x0[:, c] + slope * d[:, drv] * scale[drv] / scale[c]
        return xc

    for _ in range(steps):
        xc = apply(delta)
        s = model.logit_tensor(xc)
        loss = (10.0 * torch.relu(s - target) + 0.1 * (delta * mask).pow(2).sum(1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if constrained:
                delta.copy_(torch.maximum(torch.minimum(delta, hi_t), lo_t))
    with torch.no_grad():
        xc = apply(delta).numpy()
        valid = model.logit_tensor(torch.as_tensor(xc)).numpy() < threshold_logit
    return {"x_cf": xc, "delta": xc - X, "valid": valid, "allowed": allowed}


def plausibility(x_cf: np.ndarray, X_train: np.ndarray, X_ref_raw_q: tuple[np.ndarray, np.ndarray],
                 mean: np.ndarray, scale: np.ndarray, seed: int) -> dict:
    """LOF inlier, Mahalanobis inlier and empirical-range checks for counterfactual points."""
    rng = np.random.default_rng(seed)
    ref = X_train[rng.choice(len(X_train), min(4000, len(X_train)), replace=False)]
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(ref)
    lof_in = lof.predict(x_cf) == 1
    cov_inv = np.linalg.pinv(np.cov(ref, rowvar=False))
    mu = ref.mean(0)
    d2 = np.einsum("ij,jk,ik->i", x_cf - mu, cov_inv, x_cf - mu)
    maha_in = d2 <= stats.chi2.ppf(0.99, df=x_cf.shape[1])
    raw = x_cf * scale + mean
    lo, hi = X_ref_raw_q
    in_range = ((raw >= lo) & (raw <= hi)).all(axis=1)
    return {"lof_inlier": lof_in, "mahalanobis_inlier": maha_in, "within_empirical_range": in_range}


def summarise_counterfactuals(res: dict, plaus: dict, names: list[str], scale: np.ndarray, actionable: list[str],
                              coupled: list[str] | None = None) -> dict:
    v = res["valid"]
    d = res["delta"]
    changed = np.abs(d) > 0.05
    modifiable = set(actionable) | set(coupled or [])
    immutable = np.array([n not in modifiable for n in names])
    out = {
        "n": int(len(v)),
        "validity": float(v.mean()) if len(v) else np.nan,
        "l1_distance_sd": float(np.abs(d[v]).sum(1).mean()) if v.any() else np.nan,
        "n_features_changed": float(changed[v].sum(1).mean()) if v.any() else np.nan,
        "changes_immutable_feature": float(changed[v][:, immutable].any(1).mean()) if v.any() else np.nan,
        "lof_inlier": float(plaus["lof_inlier"][v].mean()) if v.any() else np.nan,
        "mahalanobis_inlier": float(plaus["mahalanobis_inlier"][v].mean()) if v.any() else np.nan,
        "within_empirical_range": float(plaus["within_empirical_range"][v].mean()) if v.any() else np.nan,
    }
    out["fully_plausible"] = float((plaus["lof_inlier"] & plaus["mahalanobis_inlier"] & plaus["within_empirical_range"]
                                    & ~(changed[:, immutable].any(1)))[v].mean()) if v.any() else np.nan
    mean_change = {}
    for a in actionable + ["waist_cm"]:
        if a in names and v.any():
            j = names.index(a)
            mean_change[a] = float((d[v, j] * scale[j]).mean())
    out["mean_raw_change_actionable"] = mean_change
    return out


def logit_gradient(model, X: np.ndarray) -> np.ndarray:
    """Signed d s(x) / d x on the logit scale (does not saturate for very high or low risk)."""
    x = torch.as_tensor(X, dtype=torch.float32).requires_grad_(True)
    return torch.autograd.grad(model.logit_tensor(x).sum(), x)[0].numpy()


def patient_explanation_map(model, x: np.ndarray, names: list[str], mean: np.ndarray, scale: np.ndarray, top: int = 10) -> list[dict]:
    """E_i = {(f_j, S_ij, R_j)} (Eq. 21) for one patient: signed logit gradient x deviation from the training
    mean (gradient x input), raw value and clinical reference interval."""
    s = logit_gradient(model, x[None, :])[0] * x
    raw = x * scale + mean
    order = np.argsort(-np.abs(s))[:top]
    rows = []
    for j in order:
        ref = REFERENCE_RANGES.get(_base(names[j]))
        lo, hi, note = ref if ref else (None, None, "")
        outside = (lo is not None and raw[j] < lo) or (hi is not None and raw[j] > hi)
        rows.append({"feature": names[j], "value": float(raw[j]), "signed_contribution": float(s[j]),
                     "reference_low": lo, "reference_high": hi, "reference": note, "outside_reference": bool(outside)})
    return rows


def explain_fold(model, X_tr, X_te, y_te, names, mean, scale, maig_order_local, threshold_logit, cfg, seed) -> dict:
    """All explanation analyses for one outer fold (called by evaluation.py)."""
    ecfg = cfg["explain"]
    sal = gradient_saliency(model, X_te)
    ig = np.abs(integrated_gradients(model, X_te, ecfg["ig_steps"]))
    cfa = cumulative_feature_attribution(sal)
    ig_global = ig.mean(0) / ig.mean(0).sum()
    out = {
        "features": names,
        "saliency_global": cfa.tolist(),
        "ig_global": ig_global.tolist(),
        "fidelity": explanation_fidelity(model, X_te, y_te, np.argsort(-cfa), np.argsort(-ig_global),
                                         maig_order_local, ecfg["deletion_random_orders"], seed),
    }
    if hasattr(model, "margin_unit_auc"):
        auc_units = model.margin_unit_auc(X_te, y_te)
        if auc_units is not None:
            W = model.margin_directions()
            out["margin_units"] = {"auc": auc_units.tolist(),
                                   "mean_abs_cosine": float(np.abs(W @ W.T)[np.triu_indices(len(W), 1)].mean())
                                   if len(W) > 1 else np.nan,
                                   "top_unit_weights": dict(zip(names, W[int(np.argmax(auc_units))].tolist()))}

    # counterfactuals for patients predicted positive
    logits = model.predict_logit(X_te)
    pos = np.flatnonzero(logits >= threshold_logit)
    rng = np.random.default_rng(seed)
    pick = rng.choice(pos, min(len(pos), ecfg["max_counterfactuals_per_fold"]), replace=False) if len(pos) else pos
    raw_tr = X_tr * scale + mean
    q = (np.quantile(raw_tr, 0.005, axis=0), np.quantile(raw_tr, 0.995, axis=0))
    actionable = [a for a in ecfg["actionable"] if a in names]
    cf_out = {}
    if len(pick):
        for label, constrained in (("constrained", True), ("unconstrained", False)):
            if constrained and not actionable:
                cf_out[label] = {"n": 0, "note": "no actionable variable selected"}
                continue
            res = counterfactuals(model, X_te[pick], names, mean, scale, threshold_logit, ecfg, constrained,
                                  ecfg["cf_steps"], ecfg["cf_lr"])
            pl = plausibility(res["x_cf"], X_tr, q, mean, scale, seed)
            cf_out[label] = summarise_counterfactuals(res, pl, names, scale, actionable, list(ecfg.get("coupled", {})))
    out["counterfactuals"] = cf_out

    # patient-level explanation map: correctly identified case with median risk among true positives
    tp = np.flatnonzero((logits >= threshold_logit) & (y_te == 1))
    if len(tp):
        i = tp[np.argsort(logits[tp])[len(tp) // 2]]
        out["patient_map"] = patient_explanation_map(model, X_te[i], names, mean, scale)
    return out
