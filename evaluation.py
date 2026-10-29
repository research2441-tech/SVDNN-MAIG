"""
Leakage-free nested evaluation engine shared by every experiment.

Protocol for each of the R x K outer folds (repeated stratified K-fold):
  1. The outer-training part is split (stratified) into inner-train (80 %) and
     inner-validation (20 %). The outer test fold is not touched until step 7.
  2. Plausibility cleaning, iterative Bayesian-ridge imputation, one-hot encoding
     and z-scoring are FIT ON INNER-TRAIN ONLY and applied to validation/test.
  3. MAIG scores and the redundancy-aware selection are computed on inner-train
     only; gamma is chosen by inner-validation AUROC of a fixed reference model
     (L2 logistic regression), so the feature set is the same for every model.
  4. Class balancing (Gaussian-noise bootstrap oversampling, Eq. 17) is applied
     to inner-train only, identically for all models; predicted probabilities are
     mapped back to the training prevalence by a prior-shift correction.
  5. Each model receives the same Bayesian-optimisation (TPE) budget, maximising
     inner-validation AUROC. Neural models early-stop on the same split.
  6. The decision threshold is chosen on inner-validation (Youden J) by the same
     rule for every model.
  7. The tuned model is evaluated once on the outer test fold.
Every model is therefore evaluated R x K times on held-out data, and every
participant receives exactly one held-out prediction per repeat.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required before IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from baselines import DEFAULT_PARAMS, make_model, sequence_layout, suggest_params
from explain import explain_fold
from maig import MAIGSelector, rank_features, top_k_mask
from metrics_stats import compute_metrics, logit, prior_shift, youden_threshold
from svdnn import SVDNN_VARIANTS
from utils import derive_seed, ensure_dir, get_logger, load_json, save_json

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------------
# Preprocessing (fit on training data of the fold only)
# ---------------------------------------------------------------------------------
class FoldPreprocessor:
    def __init__(self, categorical: list[str], ranges: dict, seed: int, max_iter: int = 10):
        self.categorical = categorical
        self.ranges = ranges
        self.seed = seed
        self.max_iter = max_iter

    def _range_for(self, col: str):
        if col in self.ranges:
            return self.ranges[col]
        base = col.rsplit("_t", 1)[0]
        return self.ranges.get(base)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.numeric_:
            r = self._range_for(c)
            if r is not None:
                v = df[c]
                df.loc[(v < r[0]) | (v > r[1]), c] = np.nan
        return df

    def fit(self, df: pd.DataFrame) -> "FoldPreprocessor":
        self.numeric_ = [c for c in df.columns if c not in self.categorical]
        self.levels_ = {c: sorted(df[c].dropna().unique().tolist()) for c in self.categorical}
        clean = self._clean(df)
        self.imputer_ = IterativeImputer(estimator=BayesianRidge(), max_iter=self.max_iter, initial_strategy="median",
                                         random_state=self.seed, skip_complete=True)
        num = self.imputer_.fit_transform(clean[self.numeric_])
        full = np.hstack([num, self._onehot(df)])
        self.scaler_ = StandardScaler().fit(full)
        self.feature_names_ = self.numeric_ + [f"{c}={lvl}" for c in self.categorical for lvl in self.levels_[c]]
        self.missing_rate_ = clean[self.numeric_].isna().mean().to_dict()
        return self

    def _onehot(self, df: pd.DataFrame) -> np.ndarray:
        cols = [(df[c].to_numpy() == lvl).astype(float)[:, None] for c in self.categorical for lvl in self.levels_[c]]
        return np.hstack(cols) if cols else np.empty((len(df), 0))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self.imputer_.transform(self._clean(df)[self.numeric_])
        return self.scaler_.transform(np.hstack([num, self._onehot(df)])).astype(np.float32)

    @property
    def mean_(self) -> np.ndarray:
        return self.scaler_.mean_

    @property
    def scale_(self) -> np.ndarray:
        return self.scaler_.scale_


def noise_oversample(X: np.ndarray, y: np.ndarray, eta_range, rng: np.random.Generator):
    """Bootstrap the minority class to parity and add eta * N(0, I) noise, eta ~ U(eta_range) (Eq. 17)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    n_new = len(majority) - len(minority)
    if n_new <= 0:
        return X, y
    pick = rng.choice(minority, n_new, replace=True)
    eta = rng.uniform(*eta_range, size=(n_new, 1))
    synth = X[pick] + eta * rng.standard_normal((n_new, X.shape[1])).astype(np.float32)
    return np.vstack([X, synth]).astype(np.float32), np.concatenate([y, y[pick]])


# ---------------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------------
class FoldData:
    """Everything a model needs inside one outer fold."""

    def __init__(self, Xtr, ytr, Xtr_bal, ytr_bal, Xval, yval, Xte, yte, names, mask, layout, pi_train, cfg):
        self.Xtr, self.ytr, self.Xtr_bal, self.ytr_bal = Xtr, ytr, Xtr_bal, ytr_bal
        self.Xval, self.yval, self.Xte, self.yte = Xval, yval, Xte, yte
        self.names, self.mask, self.layout, self.pi_train, self.cfg = names, mask, layout, pi_train, cfg
        self.oversampled = cfg["cv"]["oversample"]

    def view(self, model, mask=None):
        """Matrices for a model: selected columns, or zero-masked full matrix for sequence models."""
        m = self.mask if mask is None else mask
        if getattr(model, "requires_sequence", False):
            f = m.astype(np.float32)
            return self.Xtr_bal * f, self.Xval * f, self.Xte * f
        return self.Xtr_bal[:, m], self.Xval[:, m], self.Xte[:, m]

    def correct(self, model, p):
        if self.oversampled and not getattr(model, "calibrated_on_validation", False):
            return prior_shift(p, 0.5, self.pi_train)
        return p


def fit_predict(name: str, params: dict, fd: FoldData, seed: int, mask=None):
    model = make_model(name, params, fd.cfg, seed, layout=fd.layout)
    Xb, Xv, Xt = fd.view(model, mask)
    t0 = time.time()
    model.fit(Xb, fd.ytr_bal, Xv, fd.yval)
    p_val = fd.correct(model, model.predict_score(Xv))
    p_te = fd.correct(model, model.predict_score(Xt))
    return model, p_val, p_te, time.time() - t0


def tune(name: str, fd: FoldData, seed: int):
    """Equal-budget TPE search on inner-validation AUROC.

    Returns (best params, trace, best fitted model with its validation/test predictions).
    The best trial's fitted model is kept, so the final model is exactly the selected
    configuration and no refit is needed. Test predictions are computed for every trial
    but never used for selection."""
    cfg = fd.cfg
    n_trials = cfg["tuning"]["n_trials"]
    space = cfg["search_space"].get(name)
    if not space or n_trials <= 0:
        params = dict(DEFAULT_PARAMS[name])
        return params, [], fit_predict(name, params, fd, seed)
    trace = []
    best = {"auc": -np.inf, "fit": None, "params": None}

    def objective(trial):
        params = suggest_params(trial, name, space)
        fit = fit_predict(name, params, fd, seed)
        auc = roc_auc_score(fd.yval, fit[1])
        trace.append({"params": params, "val_auc": auc})
        if auc > best["auc"]:
            best.update(auc=auc, fit=fit, params=params)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: v for k, v in DEFAULT_PARAMS[name].items() if k in space})
    study.optimize(objective, n_trials=n_trials)
    return dict(best["params"]), trace, best["fit"]


def evaluate(name, model, p_val, p_te, fd: FoldData, seconds: float, params: dict | None = None) -> tuple[dict, np.ndarray]:
    thr = youden_threshold(fd.yval, p_val)
    rec = {"params": params, "threshold": thr, "fit_seconds": seconds,
           "val_auc": float(roc_auc_score(fd.yval, p_val)),
           "test": compute_metrics(fd.yte, p_te, thr)}
    if hasattr(model, "n_epochs_"):
        rec["epochs"] = model.n_epochs_
    return rec, p_te


def choose_gamma(selector: MAIGSelector, Xtr, ytr, Xval, yval, grid) -> tuple[float, dict]:
    res = {}
    for g in grid:
        m = selector.select(g)
        lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(Xtr[:, m], ytr)
        res[g] = {"val_auc": float(roc_auc_score(yval, lr.predict_proba(Xval[:, m])[:, 1])), "n_features": int(m.sum())}
    best = max(grid, key=lambda g: (round(res[g]["val_auc"], 3), g))   # ties -> larger gamma (fewer features)
    selector.select(best)
    return best, res


# ---------------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------------
def run_nested_cv(X_df: pd.DataFrame, y: np.ndarray, cfg: dict, experiment: str, models: list[str],
                  categorical: list[str], ablations: list[str] | None = None, selectors: list[str] | None = None,
                  explain: bool = False) -> Path:
    """Run (or resume) the nested protocol. Results are written to outputs/<mode>/<experiment>/.

    Outer folds are independent and checkpointed, so they are distributed over
    `n_workers` processes; results do not depend on the number of workers."""
    log = get_logger()
    out = ensure_dir(Path(cfg["output_dir"]) / experiment)
    fold_dir = ensure_dir(out / "folds")
    R, K = cfg["cv"]["outer_repeats"], cfg["cv"]["outer_folds"]
    ablations = ablations or []
    selectors = selectors or []
    save_json({"n": len(y), "prevalence": float(y.mean()), "repeats": R, "folds": K, "models": models,
               "ablations": ablations, "selectors": selectors, "features": list(X_df.columns)}, out / "design.json")
    jobs = []
    for r in range(R):
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=derive_seed(cfg["seed"], "outer", r) % (2**31))
        for f, (tr, te) in enumerate(skf.split(X_df, y)):
            if not (fold_dir / f"r{r}_f{f}.json").exists():
                jobs.append((r, f, tr, te))
    if not jobs:
        return out
    workers = max(1, min(int(cfg.get("n_workers", 1)), len(jobs)))
    args = (X_df, y, cfg, experiment, models, categorical, ablations, selectors, explain, fold_dir)
    log.info("[%s] %d outer folds to run on %d worker(s)", experiment, len(jobs), workers)
    if workers == 1:
        for job in jobs:
            _run_fold(*job, *args)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(_run_fold, *job, *args) for job in jobs]
            for fut in as_completed(futures):
                fut.result()                      # re-raise errors from workers
    return out


def _run_fold(r, f, tr, te, X_df, y, cfg, experiment, models, categorical, ablations, selectors, explain, fold_dir):
    """One outer fold of the nested protocol (runs in the main process or in a worker)."""
    import torch

    torch.set_num_threads(max(1, int(cfg.get("n_threads", 1))))
    log = get_logger()
    seed = cfg["seed"]
    proposed = cfg.get("proposed", "svdnn")
    ckpt = fold_dir / f"r{r}_f{f}.json"
    t_fold = time.time()
    fseed = derive_seed(seed, experiment, r, f) % (2**31)
    tr_in, val = train_test_split(tr, test_size=cfg["cv"]["inner_val_fraction"], stratify=y[tr],
                                  random_state=fseed)
    prep = FoldPreprocessor(categorical, cfg.get("plausible_ranges", {}), fseed).fit(X_df.iloc[tr_in])
    Xtr, Xval, Xte = (prep.transform(X_df.iloc[i]) for i in (tr_in, val, te))
    ytr, yval, yte = y[tr_in], y[val], y[te]
    names = prep.feature_names_

    mcfg = cfg["maig"]
    selector = MAIGSelector(mcfg["n_thresholds"], mcfg["balanced"], mcfg["redundancy_eta"],
                            min_features=mcfg["min_features"]).fit(Xtr, ytr)
    gamma, gamma_trace = choose_gamma(selector, Xtr, ytr, Xval, yval, mcfg["gamma_grid"])
    mask = selector.support_.copy()

    rng = np.random.default_rng(fseed)
    if cfg["cv"]["oversample"]:
        Xtr_bal, ytr_bal = noise_oversample(Xtr, ytr, cfg["cv"]["noise_eta"], rng)
    else:
        Xtr_bal, ytr_bal = Xtr, ytr
    layout = sequence_layout(names)
    fd = FoldData(Xtr, ytr, Xtr_bal, ytr_bal, Xval, yval, Xte, yte, names, mask, layout, float(ytr.mean()), cfg)

    record = {"repeat": r, "fold": f, "n_train": len(tr_in), "n_val": len(val), "n_test": len(te),
              "gamma": gamma, "gamma_trace": gamma_trace,
              "selected_features": [names[i] for i in selector.order_],
              "maig_scores": dict(zip(names, selector.normalized_.tolist())),
              "missing_rate_train": prep.missing_rate_, "models": {}, "ablations": {}, "selectors": {}}
    preds = {"test_idx": te, "y": yte}

    proposed_params, proposed_model = None, None
    for name in models:
        if name in ("lstm_rnn", "cnn_lstm") and layout is None:
            record["models"][name] = {"not_applicable": "no repeated measurements in this dataset"}
            continue
        params, trace, (model, p_val, p_te, sec) = tune(name, fd, fseed)
        rec, p = evaluate(name, model, p_val, p_te, fd, sec, params)
        rec["tuning_trace"] = trace
        record["models"][name] = rec
        preds[name] = p
        if name == proposed:
            proposed_params, proposed_model = params, model
            record["models"][name]["threshold_raw_logit"] = float(
                logit(prior_shift(rec["threshold"], fd.pi_train, 0.5) if fd.oversampled else rec["threshold"]))
        log.info("[%s r%d f%d] %-20s AUC %.3f F1 %.3f (%.0fs)", experiment, r, f, name,
                 rec["test"]["roc_auc"], rec["test"]["f1"], sec)

    # ---- ablations: tuned SVDNN hyper-parameters, one component removed ----------
    if proposed_params is not None:
        all_mask = np.ones(len(names), dtype=bool)
        for ab in ablations:
            use_all = SVDNN_VARIANTS[ab][1]
            model, p_val, p_te, sec = fit_predict(ab, proposed_params, fd, fseed, all_mask if use_all else None)
            rec, p = evaluate(ab, model, p_val, p_te, fd, sec)
            record["ablations"][ab] = rec
            preds[f"ablation:{ab}"] = p

        # ---- selector comparison: same number of features, same models -----------
        k = int(mask.sum())
        lr_params = record["models"].get("logistic_regression", {}).get("params") or DEFAULT_PARAMS["logistic_regression"]
        for method in selectors:
            if method == "maig":
                m = mask
            else:
                order = rank_features(method, Xtr, ytr, seed=fseed)
                m = top_k_mask(order, k, len(names))
            sel_rec = {"features": [names[i] for i in np.flatnonzero(m)]}
            for mdl, prm in (("svdnn", proposed_params), ("logistic_regression", lr_params)):
                if method == "maig" and mdl in record["models"]:     # identical fit already done
                    sel_rec[mdl] = record["models"][mdl]["test"]
                    continue
                model, p_val, p_te, sec = fit_predict(mdl, prm, fd, fseed, m)
                rec, _ = evaluate(mdl, model, p_val, p_te, fd, sec)
                sel_rec[mdl] = rec["test"]
            record["selectors"][method] = sel_rec

        # ---- explanations of the proposed model ------------------------------------
        if explain:
            sel_idx = np.flatnonzero(mask)
            sel_names = [names[i] for i in sel_idx]
            maig_local = np.argsort(-selector.normalized_[sel_idx])
            record["explain"] = explain_fold(
                proposed_model, Xtr[:, mask], Xte[:, mask], yte, sel_names, prep.mean_[mask], prep.scale_[mask],
                maig_local, record["models"][proposed]["threshold_raw_logit"], cfg, fseed)

    record["fold_seconds"] = time.time() - t_fold
    np.savez_compressed(fold_dir / f"r{r}_f{f}_pred.npz", **{k: np.asarray(v) for k, v in preds.items()})
    save_json(record, ckpt)
    log.info("[%s] fold r%d f%d finished in %.0fs", experiment, r, f, record["fold_seconds"])




# ---------------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------------
def load_results(out_dir: Path) -> tuple[list[dict], list[dict]]:
    folds = sorted(out_dir.glob("folds/r*_f*.json"), key=lambda p: tuple(int(s[1:]) for s in p.stem.split("_")))
    records = [load_json(p) for p in folds]
    preds = []
    for p in folds:
        z = np.load(str(p).replace(".json", "_pred.npz"))
        preds.append({k: z[k] for k in z.files})
    return records, preds


def fold_metric_table(records: list[dict], section: str = "models") -> pd.DataFrame:
    rows = []
    for rec in records:
        for name, m in rec.get(section, {}).items():
            if "test" not in m:
                continue
            rows.append({"repeat": rec["repeat"], "fold": rec["fold"], "model": name,
                         "n_train": rec["n_train"], "n_test": rec["n_test"], **m["test"]})
    return pd.DataFrame(rows)
