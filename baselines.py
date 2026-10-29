"""
Comparator models and the model factory.

All models share one interface:  fit(X, y, X_val, y_val)  and  predict_score(X) -> P(y = 1).
They receive identical folds, identical preprocessing, identical MAIG-selected features,
identical class balancing and an identical hyper-parameter search budget (evaluation.py).

Temporal models (LSTM-RNN, CNN-LSTM). The cohort contains eight variables measured at
three clinically ordered visits (T0 pre-infection, T1 3 months, T2 12 months after
infection). These form a genuine sequence of length 3 with 8 channels; the remaining
static variables are concatenated to the final recurrent state. No ordering is imposed
on unordered tabular columns. Variables not retained by MAIG are zeroed (= training
mean after z-scoring), which is equivalent to removing them while keeping the tensor
shape. When a dataset has no repeated measurements (e.g. PIMA), the temporal models
are reported as not applicable rather than run on an artificial ordering.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch import nn

from svdnn import SVDNN, SVDNN_VARIANTS, TorchBinaryClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


# ---------------------------------------------------------------------------------
# scikit-learn / XGBoost wrappers
# ---------------------------------------------------------------------------------
class SklearnModel:
    requires_sequence = False
    calibrated_on_validation = False

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(X, y)
        return self

    def predict_score(self, X):
        return self.estimator.predict_proba(X)[:, 1]


class XGBModel(SklearnModel):
    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return self


class PlattSVM(SklearnModel):
    """RBF-SVM; decision values mapped to probabilities by Platt scaling on the inner
    validation split (which has the true training prevalence)."""

    calibrated_on_validation = True

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(X, y)
        dv = self.estimator.decision_function(X_val).reshape(-1, 1)
        self.platt_ = LogisticRegression(C=1e6, max_iter=1000).fit(dv, y_val)
        return self

    def predict_score(self, X):
        return self.platt_.predict_proba(self.estimator.decision_function(X).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------------
# Temporal neural baselines
# ---------------------------------------------------------------------------------
class _SeqNet(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, hidden: int, dropout: float, conv_filters: int | None):
        super().__init__()
        self.conv = None
        rnn_in = n_dyn
        if conv_filters:
            self.conv = nn.Sequential(nn.Conv1d(n_dyn, conv_filters, kernel_size=2, padding=1), nn.ReLU())
            rnn_in = conv_filters
        self.rnn = nn.LSTM(rnn_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:                                  # seq: (n, T, V)
            seq = self.conv(seq.transpose(1, 2)).transpose(1, 2)   # (n, T + 1, filters)
        _, (h, _) = self.rnn(seq)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(-1)


class SequenceModel(TorchBinaryClassifier):
    requires_sequence = True

    def __init__(self, hidden: int = 32, dropout: float = 0.2, filters: int | None = None, **kw):
        super().__init__(**kw)
        self.arch = dict(hidden=hidden, dropout=dropout, filters=filters)
        self.layout = None

    def set_layout(self, dyn_idx: np.ndarray, static_idx: np.ndarray) -> "SequenceModel":
        self.layout = (torch.as_tensor(dyn_idx, dtype=torch.long), torch.as_tensor(static_idx, dtype=torch.long))
        return self

    def build(self, p: int) -> nn.Module:
        dyn, stat = self.layout
        return _SeqNet(dyn.shape[0], len(stat), self.arch["hidden"], self.arch["dropout"], self.arch["filters"])

    def forward(self, xb):
        dyn, stat = self.layout
        seq = xb[:, dyn].transpose(1, 2)        # (n, V, T) -> (n, T, V)
        return self.net_(seq, xb[:, stat]), {}


def sequence_layout(feature_names: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Indices of the (variable x visit) grid and of the static columns."""
    import re

    pat = re.compile(r"^(.*)_t([0-9])$")
    groups: dict[str, dict[int, int]] = {}
    for i, n in enumerate(feature_names):
        m = pat.match(n)
        if m:
            groups.setdefault(m.group(1), {})[int(m.group(2))] = i
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return None
    T = max(len(v) for v in groups.values())
    groups = {k: v for k, v in groups.items() if len(v) == T}
    dyn = np.array([[groups[g][t] for t in sorted(groups[g])] for g in sorted(groups)])
    used = set(dyn.ravel().tolist())
    static = np.array([i for i in range(len(feature_names)) if i not in used])
    return dyn, static


# ---------------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------------
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm_rbf": "SVM (RBF)",
    "mlp": "DNN (MLP)",
    "lstm_rnn": "LSTM-RNN",
    "cnn_lstm": "CNN-LSTM",
    "svdnn": "SVDNN-MAIG (proposed)",
    "svdnn_no_maig": "SVDNN without MAIG (all features)",
    "svdnn_no_margin": "SVDNN without margin layer",
    "svdnn_margin_unsupervised": "SVDNN, margin layer without hinge/orthogonality",
    "svdnn_no_diversity": "SVDNN without orthogonality (mu = 0)",
    "svdnn_no_skip": "SVDNN without skip connection",
    "svdnn_no_jacobian": "SVDNN without input-gradient penalty",
    "svdnn_margin_only": "Margin layer only (no decision network)",
    "mlp_output_hinge": "MLP + output hinge loss",
}

DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.3},
    "xgboost": {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0},
    "svm_rbf": {"C": 1.0, "gamma": 0.01},
    "mlp": {"lr": 1e-3, "width": 128, "depth": 3, "dropout": 0.2, "weight_decay": 1e-5},
    "lstm_rnn": {"lr": 1e-3, "hidden": 32, "dropout": 0.2, "weight_decay": 1e-5},
    "cnn_lstm": {"lr": 1e-3, "filters": 32, "hidden": 32, "dropout": 0.2, "weight_decay": 1e-5},
    "svdnn": {"lr": 1e-3, "width": 128, "depth": 3, "dropout": 0.2, "weight_decay": 1e-5, "n_margin": 16,
              "C": 10.0, "alpha": 0.3, "mu": 0.1, "lam": 1e-3},
}


def suggest_params(trial, name: str, space: dict) -> dict:
    params = {}
    for key, (lo, hi, kind) in space.items():
        if kind == "int":
            params[key] = trial.suggest_int(key, int(lo), int(hi))
        elif kind == "log":
            params[key] = trial.suggest_float(key, float(lo), float(hi), log=True)
        else:
            params[key] = trial.suggest_float(key, float(lo), float(hi))
    return params


def make_model(name: str, params: dict, cfg: dict, seed: int, layout=None):
    tr = cfg["training"]
    nt = cfg.get("n_threads")
    torch_kw = dict(max_epochs=tr["max_epochs"], patience=tr["patience"], batch_size=tr["batch_size"],
                    lr_scheduler_patience=tr["lr_scheduler_patience"], lr_scheduler_factor=tr["lr_scheduler_factor"],
                    seed=seed, n_threads=nt)
    p = dict(params)
    if name == "logistic_regression":
        return SklearnModel(LogisticRegression(C=p["C"], max_iter=5000))
    if name == "random_forest":
        return SklearnModel(RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                                   min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                                   n_jobs=nt, random_state=seed))
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required (pip install xgboost)")
        return XGBModel(XGBClassifier(**p, tree_method="hist", eval_metric="auc", early_stopping_rounds=50,
                                      n_jobs=nt, random_state=seed))
    if name == "svm_rbf":
        return PlattSVM(SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", cache_size=1000, random_state=seed))
    if name == "mlp":
        return SVDNN(use_margin=False, alpha=0.0, mu=0.0, lam=0.0, width=p["width"], depth=p["depth"],
                     dropout=p["dropout"], lr=p["lr"], weight_decay=p["weight_decay"], **torch_kw)
    if name in ("lstm_rnn", "cnn_lstm"):
        if layout is None:
            raise ValueError(f"{name} needs repeated measurements (sequence layout).")
        m = SequenceModel(hidden=p["hidden"], dropout=p["dropout"], filters=p.get("filters") if name == "cnn_lstm" else None,
                          lr=p["lr"], weight_decay=p["weight_decay"], **torch_kw)
        return m.set_layout(*layout)
    if name in SVDNN_VARIANTS:
        overrides, _ = SVDNN_VARIANTS[name]
        q = {**DEFAULT_PARAMS["svdnn"], **p, **overrides}
        return SVDNN(n_margin=q["n_margin"], width=q["width"], depth=q["depth"], dropout=q["dropout"], C=q["C"],
                     alpha=q["alpha"], mu=q["mu"], lam=q["lam"], use_margin=q.get("use_margin", True),
                     skip=q.get("skip", True), output_hinge=q.get("output_hinge", False), lr=q["lr"],
                     weight_decay=q["weight_decay"], **torch_kw)
    raise ValueError(f"Unknown model '{name}'")
