"""
Unit tests (run with:  pytest -q test_pipeline.py). They verify the properties claimed
in the manuscript: generator determinism, absence of label circularity, MAIG properties,
fold-confined preprocessing, the SVDNN objective and the statistical utilities.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from baselines import make_model, sequence_layout
from data_generator import FEATURE_COLUMNS, LABEL, generate_cohort, simulate_source_population
from evaluation import FoldPreprocessor, noise_oversample
from maig import MAIGSelector, maig_scores, rank_features
from metrics_stats import (corrected_resampled_ttest, delong_test, holm_adjust, improvement, prior_shift)
from svdnn import SVDNN, SVDNN_VARIANTS
from utils import load_config


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = load_config("config.yaml", "quick")
    c["output_dir"] = str(tmp_path_factory.mktemp("out"))
    c["data"].update({"n_source": 12000, "n_cases": 300, "n_controls": 600})
    return c


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.5 * rng.standard_normal(1500) > 0.6).astype(int)
    return X.astype(np.float32), y


# ------------------------------------------------------------------ generator
def test_generator_is_deterministic(cfg, tmp_path):
    s1 = generate_cohort(cfg)
    c2 = copy.deepcopy(cfg)
    c2["output_dir"] = str(tmp_path)
    s2 = generate_cohort(c2)
    assert s1["cohort_sha256"] == s2["cohort_sha256"]
    assert s1["n_features"] == 52


def test_released_columns_exclude_outcome_information(cfg):
    df = pd.read_csv(f"{cfg['output_dir']}/data/cohort.csv")
    assert set(df.columns) == set(FEATURE_COLUMNS) | {LABEL, "patient_id"}
    assert not any(c.endswith("_t3") for c in df.columns)
    for t in ("t0", "t1", "t2"):   # prevalent cases excluded: no diagnostic-range HbA1c (ignoring unit errors)
        v = df[f"hba1c_{t}"]
        assert (v[(v > 3) & (v < 18)] < cfg["data"]["ada_hba1c"]).all()


def test_label_is_not_a_deterministic_function_of_predictors(cfg):
    obs, lat = simulate_source_population(20000, cfg, np.random.default_rng(1))
    keep = lat["prevalent"] == 0
    X = obs.loc[keep, FEATURE_COLUMNS].drop(columns="ethnicity").to_numpy()
    y = obs.loc[keep, LABEL].to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    p = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=3, method="predict_proba")[:, 1]
    assert roc_auc_score(y, p) < 0.97          # a circular label would give ~1.0
    assert 0.02 < y.mean() < 0.20               # source incidence in a plausible range


# ------------------------------------------------------------------ MAIG
def test_maig_bounds_and_signal(toy):
    X, y = toy
    s = maig_scores(X, y)
    assert np.all(s >= 0) and np.all(s <= 1 + 1e-9)
    assert np.argmax(s) == 0 and s[0] > s[2:].max()


def test_maig_monotone_invariance(toy):
    X, y = toy
    Xt = X.copy().astype(float)
    Xt[:, 0] = np.exp(Xt[:, 0]) * 3 + 7          # strictly increasing transform
    assert np.allclose(maig_scores(X, y)[0], maig_scores(Xt, y)[0])


def test_maig_balanced_prevalence_invariance(toy):
    X, y = toy
    rng = np.random.default_rng(1)
    neg = np.flatnonzero(y == 0)
    keep = np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, len(neg) // 3, replace=False)])
    a = maig_scores(X, y, balanced=True)[0]
    b = maig_scores(X[keep], y[keep], balanced=True)[0]
    assert abs(a - b) / a < 0.15


def test_maig_selector_drops_duplicates(toy):
    X, y = toy
    Xd = np.hstack([X, X[:, [0]] + 1e-3 * np.random.default_rng(2).standard_normal((len(y), 1))])
    sel = MAIGSelector(redundancy_eta=1.0, gamma=0.2, min_features=1).fit(Xd, y)
    assert not (sel.support_[0] and sel.support_[-1])


@pytest.mark.parametrize("method", ["information_gain", "mutual_information", "relieff", "mrmr", "anova_f", "best_split_ig"])
def test_reference_rankers_find_signal(toy, method):
    X, y = toy
    assert rank_features(method, X, y)[0] in (0, 1)


# ------------------------------------------------------------------ fold-confined preprocessing
def test_preprocessor_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200), "ethnicity": ["x"] * 100 + ["y"] * 100})
    te = pd.DataFrame({"a": rng.normal(50, 1, 50), "b": rng.normal(5, 2, 50), "ethnicity": ["x"] * 50})
    tr.loc[:10, "a"] = np.nan
    p = FoldPreprocessor(["ethnicity"], {}, 0).fit(tr)
    mean_before = p.mean_.copy()
    out = p.transform(te)
    assert np.allclose(p.mean_, mean_before)                 # transform never refits
    assert out[:, 0].mean() > 20                              # test shift is not normalised away
    assert p.feature_names_ == ["a", "b", "ethnicity=x", "ethnicity=y"]


def test_oversampling_balances_training_only(toy):
    X, y = toy
    Xb, yb = noise_oversample(X, y, (0.01, 0.05), np.random.default_rng(0))
    assert yb.mean() == pytest.approx(0.5)
    assert np.allclose(Xb[: len(X)], X)


# ------------------------------------------------------------------ models
def test_svdnn_learns_and_exposes_margin_units(toy):
    X, y = toy
    m = SVDNN(n_margin=4, width=16, depth=2, max_epochs=15, patience=5, seed=0).fit(X[:1000], y[:1000], X[1000:], y[1000:])
    assert roc_auc_score(y[1000:], m.predict_score(X[1000:])) > 0.85
    assert len(m.margin_unit_auc(X[1000:], y[1000:])) == 4
    W = m.margin_directions()
    assert np.allclose(np.linalg.norm(W, axis=1), 1)


@pytest.mark.parametrize("variant", list(SVDNN_VARIANTS))
def test_all_variants_train(toy, cfg, variant):
    X, y = toy
    c = copy.deepcopy(cfg)
    c["training"].update({"max_epochs": 2, "patience": 1})
    m = make_model(variant, {}, c, seed=0)
    m.fit(X[:500], y[:500], X[500:700], y[500:700])
    p = m.predict_score(X[700:800])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_sequence_layout():
    names = ["age", "bmi_t0", "bmi_t1", "bmi_t2", "hba1c_t0", "hba1c_t1", "hba1c_t2", "sex_male"]
    dyn, stat = sequence_layout(names)
    assert dyn.shape == (2, 3)
    assert list(stat) == [0, 7]
    assert sequence_layout(["age", "bmi"]) is None


# ------------------------------------------------------------------ statistics
def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    a = 0.85 + 0.01 * rng.standard_normal(25)
    b = a - 0.004 + 0.005 * rng.standard_normal(25)
    from scipy import stats

    naive = stats.ttest_rel(a, b).pvalue
    corr = corrected_resampled_ttest(a, b, n_train=8000, n_test=2700)["p"]
    assert corr > naive


def test_holm_and_delong_and_prior_shift():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.all(adj >= [0.01, 0.04, 0.03]) and adj.max() <= 1
    y = np.r_[np.zeros(50), np.ones(50)]
    p = np.linspace(0, 1, 100)
    assert delong_test(y, p, p)["p"] == pytest.approx(1.0)
    assert np.allclose(prior_shift(p[1:-1], 0.3, 0.3), p[1:-1])
    imp = improvement(0.872, 0.813)
    assert imp["absolute_pp"] == pytest.approx(5.9, abs=0.05)
    assert imp["relative_pct"] == pytest.approx(7.26, abs=0.05)
