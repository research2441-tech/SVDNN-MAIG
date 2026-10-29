"""
Mean Absolute Information Gain (MAIG) and the reference feature-ranking methods
used in the comparative study (IG, MI, ReliefF, mRMR, ANOVA-F).

MAIG definition (manuscript Eqs. 1-5, made fully explicit)
---------------------------------------------------------
Let w_i = omega_{y_i} be per-sample weights. With prior balancing,
omega_c = 1 / (2 pi_c), so the weighted class distribution is (0.5, 0.5)
regardless of prevalence; without balancing omega_c = 1.

    H_w(Y)            weighted Shannon entropy of the label (bits)               (Eq. 1)
    theta_t           t-th empirical quantile of f_j, t = 1..k (levels t/(k+1))
    S_t^-, S_t^+      partitions {f_j <= theta_t}, {f_j > theta_t}
    rho_t^s           weighted mass of partition s
    MAIG(f_j) = 1/k * sum_t sum_{s in {-,+}} rho_t^s | H_w(Y) - H_w(Y | S_t^s) |     (Eq. 3)
    G_hat(f_j) = MAIG(f_j) / sum_i MAIG(f_i)                                         (Eq. 4)

Properties (verified in test_pipeline.py):
  P1  0 <= MAIG <= H_w(Y) (= 1 bit when balanced)
  P2  invariant to strictly increasing transformations of f_j (rank based)
  P3  with balancing, invariant in expectation to the class prevalence
  P4  averaging over k thresholds reduces the variance of the single best-split
      information gain, whose maximum over thresholds is biased upward under noise
The one-sided form of the original Eq. 3 (|H(Y) - H(Y | f <= theta)| only) is
available as variant="one_sided" for comparison.

Selection rule (Eq. 5, redundancy-aware)
----------------------------------------
MAIG is univariate, so redundancy is handled explicitly. Features are added
greedily by the adjusted score
    G_adj(f_j | S) = G_hat(f_j) * (1 - eta * max_{s in S} |rho_spearman(f_j, f_s)|)
and retained while G_adj >= gamma * max_i G_hat(f_i) (minimum `min_features`).
gamma is chosen on the inner validation split of each training fold.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.feature_selection import f_classif, mutual_info_classif

EPS = 1e-12


def _entropy(p1: np.ndarray) -> np.ndarray:
    p1 = np.clip(p1, EPS, 1 - EPS)
    return -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))


def _class_weights(y: np.ndarray, balanced: bool) -> np.ndarray:
    if not balanced:
        return np.ones_like(y, dtype=float)
    pi1 = y.mean()
    return np.where(y == 1, 0.5 / max(pi1, EPS), 0.5 / max(1 - pi1, EPS))


def maig_scores(X: np.ndarray, y: np.ndarray, n_thresholds: int = 19, balanced: bool = True,
                variant: str = "two_sided") -> np.ndarray:
    """MAIG score of every column of X (no missing values allowed)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if np.isnan(X).any():
        raise ValueError("MAIG expects imputed data (fit it inside the training fold after imputation).")
    w = _class_weights(y, balanced)
    w1, w0 = w * y, w * (1 - y)
    tot1, tot0 = w1.sum(), w0.sum()
    h_y = _entropy(np.array([tot1 / (tot1 + tot0)]))[0]
    levels = np.arange(1, n_thresholds + 1) / (n_thresholds + 1)
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        c1, c0 = np.cumsum(w1[order]), np.cumsum(w0[order])
        thetas = np.unique(np.quantile(xs, levels, method="lower"))
        pos = np.searchsorted(xs, thetas, side="right")
        pos = pos[(pos > 0) & (pos < len(xs))]
        if pos.size == 0:
            continue
        l1, l0 = c1[pos - 1], c0[pos - 1]
        r1, r0 = tot1 - l1, tot0 - l0
        h_l = _entropy(l1 / (l1 + l0 + EPS))
        h_r = _entropy(r1 / (r1 + r0 + EPS))
        if variant == "one_sided":
            contrib = np.abs(h_y - h_l)
        else:
            rho_l = (l1 + l0) / (tot1 + tot0)
            contrib = rho_l * np.abs(h_y - h_l) + (1 - rho_l) * np.abs(h_y - h_r)
        scores[j] = contrib.mean()
    return scores


def spearman_abs(X: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(stats.rankdata, 0, X)
    c = np.corrcoef(ranks, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    return np.abs(c)


class MAIGSelector:
    """Redundancy-aware MAIG feature selector (fit on training data only)."""

    def __init__(self, n_thresholds: int = 19, balanced: bool = True, redundancy_eta: float = 0.5,
                 gamma: float = 0.10, min_features: int = 5, variant: str = "two_sided"):
        self.n_thresholds = n_thresholds
        self.balanced = balanced
        self.redundancy_eta = redundancy_eta
        self.gamma = gamma
        self.min_features = min_features
        self.variant = variant

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MAIGSelector":
        self.scores_ = maig_scores(X, y, self.n_thresholds, self.balanced, self.variant)
        total = self.scores_.sum()
        self.normalized_ = self.scores_ / total if total > 0 else np.full_like(self.scores_, 1 / len(self.scores_))
        self.corr_ = spearman_abs(X) if self.redundancy_eta > 0 else None
        self.select(self.gamma)
        return self

    def select(self, gamma: float) -> np.ndarray:
        """(Re)apply the greedy selection for a given gamma without recomputing scores."""
        g = self.normalized_
        d = len(g)
        remaining = list(range(d))
        selected, adjusted_at_entry = [], []
        cutoff = gamma * g.max()
        while remaining:
            if selected and self.corr_ is not None:
                red = self.corr_[np.ix_(remaining, selected)].max(axis=1)
                adj = g[remaining] * (1 - self.redundancy_eta * red)
            else:
                adj = g[remaining].copy()
            k = int(np.argmax(adj))
            if adj[k] < cutoff and len(selected) >= self.min_features:
                break
            selected.append(remaining.pop(k))
            adjusted_at_entry.append(float(adj[k]))
        self.gamma = gamma
        self.order_ = np.array(selected)
        self.adjusted_scores_ = np.array(adjusted_at_entry)
        self.support_ = np.zeros(d, dtype=bool)
        self.support_[self.order_] = True
        return self.support_


# ---------------------------------------------------------------------------
# Reference rankers. Each returns a score per feature (higher = more relevant),
# except mRMR, which returns a greedy order.
# ---------------------------------------------------------------------------
def information_gain(X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Conventional IG (Eq. 2) after equal-frequency discretisation."""
    h_y = _entropy(np.array([y.mean()]))[0]
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        r = stats.rankdata(X[:, j], method="average") / len(y)
        bins = np.minimum((r * n_bins).astype(int), n_bins - 1)
        cond = 0.0
        for b in np.unique(bins):
            m = bins == b
            cond += m.mean() * _entropy(np.array([y[m].mean()]))[0]
        out[j] = h_y - cond
    return out


def best_split_information_gain(X: np.ndarray, y: np.ndarray, n_thresholds: int = 19) -> np.ndarray:
    """Max information gain over the same k quantile thresholds used by MAIG (unbalanced)."""
    h_y = _entropy(np.array([y.mean()]))[0]
    levels = np.arange(1, n_thresholds + 1) / (n_thresholds + 1)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        best = 0.0
        for th in np.unique(np.quantile(X[:, j], levels, method="lower")):
            m = X[:, j] <= th
            if m.all() or (~m).all():
                continue
            cond = m.mean() * _entropy(np.array([y[m].mean()]))[0] + (~m).mean() * _entropy(np.array([y[~m].mean()]))[0]
            best = max(best, h_y - cond)
        out[j] = best
    return out


def mutual_information(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    return mutual_info_classif(X, y, n_neighbors=3, random_state=seed)


def anova_f(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    f, _ = f_classif(X, y)
    return np.nan_to_num(f)


def relieff(X: np.ndarray, y: np.ndarray, n_neighbors: int = 10, n_samples: int = 1000, seed: int = 0) -> np.ndarray:
    """ReliefF for binary outcomes (Kononenko, 1994) with Manhattan distance on min-max scaled data."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    Z = (X - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    m = min(n_samples, len(y))
    idx = rng.choice(len(y), m, replace=False)
    weights = np.zeros(X.shape[1])
    for start in range(0, m, 250):
        batch = idx[start:start + 250]
        dist = cdist(Z[batch], Z, metric="cityblock")
        dist[np.arange(len(batch)), batch] = np.inf
        for row, i in enumerate(batch):
            same = np.flatnonzero(y == y[i])
            other = np.flatnonzero(y != y[i])
            hits = same[np.argpartition(dist[row, same], min(n_neighbors, len(same) - 1))[:n_neighbors]]
            miss = other[np.argpartition(dist[row, other], min(n_neighbors, len(other) - 1))[:n_neighbors]]
            weights -= np.abs(Z[hits] - Z[i]).mean(0)
            weights += np.abs(Z[miss] - Z[i]).mean(0)
    return weights / m


def mrmr_order(X: np.ndarray, y: np.ndarray, k: int | None = None, seed: int = 0) -> np.ndarray:
    """mRMR (MID criterion). Relevance: kNN mutual information; redundancy:
    Gaussian-copula mutual information from Spearman correlations."""
    rel = mutual_information(X, y, seed)
    rho = np.clip(spearman_abs(X), 0, 0.999)
    red = -0.5 * np.log(1 - rho**2)
    d = X.shape[1]
    k = d if k is None else min(k, d)
    selected = [int(np.argmax(rel))]
    remaining = [j for j in range(d) if j != selected[0]]
    while len(selected) < k:
        score = rel[remaining] - red[np.ix_(remaining, selected)].mean(axis=1)
        j = remaining.pop(int(np.argmax(score)))
        selected.append(j)
    return np.array(selected)


SELECTORS = ["maig", "information_gain", "best_split_ig", "mutual_information", "relieff", "mrmr", "anova_f"]


def rank_features(method: str, X: np.ndarray, y: np.ndarray, seed: int = 0, maig_kwargs: dict | None = None) -> np.ndarray:
    """Return feature indices ordered from most to least relevant."""
    if method == "mrmr":
        return mrmr_order(X, y, seed=seed)
    if method == "maig":
        s = maig_scores(X, y, **(maig_kwargs or {}))
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
    else:
        raise ValueError(f"Unknown selector '{method}'")
    return np.argsort(-s, kind="mergesort")


def top_k_mask(order: np.ndarray, k: int, d: int) -> np.ndarray:
    mask = np.zeros(d, dtype=bool)
    mask[order[:k]] = True
    return mask
