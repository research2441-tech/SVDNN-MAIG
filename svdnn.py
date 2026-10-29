"""
Support Vectorized Decision Neural Network (SVDNN).

Complete formulation (manuscript Section 3, rewritten in full)
--------------------------------------------------------------
Input        x in R^p : z-scored values of the MAIG-selected features F* (p = |F*|).
Margin layer psi_k(x) = w_k^T x + b_k,  k = 1..m                                      (Eq. 6)
             A bank of m soft-margin linear discriminants. Each unit is trained as a
             linear SVM through its own hinge loss (Eqs. 7-8) and the units are kept
             mutually orthogonal, so psi(x) = [psi_1..psi_m] is a vector of signed
             functional margins along m distinct max-margin directions.
Decision     h^0 = [psi(x) ; x]        (skip connection keeps raw features available)
network      h^l = Dropout(ReLU(LN(W^l h^{l-1} + b^l))),  l = 1..L                    (Eq. 9)
Output       s(x) = w_o^T h^L + b_o,   p_hat(x) = sigmoid(s(x))                         (Eq. 10)
             (sigmoid = two-class softmax).

Objective    L = beta L_CE + alpha L_SVM + mu L_div + lambda L_J                        (Eq. 12)
  L_CE   = -1/n sum_i [y_i log p_hat_i + (1 - y_i) log(1 - p_hat_i)]
  L_SVM  = 1/m sum_k [ ||w_k||^2 / (2 C n) + 1/n sum_i max(0, 1 - y~_i psi_k(x_i)) ]    (Eqs. 7-8 / (C n))
           with y~ = 2y - 1 in {-1, +1}
  L_div  = ||W_bar W_bar^T - I_m||_F^2 / (m (m - 1)),  W_bar = row-normalised [w_1..w_m]^T
  L_J    = 1/(n p) sum_i ||d s(x_i) / d x_i||_2^2          (input-gradient / Jacobian penalty)
Optimiser    Adam (Eq. 13) with ReduceLROnPlateau and early stopping on validation AUROC.

Why this differs from "an MLP with a hinge loss": in an MLP the first layer is an
unconstrained feature extractor and a hinge loss, if any, acts on the output only
(baseline `mlp_output_hinge`). In SVDNN every first-layer unit is itself a
supervised max-margin classifier with its own slack variables and margin
regulariser, and orthogonality forces the bank to cover different discriminative
directions. The downstream network therefore operates in a "margin coordinate
system" whose axes are individually interpretable linear decision rules. Each
component is removed in turn in the ablation study (run_experiments.py).

Note on the saliency term: the original Eq. 12 used lambda * sum_j (1 - S_j)^2, which
rewards every feature for being salient and is unbounded below for S_j > 1. It is
replaced by the input-gradient penalty L_J, a standard, well-posed regulariser that
smooths the decision function and stabilises gradient explanations.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class TorchBinaryClassifier:
    """Training loop shared by every neural model in the study (SVDNN, MLP, LSTM, CNN-LSTM)."""

    requires_sequence = False

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-5, max_epochs: int = 120, patience: int = 15,
                 batch_size: int = 256, lr_scheduler_patience: int = 5, lr_scheduler_factor: float = 0.5,
                 seed: int = 0, n_threads: int | None = None, **arch):
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.sched_patience = lr_scheduler_patience
        self.sched_factor = lr_scheduler_factor
        self.seed = seed
        self.n_threads = n_threads
        self.arch = arch
        self.history_: list[dict] = []

    # --- to be implemented by subclasses ---------------------------------------------
    def build(self, p: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, xb: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self.net_(xb).squeeze(-1), {}

    def extra_loss(self, xb: torch.Tensor, yb: torch.Tensor, logit: torch.Tensor, aux: dict, n_train: int) -> torch.Tensor:
        return torch.zeros((), device=logit.device)

    # ---------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "TorchBinaryClassifier":
        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        self.p_ = X.shape[1]
        self.net_ = self.build(self.p_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=self.sched_factor,
                                                           patience=self.sched_patience)
        bce = nn.BCEWithLogitsLoss()
        best_auc, best_state, wait = -np.inf, None, 0
        n = len(y)
        start = time.time()
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = rng.permutation(n)
            total = 0.0
            for b in range(0, n, self.batch_size):
                idx = perm[b:b + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                logit, aux = self.forward(xb)
                loss = self.arch.get("beta", 1.0) * bce(logit, yb) + self.extra_loss(xb, yb, logit, aux, n)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            val_auc = roc_auc_score(y_val, self.predict_score(X_val))
            sched.step(val_auc)
            self.history_.append({"epoch": epoch, "train_loss": total / n, "val_auc": val_auc})
            if val_auc > best_auc + 1e-4:
                best_auc, best_state, wait = val_auc, copy.deepcopy(self.net_.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_val_auc_ = best_auc
        self.n_epochs_ = len(self.history_)
        self.fit_seconds_ = time.time() - start
        return self

    @torch.no_grad()
    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        out = []
        X_t = torch.as_tensor(X, dtype=torch.float32)
        for b in range(0, len(X_t), 4096):
            out.append(self.forward(X_t[b:b + 4096])[0])
        return torch.cat(out).numpy()

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logit(X)))

    def logit_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable logit for explanation methods."""
        self.net_.eval()
        return self.forward(x)[0]


# =====================================================================================
class SVDNNNet(nn.Module):
    def __init__(self, p: int, n_margin: int, width: int, depth: int, dropout: float,
                 use_margin: bool = True, skip: bool = True):
        super().__init__()
        self.use_margin = use_margin and n_margin > 0
        self.skip = skip or not self.use_margin
        if self.use_margin:
            self.margin = nn.Linear(p, n_margin)
        in_dim = (n_margin if self.use_margin else 0) + (p if self.skip else 0)
        layers: list[nn.Module] = []
        for l in range(depth):
            layers += [nn.Linear(in_dim if l == 0 else width, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(width if depth > 0 else in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        psi = self.margin(x) if self.use_margin else None
        parts = ([psi] if psi is not None else []) + ([x] if self.skip else [])
        h = torch.cat(parts, dim=1)
        return self.out(self.body(h)).squeeze(-1), psi


class SVDNN(TorchBinaryClassifier):
    """SVDNN and all ablation variants (switch components off through the flags)."""

    def __init__(self, n_margin: int = 16, width: int = 128, depth: int = 3, dropout: float = 0.2,
                 C: float = 10.0, alpha: float = 0.5, mu: float = 0.1, lam: float = 1e-3, beta: float = 1.0,
                 use_margin: bool = True, skip: bool = True, output_hinge: bool = False, **kw):
        super().__init__(**kw)
        self.arch = dict(n_margin=n_margin, width=width, depth=depth, dropout=dropout, C=C, alpha=alpha, mu=mu,
                         lam=lam, beta=beta, use_margin=use_margin, skip=skip, output_hinge=output_hinge)

    def build(self, p: int) -> nn.Module:
        a = self.arch
        return SVDNNNet(p, a["n_margin"], a["width"], a["depth"], a["dropout"], a["use_margin"], a["skip"])

    def forward(self, xb: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logit, psi = self.net_(xb)
        return logit, {"psi": psi}

    def extra_loss(self, xb, yb, logit, aux, n_train):
        a = self.arch
        loss = torch.zeros((), device=logit.device)
        y_pm = 2 * yb - 1
        psi = aux["psi"]
        if psi is not None and a["alpha"] > 0:
            W = self.net_.margin.weight                                   # (m, p)
            hinge = torch.relu(1 - y_pm[:, None] * psi).mean()            # (1/m)(1/n) sum xi
            reg = (W.pow(2).sum(1) / (2 * a["C"] * n_train)).mean()       # (1/m) sum ||w_k||^2 / (2Cn)
            loss = loss + a["alpha"] * (hinge + reg)
        if psi is not None and a["mu"] > 0 and psi.shape[1] > 1:
            W = self.net_.margin.weight
            Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
            m = W.shape[0]
            gram = Wn @ Wn.T - torch.eye(m, device=W.device)
            loss = loss + a["mu"] * gram.pow(2).sum() / (m * (m - 1))
        if a["output_hinge"] and a["alpha"] > 0:
            loss = loss + a["alpha"] * torch.relu(1 - y_pm * logit).mean()
        if a["lam"] > 0:
            x_req = xb.detach().requires_grad_(True)
            s, _ = self.net_(x_req)
            grad = torch.autograd.grad(s.sum(), x_req, create_graph=True)[0]
            loss = loss + a["lam"] * grad.pow(2).sum(1).mean() / xb.shape[1]
        return loss

    # --- interpretability helpers ----------------------------------------------------
    def margin_directions(self) -> np.ndarray | None:
        """Unit-norm weight vectors of the m margin units (rows), in z-score units."""
        if not self.net_.use_margin:
            return None
        W = self.net_.margin.weight.detach().numpy()
        return W / np.linalg.norm(W, axis=1, keepdims=True)

    @torch.no_grad()
    def margin_unit_auc(self, X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """Stand-alone AUROC of every margin unit (each is a linear SVM)."""
        if not self.net_.use_margin:
            return None
        psi = self.net_.margin(torch.as_tensor(X, dtype=torch.float32)).numpy()
        return np.array([roc_auc_score(y, psi[:, k]) for k in range(psi.shape[1])])


SVDNN_VARIANTS = {
    # name: (flag overrides on top of the tuned SVDNN hyper-parameters, uses all features?)
    "svdnn": ({}, False),
    "svdnn_no_maig": ({}, True),
    "svdnn_no_margin": ({"use_margin": False}, False),
    "svdnn_margin_unsupervised": ({"alpha": 0.0, "mu": 0.0}, False),
    "svdnn_no_diversity": ({"mu": 0.0}, False),
    "svdnn_no_skip": ({"skip": False}, False),
    "svdnn_no_jacobian": ({"lam": 0.0}, False),
    "svdnn_margin_only": ({"depth": 0, "skip": False}, False),
    "mlp_output_hinge": ({"use_margin": False, "output_hinge": True, "mu": 0.0}, False),
}
