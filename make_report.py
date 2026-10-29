"""
Builds every manuscript table and figure from the saved fold-level results.

All numbers in the text, tables and figures come from ONE results store
(outputs/<mode>/), which removes the inconsistencies between text, tables and
figures noted in review. results_summary.json additionally contains ready-to-paste
sentences (e.g. absolute vs relative F1 improvement, FNR reduction with CI).

Outputs: outputs/<mode>/report/{tables/*.csv, tables/*.md, figures/*.png|pdf, results_summary.json}
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from baselines import DISPLAY_NAMES  # noqa: E402
from evaluation import fold_metric_table, load_results  # noqa: E402
from metrics_stats import (bootstrap_sd_ci, brown_forsythe, calibration_curve_quantile, cliffs_delta,  # noqa: E402
                           cohens_dz, corrected_resampled_ttest, delong_test, holm_adjust, improvement,
                           net_benefit, paired_bootstrap_rate_reduction, population_ppv_npv, prior_shift,
                           sensitivity_at_specificity, wilcoxon_p)
from utils import ensure_dir, get_logger, load_json, save_json  # noqa: E402

# fixed categorical order (validated palette, light mode); colour follows the model, never its rank
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["svdnn", "logistic_regression", "random_forest", "xgboost", "svm_rbf", "mlp", "lstm_rnn", "cnn_lstm"]
COLORS = dict(zip(MODEL_ORDER, PALETTE))
LINESTYLES = dict(zip(MODEL_ORDER, ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]))
ROLE_COLORS = {"glycaemic_marker": "#2a78d6", "upstream_determinant": "#1baf7a",
               "non_causal_correlate": "#eb6834", "independent": "#8a8984"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 7, "savefig.dpi": 300,
    "figure.dpi": 110, "lines.linewidth": 1.6,
})

MAIN_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc", "brier",
                "calibration_intercept", "calibration_slope"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
                 "f1": "F1", "roc_auc": "AUROC", "pr_auc": "AUPRC", "mcc": "MCC", "brier": "Brier",
                 "calibration_intercept": "Cal. intercept", "calibration_slope": "Cal. slope", "fnr": "FNR",
                 "balanced_accuracy": "Balanced acc.", "ece": "ECE", "log_loss": "Log loss"}
LOWER_IS_BETTER = {"brier", "fnr", "fpr", "log_loss", "ece"}


def name(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


def save_fig(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, path: Path, floatfmt: str = ".3f") -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:  # tabulate missing
        md = df.to_string(index=False)
    path.with_suffix(".md").write_text(md + "\n", encoding="utf-8")


def mean_ci(x, n_train, n_test, alpha=0.05):
    """Mean with corrected-resampled 95% CI (accounts for overlapping training sets)."""
    x = np.asarray(x, dtype=float)
    J = len(x)
    if J < 2:
        return x.mean(), np.nan, np.nan
    se = np.sqrt((1 / J + n_test / n_train) * x.var(ddof=1))
    h = stats.t.ppf(1 - alpha / 2, J - 1) * se
    return x.mean(), x.mean() - h, x.mean() + h


# =====================================================================================
# Pooled out-of-fold predictions
# =====================================================================================
def pooled_predictions(records, preds, key, n):
    """Return (y, P[n, R] probabilities, B[n, R] binary predictions) for a model key."""
    R = max(r["repeat"] for r in records) + 1
    P = np.full((n, R), np.nan)
    B = np.full((n, R), np.nan)
    y = np.full(n, -1)
    for rec, pr in zip(records, preds):
        if key not in pr:
            continue
        section = "ablations" if key.startswith("ablation:") else "models"
        mname = key.split(":", 1)[1] if ":" in key else key
        thr = rec[section][mname]["threshold"]
        idx = pr["test_idx"]
        P[idx, rec["repeat"]] = pr[key]
        B[idx, rec["repeat"]] = (pr[key] >= thr).astype(float)
        y[idx] = pr["y"]
    return y, P, B


# =====================================================================================
# Main experiment
# =====================================================================================
def report_main(cfg, out_tab, out_fig, summary):
    log = get_logger()
    root = Path(cfg["output_dir"])
    records, preds = load_results(root / "main")
    if not records:
        log.info("No main results found.")
        return
    proposed = cfg["proposed"]
    fm = fold_metric_table(records)
    models = [m for m in MODEL_ORDER if m in fm["model"].unique()]
    n_train, n_test = fm["n_train"].mean(), fm["n_test"].mean()
    J = fm.groupby("model").size().max()
    n = max(int(p["test_idx"].max()) for p in preds) + 1
    data_summary = load_json(root / "data" / "generator_summary.json")
    pi_pop = cfg["cv"].get("population_prevalence") or data_summary["source_incidence"]
    pi_sample = data_summary["sample_prevalence"]
    summary["evaluation_protocol"] = {
        "outer_repeats": cfg["cv"]["outer_repeats"], "outer_folds": cfg["cv"]["outer_folds"],
        "held_out_evaluations_per_model": int(J), "mean_inner_train": n_train, "mean_test": n_test,
        "hyperparameter_trials_per_model_per_fold": cfg["tuning"]["n_trials"],
        "sentence": (f"Each model was evaluated on {J} held-out outer folds ({cfg['cv']['outer_repeats']} repeats x "
                     f"{cfg['cv']['outer_folds']}-fold stratified cross-validation); every participant received one "
                     f"held-out prediction per repeat. All preprocessing, MAIG selection, oversampling, tuning "
                     f"({cfg['tuning']['n_trials']} TPE trials per model and fold) and threshold selection were "
                     f"performed inside the training part of each outer fold."),
    }

    # ---------------- Table: performance --------------------------------------------
    rows = []
    for m in models:
        d = fm[fm.model == m]
        row = {"Model": name(m)}
        for met in MAIN_METRICS:
            mu, lo, hi = mean_ci(d[met], n_train, n_test)
            row[METRIC_LABELS[met]] = f"{mu:.3f} ± {d[met].std(ddof=1):.3f}"
            row[f"{METRIC_LABELS[met]} 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    perf = pd.DataFrame(rows)
    save_table(perf, out_tab / "table3_performance")
    agg = fm.groupby("model")[MAIN_METRICS + ["fnr", "fpr", "ece"]].mean()
    summary["mean_metrics"] = agg.round(4).to_dict("index")

    # ---------------- Pairwise statistics vs proposed ----------------------------------
    comp_rows = []
    for met in cfg["stats"]["compare_metrics"]:
        pv_c, pv_w, block = [], [], []
        for m in models:
            if m == proposed:
                continue
            a = fm[fm.model == proposed].sort_values(["repeat", "fold"])[met].to_numpy()
            b = fm[fm.model == m].sort_values(["repeat", "fold"])[met].to_numpy()
            t = corrected_resampled_ttest(a, b, n_train, n_test)
            block.append({"Metric": METRIC_LABELS[met], "Comparator": name(m),
                          "Proposed mean": a.mean(), "Comparator mean": b.mean(), "Difference": t["mean_diff"],
                          "95% CI (corrected)": f"[{t['ci_low']:.3f}, {t['ci_high']:.3f}]",
                          "p (corrected t)": t["p"], "p (Wilcoxon)": wilcoxon_p(a, b),
                          "Cohen d_z": cohens_dz(a, b), "Cliff delta": cliffs_delta(a, b)})
            pv_c.append(t["p"])
            pv_w.append(block[-1]["p (Wilcoxon)"])
        hc, hw = holm_adjust(pv_c), holm_adjust(pv_w)
        for r, a1, a2 in zip(block, hc, hw):
            r["p Holm (corrected t)"] = a1
            r["p Holm (Wilcoxon)"] = a2
            comp_rows.append(r)
    comp = pd.DataFrame(comp_rows)
    save_table(comp, out_tab / "table_pairwise_statistics", floatfmt=".4f")
    friedman = {}
    for met in ["roc_auc", "f1", "mcc"]:
        pivot = fm.pivot_table(index=["repeat", "fold"], columns="model", values=met)[models]
        friedman[met] = float(stats.friedmanchisquare(*[pivot[c] for c in models]).pvalue)
    summary["friedman_p"] = friedman

    # ---------------- Improvement statements -----------------------------------------
    baselines = [m for m in models if m != proposed]
    best_base = {met: (max if met not in LOWER_IS_BETTER else min)(baselines, key=lambda m: agg.loc[m, met])
                 for met in ["f1", "roc_auc", "mcc", "recall"]}
    imp = {}
    for met, bb in best_base.items():
        imp[met] = {"best_baseline": bb, **improvement(agg.loc[proposed, met], agg.loc[bb, met]),
                    "proposed": agg.loc[proposed, met], "baseline": agg.loc[bb, met]}
    if "cnn_lstm" in agg.index:
        imp["f1_vs_cnn_lstm"] = {"best_baseline": "cnn_lstm", **improvement(agg.loc[proposed, "f1"], agg.loc["cnn_lstm", "f1"]),
                                 "proposed": agg.loc[proposed, "f1"], "baseline": agg.loc["cnn_lstm", "f1"]}
    summary["improvements"] = imp
    f1 = imp["f1"]
    summary["sentences"] = {
        "f1": (f"SVDNN-MAIG reached a mean F1 of {f1['proposed']:.3f} versus {f1['baseline']:.3f} for the strongest "
               f"comparator ({name(f1['best_baseline'])}): an absolute difference of {f1['absolute_pp']:+.1f} percentage "
               f"points, i.e. a relative change of {f1['relative_pct']:+.1f}%."),
    }

    # ---------------- FNR and threshold-free sensitivity -----------------------------
    yP, Pp, Bp = pooled_predictions(records, preds, proposed, n)
    fnr_rows, sens_rows, delong_rows = [], [], []
    for m in baselines:
        _, Pm, Bm = pooled_predictions(records, preds, m, n)
        r = paired_bootstrap_rate_reduction(yP, Bm, Bp, cfg["stats"]["n_bootstrap"], cfg["seed"], "fnr")
        fnr_rows.append({"Comparator": name(m), "Comparator FNR": r["fnr_baseline"], "SVDNN-MAIG FNR": r["fnr_proposed"],
                         "Absolute reduction (pp)": 100 * r["absolute_reduction"],
                         "95% CI (pp)": f"[{100 * r['absolute_ci'][0]:.1f}, {100 * r['absolute_ci'][1]:.1f}]",
                         "Relative reduction (%)": 100 * r["relative_reduction"],
                         "95% CI (%)": f"[{100 * r['relative_ci'][0]:.1f}, {100 * r['relative_ci'][1]:.1f}]"})
        for spec in (0.80, 0.90):
            sp = [sensitivity_at_specificity(yP, Pp[:, j], spec) for j in range(Pp.shape[1])]
            sm = [sensitivity_at_specificity(yP, Pm[:, j], spec) for j in range(Pm.shape[1])]
            sens_rows.append({"Comparator": name(m), "Specificity": spec, "Sensitivity SVDNN-MAIG": np.mean(sp),
                              "Sensitivity comparator": np.mean(sm), "Difference": np.mean(sp) - np.mean(sm)})
        dl = [delong_test(yP, Pp[:, j], Pm[:, j]) for j in range(Pp.shape[1])]
        delong_rows.append({"Comparator": name(m), "AUROC SVDNN-MAIG (pooled)": np.mean([d["auc1"] for d in dl]),
                            "AUROC comparator (pooled)": np.mean([d["auc2"] for d in dl]),
                            "DeLong p (median over repeats)": float(np.median([d["p"] for d in dl])),
                            "DeLong p (max over repeats)": float(np.max([d["p"] for d in dl]))})
    fnr = pd.DataFrame(fnr_rows)
    save_table(fnr, out_tab / "table_false_negative_rates")
    save_table(pd.DataFrame(sens_rows), out_tab / "table_sensitivity_at_fixed_specificity")
    save_table(pd.DataFrame(delong_rows), out_tab / "table_delong", floatfmt=".4f")
    best_fnr = fnr.sort_values("Comparator FNR").iloc[0]
    summary["sentences"]["fnr"] = (
        f"At the inner-validation Youden threshold, the false-negative rate of SVDNN-MAIG was {best_fnr['SVDNN-MAIG FNR']:.3f} "
        f"versus {best_fnr['Comparator FNR']:.3f} for {best_fnr['Comparator']} (lowest-FNR comparator): absolute reduction "
        f"{best_fnr['Absolute reduction (pp)']:.1f} pp {best_fnr['95% CI (pp)']}, relative reduction "
        f"{best_fnr['Relative reduction (%)']:.1f}% {best_fnr['95% CI (%)']}. FNR depends on the operating threshold; "
        f"threshold-free comparisons are given as sensitivity at fixed specificity.")

    # ---------------- Calibration and population prevalence --------------------------
    cal_rows = []
    for m in models:
        d = fm[fm.model == m]
        sens, spec = d["recall"].mean(), d["specificity"].mean()
        ppv, npv = population_ppv_npv(sens, spec, pi_pop)
        cal_rows.append({"Model": name(m), "Brier": d["brier"].mean(), "Cal. intercept": d["calibration_intercept"].mean(),
                         "Cal. slope": d["calibration_slope"].mean(), "ECE": d["ece"].mean(),
                         f"PPV at population incidence ({pi_pop:.3f})": ppv,
                         f"NPV at population incidence ({pi_pop:.3f})": npv})
    save_table(pd.DataFrame(cal_rows), out_tab / "table_calibration")
    summary["prevalence"] = {"sample_prevalence": pi_sample, "source_population_incidence": pi_pop,
                             "sentence": (f"The analysis sample is case-enriched (prevalence {pi_sample:.3f}); it is neither "
                                          f"balanced nor representative of the source population, whose 12-month incidence "
                                          f"was {pi_pop:.3f}. Predictive values at the population incidence are reported "
                                          f"separately.")}

    # ---------------- Variability (replaces 'robustness' claim) ----------------------
    var_rows = []
    for m in models:
        row = {"Model": name(m)}
        for met in ["roc_auc", "f1", "mcc"]:
            x = fm[fm.model == m][met].to_numpy()
            lo, hi = bootstrap_sd_ci(x, cfg["stats"]["n_bootstrap"], cfg["seed"])
            row[f"SD {METRIC_LABELS[met]}"] = f"{x.std(ddof=1):.4f} [{lo:.4f}, {hi:.4f}]"
            if m != proposed:
                row[f"Brown-Forsythe p ({METRIC_LABELS[met]})"] = brown_forsythe(
                    fm[fm.model == proposed][met], fm[fm.model == m][met])
        var_rows.append(row)
    save_table(pd.DataFrame(var_rows), out_tab / "table5_fold_variability", floatfmt=".4f")

    # ---------------- Ablation --------------------------------------------------------
    ab = fold_metric_table(records, "ablations")
    if not ab.empty:
        full = fm[fm.model == proposed].sort_values(["repeat", "fold"])
        rows = [{"Variant": name(proposed), "AUROC": full["roc_auc"].mean(), "F1": full["f1"].mean(),
                 "MCC": full["mcc"].mean(), "Brier": full["brier"].mean(), "ΔAUROC vs full": 0.0,
                 "95% CI": "", "p (corrected t)": np.nan}]
        for v in cfg["ablations"]:
            d = ab[ab.model == v].sort_values(["repeat", "fold"])
            if d.empty:
                continue
            t = corrected_resampled_ttest(d["roc_auc"].to_numpy(), full["roc_auc"].to_numpy(), n_train, n_test)
            rows.append({"Variant": name(v), "AUROC": d["roc_auc"].mean(), "F1": d["f1"].mean(), "MCC": d["mcc"].mean(),
                         "Brier": d["brier"].mean(), "ΔAUROC vs full": t["mean_diff"],
                         "95% CI": f"[{t['ci_low']:.4f}, {t['ci_high']:.4f}]", "p (corrected t)": t["p"]})
        abl = pd.DataFrame(rows)
        save_table(abl, out_tab / "table_ablation", floatfmt=".4f")
        fig_ablation(abl, out_fig / "fig_ablation")

    # ---------------- Feature-ranking method comparison ------------------------------
    sel_rows = []
    for method in cfg["selector_comparison"]:
        vals = [rec["selectors"][method] for rec in records if method in rec.get("selectors", {})]
        if not vals:
            continue
        maig_sets = [set(rec["selectors"]["maig"]["features"]) for rec in records if "maig" in rec.get("selectors", {})]
        jac = [len(set(v["features"]) & s) / len(set(v["features"]) | s) for v, s in zip(vals, maig_sets)]
        sel_rows.append({"Ranking method": method, "Features (k)": np.mean([len(v["features"]) for v in vals]),
                         "SVDNN AUROC": np.mean([v["svdnn"]["roc_auc"] for v in vals]),
                         "SVDNN AUROC SD": np.std([v["svdnn"]["roc_auc"] for v in vals], ddof=1),
                         "SVDNN F1": np.mean([v["svdnn"]["f1"] for v in vals]),
                         "LR AUROC": np.mean([v["logistic_regression"]["roc_auc"] for v in vals]),
                         "LR F1": np.mean([v["logistic_regression"]["f1"] for v in vals]),
                         "Jaccard with MAIG set": np.mean(jac)})
    if sel_rows:
        sel = pd.DataFrame(sel_rows)
        save_table(sel, out_tab / "table_feature_ranking_methods")
        if "maig" in sel["Ranking method"].values:
            maig_vals = [rec["selectors"]["maig"]["svdnn"]["roc_auc"] for rec in records]
            pv = []
            for method in sel["Ranking method"]:
                if method == "maig":
                    pv.append(np.nan)
                    continue
                other = [rec["selectors"][method]["svdnn"]["roc_auc"] for rec in records]
                pv.append(corrected_resampled_ttest(maig_vals, other, n_train, n_test)["p"])
            sel["p vs MAIG (SVDNN AUROC)"] = pv
            save_table(sel, out_tab / "table_feature_ranking_methods")

    # ---------------- Feature importance (Table 4 replacement) ------------------------
    dictionary = pd.read_csv(root / "data" / "data_dictionary.csv").set_index("variable")
    feat = feature_importance_table(records, dictionary)
    save_table(feat.head(15), out_tab / "table4_top_features")
    feat.to_csv(out_tab / "feature_importance_all.csv", index=False)
    fig_feature_importance(feat.head(15), out_fig / "fig7_feature_importance")
    role_summary = feat.head(15)["Causal role (SCM)"].value_counts().to_dict()
    summary["top15_roles"] = role_summary
    parts = [f"{v} {'was a' if v == 1 else 'were'} {k.replace('_', ' ')}{'' if v == 1 else 's'}"
             for k, v in role_summary.items()]
    summary["sentences"]["causality"] = (
        "Among the 15 variables with the highest MAIG scores, " + "; ".join(parts)
        + " in the data-generating model. Rankings and attributions therefore reflect predictive association, "
          "not causal effect.")

    # ---------------- Explanation validation -----------------------------------------
    expl_tables(records, out_tab, out_fig, summary)

    # ---------------- Figures ---------------------------------------------------------
    fig_roc_pr(records, preds, models, n, agg, out_fig)
    fig_calibration_dca(records, preds, models, n, pi_sample, pi_pop, root, out_fig)
    fig_confusion(yP, Bp, out_fig / "fig5_confusion_matrix")
    fig_fold_distributions(fm, models, out_fig / "fig_fold_distributions")
    fig_correlation(root, feat, out_fig / "fig4_correlation_heatmap")


def feature_importance_table(records, dictionary):
    rows = {}
    n_folds = len(records)
    for rec in records:
        for f, s in rec["maig_scores"].items():
            rows.setdefault(f, {"maig": [], "selected": 0, "saliency": [], "ig": []})["maig"].append(s)
        for f in rec["selected_features"]:
            rows[f]["selected"] += 1
        ex = rec.get("explain")
        if ex:
            for f, s, g in zip(ex["features"], ex["saliency_global"], ex["ig_global"]):
                rows[f]["saliency"].append(s)
                rows[f]["ig"].append(g)
    out = []
    for f, d in rows.items():
        base = f.split("=")[0]
        role = dictionary.loc[base, "causal_role"] if base in dictionary.index else "upstream_determinant"
        out.append({"Feature": f, "MAIG (normalised)": np.mean(d["maig"]), "MAIG SD": np.std(d["maig"]),
                    "Selection frequency": d["selected"] / n_folds,
                    # unselected folds contribute zero attribution
                    "Gradient saliency (CFA)": np.sum(d["saliency"]) / n_folds,
                    "Integrated gradients": np.sum(d["ig"]) / n_folds,
                    "Causal role (SCM)": role})
    df = pd.DataFrame(out).sort_values("MAIG (normalised)", ascending=False).reset_index(drop=True)
    df.insert(0, "MAIG rank", np.arange(1, len(df) + 1))
    df["Saliency rank"] = df["Gradient saliency (CFA)"].rank(ascending=False, method="min").astype(int)
    return df


def expl_tables(records, out_tab, out_fig, summary):
    ex = [r["explain"] for r in records if r.get("explain")]
    if not ex:
        return
    # fidelity
    rows = []
    for meth in ["saliency", "integrated_gradients", "maig", "random"]:
        a = [e["fidelity"][meth]["aopc"] for e in ex]
        rows.append({"Attribution ordering": meth, "AOPC mean": np.mean(a), "AOPC SD": np.std(a, ddof=1) if len(a) > 1 else np.nan})
    fid = pd.DataFrame(rows)
    sal_a = np.array([e["fidelity"]["saliency"]["aopc"] for e in ex])
    rnd_a = np.array([e["fidelity"]["random"]["aopc"] for e in ex])
    fid["p vs random (Wilcoxon)"] = [wilcoxon_p([e["fidelity"][m]["aopc"] for e in ex], rnd_a) if m != "random" else np.nan
                                      for m in fid["Attribution ordering"]]
    save_table(fid, out_tab / "table_explanation_fidelity", floatfmt=".4f")
    # stability across folds (union of features; absent = 0)
    feats = sorted({f for e in ex for f in e["features"]})
    M = np.zeros((len(ex), len(feats)))
    for i, e in enumerate(ex):
        for f, s in zip(e["features"], e["saliency_global"]):
            M[i, feats.index(f)] = s
    rhos = [stats.spearmanr(M[i], M[j]).correlation for i, j in itertools.combinations(range(len(ex)), 2)]
    summary["explanation"] = {"saliency_aopc": float(sal_a.mean()), "random_aopc": float(rnd_a.mean()),
                              "saliency_rank_stability_spearman": float(np.nanmean(rhos)) if rhos else np.nan}
    # margin units
    mu = [e["margin_units"] for e in ex if "margin_units" in e]
    if mu:
        summary["margin_units"] = {"mean_unit_auc": float(np.mean([np.mean(m["auc"]) for m in mu])),
                                   "best_unit_auc": float(np.mean([np.max(m["auc"]) for m in mu])),
                                   "mean_abs_cosine_between_units": float(np.nanmean([m["mean_abs_cosine"] for m in mu]))}
    # counterfactuals
    cf_rows = []
    for kind in ["constrained", "unconstrained"]:
        vals = [e["counterfactuals"][kind] for e in ex if kind in e.get("counterfactuals", {}) and e["counterfactuals"][kind].get("n")]
        if not vals:
            continue
        row = {"Counterfactual type": kind}
        for k in ["validity", "l1_distance_sd", "n_features_changed", "changes_immutable_feature", "lof_inlier",
                  "mahalanobis_inlier", "within_empirical_range", "fully_plausible"]:
            row[k] = float(np.nanmean([v[k] for v in vals]))
        acts = {}
        for v in vals:
            for a, c in v.get("mean_raw_change_actionable", {}).items():
                acts.setdefault(a, []).append(c)
        for a, c in acts.items():
            row[f"mean change {a}"] = float(np.mean(c))
        cf_rows.append(row)
    if cf_rows:
        cf = pd.DataFrame(cf_rows)
        save_table(cf, out_tab / "table_counterfactual_plausibility")
        fig_counterfactual(cf, out_fig / "fig_counterfactual_plausibility")
    pm = next((e["patient_map"] for e in ex if e.get("patient_map")), None)
    if pm:
        pd.DataFrame(pm).to_csv(out_tab / "patient_explanation_map.csv", index=False)
        fig_patient_map(pm, out_fig / "fig8_patient_explanation_map")
    fig_deletion(ex, out_fig / "fig_deletion_curves")


# =====================================================================================
# Figures
# =====================================================================================
def fig_roc_pr(records, preds, models, n, agg, out_fig):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for m in models:
        y, P, _ = pooled_predictions(records, preds, m, n)
        grid = np.linspace(0, 1, 201)
        tprs, precs = [], []
        for j in range(P.shape[1]):
            fpr, tpr, _ = roc_curve(y, P[:, j])
            tprs.append(np.interp(grid, fpr, tpr))
            pr, rc, _ = precision_recall_curve(y, P[:, j])
            precs.append(np.interp(grid, rc[::-1], pr[::-1]))
        lw = 2.2 if m == "svdnn" else 1.3
        axes[0].plot(grid, np.mean(tprs, 0), color=COLORS[m], ls=LINESTYLES[m], lw=lw,
                     label=f"{name(m)} ({agg.loc[m, 'roc_auc']:.3f})")
        axes[1].plot(grid, np.mean(precs, 0), color=COLORS[m], ls=LINESTYLES[m], lw=lw,
                     label=f"{name(m)} ({agg.loc[m, 'pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC (mean over repeats; legend: mean AUROC)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (legend: mean AUPRC)")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    save_fig(fig, out_fig / "fig6_roc_pr")


def fig_calibration_dca(records, preds, models, n, pi_sample, pi_pop, root, out_fig):
    latent = pd.read_csv(root / "data" / "cohort_latent_audit.csv")
    w = latent["sampling_weight"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    thr = np.linspace(0.02, 0.6, 59)
    for m in models:
        y, P, _ = pooled_predictions(records, preds, m, n)
        c = calibration_curve_quantile(y, P[:, 0], 10)
        lw = 2.2 if m == "svdnn" else 1.2
        axes[0].plot(c["predicted"], c["observed"], marker="o", ms=3, color=COLORS[m], ls=LINESTYLES[m], lw=lw, label=name(m))
        nb = np.mean([net_benefit(y, P[:, j], thr)["model"] for j in range(P.shape[1])], 0)
        axes[1].plot(thr, nb, color=COLORS[m], ls=LINESTYLES[m], lw=lw, label=name(m))
        P_pop = prior_shift(P, pi_sample, pi_pop)
        nbp = np.mean([net_benefit(y, P_pop[:, j], thr, sample_weight=w)["model"] for j in range(P.shape[1])], 0)
        axes[2].plot(thr, nbp, color=COLORS[m], ls=LINESTYLES[m], lw=lw)
    y0 = pooled_predictions(records, preds, models[0], n)[0]
    ref = net_benefit(y0, np.zeros(n), thr)
    refp = net_benefit(y0, np.zeros(n), thr, sample_weight=w)
    for ax, r in ((axes[1], ref), (axes[2], refp)):
        ax.plot(thr, r["treat_all"], color=MUTED, lw=1, ls="--", label="Treat all")
        ax.axhline(0, color=INK, lw=0.8, label="Treat none")
        ax.set_ylim(-0.05, max(0.05, max(r["treat_all"]) * 1.4 + 0.02))
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    axes[0].set(xlabel="Predicted probability", ylabel="Observed proportion", title="Calibration (repeat 1, deciles)")
    axes[1].set(xlabel="Threshold probability", ylabel="Net benefit", title=f"Decision curve, sample (prev. {pi_sample:.2f})")
    axes[2].set(xlabel="Threshold probability", ylabel="Net benefit", title=f"Decision curve, population (inc. {pi_pop:.3f})")
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    save_fig(fig, out_fig / "fig_calibration_decision_curves")


def fig_confusion(y, B, path):
    tn = fp = fn = tp = 0
    for j in range(B.shape[1]):
        b = B[:, j]
        tp += ((b == 1) & (y == 1)).sum(); fn += ((b == 0) & (y == 1)).sum()
        tn += ((b == 0) & (y == 0)).sum(); fp += ((b == 1) & (y == 0)).sum()
    M = np.array([[tn, fp], [fn, tp]], dtype=float)
    N = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.imshow(N, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{N[i, j]:.3f}\n(n = {int(M[i, j] / B.shape[1])})", ha="center", va="center",
                    color="white" if N[i, j] > 0.5 else INK, fontsize=8)
    ax.set_xticks([0, 1], ["Predicted no T2DM", "Predicted T2DM"])
    ax.set_yticks([0, 1], ["No T2DM", "Incident T2DM"])
    ax.set_title("SVDNN-MAIG, row-normalised\n(pooled held-out predictions; n per repeat)")
    save_fig(fig, path)


def fig_fold_distributions(fm, models, path):
    mets = ["roc_auc", "f1", "mcc", "brier"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    rng = np.random.default_rng(0)
    for ax, met in zip(axes, mets):
        data = [fm[fm.model == m][met].to_numpy() for m in models]
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], models):
            patch.set(facecolor=COLORS[m] + "40", edgecolor=COLORS[m])
        for med in bp["medians"]:
            med.set(color=INK)
        for i, (d, m) in enumerate(zip(data, models)):
            ax.scatter(d, i + 1 + rng.uniform(-0.15, 0.15, len(d)), s=8, color=COLORS[m], zorder=3, linewidths=0)
        ax.set_yticks(range(1, len(models) + 1), [name(m) if ax is axes[0] else "" for m in models])
        ax.set_title(METRIC_LABELS[met] + (" (lower is better)" if met in LOWER_IS_BETTER else ""))
    fig.suptitle("Distribution over held-out outer folds", fontsize=9)
    save_fig(fig, path)


def fig_ablation(abl, path):
    d = abl.iloc[1:].copy()
    lo = d["95% CI"].str.extract(r"\[(.*), (.*)\]").astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(d) + 1.0))
    yv = np.arange(len(d))
    ax.errorbar(d["ΔAUROC vs full"], yv, xerr=[d["ΔAUROC vs full"] - lo[0], lo[1] - d["ΔAUROC vs full"]],
                fmt="o", color=PALETTE[0], ms=5, capsize=2, lw=1.2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yv, d["Variant"])
    ax.invert_yaxis()
    ax.set_xlabel("Δ AUROC relative to full SVDNN-MAIG (corrected 95% CI)")
    save_fig(fig, path)


def fig_feature_importance(top, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True)
    yv = np.arange(len(top))
    cols = [ROLE_COLORS.get(r, MUTED) for r in top["Causal role (SCM)"]]
    axes[0].barh(yv, top["MAIG (normalised)"], xerr=top["MAIG SD"], color=cols, height=0.7, error_kw={"lw": 0.8})
    axes[1].barh(yv, top["Gradient saliency (CFA)"], color=cols, height=0.7)
    axes[0].set_yticks(yv, top["Feature"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Normalised MAIG (mean ± SD over folds)", title="MAIG ranking (training folds)")
    axes[1].set(xlabel="Cumulative gradient attribution (held-out folds)", title="SVDNN gradient attribution")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ROLE_COLORS.values()]
    axes[1].legend(handles, [k.replace("_", " ") for k in ROLE_COLORS], title="Role in data-generating model",
                   loc="lower right", fontsize=6, title_fontsize=6)
    save_fig(fig, path)


def fig_deletion(ex, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    styles = {"saliency": (PALETTE[0], "-"), "integrated_gradients": (PALETTE[1], "--"),
              "maig": (PALETTE[2], "-."), "random": (MUTED, ":")}
    for meth, (c, ls) in styles.items():
        L = min(len(e["fidelity"][meth]["auc_curve"]) for e in ex)
        curve = np.mean([e["fidelity"][meth]["auc_curve"][:L] for e in ex], 0)
        ax.plot(range(L), curve, color=c, ls=ls, label=meth.replace("_", " "), marker="o", ms=3)
    ax.set(xlabel="Number of top-ranked features masked", ylabel="AUROC on held-out fold",
           title="Deletion test of explanation fidelity")
    ax.legend()
    save_fig(fig, path)


def fig_counterfactual(cf, path):
    keys = ["validity", "lof_inlier", "mahalanobis_inlier", "within_empirical_range", "changes_immutable_feature",
            "fully_plausible"]
    labels = ["Valid (class flipped)", "LOF inlier", "Mahalanobis inlier", "Within empirical range",
              "Changes immutable variable", "Fully plausible"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(len(keys))
    for i, (_, r) in enumerate(cf.iterrows()):
        ax.bar(x + (i - 0.5) * 0.38, [r[k] for k in keys], width=0.36, color=PALETTE[i], label=r["Counterfactual type"])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set(ylabel="Proportion of counterfactuals", ylim=(0, 1.05), title="Counterfactual plausibility")
    ax.legend()
    save_fig(fig, path)


def fig_patient_map(pm, path):
    df = pd.DataFrame(pm)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    yv = np.arange(len(df))
    cols = [PALETTE[7] if s > 0 else PALETTE[0] for s in df["signed_contribution"]]
    ax.barh(yv, df["signed_contribution"], color=cols, height=0.65)
    labels = [f"{f} = {v:.2f}{'  (outside ref.)' if o else ''}" for f, v, o in
              zip(df["feature"], df["value"], df["outside_reference"])]
    ax.set_yticks(yv, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlabel="Gradient x input on the logit scale (contribution relative to training mean)",
           title="Patient-level explanation map (red: raises predicted risk)")
    save_fig(fig, path)


def fig_correlation(root, feat, path):
    df = pd.read_csv(root / "data" / "cohort.csv")
    top = [f for f in feat["Feature"] if f in df.columns and df[f].dtype != object][:30]
    C = df[top].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(top)), top, rotation=90, fontsize=6)
    ax.set_yticks(range(len(top)), top, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman correlation")
    ax.set_title("Pairwise correlation of the 30 highest-MAIG variables (released cohort)")
    save_fig(fig, path)


# =====================================================================================
# Other stages
# =====================================================================================
def report_data(cfg, out_tab, summary):
    root = Path(cfg["output_dir"]) / "data"
    if not (root / "data_dictionary.csv").exists():
        return
    d = pd.read_csv(root / "data_dictionary.csv")
    save_table(d, out_tab / "table2_data_dictionary")
    summary["dataset"] = load_json(root / "generator_summary.json")


def report_leakage(cfg, out_tab, out_fig, summary):
    p = Path(cfg["output_dir"]) / "leakage" / "leakage_audit.json"
    if not p.exists():
        return
    a = load_json(p)
    rows = [
        ("Best single released variable", a["max_single_feature_auc"]),
        ("Observed variables, logistic regression (5-fold)", a["observed_features_lr_auc"]),
        ("Observed variables, gradient boosting (5-fold)", a["observed_features_hgb_auc"]),
        ("Latent T2 state oracle (ceiling, not observable)", a["oracle_latent_t2_auc"]),
        ("Latent T3 state (defines label; after prediction time)", a["oracle_latent_t3_auc"]),
        ("Positive control: circular label from T2 predictors", a["circular_label_positive_control"]["hgb_auc"]),
        ("Permuted labels, full nested pipeline (LR)", a["permuted_label_nested_auc"]["logistic_regression"]["mean"]),
        ("Permuted labels, full nested pipeline (SVDNN)", a["permuted_label_nested_auc"]["svdnn"]["mean"]),
        ("Noise features: selection before CV", a["selection_bias_demo"]["selection_before_cv_auc"]),
        ("Noise features: selection nested in CV", a["selection_bias_demo"]["selection_nested_in_cv_auc"]),
    ]
    df = pd.DataFrame(rows, columns=["Audit", "AUROC"])
    save_table(df, out_tab / "table_leakage_audit")
    summary["leakage_audit"] = {r[0]: r[1] for r in rows}
    summary["leakage_audit"]["prediabetes_rule_t2"] = a["prediabetes_rule_t2"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    yv = np.arange(len(df))
    ax.barh(yv, df["AUROC"], color=[PALETTE[0]] * 3 + [MUTED] * 2 + [PALETTE[7]] + [PALETTE[2]] * 4, height=0.65)
    ax.axvline(0.5, color=INK, lw=0.8, ls=":")
    ax.set_yticks(yv, df["Audit"])
    ax.invert_yaxis()
    ax.set(xlim=(0.4, 1.0), xlabel="AUROC", title="Label-circularity and leakage audit")
    for i, v in enumerate(df["AUROC"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7, color=INK)
    save_fig(fig, out_fig / "fig_leakage_audit")


def report_robustness(cfg, out_tab, out_fig, summary):
    p = Path(cfg["output_dir"]) / "robustness" / "maig_robustness_raw.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    g = df.groupby(["factor", "level", "method"])
    s = g[["relevance_auc", "precision_at_k", "spearman_truth", "stability"]].mean().reset_index()
    ci = g["relevance_auc"].agg(lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))).reset_index(name="relevance_auc_ci95")
    s = s.merge(ci, on=["factor", "level", "method"])
    save_table(s, out_tab / "table_maig_robustness")
    # paired test MAIG vs each method on relevance AUC across all conditions/replicates
    piv = df.pivot_table(index=["factor", "level", "replicate"], columns="method", values="relevance_auc")
    rows = []
    for m in piv.columns:
        if m == "maig":
            continue
        d = piv["maig"] - piv[m]
        rows.append({"Comparator": m, "Mean Δ relevance AUROC (MAIG - comparator)": d.mean(),
                     "Wilcoxon p": wilcoxon_p(piv["maig"], piv[m]), "MAIG better (share of datasets)": (d > 0).mean()})
    save_table(pd.DataFrame(rows), out_tab / "table_maig_robustness_tests", floatfmt=".4f")
    methods = ["maig", "maig_unbalanced", "information_gain", "best_split_ig", "mutual_information", "relieff", "mrmr", "anova_f"]
    colors = dict(zip(methods, PALETTE))
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    titles = {"prevalence": "Prevalence (class imbalance)", "label_noise": "Label-flip noise",
              "correlation": "Inter-predictor correlation", "feature_noise": "Predictor measurement noise"}
    for col, factor in enumerate(["prevalence", "label_noise", "correlation", "feature_noise"]):
        for row, met in enumerate(["relevance_auc", "stability"]):
            ax = axes[row, col]
            for m in methods:
                d = s[(s.factor == factor) & (s.method == m)].sort_values("level")
                ax.plot(d["level"], d[met], marker="o", ms=3, color=colors[m], lw=2.2 if m == "maig" else 1.1,
                        ls="-" if m.startswith("maig") else "--", label=m.replace("_", " "))
            ax.set(xlabel=titles[factor], ylabel="Relevance AUROC" if met == "relevance_auc" else "Top-k stability (Jaccard)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Controlled ranking-robustness study (ground truth known)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_fig / "fig_maig_robustness")


def report_external(cfg, out_tab, out_fig, summary):
    root = Path(cfg["output_dir"])
    rows = []
    for ds in ["external_pima", "external_brfss"]:
        if not (root / ds).exists():
            continue
        recs, _ = load_results(root / ds)
        if not recs:
            continue
        fm = fold_metric_table(recs)
        ab = fold_metric_table(recs, "ablations")
        both = pd.concat([fm, ab])
        n_tr, n_te = fm["n_train"].mean(), fm["n_test"].mean()
        prop = fm[fm.model == "svdnn"].sort_values(["repeat", "fold"])["roc_auc"].to_numpy()
        for m in both["model"].unique():
            d = both[both.model == m].sort_values(["repeat", "fold"])
            p = np.nan if m == "svdnn" else corrected_resampled_ttest(prop, d["roc_auc"].to_numpy(), n_tr, n_te)["p"]
            rows.append({"Dataset": ds.replace("external_", "").upper(), "Model": name(m),
                         "AUROC": f"{d['roc_auc'].mean():.3f} ± {d['roc_auc'].std(ddof=1):.3f}",
                         "F1": f"{d['f1'].mean():.3f} ± {d['f1'].std(ddof=1):.3f}",
                         "MCC": f"{d['mcc'].mean():.3f}", "Brier": f"{d['brier'].mean():.3f}",
                         "Recall": f"{d['recall'].mean():.3f}", "p vs SVDNN-MAIG (AUROC)": p,
                         "auroc_mean": d["roc_auc"].mean()})
    if not rows:
        return
    df = pd.DataFrame(rows)
    save_table(df.drop(columns="auroc_mean"), out_tab / "table_external_validation", floatfmt=".4f")
    summary["external"] = {f"{r['Dataset']}|{r['Model']}": r["auroc_mean"] for r in rows}
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(df) + 1))
    yv = np.arange(len(df))
    ax.barh(yv, df["auroc_mean"], color=[PALETTE[0] if "proposed" in m else MUTED for m in df["Model"]], height=0.6)
    ax.set_yticks(yv, [f"{d}: {m}" for d, m in zip(df["Dataset"], df["Model"])])
    ax.invert_yaxis()
    ax.set(xlim=(0.5, 1.0), xlabel="Mean AUROC over held-out folds", title="Validation on real patient data")
    save_fig(fig, out_fig / "fig_external_validation")


def build_report(cfg: dict) -> dict:
    root = Path(cfg["output_dir"])
    out = ensure_dir(root / "report")
    out_tab, out_fig = ensure_dir(out / "tables"), ensure_dir(out / "figures")
    summary: dict = {"mode": cfg["mode"]}
    report_data(cfg, out_tab, summary)
    report_main(cfg, out_tab, out_fig, summary)
    report_leakage(cfg, out_tab, out_fig, summary)
    report_robustness(cfg, out_tab, out_fig, summary)
    report_external(cfg, out_tab, out_fig, summary)
    save_json(summary, out / "results_summary.json")
    get_logger().info("Report written to %s", out)
    return summary
