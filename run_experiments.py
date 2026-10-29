"""
Single entry point that reproduces every result, table and figure of the manuscript.

    python run_experiments.py --mode quick            # smoke test (minutes)
    python run_experiments.py --mode full             # full study (hours, resumable)
    python run_experiments.py --mode full --stages main,report

Stages
  data        generate the synthetic cohort, data dictionary and checksum
  main        nested repeated CV: 8 models + 8 ablations + 6 feature rankers + explanations
  leakage     label-circularity and information-leakage audit
  robustness  controlled MAIG vs IG / MI / ReliefF / mRMR / ANOVA robustness study
  external    real-patient validation (PIMA; BRFSS 2015 if the CSV is provided)
  report      tables (CSV + Markdown), figures (PNG + PDF) and results_summary.json

Every stage writes to outputs/<mode>/ and skips work that is already complete, so an
interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import time

from utils import environment_info, get_logger, load_config, save_json, set_seed

STAGES = ["data", "main", "leakage", "robustness", "external", "report"]


def stage_data(cfg):
    from data_generator import generate_cohort

    return generate_cohort(cfg)


def stage_main(cfg):
    from data_generator import CATEGORICAL_COLUMNS, load_cohort
    from evaluation import run_nested_cv

    X, y, _ = load_cohort(cfg)
    run_nested_cv(X, y, cfg, "main", cfg["models"], CATEGORICAL_COLUMNS, ablations=cfg["ablations"],
                  selectors=cfg["selector_comparison"], explain=True)


def stage_leakage(cfg):
    from leakage_audit import run_leakage_audit

    run_leakage_audit(cfg)


def stage_robustness(cfg):
    from maig_robustness import run_robustness

    run_robustness(cfg)


def stage_external(cfg):
    from external_validation import run_external

    run_external(cfg)


def stage_report(cfg):
    from make_report import build_report

    build_report(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="quick", choices=["quick", "full"])
    parser.add_argument("--stages", default="all", help="comma-separated subset of: " + ",".join(STAGES))
    args = parser.parse_args()

    cfg = load_config(args.config, args.mode)
    set_seed(cfg["seed"])
    try:
        import torch

        torch.set_num_threads(cfg.get("n_threads") or 1)
    except ImportError:  # pragma: no cover
        pass
    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    log = get_logger()
    save_json({"mode": args.mode, "stages": stages, "config": cfg, "environment": environment_info()},
              f"{cfg['output_dir']}/run_manifest.json")
    for s in stages:
        if s not in STAGES:
            raise ValueError(f"Unknown stage '{s}'")
        t0 = time.time()
        log.info("=== stage: %s (mode=%s) ===", s, args.mode)
        globals()[f"stage_{s}"](cfg)
        log.info("=== stage %s done in %.1f min ===", s, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
