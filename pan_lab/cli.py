"""
pan_lab.cli — command-line runner.

Usage:
    python -m pan_lab experiments/k8_sweep.yaml
    python -m pan_lab experiments/k8_sweep.yaml --dry-run
    python -m pan_lab --list

    # Ad-hoc experiment without a YAML file:
    python -m pan_lab --ad-hoc compare --p 113 --k 9 --steps 50000

    # Re-plot from existing CSVs without retraining:
    python -m pan_lab --replot results/tier3
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from pan_lab.config      import RunConfig
from pan_lab.experiments import (
    EXPERIMENT_REGISTRY,
    run_experiment,
    run_from_yaml,
)


def _replot(dir_path: str) -> None:
    """Regenerate all figures from the CSVs in `dir_path`."""
    import pandas as pd
    from pan_lab.plots import (
        plot_ablation_bars,
        plot_freq_trajectories,
        plot_metric_formation_curves,
        plot_metric_peak_timescales,
        plot_metric_spectra,
        plot_parameter_efficiency,
        plot_slot_census,
        plot_sweep_reliability,
        plot_training_curves,
    )
    runs   = pd.read_csv(os.path.join(dir_path, "runs.csv"))
    curves_path = os.path.join(dir_path, "curves.csv")
    curves = pd.read_csv(curves_path) if os.path.exists(curves_path) else pd.DataFrame()
    if not curves.empty:
        plot_training_curves(curves, runs,
            os.path.join(dir_path, "curves.png"))

    for col in ("k_freqs", "diversity_weight", "weight_decay", "p",
                "d_model", "freq_init"):
        if col in runs.columns and runs[col].nunique() > 1:
            plot_sweep_reliability(runs, group_by=col,
                out_path=os.path.join(dir_path, f"reliability_{col}.png"))

    cp_path = os.path.join(dir_path, "checkpoints.csv")
    if os.path.exists(cp_path):
        cp = pd.read_csv(cp_path)
        for rid in cp["run_id"].unique():
            plot_freq_trajectories(cp[cp["run_id"] == rid], runs,
                os.path.join(dir_path, f"freq_trajectories_{rid}.png"))

    sl_path = os.path.join(dir_path, "slots.csv")
    if os.path.exists(sl_path):
        plot_slot_census(pd.read_csv(sl_path),
            os.path.join(dir_path, "slot_census.png"))

    ab_path = os.path.join(dir_path, "ablations.csv")
    if os.path.exists(ab_path):
        plot_ablation_bars(pd.read_csv(ab_path),
            os.path.join(dir_path, "ablations.png"))

    plot_parameter_efficiency(runs,
        os.path.join(dir_path, "param_efficiency.png"))

    m_path = os.path.join(dir_path, "metrics.csv")
    if os.path.exists(m_path):
        metrics_df = pd.read_csv(m_path)
        if not metrics_df.empty:
            plot_metric_formation_curves(metrics_df, runs,
                os.path.join(dir_path, "metric_formation_curves.png"))

    sp_path = os.path.join(dir_path, "metrics_spectra.csv")
    if os.path.exists(sp_path):
        sp = pd.read_csv(sp_path)
        if not sp.empty:
            plot_metric_spectra(sp,
                os.path.join(dir_path, "metric_spectra.png"))

    pk_path = os.path.join(dir_path, "metrics_peaks.csv")
    if os.path.exists(pk_path):
        pk = pd.read_csv(pk_path)
        if not pk.empty:
            plot_metric_peak_timescales(pk,
                os.path.join(dir_path, "metric_peak_timescales.png"))

    print(f"re-plotted: {dir_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pan_lab",
        description="Run PAN experiments from YAML specs.")
    p.add_argument("yaml_path", nargs="?", type=str,
        help="Path to experiment YAML (or omit when using --list/--replot).")
    p.add_argument("--dry-run", action="store_true",
        help="Print the plan and exit without training.")
    p.add_argument("--list", action="store_true",
        help="List registered experiment names and exit.")
    p.add_argument("--replot", type=str, metavar="DIR",
        help="Regenerate plots from CSVs in DIR and exit.")

    # Ad-hoc mode — build a RunConfig from args and dispatch.
    p.add_argument("--ad-hoc", type=str, metavar="NAME",
        help="Name of experiment to run without a YAML file.")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=None,
        help="Run grid sweep with N parallel workers (overrides options.workers).")

    args = p.parse_args(argv)

    if args.list:
        print("Registered experiments:")
        for name in sorted(EXPERIMENT_REGISTRY):
            print(f"  - {name}")
        return 0

    if args.replot:
        _replot(args.replot)
        return 0

    if args.ad_hoc:
        out_dir = args.out_dir or f"results/{args.ad_hoc}"
        base    = RunConfig(p=args.p, k_freqs=args.k,
                            n_steps=args.steps, seed=args.seed)
        run_experiment(args.ad_hoc, base, out_dir,
                       dry_run=args.dry_run)
        return 0

    if not args.yaml_path:
        p.error("Provide a YAML path, --ad-hoc NAME, --list, or --replot DIR")

    run_from_yaml(args.yaml_path,
                   force_dry_run=True if args.dry_run else None,
                   workers_override=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
