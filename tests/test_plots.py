"""
Smoke tests for the metric-driven plot functions:
    plot_metric_formation_curves
    plot_metric_spectra
    plot_metric_peak_timescales

Each test synthesizes a minimal DataFrame and asserts the PNG is
written and non-trivial (> 1 KB). We do not inspect pixel contents —
that's what hand-review is for.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pan_lab.plots import (
    DEFAULT_FORMATION_METRICS,
    plot_metric_formation_curves,
    plot_metric_peak_timescales,
    plot_metric_spectra,
)


# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_metrics_df(n_runs: int = 2, n_rows: int = 10) -> pd.DataFrame:
    """Tiny stand-in for a real metrics.csv: step × a handful of columns."""
    rows = []
    for r in range(n_runs):
        for i in range(n_rows):
            step = i * 50
            rows.append({
                "run_id":                        f"run-{r}",
                "step":                          step,
                "enc0_snap_mean":                0.5 - 0.05 * i,
                "clock_compliance":              0.1 * i,
                "mix_row_eff_n_mean":            6 - 0.4 * i,
                "active_freq_count":             5 - (i // 3),
                "decoder_fourier_peak_mean":     0.05 * i,
                # Expensive metrics at step 0 and step-4 only
                "gate_linear_acc":               0.2 * i if i in (0, 4, 9) else np.nan,
                "gate_decoder_gap":              0.05 * i if i in (0, 4, 9) else np.nan,
                "sifp16_acc":                    0.1 * i if i in (0, 4, 9) else np.nan,
            })
    return pd.DataFrame(rows)


def _synthetic_runs_df(n_runs: int = 2) -> pd.DataFrame:
    return pd.DataFrame([
        {"run_id": f"run-{r}", "grok_step": 200, "model_kind": "pan"}
        for r in range(n_runs)
    ])


def _synthetic_spectra_df(n_runs: int = 2) -> pd.DataFrame:
    rows = []
    for r in range(n_runs):
        for metric in ("enc0_snap_mean", "clock_compliance",
                        "mix_row_eff_n_mean"):
            for i, f in enumerate([0.0, 0.002, 0.005, 0.01]):
                ts = float("inf") if f == 0 else 1.0 / f
                rows.append({
                    "run_id":               f"run-{r}",
                    "metric":               metric,
                    "freq_cycles_per_step": f,
                    "power":                1.0 / (i + 1) ** 2,
                    "timescale_steps":      ts,
                })
    return pd.DataFrame(rows)


def _synthetic_peaks_df(n_runs: int = 3) -> pd.DataFrame:
    rows = []
    for r in range(n_runs):
        rows.append({
            "run_id": f"run-{r}", "metric": "enc0_snap_mean",
            "peak_freq": 0.01, "peak_power": 1.0, "peak_timescale_steps": 100.0,
        })
        rows.append({
            "run_id": f"run-{r}", "metric": "clock_compliance",
            "peak_freq": 0.002, "peak_power": 0.5, "peak_timescale_steps": 500.0,
        })
        rows.append({
            "run_id": f"run-{r}", "metric": "mix_row_eff_n_mean",
            "peak_freq": 0.005, "peak_power": 0.3, "peak_timescale_steps": 200.0,
        })
    return pd.DataFrame(rows)


def _png_ok(p) -> bool:
    return p.exists() and p.stat().st_size > 1024


# ─────────────────────────────────────────────────────────────────────────────
# plot_metric_formation_curves
# ─────────────────────────────────────────────────────────────────────────────
def test_plot_metric_formation_curves_writes_png(tmp_outdir):
    out = tmp_outdir / "mfc.png"
    plot_metric_formation_curves(
        _synthetic_metrics_df(), _synthetic_runs_df(), str(out),
    )
    assert _png_ok(out)


def test_plot_metric_formation_curves_handles_sparse_expensive(tmp_outdir):
    """Expensive metrics populated at only 3/10 rows should still plot."""
    df = _synthetic_metrics_df()
    # Ensure gate_linear_acc really is mostly NaN
    assert df["gate_linear_acc"].isna().sum() > 0
    out = tmp_outdir / "mfc_sparse.png"
    plot_metric_formation_curves(
        df, _synthetic_runs_df(), str(out),
        metrics=["gate_linear_acc", "gate_decoder_gap"],
    )
    assert _png_ok(out)


def test_plot_metric_formation_curves_skips_missing_columns(tmp_outdir):
    """If a requested metric isn't a column, skip it silently."""
    df = _synthetic_metrics_df()[["run_id", "step", "clock_compliance"]]
    out = tmp_outdir / "mfc_missing.png"
    plot_metric_formation_curves(
        df, _synthetic_runs_df(), str(out),
        metrics=["clock_compliance", "nonexistent_metric"],
    )
    assert _png_ok(out)


def test_plot_metric_formation_curves_empty_noop(tmp_outdir):
    out = tmp_outdir / "mfc_empty.png"
    plot_metric_formation_curves(
        pd.DataFrame(), _synthetic_runs_df(), str(out),
    )
    assert not out.exists()


# ─────────────────────────────────────────────────────────────────────────────
# plot_metric_spectra
# ─────────────────────────────────────────────────────────────────────────────
def test_plot_metric_spectra_writes_png(tmp_outdir):
    out = tmp_outdir / "spec.png"
    plot_metric_spectra(_synthetic_spectra_df(), str(out))
    assert _png_ok(out)


def test_plot_metric_spectra_per_run_mode(tmp_outdir):
    out = tmp_outdir / "spec_per_run.png"
    plot_metric_spectra(_synthetic_spectra_df(), str(out), aggregate="per_run")
    assert _png_ok(out)


def test_plot_metric_spectra_empty_noop(tmp_outdir):
    out = tmp_outdir / "spec_empty.png"
    plot_metric_spectra(pd.DataFrame(), str(out))
    assert not out.exists()


# ─────────────────────────────────────────────────────────────────────────────
# plot_metric_peak_timescales
# ─────────────────────────────────────────────────────────────────────────────
def test_plot_metric_peak_timescales_writes_png(tmp_outdir):
    out = tmp_outdir / "peaks.png"
    plot_metric_peak_timescales(_synthetic_peaks_df(), str(out))
    assert _png_ok(out)


def test_plot_metric_peak_timescales_empty_noop(tmp_outdir):
    out = tmp_outdir / "peaks_empty.png"
    plot_metric_peak_timescales(pd.DataFrame(), str(out))
    assert not out.exists()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end through grid_sweep PLOT_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
def test_end_to_end_plots_via_grid_sweep(tmp_outdir, tiny_cfg):
    """Tiny grid_sweep with plots declared in YAML should write 3 new PNGs."""
    from pan_lab.experiments import run_experiment

    rep = run_experiment(
        "grid_sweep",
        tiny_cfg.with_overrides(n_steps=200, log_every=50, early_stop=False),
        str(tmp_outdir),
        dry_run=False,
        grid={"seed": [0, 1]},
        options={"ablations": False, "metrics_expensive_every": 50,
                  "metrics_gate_decode_max_rows": 200},
        plots=[
            "metric_formation_curves",
            "metric_spectra",
            "metric_peak_timescales",
        ],
    )
    assert (tmp_outdir / "metric_formation_curves.png").exists()
    assert (tmp_outdir / "metric_spectra.png").exists()
    assert (tmp_outdir / "metric_peak_timescales.png").exists()
