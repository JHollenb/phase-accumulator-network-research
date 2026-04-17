"""
pan_lab.training_dynamics — post-hoc Fourier analysis of per-eval metric
time series.

Input: the metrics.csv (wide) produced by MetricsLogger.
Output: a long-format spectrum per (run_id, metric) and a peak-summary
per (run_id, metric).

The hypothesis this module exists to test is that grokking is
oscillatory and multi-timescale — encoder frequencies lock fast,
mixing matrix sparsifies slowly, decoder Clock projection lands in
between — and that those timescales show up as distinct peaks in the
DFT of each metric's `x(t)` curve.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# Columns in metrics.csv that aren't numeric scalars or shouldn't be DFT'd.
_NON_SPECTRAL_COLS = {
    "run_id", "step",
    "enc0_active_n", "enc1_active_n",
    "active_freq_set",
}


# ─────────────────────────────────────────────────────────────────────────────
def training_dynamics_spectrum(
    x_t: Iterable[float],
    eval_interval_steps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DFT of a metric time series, quadratic-detrended and Hann-windowed.

    Returns (freqs in cycles/step, power). Empty arrays when the series
    is too short (< 4 finite samples) or fully NaN — callers should
    handle these gracefully.

    The quadratic detrend removes DC plus linear and quadratic drift,
    so a pure monotone rise (e.g. `snap_mean` steadily decaying) does
    not dominate the spectrum. The Hann window reduces leakage at bin
    edges on a finite-length series.
    """
    x = np.asarray(list(x_t), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.array([]), np.array([])
    t     = np.arange(len(x), dtype=float)
    coef  = np.polyfit(t, x, deg=2)
    xd    = x - np.polyval(coef, t)
    xw    = xd * np.hanning(len(xd))
    fft   = np.fft.rfft(xw)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(x), d=float(eval_interval_steps))
    return freqs, power


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics_spectra(
    metrics_df: pd.DataFrame,
    metric_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    One long-format spectrum row per (run_id, metric, freq).

    Columns: run_id, metric, freq_cycles_per_step, power, timescale_steps.

    `eval_interval_steps` is derived per run as median(diff(step)), so
    this adapts to whatever `log_every` the run used. Runs with fewer
    than 4 finite samples of a metric contribute no rows for it.
    """
    if metrics_df.empty:
        return pd.DataFrame(columns=[
            "run_id", "metric", "freq_cycles_per_step",
            "power", "timescale_steps",
        ])

    if metric_columns is None:
        metric_columns = [
            c for c in metrics_df.columns
            if c not in _NON_SPECTRAL_COLS
            and pd.api.types.is_numeric_dtype(metrics_df[c])
        ]
    else:
        metric_columns = list(metric_columns)

    rows = []
    for run_id, g in metrics_df.groupby("run_id", sort=False):
        g = g.sort_values("step")
        steps = g["step"].to_numpy()
        if len(steps) < 4:
            continue
        diffs = np.diff(steps)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            continue
        eval_interval = float(np.median(diffs))

        for metric in metric_columns:
            freqs, power = training_dynamics_spectrum(
                g[metric].to_numpy(), eval_interval,
            )
            if freqs.size == 0:
                continue
            with np.errstate(divide="ignore"):
                timescales = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-30), np.inf)
            for f, p, ts in zip(freqs, power, timescales):
                rows.append({
                    "run_id":               run_id,
                    "metric":               metric,
                    "freq_cycles_per_step": float(f),
                    "power":                float(p),
                    "timescale_steps":      float(ts),
                })

    return pd.DataFrame(rows, columns=[
        "run_id", "metric", "freq_cycles_per_step",
        "power", "timescale_steps",
    ])


# ─────────────────────────────────────────────────────────────────────────────
def summarize_metrics_spectra(spectra_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (run_id, metric) with the dominant non-DC peak.

    Columns: run_id, metric, peak_freq, peak_power, peak_timescale_steps.
    The DC bin (freq == 0) is excluded so peaks describe oscillations,
    not constant offsets left over after detrending.
    """
    if spectra_df.empty:
        return pd.DataFrame(columns=[
            "run_id", "metric", "peak_freq",
            "peak_power", "peak_timescale_steps",
        ])

    df = spectra_df[spectra_df["freq_cycles_per_step"] > 0]
    if df.empty:
        return pd.DataFrame(columns=[
            "run_id", "metric", "peak_freq",
            "peak_power", "peak_timescale_steps",
        ])

    idx = df.groupby(["run_id", "metric"])["power"].idxmax()
    peaks = df.loc[idx, ["run_id", "metric",
                          "freq_cycles_per_step", "power", "timescale_steps"]]
    return peaks.rename(columns={
        "freq_cycles_per_step": "peak_freq",
        "power":                "peak_power",
        "timescale_steps":      "peak_timescale_steps",
    }).reset_index(drop=True)
