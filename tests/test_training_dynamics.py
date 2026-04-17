"""
Tests for pan_lab.training_dynamics — post-hoc DFT of metric time
series.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pan_lab.training_dynamics import (
    compute_metrics_spectra,
    summarize_metrics_spectra,
    training_dynamics_spectrum,
)


# ─────────────────────────────────────────────────────────────────────────────
# training_dynamics_spectrum
# ─────────────────────────────────────────────────────────────────────────────
def test_spectrum_of_pure_sine_peaks_at_correct_freq():
    # x(t) = sin(2π · (1/500) · t) sampled every `dt` steps for T evals.
    dt = 50
    T  = 400
    period_steps = 500
    t = np.arange(T) * dt
    x = np.sin(2 * math.pi * t / period_steps)

    freqs, power = training_dynamics_spectrum(x, eval_interval_steps=dt)
    # Skip DC, find peak
    peak_idx  = int(np.argmax(power[1:]) + 1)
    peak_freq = freqs[peak_idx]
    assert peak_freq == pytest.approx(1.0 / period_steps, rel=0.05)


def test_spectrum_detrends_linear_drift():
    # Pure linear drift: after quadratic detrend, power should be tiny.
    T  = 200
    dt = 10
    x  = 0.01 * np.arange(T)
    _, power = training_dynamics_spectrum(x, eval_interval_steps=dt)
    # Total power should be essentially zero after detrending a polynomial
    # of degree ≤ 2 with a polynomial detrender of degree 2.
    assert power.sum() < 1e-10


def test_spectrum_too_short_returns_empty():
    freqs, power = training_dynamics_spectrum([1.0, 2.0, 3.0], eval_interval_steps=1)
    assert freqs.size == 0
    assert power.size == 0


def test_spectrum_all_nan_returns_empty():
    freqs, power = training_dynamics_spectrum(
        [float("nan")] * 10, eval_interval_steps=1,
    )
    assert freqs.size == 0
    assert power.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics_spectra
# ─────────────────────────────────────────────────────────────────────────────
def _make_mock_metrics_df(n_runs: int = 2, n_rows: int = 100, dt: int = 50):
    """Sine at period=500 steps + linear drift per run."""
    rows = []
    for r in range(n_runs):
        steps = np.arange(n_rows) * dt
        sine  = np.sin(2 * math.pi * steps / 500.0)
        drift = 0.01 * np.arange(n_rows)
        for i, s in enumerate(steps):
            rows.append({
                "run_id": f"run-{r}",
                "step":   int(s),
                "snap_mean":          float(sine[i] + drift[i]),
                "clock_compliance":   float(sine[i] * 0.5),
                "active_freq_set":    "1,2,3",      # non-numeric — excluded
                "enc0_active_n":      "1,2,3",      # non-numeric — excluded
            })
    return pd.DataFrame(rows)


def test_compute_metrics_spectra_long_format_schema():
    df = _make_mock_metrics_df(n_runs=2, n_rows=100, dt=50)
    spec = compute_metrics_spectra(df)

    assert set(spec.columns) == {
        "run_id", "metric", "freq_cycles_per_step",
        "power", "timescale_steps",
    }
    # Two numeric metrics (snap_mean, clock_compliance) × two runs,
    # each producing rfftfreq(100, d=50) = 51 bins.
    expected_rows = 2 * 2 * 51
    assert len(spec) == expected_rows
    # Non-numeric cols must not appear
    assert "active_freq_set" not in spec["metric"].unique()
    assert "enc0_active_n"   not in spec["metric"].unique()
    assert "step"            not in spec["metric"].unique()


def test_compute_metrics_spectra_adapts_to_eval_interval():
    # Two runs with different log_every → different freq grids
    df_fast = _make_mock_metrics_df(n_runs=1, n_rows=100, dt=10)
    df_slow = _make_mock_metrics_df(n_runs=1, n_rows=100, dt=100)
    df_fast["run_id"] = "fast"
    df_slow["run_id"] = "slow"
    df = pd.concat([df_fast, df_slow], ignore_index=True)

    spec = compute_metrics_spectra(df)
    fast_freqs = spec[spec.run_id == "fast"]["freq_cycles_per_step"].max()
    slow_freqs = spec[spec.run_id == "slow"]["freq_cycles_per_step"].max()
    # Nyquist of fast run is 1/(2*10); of slow run is 1/(2*100).
    assert fast_freqs > slow_freqs


def test_compute_metrics_spectra_too_short_returns_empty():
    df = _make_mock_metrics_df(n_runs=1, n_rows=3)
    spec = compute_metrics_spectra(df)
    assert spec.empty
    # Columns still present so downstream consumers don't crash
    assert list(spec.columns) == [
        "run_id", "metric", "freq_cycles_per_step",
        "power", "timescale_steps",
    ]


def test_compute_metrics_spectra_empty_input():
    spec = compute_metrics_spectra(pd.DataFrame())
    assert spec.empty


# ─────────────────────────────────────────────────────────────────────────────
# summarize_metrics_spectra
# ─────────────────────────────────────────────────────────────────────────────
def test_summarize_metrics_spectra_one_row_per_pair():
    df = _make_mock_metrics_df(n_runs=2, n_rows=100, dt=50)
    spec = compute_metrics_spectra(df)
    peaks = summarize_metrics_spectra(spec)
    assert set(peaks.columns) == {
        "run_id", "metric", "peak_freq",
        "peak_power", "peak_timescale_steps",
    }
    # 2 runs × 2 metrics
    assert len(peaks) == 4


def test_summarize_metrics_spectra_peak_at_expected_frequency():
    df = _make_mock_metrics_df(n_runs=1, n_rows=100, dt=50)
    spec  = compute_metrics_spectra(df)
    peaks = summarize_metrics_spectra(spec)
    sine_peak = peaks[peaks["metric"] == "snap_mean"]["peak_freq"].iloc[0]
    # peak should be near 1/500 cycles/step (sine's period)
    assert sine_peak == pytest.approx(1.0 / 500, rel=0.1)


def test_summarize_empty_returns_empty():
    empty = pd.DataFrame(columns=[
        "run_id", "metric", "freq_cycles_per_step",
        "power", "timescale_steps",
    ])
    out = summarize_metrics_spectra(empty)
    assert out.empty
    assert list(out.columns) == [
        "run_id", "metric", "peak_freq",
        "peak_power", "peak_timescale_steps",
    ]
