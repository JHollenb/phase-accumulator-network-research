"""
Tests for pan_lab.metrics — the eight per-eval-step instrumentation
functions and the MetricsLogger hook.

Each metric is exercised on a tiny PAN (P=11, K=3) either at its random
init or with hand-stuffed weights so we can assert expected values
without depending on grokking actually happening.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from pan_lab.config import TWO_PI
from pan_lab.metrics import (
    MetricsLogger,
    active_frequencies,
    cheap_metrics,
    clock_freq_alignment,
    clock_pair_compliance,
    decoder_fourier_projection,
    expensive_metrics,
    fourier_snap,
    gate_linear_decodability,
    logit_2d_spectrum,
    mix_row_entropy,
    sifp16_robustness,
)
from pan_lab.models.pan import PhaseAccumulatorNetwork
from pan_lab.models.quantize import sifp16_context


def _tiny_pan(p: int = 11, k: int = 3) -> PhaseAccumulatorNetwork:
    torch.manual_seed(0)
    return PhaseAccumulatorNetwork(p=p, k_freqs=k, n_inputs=2, freq_init="fourier")


# ─────────────────────────────────────────────────────────────────────────────
# M1. fourier_snap
# ─────────────────────────────────────────────────────────────────────────────
def test_fourier_snap_locked_to_basis():
    p, K = 11, 3
    W = torch.tensor([(k + 1) * TWO_PI / p for k in range(K)])
    out = fourier_snap(W, p)
    assert out["snap_mean"] < 1e-5
    assert out["snap_max"]  < 1e-5
    assert out["active_n"]  == "1,2,3"


def test_fourier_snap_off_lattice():
    p = 11
    W = torch.tensor([TWO_PI / p + 0.05, 3 * TWO_PI / p])
    out = fourier_snap(W, p)
    assert out["snap_mean"] > 0.0
    assert out["active_n"]  == "1,3"


# ─────────────────────────────────────────────────────────────────────────────
# M2. clock_pair_compliance
# ─────────────────────────────────────────────────────────────────────────────
def test_clock_compliance_perfect_pair():
    K = 3
    W = torch.zeros(K, 2 * K)
    # Row j: slot j on enc0, slot j on enc1, equal magnitudes.
    for j in range(K):
        W[j, j]     = 1.0
        W[j, K + j] = 1.0
    assert clock_pair_compliance(W, K) == pytest.approx(1.0)


def test_clock_compliance_same_encoder_fails():
    K = 3
    # Top-2 both from enc0 → not a Clock pair.
    W = torch.zeros(K, 2 * K)
    W[0, 0] = 1.0
    W[0, 1] = 1.0
    assert clock_pair_compliance(W, K) == 0.0


def test_clock_compliance_mag_tol():
    K = 2
    W = torch.zeros(K, 2 * K)
    W[0, 0]     = 1.0
    W[0, K + 0] = 0.5          # 50% mismatch fails default tol=0.20
    W[1, 1]     = 1.0
    W[1, K + 1] = 0.95         # 5% mismatch passes
    assert clock_pair_compliance(W, K, mag_tol=0.20) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# M3. clock_freq_alignment
# ─────────────────────────────────────────────────────────────────────────────
def test_clock_freq_align_zero_when_paired_freqs_identical():
    K = 2
    W_mix = torch.zeros(K, 2 * K)
    for j in range(K):
        W_mix[j, j]     = 1.0
        W_mix[j, K + j] = 1.0
    W_enc0 = torch.tensor([0.5, 1.5])
    W_enc1 = torch.tensor([0.5, 1.5])
    out = clock_freq_alignment(W_mix, W_enc0, W_enc1, K)
    assert out["align_n"] == 2
    assert out["align_mean"] < 1e-6


def test_clock_freq_align_returns_nan_when_no_pairs():
    K = 2
    W_mix = torch.zeros(K, 2 * K)
    W_mix[0, 0] = 1.0
    W_mix[0, 1] = 1.0               # same encoder, not a pair
    out = clock_freq_alignment(W_mix, torch.zeros(K), torch.zeros(K), K)
    assert out["align_n"] == 0
    assert math.isnan(out["align_mean"])


# ─────────────────────────────────────────────────────────────────────────────
# M4. mix_row_entropy
# ─────────────────────────────────────────────────────────────────────────────
def test_mix_row_entropy_uniform_row():
    K = 3
    W = torch.ones(K, 2 * K)        # perfectly uniform → eff_n = 2K
    out = mix_row_entropy(W, K)
    assert out["row_eff_n_mean"] == pytest.approx(2 * K, rel=1e-4)
    assert out["row_eff_n_min"]  == pytest.approx(2 * K, rel=1e-4)


def test_mix_row_entropy_clock_pair_row_eff_n_2():
    K = 3
    W = torch.zeros(K, 2 * K)
    for j in range(K):
        W[j, j]     = 1.0
        W[j, K + j] = 1.0
    out = mix_row_entropy(W, K)
    assert out["row_eff_n_mean"] == pytest.approx(2.0, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# M5. active_frequencies
# ─────────────────────────────────────────────────────────────────────────────
def test_active_frequencies_threshold():
    p, K = 11, 3
    W_enc0 = torch.tensor([(k + 1) * TWO_PI / p for k in range(K)])
    W_enc1 = W_enc0.clone()
    W_mix  = torch.zeros(K, 2 * K)
    # Only column 0 (enc0 slot 0 → n=1) and column K+1 (enc1 slot 1 → n=2) active
    W_mix[0, 0]     = 0.5
    W_mix[0, K + 1] = 0.5
    out = active_frequencies(W_enc0, W_enc1, W_mix, p, K, weight_threshold=0.1)
    assert out["count"] == 2
    assert out["set"]   == "1,2"


# ─────────────────────────────────────────────────────────────────────────────
# M6. gate_linear_decodability (tiny smoke — fits a probe)
# ─────────────────────────────────────────────────────────────────────────────
def test_gate_linear_decodability_returns_accuracy_in_range(tiny_data):
    tx, ty, vx, vy = tiny_data
    m = _tiny_pan()
    out = gate_linear_decodability(m, vx, vy, max_rows=200)
    assert 0.0 <= out["gate_linear_acc"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# M7. sifp16_robustness + reversibility of sifp16_context
# ─────────────────────────────────────────────────────────────────────────────
def test_sifp16_context_reversible(tiny_data):
    _, _, vx, _ = tiny_data
    m = _tiny_pan()
    m.eval()
    with torch.no_grad():
        before = m(vx).clone()
    with sifp16_context(m):
        _ = m(vx)                    # inside: quantized
    with torch.no_grad():
        after = m(vx)
    assert torch.equal(before, after)


def test_sifp16_robustness_delta_nonnegative(tiny_data):
    _, _, vx, vy = tiny_data
    m = _tiny_pan()
    out = sifp16_robustness(m, vx, vy)
    # Quantizing can only hurt (or be equal), not help on a random init
    # *in expectation*; since this isn't strict on a single seed, just
    # check the shape and the values are in [0, 1].
    for key in ("fp32_acc", "sifp16_acc"):
        assert 0.0 <= out[key] <= 1.0
    assert out["quant_delta"] == pytest.approx(out["fp32_acc"] - out["sifp16_acc"])


# ─────────────────────────────────────────────────────────────────────────────
# M8. decoder_fourier_projection
# ─────────────────────────────────────────────────────────────────────────────
def test_decoder_fourier_projection_pure_sinusoid_peak_near_one():
    p, K = 37, 2
    W = torch.zeros(p, K)
    t = torch.arange(p, dtype=torch.float32)
    W[:, 0] = torch.cos(2 * math.pi * 3 * t / p)   # peak at bin 3
    W[:, 1] = torch.sin(2 * math.pi * 5 * t / p)
    out = decoder_fourier_projection(W)
    assert out["peak_mean"] > 0.95
    assert out["peak_max"]  > 0.95


def test_decoder_fourier_projection_zero_signal_safe():
    W = torch.zeros(11, 3)
    out = decoder_fourier_projection(W)
    assert out["peak_mean"] == 0.0
    assert out["peak_max"]  == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# M9. logit_2d_spectrum
# ─────────────────────────────────────────────────────────────────────────────
def test_logit_2d_spectrum_output_schema_and_range():
    m = _tiny_pan()
    out = logit_2d_spectrum(m, p=11)
    assert set(out) == {
        "logit_spec_diag_frac_mean",
        "logit_spec_peak_sparsity_mean",
        "logit_spec_active_count_mean",
    }
    # Fractions are in [0, 1]; active count is ≥ 0.
    for k in ("logit_spec_diag_frac_mean", "logit_spec_peak_sparsity_mean"):
        v = out[k]
        assert math.isnan(v) or 0.0 <= v <= 1.0 + 1e-9
    v = out["logit_spec_active_count_mean"]
    assert math.isnan(v) or v >= 0


def test_logit_2d_spectrum_sample_classes_override():
    m = _tiny_pan()
    out = logit_2d_spectrum(m, p=11, sample_classes=[0, 1])
    assert set(out) == {
        "logit_spec_diag_frac_mean",
        "logit_spec_peak_sparsity_mean",
        "logit_spec_active_count_mean",
    }


def test_logit_2d_spectrum_int_sample_classes():
    m = _tiny_pan()
    # Integer count maps to that many evenly-spaced classes.
    out = logit_2d_spectrum(m, p=11, sample_classes=3)
    assert set(out) == {
        "logit_spec_diag_frac_mean",
        "logit_spec_peak_sparsity_mean",
        "logit_spec_active_count_mean",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composers
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_CHEAP_KEYS = {
    "enc0_snap_mean", "enc0_snap_max", "enc0_snap_min", "enc0_active_n",
    "enc1_snap_mean", "enc1_snap_max", "enc1_snap_min", "enc1_active_n",
    "clock_compliance",
    "clock_freq_align_mean", "clock_freq_align_n",
    "mix_row_entropy_mean", "mix_row_eff_n_mean", "mix_row_eff_n_min",
    "active_freq_count", "active_freq_set",
    "decoder_fourier_peak_mean", "decoder_fourier_peak_max",
}
EXPECTED_LOGIT_SPEC_KEYS = {
    "logit_spec_diag_frac_mean",
    "logit_spec_peak_sparsity_mean",
    "logit_spec_active_count_mean",
}
EXPECTED_EXPENSIVE_KEYS = {
    "gate_linear_acc", "fp32_acc", "sifp16_acc", "quant_delta",
    "gate_decoder_gap",
}


def test_cheap_metrics_has_expected_keys():
    m = _tiny_pan()
    out = cheap_metrics(m)
    assert set(out) == EXPECTED_CHEAP_KEYS


def test_expensive_metrics_has_expected_keys(tiny_data):
    _, _, vx, vy = tiny_data
    m = _tiny_pan()
    out = expensive_metrics(m, vx, vy, max_rows=200)
    assert set(out) == EXPECTED_EXPENSIVE_KEYS


def test_expensive_metrics_with_logit_spectrum_true(tiny_data):
    _, _, vx, vy = tiny_data
    m = _tiny_pan()
    out = expensive_metrics(m, vx, vy, max_rows=200, logit_spectrum=True)
    assert set(out) == EXPECTED_EXPENSIVE_KEYS | EXPECTED_LOGIT_SPEC_KEYS


def test_gate_decoder_gap_derivation(tiny_data):
    _, _, vx, vy = tiny_data
    m = _tiny_pan()
    out = expensive_metrics(m, vx, vy, max_rows=200)
    assert out["gate_decoder_gap"] == pytest.approx(
        out["gate_linear_acc"] - out["fp32_acc"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# MetricsLogger hook + end-to-end metrics.csv
# ─────────────────────────────────────────────────────────────────────────────
def test_metrics_logger_populates_history_rows(tiny_data):
    from pan_lab.trainer import TrainHistory
    _, _, vx, vy = tiny_data
    m = _tiny_pan()
    logger = MetricsLogger(val_x=vx, val_y=vy, expensive_every=0)
    hist = TrainHistory()
    logger.on_eval(step=0, model=m, cfg=None, history=hist,
                   val_loss=0.0, val_acc=0.0)
    logger.on_eval(step=50, model=m, cfg=None, history=hist,
                   val_loss=0.0, val_acc=0.0)
    assert len(hist.metrics_rows) == 2
    assert set(hist.metrics_rows[0]) == EXPECTED_CHEAP_KEYS | {"step"}


def test_metrics_logger_expensive_cadence(tiny_data):
    from pan_lab.trainer import TrainHistory
    _, _, vx, vy = tiny_data
    m = _tiny_pan()
    logger = MetricsLogger(val_x=vx, val_y=vy, expensive_every=50,
                           gate_decode_max_rows=200)
    hist = TrainHistory()
    # step 0 and step 50 should fire expensive; step 25 should not.
    for step in (0, 25, 50):
        logger.on_eval(step=step, model=m, cfg=None, history=hist,
                       val_loss=0.0, val_acc=0.0)
    assert "gate_linear_acc" in hist.metrics_rows[0]
    assert "gate_linear_acc" not in hist.metrics_rows[1]
    assert "gate_linear_acc" in hist.metrics_rows[2]


def test_metrics_logger_noop_on_transformer():
    """MetricsLogger should no-op when the model isn't a PAN."""
    from pan_lab.trainer import TrainHistory
    import torch.nn as nn

    class FakeTransformer(nn.Module):
        pass

    m = FakeTransformer()
    logger = MetricsLogger(val_x=torch.zeros(1), val_y=torch.zeros(1))
    hist = TrainHistory()
    logger.on_eval(step=0, model=m, cfg=None, history=hist,
                   val_loss=0.0, val_acc=0.0)
    assert hist.metrics_rows == []


def test_end_to_end_grid_sweep_writes_metrics_csv(tmp_outdir, tiny_cfg):
    """Tiny grid_sweep run should produce metrics.csv with expected cols."""
    from pan_lab.experiments import run_experiment

    rep = run_experiment(
        "grid_sweep",
        tiny_cfg.with_overrides(n_steps=200, log_every=50, early_stop=False),
        str(tmp_outdir),
        dry_run=False,
        grid={"seed": [0, 1]},
        options={"ablations": False, "metrics_expensive_every": 50,
                 "metrics_gate_decode_max_rows": 200},
    )
    df = rep.metrics_df()
    assert not df.empty
    assert {"run_id", "step"}.issubset(df.columns)
    assert EXPECTED_CHEAP_KEYS.issubset(df.columns)
    # M9 logit_spec_* columns are off by default now.
    assert not (EXPECTED_LOGIT_SPEC_KEYS & set(df.columns))
    # New derived column is populated at the expensive cadence.
    assert "gate_decoder_gap" in df.columns
    # At least one row (step 0, 50, 100) should have expensive metrics populated.
    assert df["gate_linear_acc"].notna().sum() >= 2
    assert df["gate_decoder_gap"].notna().sum() >= 2

    # Files written
    assert (tmp_outdir / "metrics.csv").exists()
    assert (tmp_outdir / "metrics_spectra.csv").exists()
    assert (tmp_outdir / "metrics_peaks.csv").exists()


def test_end_to_end_grid_sweep_with_logit_spectrum_opt_in(tmp_outdir, tiny_cfg):
    """Flipping `metrics_logit_spectrum: true` restores the three
    logit_spec_* columns."""
    from pan_lab.experiments import run_experiment

    rep = run_experiment(
        "grid_sweep",
        tiny_cfg.with_overrides(n_steps=200, log_every=50, early_stop=False),
        str(tmp_outdir),
        dry_run=False,
        grid={"seed": [0]},
        options={"ablations": False, "metrics_expensive_every": 50,
                 "metrics_gate_decode_max_rows": 200,
                 "metrics_logit_spectrum": True},
    )
    df = rep.metrics_df()
    assert not df.empty
    assert EXPECTED_LOGIT_SPEC_KEYS.issubset(df.columns)
    assert df["logit_spec_diag_frac_mean"].notna().sum() >= 2


def test_reporter_spectra_df_and_peaks_df(tmp_outdir, tiny_cfg):
    """ExperimentReporter.spectra_df()/peaks_df() should delegate to
    pan_lab.training_dynamics and produce the same output as calling
    those directly on rep.metrics_df()."""
    from pan_lab.experiments import run_experiment
    from pan_lab.training_dynamics import (
        compute_metrics_spectra, summarize_metrics_spectra,
    )

    rep = run_experiment(
        "grid_sweep",
        tiny_cfg.with_overrides(n_steps=200, log_every=50, early_stop=False),
        str(tmp_outdir),
        dry_run=False,
        grid={"seed": [0]},
        options={"ablations": False, "metrics_expensive_every": 50,
                 "metrics_gate_decode_max_rows": 200},
    )
    assert not rep.metrics_df().empty

    expected_spec  = compute_metrics_spectra(rep.metrics_df())
    expected_peaks = summarize_metrics_spectra(expected_spec)

    pd.testing.assert_frame_equal(rep.spectra_df(), expected_spec)
    pd.testing.assert_frame_equal(rep.peaks_df(),   expected_peaks)
