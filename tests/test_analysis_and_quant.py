"""Tests for pan_lab.analysis and pan_lab.models.quantize."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pan_lab import PhaseAccumulatorNetwork
from pan_lab.analysis import (
    ablation_test,
    compute_frequency_errors,
    detect_mode_collapse,
    fourier_concentration,
    slot_activation_census,
)
from pan_lab.config   import SIFP16_QUANT_ERR
from pan_lab.data     import make_modular_dataset
from pan_lab.models.quantize import (
    apply_sifp16_to_pan,
    quantize_phase_sifp16,
)


def test_fourier_concentration_spikes_on_pure_sinusoid():
    """A single sinusoid concentrates all FFT energy in one bin -> conc ~= 1."""
    n = 64
    W = torch.cos(2 * np.pi * 3 * torch.arange(n).float() / n).unsqueeze(-1)
    conc = fourier_concentration(W, top_k=2)   # DC + the spike
    assert conc > 0.99


def test_fourier_concentration_low_on_noise():
    torch.manual_seed(0)
    W = torch.randn(256, 8)
    conc = fourier_concentration(W, top_k=2)
    # top-2 bins of a white-noise signal should be a small fraction.
    assert conc < 0.2


def test_compute_frequency_errors_has_convergence_mask():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    info = compute_frequency_errors(m)
    # fresh model with Fourier init: errors are exactly 0
    assert info["error_0"].max() < 1e-6
    assert info["converged_0"].all()
    assert info["sifp16_quant_err"] == SIFP16_QUANT_ERR


def test_mode_collapse_detects_collapse():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    # Force every output channel to favor input slot 0
    with torch.no_grad():
        m.phase_mix.weight.data.zero_()
        m.phase_mix.weight.data[:, 0] = 1.0
    assert detect_mode_collapse(m) is True


def test_mode_collapse_not_triggered_on_diverse_mix():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    # Distinct dominant slot per output channel
    with torch.no_grad():
        m.phase_mix.weight.data.zero_()
        m.phase_mix.weight.data[0, 0] = 1.0
        m.phase_mix.weight.data[1, 2] = 1.0
        m.phase_mix.weight.data[2, 4] = 1.0
    assert detect_mode_collapse(m) is False


def test_ablation_test_returns_all_intervention_accs():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    _, _, vx, vy = make_modular_dataset(p=11, seed=0, device="cpu")
    out = ablation_test(m, vx, vy)
    for k in ("baseline", "zero_phase_mixing",
              "randomize_frequencies", "zero_ref_phases"):
        assert k in out
        assert 0.0 <= out[k] <= 1.0


def test_ablation_test_does_not_mutate_model():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    _, _, vx, vy = make_modular_dataset(p=11, seed=0, device="cpu")
    before = {n: p.data.clone() for n, p in m.named_parameters()}
    ablation_test(m, vx, vy)
    for n, p in m.named_parameters():
        assert torch.allclose(before[n], p.data), f"{n} was mutated"


def test_slot_census_dataframe_schema():
    models = [
        PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2),
        PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2),
    ]
    df = slot_activation_census(models)
    expected = {"model_idx", "encoder", "k", "theoretical",
                "learned", "learned_raw", "error", "converged"}
    assert expected.issubset(df.columns)
    # 2 models * 2 encoders * 3 k = 12 rows
    assert len(df) == 12


# ── Quantization ─────────────────────────────────────────────────────────
def test_sifp16_quantizer_error_is_within_one_lsb():
    x = torch.rand(1000) * (2 * np.pi)
    q = quantize_phase_sifp16(x)
    # error per element <= one LSB (half an LSB on average)
    assert (q - x).abs().max().item() <= SIFP16_QUANT_ERR + 1e-6


def test_sifp16_quantizer_gradient_is_identity():
    # Build the leaf tensor directly — `rand(...) * 2pi` would make x
    # non-leaf, and .grad never gets populated on non-leaves.
    x = (torch.rand(16) * (2 * np.pi)).detach().requires_grad_(True)
    q = quantize_phase_sifp16(x)
    q.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_sifp16_applied_to_pan_changes_forward():
    torch.manual_seed(0)
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (16, 2))
    fp = m(x)
    apply_sifp16_to_pan(m)
    q  = m(x)
    # Quantization must not wreck logits at this tiny scale, but should
    # produce a measurably different output than the fp path.
    assert (fp - q).abs().max().item() > 0
    assert (fp - q).abs().max().item() < 1.0


def test_sifp16_applied_is_idempotent():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    apply_sifp16_to_pan(m)
    apply_sifp16_to_pan(m)     # second call should no-op
    assert getattr(m, "_sifp16_applied", False) is True
