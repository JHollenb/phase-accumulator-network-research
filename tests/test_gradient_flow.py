"""
Gradient-flow tests. These are the most important tests in the suite —
they verify that the mechanisms behind the paper's claims are actually
operating.

Critical test: `test_diversity_reg_grads_reach_encoder_freqs`.

The original pan.py computed phi_a/phi_b under torch.no_grad() and then
fed them into the Gram-matrix diversity penalty. That detached the
encoder frequencies from the diversity gradient, so the penalty could
only ever push phase_mix.weight — never the encoder freqs themselves.

The fix in pan_lab/trainer.py routes DW through model.mix_features(),
which does the encoding with autograd enabled. These tests confirm that
the encoder.freq gradient is non-zero when DW > 0.
"""
from __future__ import annotations

import pytest
import torch

from pan_lab import PhaseAccumulatorNetwork
from pan_lab.config import TWO_PI


def _diversity_penalty(mix_out: torch.Tensor) -> torch.Tensor:
    """Same formula the trainer uses."""
    mix_norm = mix_out - mix_out.mean(0, keepdim=True)
    norms    = mix_norm.norm(dim=0, keepdim=True).clamp(min=1e-6)
    mix_norm = mix_norm / norms
    gram     = mix_norm.T @ mix_norm / mix_out.shape[0]
    eye      = torch.eye(gram.shape[0], device=gram.device)
    return (gram - eye).pow(2).sum() / gram.shape[0]


def test_mix_features_routes_grads_to_encoder_freqs():
    """
    The critical regression test. When DW is applied through
    mix_features(), autograd must populate encoder[i].freq.grad with a
    non-zero tensor — not None, not all-zero.
    """
    torch.manual_seed(0)
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (32, 2))

    mix_out = m.mix_features(x)
    loss    = _diversity_penalty(mix_out)
    loss.backward()

    for i, enc in enumerate(m.encoders):
        assert enc.freq.grad is not None, f"encoder[{i}].freq.grad is None"
        assert enc.freq.grad.abs().sum() > 0, (
            f"encoder[{i}].freq.grad is all zeros — diversity reg is not "
            "reaching the encoder frequencies (the old bug)."
        )
    assert m.phase_mix.weight.grad is not None
    assert m.phase_mix.weight.grad.abs().sum() > 0


def test_no_grad_encoding_drops_encoder_grads():
    """
    Demonstrate the old bug is real: if phi_a/phi_b are computed under
    torch.no_grad() and only the phase_mix stage gets autograd, the
    encoder.freq.grad ends up None. This is what pan.py was doing.
    """
    torch.manual_seed(0)
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (32, 2))

    with torch.no_grad():                      # ← the old bug
        phis = [enc(x[:, i]) for i, enc in enumerate(m.encoders)]
    concat  = torch.cat(phis, dim=-1)
    mix_out = m.phase_mix(concat)              # phase_mix.weight still has grad
    loss    = _diversity_penalty(mix_out)
    loss.backward()

    # Encoder freqs never saw gradient from the diversity term.
    for enc in m.encoders:
        assert enc.freq.grad is None or enc.freq.grad.abs().sum() == 0, (
            "Encoder freqs somehow got gradient from detached-DW path — "
            "this should NOT happen and would invalidate the fix rationale."
        )


def test_ce_loss_grads_all_parameters():
    """Standard CE loss should touch every trainable parameter."""
    torch.manual_seed(0)
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (32, 2))
    y = torch.randint(0, 11, (32,))

    logits = m(x)
    loss   = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no grad"


def test_phase_gate_remainder_preserves_grad():
    """
    The ref_phase wrap via torch.remainder is the fix for the Adam
    overshoot issue. Confirm that gradient still flows through the
    remainder (should be identity almost everywhere).
    """
    torch.manual_seed(0)
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    # push ref_phase well outside [0, 2pi) to exercise the wrap
    with torch.no_grad():
        m.phase_gate.ref_phase.data = torch.tensor([10.0, -7.0, 15.0])
    x = torch.randint(0, 11, (16, 2))
    y = torch.randint(0, 11, (16,))
    logits = m(x)
    loss   = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    g = m.phase_gate.ref_phase.grad
    assert g is not None and g.abs().sum() > 0
