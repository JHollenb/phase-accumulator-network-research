"""
pan_lab.models.quantize — SIFP-16 phase quantization for PAN.

The SPF-32 format stores phase as an unsigned 16-bit integer, so the
quantization error on any phase is at most 2pi / 65536 ~ 9.6e-5 rad.
The paper claims PAN frequencies converge to well within this error.

This module provides `quantize_phase_sifp16`, a straight-through
quantizer that rounds phases to 16-bit grid points. Swap it into the
PAN forward pass to simulate SIFP-16 inference accuracy without
actually building the hardware.

Experiment E in the experiment list: at the grokked solution, quantize
phases to 16-bit and measure val accuracy. If accuracy is preserved,
the hardware claim is vindicated.
"""
from __future__ import annotations

import contextlib

import torch

from pan_lab.config import PHASE_SCALE, PHASE_SCALE_F, TWO_PI


class _QuantizePhaseSTE(torch.autograd.Function):
    """Straight-through estimator: round to 16-bit grid on forward, identity on backward."""

    @staticmethod
    def forward(ctx, phase: torch.Tensor) -> torch.Tensor:
        wrapped = torch.remainder(phase, TWO_PI)
        # scale to [0, 2^16), round, scale back
        idx     = torch.round(wrapped * (PHASE_SCALE_F / TWO_PI))
        idx     = idx % PHASE_SCALE
        return idx * (TWO_PI / PHASE_SCALE_F)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_phase_sifp16(phase: torch.Tensor) -> torch.Tensor:
    """
    Round phases to the SIFP-16 grid (16-bit unsigned, 2pi/65536 resolution).
    Gradient is identity (straight-through).
    """
    return _QuantizePhaseSTE.apply(phase)


def apply_sifp16_to_pan(pan_model) -> None:
    """
    Monkey-patch a PAN's PhaseEncoder and PhaseMixingLayer to quantize their
    outputs. Useful for post-hoc inference-time quantization experiments
    (does not affect training). Idempotent — calling twice is a no-op.
    """
    # Tag classes on the module so the quantization state is visible.
    if getattr(pan_model, "_sifp16_applied", False):
        return

    from pan_lab.models.pan import PhaseEncoder, PhaseMixingLayer

    for m in pan_model.modules():
        if isinstance(m, (PhaseEncoder, PhaseMixingLayer)):
            # Wrap forward to quantize its output.
            orig_forward = m.forward

            def wrapped(x, _f=orig_forward):
                return quantize_phase_sifp16(_f(x))

            m.forward = wrapped    # type: ignore[assignment]
    pan_model._sifp16_applied = True


@contextlib.contextmanager
def sifp16_context(pan_model):
    """
    Temporarily route PAN phase outputs through SIFP-16 quantization.
    Unlike `apply_sifp16_to_pan`, this reverses the monkey-patch on exit
    so the live model is unchanged — which is what instrumentation (M7)
    needs.
    """
    from pan_lab.models.pan import PhaseEncoder, PhaseMixingLayer

    patched = []
    try:
        for m in pan_model.modules():
            if isinstance(m, (PhaseEncoder, PhaseMixingLayer)):
                orig = m.forward
                patched.append((m, orig))

                def wrapped(x, _f=orig):
                    return quantize_phase_sifp16(_f(x))

                m.forward = wrapped    # type: ignore[assignment]
        yield pan_model
    finally:
        for m, orig in patched:
            m.forward = orig           # type: ignore[assignment]
