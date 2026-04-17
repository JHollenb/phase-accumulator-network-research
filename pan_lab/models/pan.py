"""
pan_lab.models.pan — Phase Accumulator Network.

The core primitive is phase accumulation: numbers are represented as points
on S^1 (unit circle), and modular addition of inputs becomes addition of
phases. The forward pass is:

    encode:  a    -> phi_k(a) = (a * f_k) mod 2pi                  for k = 1..K
    mix:     phi  -> psi_j    = (Sum_i W[j,i] phi_i) mod 2pi       (over K outputs)
    gate:    psi  -> g_j      = (1 + cos(psi_j - phi_ref_j)) / 2
    decode:  g    -> logits   = W_dec g + b_dec

Differences from the original pan.py in pan_paper.md:

1. Fixed diversity regularization bug.
   The old code computed phi_a/phi_b inside torch.no_grad() and then fed
   them into the Gram penalty. That silently dropped gradients into the
   encoder frequencies from the diversity term. The fix is to compute the
   encoded phases under autograd so the full mixing matrix gets its
   correct regularization signal. The new per-batch API is `mix_features`
   which returns exactly the tensor the diversity penalty needs.

2. Generalized to N inputs.
   mod_add and mod_mul have N=2. The two-step task (a+b)*c mod P needs N=3.
   A single `n_inputs` arg replaces the hard-coded encoder_a/encoder_b.

3. freq_init = "fourier" | "random".
   The old code initialized to the theoretical Fourier basis k*2pi/P.
   Experiment H in the roadmap critique is "does random init still grok?"
   — a proper ablation of whether the good init is doing the work.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pan_lab.config import TWO_PI, PHASE_SCALE, SIFP16_QUANT_ERR


# ─────────────────────────────────────────────────────────────────────────────
class PhaseEncoder(nn.Module):
    """
    Encodes an integer token a in [0, P) as K phases:
        phi_k(a) = a * f_k mod 2pi

    The f_k are learned. Initialization is either the natural Fourier basis
    (f_k = k * 2pi / P for k = 1..K) or uniform random in [0, 2pi).
    """

    def __init__(
        self,
        p:         int,
        k_freqs:   int,
        freq_init: Literal["fourier", "random"] = "fourier",
    ):
        super().__init__()
        self.p       = p
        self.k_freqs = k_freqs
        self.freq_init = freq_init

        if freq_init == "fourier":
            init = torch.tensor(
                [(k + 1) * TWO_PI / p for k in range(k_freqs)],
                dtype=torch.float32,
            )
        elif freq_init == "random":
            init = torch.rand(k_freqs) * TWO_PI
        else:
            raise ValueError(f"Unknown freq_init: {freq_init!r}")
        self.freq = nn.Parameter(init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (batch,) long
        returns: (batch, K) float in [0, 2pi)
        """
        a = tokens.float().unsqueeze(-1)     # (B, 1)
        f = self.freq.unsqueeze(0)           # (1, K)
        return (a * f) % TWO_PI


# ─────────────────────────────────────────────────────────────────────────────
class PhaseMixingLayer(nn.Module):
    """
    Linear phase mixing: for each output j,
        psi_j = (Sum_i W[j,i] * phi_i) mod 2pi

    In hardware this is a 16-bit integer multiply-accumulate modulo 2^16.
    In software we work in float [0, 2pi) for differentiability; the mod
    is applied at the output.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        w = torch.randn(n_out, n_in) * 0.1 + (1.0 / n_in)
        self.weight = nn.Parameter(w)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """phases: (B, n_in) in [0, 2pi). returns (B, n_out) in [0, 2pi)."""
        return F.linear(phases, self.weight) % TWO_PI


# ─────────────────────────────────────────────────────────────────────────────
class PhaseGate(nn.Module):
    """
    Cosine gate: fires maximally when input phase matches a learned ref.

        g_j = (1 + cos(phi_j - phi_ref_j)) / 2        in [0, 1]

    ref_phase wrapping — ref_phase lives on S^1 but Adam treats it as an
    unconstrained real parameter. Without wrapping, momentum pushes the
    stored value outside [0, 2pi), causing gradient spikes at the cosine
    inflection points. The stored parameter is left free; the *effective*
    phase used in forward is always wrapped. torch.remainder has gradient 1
    almost everywhere, so autograd is unaffected.
    """

    def __init__(self, n_phases: int):
        super().__init__()
        self.ref_phase = nn.Parameter(torch.rand(n_phases) * TWO_PI)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        ref  = torch.remainder(self.ref_phase, TWO_PI)
        diff = phases - ref.unsqueeze(0)
        return (1.0 + torch.cos(diff)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
class PhaseAccumulatorNetwork(nn.Module):
    """
    Full PAN. Architecture:

        inputs (B, N) long, each in [0, P)
            | N independent PhaseEncoders, each -> (B, K)
            | concat             -> (B, N*K)
            | PhaseMixingLayer   -> (B, K)
            | PhaseGate          -> (B, K)
            | Linear(K -> P)     -> logits

    N is `n_inputs` — 2 for mod_add/mod_mul, 3 for the two-step task.
    """

    def __init__(
        self,
        p:         int,
        k_freqs:   int = 9,
        n_inputs:  int = 2,
        freq_init: Literal["fourier", "random"] = "fourier",
    ):
        super().__init__()
        self.p         = p
        self.k_freqs   = k_freqs
        self.n_inputs  = n_inputs
        self.freq_init = freq_init

        # One encoder per input position — allows per-input frequency learning.
        self.encoders = nn.ModuleList([
            PhaseEncoder(p, k_freqs, freq_init=freq_init)
            for _ in range(n_inputs)
        ])
        self.phase_mix  = PhaseMixingLayer(n_inputs * k_freqs, k_freqs)
        self.phase_gate = PhaseGate(k_freqs)
        self.decoder    = nn.Linear(k_freqs, p, bias=True)

        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    # ── Convenience accessors for legacy call sites ───────────────────────
    @property
    def encoder_a(self) -> PhaseEncoder:
        return self.encoders[0]

    @property
    def encoder_b(self) -> PhaseEncoder:
        if self.n_inputs < 2:
            raise AttributeError("encoder_b undefined when n_inputs < 2")
        return self.encoders[1]

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs : (B, N) long
        returns: (B, P) logits
        """
        phases = [enc(inputs[:, i]) for i, enc in enumerate(self.encoders)]
        concat = torch.cat(phases, dim=-1)
        mixed  = self.phase_mix(concat)
        gates  = self.phase_gate(mixed)
        return self.decoder(gates)

    def mix_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return the K-dim mixed-phase output (B, K) — the tensor the diversity
        regularizer penalizes. Autograd connects back to the encoder
        frequencies and mixing weights, so the penalty regularizes both.

        Equivalent to the first three stages of forward() without the gate
        and decoder.
        """
        phases = [enc(inputs[:, i]) for i, enc in enumerate(self.encoders)]
        concat = torch.cat(phases, dim=-1)
        return self.phase_mix(concat)

    def get_gates(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return the post-gate activations (B, K). Useful for mechanistic
        probes (e.g. gate linear-decodability) that need the tensor just
        before the decoder.
        """
        phases = [enc(inputs[:, i]) for i, enc in enumerate(self.encoders)]
        concat = torch.cat(phases, dim=-1)
        mixed  = self.phase_mix(concat)
        return self.phase_gate(mixed)

    # ── Introspection ──────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learned_frequencies(self) -> dict:
        """
        Per-encoder learned frequencies, wrapped into [0, 2pi), plus
        angular errors to the theoretical Fourier basis k*2pi/P.
        """
        theoretical = np.array(
            [(k + 1) * TWO_PI / self.p for k in range(self.k_freqs)]
        )

        def _angle_err(learned: np.ndarray, theory: np.ndarray) -> np.ndarray:
            diff = np.abs(learned - theory) % TWO_PI
            return np.minimum(diff, TWO_PI - diff)

        out = {"theoretical": theoretical, "sifp16_quant_err": SIFP16_QUANT_ERR}
        for i, enc in enumerate(self.encoders):
            raw     = enc.freq.detach().cpu().numpy()
            wrapped = raw % TWO_PI
            key_raw     = f"learned_{i}_raw"
            key_wrapped = f"learned_{i}"
            key_err     = f"error_{i}"
            out[key_raw]     = raw
            out[key_wrapped] = wrapped
            out[key_err]     = _angle_err(wrapped, theoretical)

        # Legacy aliases for mod_add/mod_mul callers
        if self.n_inputs >= 1:
            out["learned_a_raw"] = out["learned_0_raw"]
            out["learned_a"]     = out["learned_0"]
            out["error_a"]       = out["error_0"]
        if self.n_inputs >= 2:
            out["learned_b_raw"] = out["learned_1_raw"]
            out["learned_b"]     = out["learned_1"]
            out["error_b"]       = out["error_1"]

        return out
