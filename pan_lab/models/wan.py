"""
pan_lab.models.wan — Walsh Accumulator Network.

The PAN primitive is phase addition in Z/PZ realised through the character
map x -> exp(i * 2pi * k * x / P). WAN is the same construction with the
underlying group replaced by F_2^n: characters are now real-valued Walsh
functions chi_s(x) = (-1)^(s·x), where s·x = XOR over bits of (s AND x).

Concretely, WAN keeps PAN's four-stage pipeline:

    encode:  bits(x)  -> v_k(x) = sum_i m[k,i] * x_i            for k = 1..K
    mix:     v        -> u_j    = sum_i W[j,i] v_i              (over K outputs)
    gate:    u        -> g_j    = (1 + cos(pi * (u_j - u_ref_j))) / 2
    decode:  g        -> logits = W_dec g + b_dec

The Walsh analog of the Fourier phase "mod 2pi" is "mod 2" — it is
implicit in cos(pi * .) and we never wrap v explicitly. The gate's
reference parameter is wrapped to [0, 2) in forward, mirroring PAN's
wrap of ref_phase to [0, 2pi) (see the docstring on `WalshGate`).

Parallels with PAN (one-to-one):

    PAN                           WAN
    ---                           ---
    phi_k(a) = k*2pi*a / P        v_k(x) = sum_i m[k,i] * x_i
    period 2pi, wrap with %       period 2, wrap with %
    cos(phase - ref_phase)        cos(pi * (v - v_ref))
    PhaseEncoder.freq             WalshEncoder.mask_logits (sigmoided to [0,1])
    PhaseMixingLayer              WalshMixingLayer    (no % at layer boundary;
                                                        cos(pi·.) is 2-periodic)
    PhaseGate.ref_phase in [0,2pi) WalshGate.ref_v   in [0, 2)
    freq_init = fourier|random    mask_init = onehot|random|parity

Mask relaxation: the natural discrete target for each mask is a binary
vector in {0,1}^n_bits. During training we relax to sigmoid(logits),
which approaches {0,1} as the logits saturate. No hard Gumbel / ST
estimator is needed — the gate's cosine keeps gradients alive across
the whole relaxation range, exactly as PAN's cosine gate does.

The gradient pathology identified in the spec ("collapse at perfect
integer weights") is already handled by the continuous relaxation plus
the gate: at discrete masks the encoder output is integer-valued but
cos(pi * .) is still smooth. No special wrapping of mixing weights is
required; the same periodicity argument as PAN's ref_phase wrap applies.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


WALSH_PERIOD = 2.0      # cos(pi * v) has period 2 in v; parallels TWO_PI for PAN


# ─────────────────────────────────────────────────────────────────────────────
def _bits_of(tokens: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Little-endian bit expansion of (B,) int tokens into (B, n_bits) float.
    Tokens are expected to lie in [0, 2^n_bits).
    """
    device = tokens.device
    powers = torch.tensor(
        [1 << i for i in range(n_bits)], device=device, dtype=torch.long,
    )
    return (tokens.unsqueeze(-1).div(powers, rounding_mode="floor") % 2).float()


# ─────────────────────────────────────────────────────────────────────────────
class WalshEncoder(nn.Module):
    """
    Encode an n_bits integer x as K Walsh "phases":

        v_k(x) = sum_i m[k, i] * x_i           (x_i in {0, 1})

    The sign character is e_k(x) = cos(pi * v_k(x)); it equals +/- 1 when
    both m[k,:] and x are binary. During training m = sigmoid(logits),
    which relaxes each mask bit smoothly onto [0, 1].

    Initialization choices (analogous to PAN's freq_init):
      - "onehot"  : mask_k selects bit (k mod n_bits), logits are ±c
      - "random"  : small-scale gaussian logits; no structure
      - "parity"  : every mask is all-ones (useful when the target is
                    the global parity and you want a "correct" init to
                    verify training mechanics)
    """

    def __init__(
        self,
        n_bits:    int,
        k_freqs:   int,
        mask_init: Literal["onehot", "random", "parity"] = "onehot",
        logit_scale: float = 4.0,
    ):
        super().__init__()
        self.n_bits      = n_bits
        self.k_freqs     = k_freqs
        self.mask_init   = mask_init
        self.logit_scale = logit_scale

        if mask_init == "onehot":
            init = torch.full((k_freqs, n_bits), -logit_scale)
            for k in range(k_freqs):
                init[k, k % n_bits] = logit_scale
        elif mask_init == "random":
            init = torch.randn(k_freqs, n_bits) * 0.5
        elif mask_init == "parity":
            init = torch.full((k_freqs, n_bits), logit_scale)
        else:
            raise ValueError(f"Unknown mask_init: {mask_init!r}")
        self.mask_logits = nn.Parameter(init)

    # Interface parity with PhaseEncoder.freq — callers that probe the
    # learned "frequencies" of an encoder treat `mask` as the WAN analog.
    @property
    def mask(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        bits : (B, n_bits) float in {0, 1}
        returns: (B, K) float — the Walsh phase vector
        """
        return bits @ self.mask.T


# ─────────────────────────────────────────────────────────────────────────────
class WalshMixingLayer(nn.Module):
    """
    Linear Walsh mixing: for each output j,
        u_j = sum_i W[j, i] * v_i

    No bias. There is no explicit `mod 2` because cos(pi * .) is already
    2-periodic, just as PAN's mix has no explicit cos and cos(.) is
    2pi-periodic. When both the masks and W collapse to integer values
    the effective mix over sign characters is exact XOR of the selected
    input signs.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        w = torch.randn(n_out, n_in) * 0.1 + (1.0 / n_in)
        self.weight = nn.Parameter(w)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.linear(v, self.weight)


# ─────────────────────────────────────────────────────────────────────────────
class WalshGate(nn.Module):
    """
    Cosine gate in Walsh space:

        g_j = (1 + cos(pi * (u_j - u_ref_j))) / 2     in [0, 1]

    ref_v lives on R/2Z but Adam treats it as an unconstrained scalar.
    Without wrapping, momentum pushes the stored value outside [0, 2)
    and into cosine-inflection regions where gradients spike. The fix
    matches PAN exactly: keep the parameter free, wrap it at forward
    time with torch.remainder (gradient 1 almost everywhere).
    """

    def __init__(self, n_phases: int):
        super().__init__()
        self.ref_v = nn.Parameter(torch.rand(n_phases) * WALSH_PERIOD)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        ref  = torch.remainder(self.ref_v, WALSH_PERIOD)
        diff = v - ref.unsqueeze(0)
        return (1.0 + torch.cos(math.pi * diff)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
class WalshAccumulatorNetwork(nn.Module):
    """
    Full WAN. Architecture (N inputs, K channels, C classes):

        inputs (B, N) long, each in [0, 2^n_bits)
            | _bits_of          -> (B, N, n_bits)
            | N WalshEncoders   -> N x (B, K)
            | concat            -> (B, N*K)
            | WalshMixingLayer  -> (B, K)
            | WalshGate         -> (B, K)
            | Linear(K -> C)    -> logits

    N is `n_inputs` (1 for single-input Walsh tasks like parity /
    popcount, 2 for XOR / arithmetic pairs). C is `n_classes` — for
    parity/XOR of a single subset this is 2; for popcount mod m it is
    m; for full XOR of two n-bit words it is 2^n_bits.
    """

    def __init__(
        self,
        n_bits:    int,
        k_freqs:   int = 4,
        n_inputs:  int = 1,
        n_classes: int = 2,
        mask_init: Literal["onehot", "random", "parity"] = "onehot",
    ):
        super().__init__()
        self.n_bits    = n_bits
        self.k_freqs   = k_freqs
        self.n_inputs  = n_inputs
        self.n_classes = n_classes
        self.mask_init = mask_init

        self.encoders = nn.ModuleList([
            WalshEncoder(n_bits, k_freqs, mask_init=mask_init)
            for _ in range(n_inputs)
        ])
        self.walsh_mix  = WalshMixingLayer(n_inputs * k_freqs, k_freqs)
        self.walsh_gate = WalshGate(k_freqs)
        self.decoder    = nn.Linear(k_freqs, n_classes, bias=True)

        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    # ── Forward ────────────────────────────────────────────────────────────
    def _encode_all(self, inputs: torch.Tensor) -> torch.Tensor:
        phases = [
            enc(_bits_of(inputs[:, i], self.n_bits))
            for i, enc in enumerate(self.encoders)
        ]
        return torch.cat(phases, dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs : (B, N) long in [0, 2^n_bits)
        returns: (B, n_classes) logits
        """
        concat = self._encode_all(inputs)
        mixed  = self.walsh_mix(concat)
        gates  = self.walsh_gate(mixed)
        return self.decoder(gates)

    def mix_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        The (B, K) tensor the diversity regularizer penalizes. Autograd
        connects back to encoder masks and mixing weights, so the
        off-diagonal Gram penalty regularizes both — same contract as
        PhaseAccumulatorNetwork.mix_features.
        """
        return self.walsh_mix(self._encode_all(inputs))

    def get_gates(self, inputs: torch.Tensor) -> torch.Tensor:
        """Post-gate activations (B, K) — the tensor just before decoder."""
        return self.walsh_gate(self.walsh_mix(self._encode_all(inputs)))

    # ── Introspection ──────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_learned_masks(self) -> dict:
        """
        Per-encoder learned masks (post-sigmoid) and their binarized form.
        Mirrors PAN's `get_learned_frequencies` but returns masks instead
        of phases. Useful for mechanistic probes (which bits each Walsh
        channel reads).
        """
        out: dict = {"n_bits": self.n_bits}
        for i, enc in enumerate(self.encoders):
            logits = enc.mask_logits.detach().cpu().numpy()
            mask   = 1.0 / (1.0 + np.exp(-logits))
            binary = (mask > 0.5).astype(np.int64)
            out[f"logits_{i}"]   = logits
            out[f"mask_{i}"]     = mask
            out[f"binary_{i}"]   = binary
            out[f"popcount_{i}"] = binary.sum(axis=1)
        return out
