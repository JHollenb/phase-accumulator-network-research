"""
pan_lab.metrics — per-eval-step mechanistic instrumentation of a PAN.

Collects eight metrics whose *formation curves* across training are the
mechanistic story the paper currently only measures at convergence:

  M1  fourier_snap              — per-slot distance to nearest n·2π/p
  M2  clock_pair_compliance     — fraction of mix rows whose top-2 weights
                                  come from different encoders at matched
                                  magnitudes
  M3  clock_freq_alignment      — circular freq distance within the pairs
                                  that M2 picked out (guards M2 against
                                  false positives)
  M4  mix_row_entropy           — effective number of contributing slots
                                  per output row; approaches 2 as the
                                  circuit sparsifies into Clock pairs
  M5  active_frequencies        — distinct integer Fourier modes used
                                  above a mixing-weight threshold
  M6  gate_linear_decodability  — multinomial logistic-regression acc on
                                  (gate -> label); how sufficient the gate
                                  representation is
  M7  sifp16_robustness         — val acc when every phase is rounded to
                                  the SIFP-16 lattice
  M8  decoder_fourier_projection — fraction of per-column decoder energy
                                  concentrated in its peak DFT bin

Every metric is a pure function of model weights (M1–M5, M8) or needs
one extra forward pass (M6, M7). Gradients are off for all model reads;
M6's linear probe fits internally via sklearn.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from pan_lab.config import TWO_PI
from pan_lab.models.quantize import sifp16_context


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _snap_to_lattice(W_enc: torch.Tensor, p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each scalar frequency in W_enc, find the nearest integer Fourier
    mode n·2π/p for n ∈ [1, p//2]. Returns (dist, n) tensors of shape (K,).
    Distance is circular on [0, π].
    """
    freqs   = torch.remainder(W_enc, TWO_PI)                           # (K,)
    lattice = torch.arange(1, p // 2 + 1, device=W_enc.device) * (TWO_PI / p)
    diff    = freqs.unsqueeze(1) - lattice.unsqueeze(0)                # (K, P//2)
    diff    = torch.abs(torch.remainder(diff + math.pi, TWO_PI) - math.pi)
    dist, idx = diff.min(dim=1)
    n = idx + 1                                                         # 1..P//2
    return dist, n


# ─────────────────────────────────────────────────────────────────────────────
# M1. Fourier lattice snap distance (per encoder)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fourier_snap(W_enc: torch.Tensor, p: int) -> Dict[str, Any]:
    dist, n = _snap_to_lattice(W_enc, p)
    return {
        "snap_mean": float(dist.mean().item()),
        "snap_max":  float(dist.max().item()),
        "snap_min":  float(dist.min().item()),
        "active_n":  ",".join(str(int(v)) for v in n.tolist()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M2. Clock-pair compliance
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def clock_pair_compliance(W_mix: torch.Tensor, K: int, mag_tol: float = 0.20) -> float:
    """
    Fraction of output rows whose top-2 |weights| come from different
    encoders (one from the first K cols, one from the last K) and whose
    magnitudes match within `mag_tol`.
    """
    compliant = 0
    for row_idx in range(W_mix.shape[0]):
        row = W_mix[row_idx].abs()
        top2_vals, top2_idx = row.topk(2)
        i0, i1 = int(top2_idx[0].item()), int(top2_idx[1].item())
        diff_enc = (i0 < K) != (i1 < K)
        denom    = float(top2_vals[0].item())
        matched  = denom > 0 and abs(top2_vals[0].item() - top2_vals[1].item()) / denom < mag_tol
        if diff_enc and matched:
            compliant += 1
    return compliant / max(W_mix.shape[0], 1)


# ─────────────────────────────────────────────────────────────────────────────
# M3. Matched-slot frequency alignment
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def clock_freq_alignment(
    W_mix: torch.Tensor, W_enc0: torch.Tensor, W_enc1: torch.Tensor,
    K: int, mag_tol: float = 0.20,
) -> Dict[str, Any]:
    """
    For rows that pass M2, mean circular distance between the two paired
    frequencies. Low = the "pair" really is the same frequency on both
    encoders (a true Clock pair). Returns (mean, n_pairs_found).
    """
    diffs: List[float] = []
    for row_idx in range(W_mix.shape[0]):
        row = W_mix[row_idx].abs()
        top2_vals, top2_idx = row.topk(2)
        i0, i1 = int(top2_idx[0].item()), int(top2_idx[1].item())
        if (i0 < K) == (i1 < K):
            continue
        denom = float(top2_vals[0].item())
        if denom <= 0 or abs(top2_vals[0].item() - top2_vals[1].item()) / denom >= mag_tol:
            continue
        f0 = W_enc0[i0] if i0 < K else W_enc1[i0 - K]
        f1 = W_enc0[i1] if i1 < K else W_enc1[i1 - K]
        d  = float(torch.abs(torch.remainder(f0 - f1 + math.pi, TWO_PI) - math.pi).item())
        diffs.append(d)
    if not diffs:
        return {"align_mean": float("nan"), "align_n": 0}
    return {"align_mean": float(np.mean(diffs)), "align_n": len(diffs)}


# ─────────────────────────────────────────────────────────────────────────────
# M4. Mixing-matrix row entropy
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mix_row_entropy(W_mix: torch.Tensor, K: int) -> Dict[str, Any]:
    """
    Shannon entropy of |W_mix| rows (treated as probability distributions),
    plus the equivalent "effective number of contributing slots"
    eff_n = exp(entropy). A clean Clock-pair row has eff_n ≈ 2.
    """
    abs_rows = W_mix.abs() + 1e-10
    probs    = abs_rows / abs_rows.sum(dim=1, keepdim=True)
    entropy  = -(probs * probs.log()).sum(dim=1)
    eff_n    = entropy.exp()
    return {
        "row_entropy_mean": float(entropy.mean().item()),
        "row_eff_n_mean":   float(eff_n.mean().item()),
        "row_eff_n_min":    float(eff_n.min().item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M5. Distinct active frequencies
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def active_frequencies(
    W_enc0: torch.Tensor, W_enc1: torch.Tensor, W_mix: torch.Tensor,
    p: int, K: int, weight_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    For each slot, snap its learned freq to an integer Fourier mode n.
    Count the distinct n's that are used by at least one mixing-row
    entry above `weight_threshold`.
    """
    _, snap_n0 = _snap_to_lattice(W_enc0, p)
    _, snap_n1 = _snap_to_lattice(W_enc1, p)
    abs_mix = W_mix.abs()
    active: set = set()
    for row_idx in range(abs_mix.shape[0]):
        row = abs_mix[row_idx]
        for col in range(2 * K):
            if float(row[col].item()) > weight_threshold:
                n = int(snap_n0[col].item() if col < K else snap_n1[col - K].item())
                active.add(n)
    return {
        "count": len(active),
        "set":   ",".join(str(n) for n in sorted(active)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M6. Gate linear decodability
# ─────────────────────────────────────────────────────────────────────────────
def gate_linear_decodability(
    model, val_x: torch.Tensor, val_y: torch.Tensor, max_rows: int = 4000,
) -> Dict[str, Any]:
    """
    Fit a multinomial logistic regression on (gate_output -> label) over
    the val set (subsampled to `max_rows` for speed). The accuracy is a
    ceiling on what a linear decoder could achieve on the current gate
    representation — i.e. how computationally sufficient the gate layer
    has become.
    """
    from sklearn.linear_model import LogisticRegression

    model.eval()
    with torch.no_grad():
        n = val_x.shape[0]
        if n > max_rows:
            idx = torch.randperm(n, device=val_x.device)[:max_rows]
            vx, vy = val_x[idx], val_y[idx]
        else:
            vx, vy = val_x, val_y
        G = model.get_gates(vx).detach().cpu().numpy()
        y = vy.detach().cpu().numpy()

    clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
    clf.fit(G, y)
    return {"gate_linear_acc": float(clf.score(G, y))}


# ─────────────────────────────────────────────────────────────────────────────
# M7. SIFP-16 robustness
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sifp16_robustness(model, val_x: torch.Tensor, val_y: torch.Tensor) -> Dict[str, Any]:
    """
    Val acc at fp32 vs with every PAN phase output rounded to the
    SIFP-16 lattice. Delta measures how resilient the circuit is to
    hardware quantization — a proxy for "has the frequency set cleaned
    up enough to quantize cleanly?"
    """
    model.eval()
    fp32_logits = model(val_x)
    fp32_acc    = float((fp32_logits.argmax(-1) == val_y).float().mean().item())
    with sifp16_context(model):
        q_logits  = model(val_x)
        sifp16_acc = float((q_logits.argmax(-1) == val_y).float().mean().item())
    return {
        "fp32_acc":    fp32_acc,
        "sifp16_acc":  sifp16_acc,
        "quant_delta": fp32_acc - sifp16_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M8. Decoder Fourier projection
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def decoder_fourier_projection(W_dec: torch.Tensor) -> Dict[str, Any]:
    """
    For each of the K decoder columns (viewed as a length-P signal over
    class index), compute the fraction of spectral energy carried by
    its single largest DFT bin. A Clock decoder has each column purely
    sinusoidal, so peak → 1.
    """
    W = W_dec.detach().cpu().numpy()            # (P, K)
    peaks = []
    for k in range(W.shape[1]):
        fft   = np.fft.rfft(W[:, k])
        power = np.abs(fft) ** 2
        total = power.sum()
        if total < 1e-12:
            peaks.append(0.0)
            continue
        peaks.append(float(power.max() / total))
    return {
        "peak_mean": float(np.mean(peaks)),
        "peak_max":  float(np.max(peaks)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M9. Logit 2D spectrum
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def logit_2d_spectrum(
    model,
    p: int,
    sample_classes: Optional[Union[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """
    Aggregate sparsity statistics of the 2D DFT of logit(c | a, b)
    over the P×P input grid, averaged across `sample_classes`.

    A grokked PAN computing Fourier modular addition has per-class
    logit grids with power concentrated on the `f_a = f_b` diagonal
    and at a small number of (f_a, f_b) bins; a not-yet-grokked model
    has power spread roughly uniformly.

    `sample_classes`:
      - None → 8 evenly-spaced classes over [0, P)
      - int  → that many evenly-spaced classes
      - list/tuple → exactly those class indices

    Returns three scalars; all NaN if every sampled class has
    zero-power logit grid (pathological).
    """
    if sample_classes is None:
        n = min(8, int(p))
        sampled = list(np.linspace(0, p - 1, n, dtype=int))
    elif isinstance(sample_classes, int):
        n = min(int(sample_classes), int(p))
        sampled = list(np.linspace(0, p - 1, n, dtype=int))
    else:
        sampled = [int(c) for c in sample_classes]

    device = next(model.parameters()).device
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    model.eval()
    logits = model(torch.stack([a, b], dim=-1)).detach().cpu().numpy()   # (P², P)
    # reshape to (class, a, b)
    grids = logits.reshape(p, p, p).transpose(2, 0, 1)

    diag, peak, active = [], [], []
    for c in sampled:
        g     = grids[int(c)]
        power = np.abs(np.fft.fft2(g)) ** 2
        total = power.sum()
        if total < 1e-12:
            continue
        diag.append(float(np.trace(power) / total))
        flat = power.ravel()
        # top-5 bins even if p is tiny
        k = min(5, flat.size)
        peak.append(float(np.partition(flat, -k)[-k:].sum() / total))
        active.append(int((flat > 0.01 * total).sum()))

    if not diag:
        nan = float("nan")
        return {
            "logit_spec_diag_frac_mean":     nan,
            "logit_spec_peak_sparsity_mean": nan,
            "logit_spec_active_count_mean":  nan,
        }
    return {
        "logit_spec_diag_frac_mean":     float(np.mean(diag)),
        "logit_spec_peak_sparsity_mean": float(np.mean(peak)),
        "logit_spec_active_count_mean":  float(np.mean(active)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composers
# ─────────────────────────────────────────────────────────────────────────────
def cheap_metrics(model) -> Dict[str, Any]:
    """
    Bundle the weight-only metrics (M1, M2, M3, M4, M5, M8) into one
    flat dict with unique column names. Assumes `model` is a PAN with
    exactly 2 encoders (mod_add / mod_mul).
    """
    p = model.p
    K = model.k_freqs
    W_enc0 = model.encoders[0].freq.detach()
    W_enc1 = model.encoders[1].freq.detach() if len(model.encoders) > 1 else W_enc0
    W_mix  = model.phase_mix.weight.detach()
    W_dec  = model.decoder.weight.detach()

    out: Dict[str, Any] = {}
    for i, W in enumerate((W_enc0, W_enc1)):
        for key, val in fourier_snap(W, p).items():
            out[f"enc{i}_{key}"] = val

    out["clock_compliance"] = clock_pair_compliance(W_mix, K)

    align = clock_freq_alignment(W_mix, W_enc0, W_enc1, K)
    out["clock_freq_align_mean"] = align["align_mean"]
    out["clock_freq_align_n"]    = align["align_n"]

    for key, val in mix_row_entropy(W_mix, K).items():
        out[f"mix_{key}"] = val

    af = active_frequencies(W_enc0, W_enc1, W_mix, p, K)
    out["active_freq_count"] = af["count"]
    out["active_freq_set"]   = af["set"]

    for key, val in decoder_fourier_projection(W_dec).items():
        out[f"decoder_fourier_{key}"] = val

    return out


def expensive_metrics(
    model,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    max_rows: int = 4000,
    logit_spectrum: bool = False,
    logit_spectrum_classes: Optional[Union[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Bundle the forward-pass-requiring metrics (M6, M7, + derived gap).

    M9 (`logit_2d_spectrum`) is OFF by default — it saturates early
    during training and costs a P² forward + per-class 2D FFT. Flip
    `logit_spectrum=True` to restore the three logit_spec_* columns.
    """
    out: Dict[str, Any] = {}
    out.update(gate_linear_decodability(model, val_x, val_y, max_rows=max_rows))
    out.update(sifp16_robustness(model, val_x, val_y))
    out["gate_decoder_gap"] = float(out["gate_linear_acc"] - out["fp32_acc"])
    if logit_spectrum:
        out.update(logit_2d_spectrum(model, model.p, logit_spectrum_classes))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hook
# ─────────────────────────────────────────────────────────────────────────────
class MetricsLogger:
    """
    Per-eval hook. On every eval, append a dict of cheap metrics to
    `history.metrics_rows`; every `expensive_every` training steps,
    additionally append the expensive ones (M6, M7). No-op on non-PAN
    models.

    Set `expensive_every=0` to skip M6 and M7 entirely.
    """

    def __init__(
        self,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        expensive_every: int = 5000,
        gate_decode_max_rows: int = 4000,
        logit_spectrum: bool = False,
        logit_spectrum_classes: Optional[Union[int, Sequence[int]]] = None,
    ):
        self.val_x = val_x
        self.val_y = val_y
        self.expensive_every = expensive_every
        self.gate_decode_max_rows = gate_decode_max_rows
        self.logit_spectrum = logit_spectrum
        self.logit_spectrum_classes = logit_spectrum_classes

    def on_eval(self, step, model, cfg, history, val_loss, val_acc):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        from pan_lab.models.pan import PhaseAccumulatorNetwork
        if not isinstance(raw, PhaseAccumulatorNetwork):
            return

        row: Dict[str, Any] = {"step": int(step), **cheap_metrics(raw)}
        if self.expensive_every and step % self.expensive_every == 0:
            row.update(expensive_metrics(
                raw, self.val_x, self.val_y,
                max_rows=self.gate_decode_max_rows,
                logit_spectrum=self.logit_spectrum,
                logit_spectrum_classes=self.logit_spectrum_classes,
            ))
        history.metrics_rows.append(row)
