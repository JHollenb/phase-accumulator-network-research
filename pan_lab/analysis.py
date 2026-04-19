"""
pan_lab.analysis — post-hoc mechanistic analysis of a trained PAN.

Functions here never mutate the model's state (they save/restore around
ablations) and they never depend on a particular experiment's config.
They take a trained model and return a dict of metrics, which the
caller can append to a DataFrame and/or feed to the plotter.

Design rule: every analyzer returns a plain-Python dict or a pandas
DataFrame — never a TrainResult or a nested object — so aggregation
stays flat.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from pan_lab.config import SIFP16_QUANT_ERR, TWO_PI
from pan_lab.models.pan import PhaseAccumulatorNetwork
from pan_lab.models.wan import WalshAccumulatorNetwork


# ─────────────────────────────────────────────────────────────────────────────
def fourier_concentration(W: torch.Tensor, top_k: int = 10) -> float:
    """
    Fraction of spectral energy of W concentrated in its top-k FFT bins.

    1.0 = pure sinusoid at a few frequencies. 0.0 = white noise.
    Computed along axis 0 (rows, typically the P dimension of a decoder
    weight matrix).
    """
    W = W.float()
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    F_ = torch.fft.fft(W, dim=0)
    energy = F_.abs() ** 2
    total  = float(energy.sum().item())
    if total < 1e-12:
        return 0.0
    k = min(top_k, energy.numel())
    return float(energy.reshape(-1).topk(k).values.sum().item() / total)


# ─────────────────────────────────────────────────────────────────────────────
def compute_frequency_errors(
    model: PhaseAccumulatorNetwork,
) -> Dict[str, np.ndarray]:
    """
    Thin wrapper over model.get_learned_frequencies() that also computes
    binary convergence masks per encoder. Returned keys include:
      - learned_<i>, learned_<i>_raw, error_<i>           per encoder
      - theoretical                                       common
      - converged_<i>                                     bool mask,
        True where error_<i> < SIFP16_QUANT_ERR
    """
    info = model.get_learned_frequencies()
    for i in range(model.n_inputs):
        info[f"converged_{i}"] = info[f"error_{i}"] < SIFP16_QUANT_ERR
    return info


# ─────────────────────────────────────────────────────────────────────────────
def ablation_test(
    model: PhaseAccumulatorNetwork,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Zero out each PAN component and measure the val-accuracy drop.
    Confirms that phase arithmetic is the active mechanism rather than
    a decoder shortcut.

    Returns a dict of {intervention_name: val_acc_after_intervention}.
    """
    model.eval()
    out: Dict[str, float] = {}

    def _acc() -> float:
        with torch.no_grad():
            logits = model(val_x)
            return float((logits.argmax(-1) == val_y).float().mean().item())

    out["baseline"] = _acc()

    if isinstance(model, PhaseAccumulatorNetwork):
        # 1. Zero phase mixing
        with torch.no_grad():
            saved = model.phase_mix.weight.data.clone()
            model.phase_mix.weight.data.zero_()
            out["zero_phase_mixing"] = _acc()
            model.phase_mix.weight.data.copy_(saved)

        # 2. Randomize frequencies (every encoder)
        with torch.no_grad():
            saved_freqs = [enc.freq.data.clone() for enc in model.encoders]
            for enc in model.encoders:
                enc.freq.data = torch.rand_like(enc.freq.data) * TWO_PI
            out["randomize_frequencies"] = _acc()
            for enc, s in zip(model.encoders, saved_freqs):
                enc.freq.data.copy_(s)

        # 3. Zero reference phases (gate becomes constant 0.5)
        with torch.no_grad():
            saved_ref = model.phase_gate.ref_phase.data.clone()
            model.phase_gate.ref_phase.data.zero_()
            out["zero_ref_phases"] = _acc()
            model.phase_gate.ref_phase.data.copy_(saved_ref)

    elif isinstance(model, WalshAccumulatorNetwork):
        # 1. Zero Walsh mixing
        with torch.no_grad():
            saved = model.walsh_mix.weight.data.clone()
            model.walsh_mix.weight.data.zero_()
            out["zero_walsh_mixing"] = _acc()
            model.walsh_mix.weight.data.copy_(saved)

        # 2. Randomize encoder masks (the WAN "frequency" analog).
        # Replace logits with gaussian noise so sigmoid(m) becomes random.
        with torch.no_grad():
            saved_logits = [enc.mask_logits.data.clone() for enc in model.encoders]
            for enc in model.encoders:
                enc.mask_logits.data = torch.randn_like(enc.mask_logits.data)
            out["randomize_masks"] = _acc()
            for enc, s in zip(model.encoders, saved_logits):
                enc.mask_logits.data.copy_(s)

        # 3. Zero gate reference — Walsh gate becomes constant 0.5 of cos(0)=1
        with torch.no_grad():
            saved_ref = model.walsh_gate.ref_v.data.clone()
            model.walsh_gate.ref_v.data.zero_()
            out["zero_ref_v"] = _acc()
            model.walsh_gate.ref_v.data.copy_(saved_ref)

    if verbose:
        for name, acc in out.items():
            print(f"  {name:>25}: acc={acc:.3f}  delta={acc-out['baseline']:+.3f}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
def detect_mode_collapse(model) -> bool:
    """
    True if the mixing layer has collapsed — every one of the K output
    channels is dominated by the same input slot. Handles PAN and WAN.
    """
    if isinstance(model, PhaseAccumulatorNetwork):
        W = model.phase_mix.weight.detach().cpu().numpy()
    elif isinstance(model, WalshAccumulatorNetwork):
        W = model.walsh_mix.weight.detach().cpu().numpy()
    else:
        return False
    dominant = [int(np.argmax(np.abs(row))) for row in W]
    return len(set(dominant)) == 1


# ─────────────────────────────────────────────────────────────────────────────
def slot_activation_census(
    models: List[PhaseAccumulatorNetwork],
    threshold: float = SIFP16_QUANT_ERR,
) -> "pd.DataFrame":
    """
    Given a list of trained PANs (e.g. one per seed), produce a DataFrame
    with one row per (model_index, encoder, k). Columns:
      - model_idx, encoder, k (1-indexed), theoretical, learned,
        error, converged (bool: error < threshold)

    This is the raw data for Experiment A in the roadmap critique:
    the frequency-slot activation census. From it you can ask
    'how many of the K=9 slots converge in each seed?' or 'is it always
    the same subset?' — which is the direct answer to the Section 5.1
    open question.
    """
    import pandas as pd

    rows = []
    for idx, m in enumerate(models):
        info = compute_frequency_errors(m)
        for i in range(m.n_inputs):
            for k in range(m.k_freqs):
                rows.append({
                    "model_idx":   idx,
                    "encoder":     i,
                    "k":           k + 1,
                    "theoretical": float(info["theoretical"][k]),
                    "learned":     float(info[f"learned_{i}"][k]),
                    "learned_raw": float(info[f"learned_{i}_raw"][k]),
                    "error":       float(info[f"error_{i}"][k]),
                    "converged":   bool(info[f"converged_{i}"][k]),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def analyze_pan(
    model: PhaseAccumulatorNetwork,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Aggregate everything in one call — frequency errors, gate phases,
    mixing-weight structure, mode-collapse flag.
    """
    info = compute_frequency_errors(model)

    mix_W = model.phase_mix.weight.detach().cpu().numpy()      # (K, N*K)

    # For each output channel j, identify the dominant input slot.
    mix_dominant = []
    for j in range(mix_W.shape[0]):
        dom    = int(np.argmax(np.abs(mix_W[j])))
        src    = dom // model.k_freqs       # encoder index
        k_idx  = dom %  model.k_freqs
        weight = float(mix_W[j, dom])
        mix_dominant.append((j, src, k_idx, weight))

    ref_raw     = model.phase_gate.ref_phase.detach().cpu().numpy()
    ref_wrapped = ref_raw % TWO_PI

    result = {
        "p":              model.p,
        "k_freqs":        model.k_freqs,
        "n_inputs":       model.n_inputs,
        "freq_info":      info,
        "mixing_weights": mix_W,
        "mix_dominant":   mix_dominant,
        "ref_phase_raw":     ref_raw,
        "ref_phase_wrapped": ref_wrapped,
        "mode_collapsed": detect_mode_collapse(model),
    }

    if verbose:
        print(f"\n── PAN analysis  p={model.p}  K={model.k_freqs}  "
              f"N={model.n_inputs} ──")
        for i in range(model.n_inputs):
            err = info[f"error_{i}"]
            conv = int(info[f"converged_{i}"].sum())
            print(f"  encoder[{i}]  mean_err={err.mean():.5f}  "
                  f"max_err={err.max():.5f}  "
                  f"converged_to_SIFP16: {conv}/{model.k_freqs}")
        print(f"  mode_collapsed: {result['mode_collapsed']}")
        for j, src, k_idx, w in mix_dominant:
            print(f"    mix[{j}] <- enc[{src}][{k_idx}]  w={w:+.3f}")

    return result
