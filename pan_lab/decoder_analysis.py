"""
decoder_analysis.py — a new PAN experiment.

Goal
====

The original `decoder_swap` experiment asked: "does swapping in a
theoretical Fourier decoder preserve accuracy?" The answer was no,
but the answer alone doesn't say why. This experiment characterizes
what the learned decoder contains that a pure Clock decoder misses.

Given a grokked PAN with effective channel frequencies f_j and gate
reference phases phi_j, the Clock circuit predicts decoder rows of
the form

    W[c, j] ~ cos(f_j * c - phi_j)

We project the learned decoder onto the span of the Clock basis
(cos AND sin at each (f_j, phi_j) pair), measure how much energy is
explained, and analyze the spectrum of the residual. The key
diagnostic: do the residual's dominant frequencies align with
*additional* Fourier basis vectors of Z_P that the mixing matrix
didn't highlight? If yes, the circuit is "Clock + extra frequencies."
If the residual is unstructured noise, the circuit is truly
Clock-only and the gap is something else (scaling, bias, etc.).

How to integrate
================

Save this file as `pan_lab/pan_lab/decoder_analysis.py`. Then add
these two lines near the top of `pan_lab/pan_lab/experiments.py`:

    # Register the decoder_analysis experiment
    from pan_lab import decoder_analysis as _decoder_analysis  # noqa: F401

That's enough — the @register("decoder_analysis") decorator at the
bottom of this file auto-adds it to EXPERIMENT_REGISTRY on import.

Then create experiments/decoder_analysis.yaml (see bottom of this
file for the YAML) and run:

    python -m pan_lab experiments/decoder_analysis.yaml
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from pan_lab.config     import DEVICE, RunConfig, TWO_PI
from pan_lab.data       import make_modular_dataset
from pan_lab.experiments import _print_plan, register
from pan_lab.models     import make_model
from pan_lab.reporting  import ExperimentReporter
from pan_lab.trainer    import train


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def _channel_effective_frequency(W_mix: np.ndarray,
                                  enc0_freq: np.ndarray,
                                  enc1_freq: np.ndarray) -> np.ndarray:
    """
    For each output channel j in the phase-mixing matrix, estimate the
    effective frequency as the circular mean of the top-1 enc0 slot's
    frequency and the top-1 enc1 slot's frequency.

    Returns (K,) float array in [0, 2pi).
    """
    K = W_mix.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for j in range(K):
        abs_row = np.abs(W_mix[j])
        top_e0  = int(np.argmax(abs_row[:K]))
        top_e1  = int(np.argmax(abs_row[K:]))
        f0 = float(enc0_freq[top_e0] % TWO_PI)
        f1 = float(enc1_freq[top_e1] % TWO_PI)
        s  = (np.sin(f0) + np.sin(f1)) / 2
        c  = (np.cos(f0) + np.cos(f1)) / 2
        out[j] = float(np.arctan2(s, c) % TWO_PI)
    return out


def _build_clock_basis(p: int, f_channels: np.ndarray,
                       phi_ref: np.ndarray) -> np.ndarray:
    """
    Build the Clock basis matrix B of shape (P, 2K).
    Columns 0..K-1:  cos(f_j * c - phi_j)
    Columns K..2K-1: sin(f_j * c - phi_j)

    Using both cos and sin lets us match any phase offset the learned
    decoder may have absorbed — without sin we'd only capture decoders
    aligned to exactly the learned phi_ref, which is overly strict.
    """
    P = p
    K = len(f_channels)
    c = np.arange(P, dtype=np.float64).reshape(P, 1)        # (P, 1)
    f = f_channels.reshape(1, K)                            # (1, K)
    phi = phi_ref.reshape(1, K)                             # (1, K)
    arg = f * c - phi                                        # (P, K)
    B = np.concatenate([np.cos(arg), np.sin(arg)], axis=1)  # (P, 2K)
    return B


def _project_onto_basis(W: np.ndarray, B: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Column-space projection: for each learned decoder column w_j (one
    per channel), find the best fit in the column span of B.

    Returns (W_proj, residual), same shape as W.
    """
    # We fit per-column: for column j of W, solve
    #   min_alpha  || B @ alpha - W[:, j] ||^2
    # in closed form via lstsq.
    alpha, *_ = np.linalg.lstsq(B, W, rcond=None)   # (2K, K)
    W_proj    = B @ alpha                            # (P, K)
    residual  = W - W_proj
    return W_proj, residual


def _residual_spectrum(residual: np.ndarray, p: int
                       ) -> dict[str, np.ndarray]:
    """
    FFT each column of the residual, take magnitudes, average across
    columns. Returns keyed arrays for csv export.
    """
    # residual shape (P, K); FFT along axis=0 gives (P, K) complex
    F = np.fft.fft(residual, axis=0)
    mag = np.abs(F).mean(axis=1)                     # (P,)
    # FFT bins correspond to k = 0..P-1 on integers; the "angular"
    # frequency at bin k is 2*pi*k/P
    return {
        "k":         np.arange(p),
        "magnitude": mag,
    }


def _evaluate_decoder(pan, W_new: np.ndarray,
                       val_x: torch.Tensor, val_y: torch.Tensor
                       ) -> float:
    """Swap W_new into pan.decoder.weight, zero bias, eval, restore."""
    saved_w = pan.decoder.weight.data.clone()
    saved_b = pan.decoder.bias.data.clone()
    with torch.no_grad():
        pan.decoder.weight.data = torch.tensor(
            W_new, dtype=torch.float32, device=DEVICE)
        pan.decoder.bias.data.zero_()
        logits = pan(val_x)
        acc = float((logits.argmax(-1) == val_y).float().mean().item())
        pan.decoder.weight.data.copy_(saved_w)
        pan.decoder.bias.data.copy_(saved_b)
    return acc


# ─────────────────────────────────────────────────────────────────────────
# The experiment
# ─────────────────────────────────────────────────────────────────────────
@register("decoder_analysis")
def exp_decoder_analysis(base: RunConfig, out_dir: str,
                          dry_run: bool = False,
                          seeds: Optional[List[int]] = None,
                          max_extra_freqs: int = 20,
                          **_) -> ExperimentReporter:
    """
    For each seed, train PAN, then decompose the learned decoder:

      0. Measure learned-decoder val_acc (the baseline).
      1. Build Clock basis from the mixing matrix and gate phases.
      2. Project learned decoder onto Clock basis.
         - explained_energy_fraction = 1 - ||R||^2 / ||W||^2
         - val_acc_clock_only       = eval(W_proj)
      3. Analyze residual spectrum (FFT per column, average magnitudes).
      4. Basis expansion: iteratively add (cos, sin) pairs at the
         top-residual-magnitude k values. For each step, refit and
         re-evaluate val_acc. Stops when val_acc within 1% of learned
         OR max_extra_freqs reached.

    Writes:
      decoder_analysis.csv          — one row per seed: summary metrics
      decoder_recovery_curve.csv    — long: (seed, n_extras, val_acc)
      decoder_residual_spectrum.csv — long: (seed, k, magnitude)
    """
    seeds = seeds or [42, 123, 456]
    cfgs  = [base.with_overrides(model_kind="pan", seed=s,
                                  weight_decay=0.01, save_model=True,
                                  label=f"danal-s{s}") for s in seeds]

    if dry_run:
        _print_plan(cfgs, "decoder_analysis")
        return ExperimentReporter("decoder_analysis", out_dir)

    rep = ExperimentReporter("decoder_analysis", out_dir)
    summary_rows:  list[dict] = []
    recovery_rows: list[dict] = []
    spectrum_rows: list[dict] = []

    for cfg in cfgs:
        tx, ty, vx, vy = make_modular_dataset(
            p=cfg.p, task_kind=cfg.task_kind,
            train_frac=cfg.train_frac, seed=cfg.seed)
        model  = make_model(cfg).to(DEVICE)
        result = train(model, cfg, tx, ty, vx, vy, verbose=True)
        rep.add_run(result, val_x=vx, val_y=vy, ablations=False)
        pan = result.model

        # ── 0. Baseline ──
        with torch.no_grad():
            acc_learned = float(
                (pan(vx).argmax(-1) == vy).float().mean().item())
            W_learned = pan.decoder.weight.detach().cpu().numpy()  # (P, K)

        if acc_learned < 0.95:
            print(f"  [{cfg.display_id()}] skipped — not grokked "
                  f"(learned={acc_learned:.3f})")
            summary_rows.append({
                "run_id":     cfg.display_id(),
                "seed":       cfg.seed,
                "grokked":    False,
                "acc_learned": acc_learned,
            })
            continue

        # ── 1. Extract circuit parameters ──
        W_mix = pan.phase_mix.weight.detach().cpu().numpy()         # (K, 2K)
        f0    = pan.encoders[0].freq.detach().cpu().numpy()
        f1    = pan.encoders[1].freq.detach().cpu().numpy()
        phi   = (pan.phase_gate.ref_phase.detach().cpu().numpy()
                  % TWO_PI)
        f_eff = _channel_effective_frequency(W_mix, f0, f1)         # (K,)
        K     = cfg.k_freqs
        P     = cfg.p

        # ── 2. Clock projection ──
        B_clock = _build_clock_basis(P, f_eff, phi)                 # (P, 2K)
        W_proj, residual = _project_onto_basis(W_learned, B_clock)
        acc_clock = _evaluate_decoder(pan, W_proj, vx, vy)

        energy_total    = float((W_learned ** 2).sum())
        energy_residual = float((residual ** 2).sum())
        explained_frac  = 1.0 - energy_residual / max(energy_total, 1e-12)

        # ── 3. Residual spectrum ──
        spec = _residual_spectrum(residual, P)
        for k, mag in zip(spec["k"], spec["magnitude"]):
            spectrum_rows.append({
                "run_id":    cfg.display_id(),
                "seed":      cfg.seed,
                "k":         int(k),
                "magnitude": float(mag),
            })

        # ── 4. Basis expansion ──
        # Sort FFT bins by magnitude (exclude DC and bins already close
        # to channel frequencies). Angular frequency at bin k is
        # 2*pi*k/P. We add cos/sin pairs at those angular freqs with
        # phi=0 (we'll refit phases via projection).
        mag = spec["magnitude"].copy()
        mag[0] = 0.0                              # exclude DC
        # Exclude bins that already correspond to an existing channel freq
        for fj in f_eff:
            # Find nearest FFT bin
            nearest_k = int(round(fj * P / TWO_PI)) % P
            mag[nearest_k] = 0.0
            # Also suppress the conjugate bin (FFT symmetry)
            conj = (-nearest_k) % P
            mag[conj] = 0.0

        # Top-residual frequencies in descending order
        candidate_order = np.argsort(mag)[::-1]
        # Take only bins in the first half (real signal -> symmetric
        # spectrum, avoid duplicating conjugates)
        candidate_order = [int(k) for k in candidate_order
                           if 0 < int(k) <= P // 2]

        # Record n_extras = 0 first (Clock-only)
        recovery_rows.append({
            "run_id":     cfg.display_id(),
            "seed":       cfg.seed,
            "n_extras":   0,
            "added_ks":   "",
            "val_acc":    acc_clock,
            "explained":  explained_frac,
        })

        added_ks: List[int] = []
        # Start basis as just Clock
        current_basis = B_clock.copy()

        n_extras_needed = None
        for step in range(1, min(max_extra_freqs, len(candidate_order)) + 1):
            k_new = candidate_order[step - 1]
            added_ks.append(k_new)
            freq_new = TWO_PI * k_new / P
            c  = np.arange(P, dtype=np.float64).reshape(P, 1)
            cos_col = np.cos(freq_new * c)
            sin_col = np.sin(freq_new * c)
            current_basis = np.concatenate(
                [current_basis, cos_col, sin_col], axis=1)

            # Refit and evaluate
            W_fit, _ = _project_onto_basis(W_learned, current_basis)
            acc_now  = _evaluate_decoder(pan, W_fit, vx, vy)

            energy_resid_now = float(((W_learned - W_fit) ** 2).sum())
            explained_now    = 1.0 - energy_resid_now / max(energy_total, 1e-12)

            recovery_rows.append({
                "run_id":     cfg.display_id(),
                "seed":       cfg.seed,
                "n_extras":   step,
                "added_ks":   ",".join(str(k) for k in added_ks),
                "val_acc":    acc_now,
                "explained":  explained_now,
            })

            if n_extras_needed is None and acc_now >= acc_learned - 0.01:
                n_extras_needed = step
                # Continue a few more for completeness
                if step >= n_extras_needed + 3:
                    break

        # ── Summary ──
        summary_rows.append({
            "run_id":              cfg.display_id(),
            "seed":                cfg.seed,
            "grokked":             True,
            "acc_learned":         acc_learned,
            "acc_clock_only":      acc_clock,
            "gap_clock":           acc_learned - acc_clock,
            "clock_explained_frac": explained_frac,
            "n_extras_for_1pct":   n_extras_needed if n_extras_needed else -1,
            "first_3_extras":      ",".join(
                str(k) for k in candidate_order[:3]),
            "f_eff_channels":      ",".join(
                f"{f:.4f}" for f in f_eff),
        })

        print(f"  [{cfg.display_id()}] "
              f"learned={acc_learned:.3f}  "
              f"clock_only={acc_clock:.3f}  "
              f"explained={explained_frac:.2%}  "
              f"n_extras={n_extras_needed}")

    rep.write_all()
    summary_df  = pd.DataFrame(summary_rows)
    recovery_df = pd.DataFrame(recovery_rows)
    spectrum_df = pd.DataFrame(spectrum_rows)

    summary_df.to_csv(os.path.join(out_dir, "decoder_analysis.csv"),
                       index=False)
    recovery_df.to_csv(
        os.path.join(out_dir, "decoder_recovery_curve.csv"), index=False)
    spectrum_df.to_csv(
        os.path.join(out_dir, "decoder_residual_spectrum.csv"), index=False)

    print("\n── Decoder analysis summary ──")
    if len(summary_df):
        print(summary_df.to_string(index=False))
    rep.write_manifest()
    return rep


# ─────────────────────────────────────────────────────────────────────────
# YAML (save as experiments/decoder_analysis.yaml)
# ─────────────────────────────────────────────────────────────────────────
_YAML_TEMPLATE = """\
# decoder_analysis — decompose the learned decoder against the Clock
# basis and characterize the residual. For each seed:
#   * Clock-only val_acc (pure Clock decoder)
#   * explained_frac    (fraction of decoder energy the Clock basis captures)
#   * n_extras          (how many additional Fourier frequencies we need
#                        to add for the decoder to match the learned baseline)
#   * residual_spectrum (per-k FFT magnitude of the residual)
experiment: decoder_analysis
out_dir:    results/decoder_analysis

base:
  p:                113
  k_freqs:          9
  n_steps:          100000
  weight_decay:     0.01
  diversity_weight: 0.01
  log_every:        500
  early_stop:       true
  save_model:       true

experiment_args:
  seeds: [42, 123, 456, 789, 999]
  max_extra_freqs: 20
"""


if __name__ == "__main__":
    # Convenience: dumping the YAML to stdout
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--print-yaml":
        print(_YAML_TEMPLATE)
