from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from pan_lab.config import DEVICE, RunConfig, TWO_PI
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs


def _channel_effective_frequency(W_mix: np.ndarray, enc0_freq: np.ndarray, enc1_freq: np.ndarray) -> np.ndarray:
    K = W_mix.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for j in range(K):
        abs_row = np.abs(W_mix[j])
        top_e0 = int(np.argmax(abs_row[:K]))
        top_e1 = int(np.argmax(abs_row[K:]))
        f0 = float(enc0_freq[top_e0] % TWO_PI)
        f1 = float(enc1_freq[top_e1] % TWO_PI)
        s = (np.sin(f0) + np.sin(f1)) / 2
        c = (np.cos(f0) + np.cos(f1)) / 2
        out[j] = float(np.arctan2(s, c) % TWO_PI)
    return out


def _build_clock_basis(p: int, f_channels: np.ndarray, phi_ref: np.ndarray) -> np.ndarray:
    P = p
    K = len(f_channels)
    c = np.arange(P, dtype=np.float64).reshape(P, 1)
    f = f_channels.reshape(1, K)
    phi = phi_ref.reshape(1, K)
    arg = f * c - phi
    return np.concatenate([np.cos(arg), np.sin(arg)], axis=1)


def _build_harmonic_basis(p: int, f_channels: np.ndarray, phi_ref: np.ndarray, harmonic_order: int = 4) -> np.ndarray:
    P = p
    K = len(f_channels)
    c = np.arange(P, dtype=np.float64).reshape(P, 1)
    cols = []
    for h in range(1, harmonic_order + 1):
        f = (h * f_channels).reshape(1, K)
        phi = phi_ref.reshape(1, K)
        arg = f * c - phi
        cols.append(np.cos(arg))
        cols.append(np.sin(arg))
    return np.concatenate(cols, axis=1)


def _project_onto_basis(W: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha, *_ = np.linalg.lstsq(B, W, rcond=None)
    W_proj = B @ alpha
    residual = W - W_proj
    return W_proj, residual


def _residual_spectrum(residual: np.ndarray, p: int) -> dict[str, np.ndarray]:
    F = np.fft.fft(residual, axis=0)
    mag = np.abs(F).mean(axis=1)
    return {"k": np.arange(p), "magnitude": mag}


def _evaluate_decoder(pan, W_new: np.ndarray, val_x: torch.Tensor, val_y: torch.Tensor) -> float:
    saved_w = pan.decoder.weight.data.clone()
    saved_b = pan.decoder.bias.data.clone()
    with torch.no_grad():
        pan.decoder.weight.data = torch.tensor(W_new, dtype=torch.float32, device=DEVICE)
        pan.decoder.bias.data.zero_()
        logits = pan(val_x)
        acc = float((logits.argmax(-1) == val_y).float().mean().item())
        pan.decoder.weight.data.copy_(saved_w)
        pan.decoder.bias.data.copy_(saved_b)
    return acc


def analyze_harmonics(pan_model, val_x: torch.Tensor, val_y: torch.Tensor, harmonic_order: int = 4) -> dict:
    with torch.no_grad():
        W_mix = pan_model.phase_mix.weight.detach().cpu().numpy()
        f0 = pan_model.encoders[0].freq.detach().cpu().numpy()
        f1 = pan_model.encoders[1].freq.detach().cpu().numpy()
        phi = pan_model.phase_gate.ref_phase.detach().cpu().numpy() % TWO_PI
        W_learned = pan_model.decoder.weight.detach().cpu().numpy()

    f_eff = _channel_effective_frequency(W_mix, f0, f1)
    out = {
        "f_eff": f_eff.tolist(),
        "harmonic_order": harmonic_order,
        "baseline_learned": None,
        "clock_only": None,
        "harmonic_fits": {},
    }

    with torch.no_grad():
        out["baseline_learned"] = float((pan_model(val_x).argmax(-1) == val_y).float().mean().item())

    B1 = _build_clock_basis(pan_model.p, f_eff, phi)
    W_fit, resid = _project_onto_basis(W_learned, B1)
    acc = _evaluate_decoder(pan_model, W_fit, val_x, val_y)
    explained = 1 - (resid**2).sum() / (W_learned**2).sum()
    out["clock_only"] = {"acc": acc, "explained_frac": float(explained), "n_basis_cols": B1.shape[1]}

    for H in range(2, harmonic_order + 1):
        BH = _build_harmonic_basis(pan_model.p, f_eff, phi, harmonic_order=H)
        W_fit, resid = _project_onto_basis(W_learned, BH)
        acc = _evaluate_decoder(pan_model, W_fit, val_x, val_y)
        explained = 1 - (resid**2).sum() / (W_learned**2).sum()
        out["harmonic_fits"][f"H={H}"] = {
            "acc": acc,
            "explained_frac": float(explained),
            "n_basis_cols": BH.shape[1],
        }
    return out


def analyze_gate_space_upper_bound(pan_model, val_x, val_y, p: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    P = p
    device = next(pan_model.parameters()).device
    a_all = torch.arange(P, device=device).repeat_interleave(P)
    b_all = torch.arange(P, device=device).repeat(P)
    x_all = torch.stack([a_all, b_all], dim=1)
    y_all = (a_all + b_all) % P

    with torch.no_grad():
        phases = [enc(x_all[:, i]) for i, enc in enumerate(pan_model.encoders)]
        concat = torch.cat(phases, dim=-1)
        mixed = pan_model.phase_mix(concat)
        gates = pan_model.phase_gate(mixed)

    G = gates.cpu().numpy()
    y = y_all.cpu().numpy()

    T = np.eye(P)[y]
    W_ols, *_ = np.linalg.lstsq(G, T, rcond=None)
    ols_acc = (np.argmax(G @ W_ols, axis=1) == y).mean()

    lr = LogisticRegression(solver="lbfgs", max_iter=5000, C=1e6, fit_intercept=True)
    lr.fit(G, y)
    lr_acc = lr.score(G, y)

    with torch.no_grad():
        learned_acc = float((pan_model(val_x).argmax(-1) == val_y).float().mean().item())

    return {
        "gate_optimal_acc": float(lr_acc),
        "gate_ols_acc": float(ols_acc),
        "learned_acc": float(learned_acc),
        "gap_from_optimal": float(learned_acc - lr_acc),
    }


class DecoderAnalysisExperiment(BaseExperiment):
    name = "decoder_analysis"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456]
        return build_pan_seed_cfgs(base, seeds, label_prefix="danal", save_model=True)

    def init_state(self, **kwargs):
        return {
            "max_extra_freqs": kwargs["exp_args"].get("max_extra_freqs", 20),
            "summary_rows": [],
            "recovery_rows": [],
            "spectrum_rows": [],
        }

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        super().handle_result(reporter, result, vx, vy, cfg, state)
        pan = result.model

        with torch.no_grad():
            acc_learned = float((pan(vx).argmax(-1) == vy).float().mean().item())
            W_learned = pan.decoder.weight.detach().cpu().numpy()

        if acc_learned < 0.95:
            print(f"  [{cfg.display_id()}] skipped — not grokked (learned={acc_learned:.3f})")
            state["summary_rows"].append(
                {"run_id": cfg.display_id(), "seed": cfg.seed, "grokked": False, "acc_learned": acc_learned}
            )
            return

        W_mix = pan.phase_mix.weight.detach().cpu().numpy()
        f0 = pan.encoders[0].freq.detach().cpu().numpy()
        f1 = pan.encoders[1].freq.detach().cpu().numpy()
        phi = pan.phase_gate.ref_phase.detach().cpu().numpy() % TWO_PI
        f_eff = _channel_effective_frequency(W_mix, f0, f1)
        P = cfg.p

        B_clock = _build_clock_basis(P, f_eff, phi)
        W_proj, residual = _project_onto_basis(W_learned, B_clock)
        acc_clock = _evaluate_decoder(pan, W_proj, vx, vy)

        energy_total = float((W_learned**2).sum())
        energy_residual = float((residual**2).sum())
        explained_frac = 1.0 - energy_residual / max(energy_total, 1e-12)

        spec = _residual_spectrum(residual, P)
        for k, mag in zip(spec["k"], spec["magnitude"]):
            state["spectrum_rows"].append(
                {"run_id": cfg.display_id(), "seed": cfg.seed, "k": int(k), "magnitude": float(mag)}
            )

        mag = spec["magnitude"].copy()
        mag[0] = 0.0
        for fj in f_eff:
            nearest_k = int(round(fj * P / TWO_PI)) % P
            mag[nearest_k] = 0.0
            conj = (-nearest_k) % P
            mag[conj] = 0.0

        harm = analyze_harmonics(pan, vx, vy, harmonic_order=4)
        gate_ub = analyze_gate_space_upper_bound(pan, vx, vy, p=cfg.p)

        candidate_order = np.argsort(mag)[::-1]
        candidate_order = [int(k) for k in candidate_order if 0 < int(k) <= P // 2]

        state["recovery_rows"].append(
            {
                "run_id": cfg.display_id(),
                "seed": cfg.seed,
                "n_extras": 0,
                "added_ks": "",
                "val_acc": acc_clock,
                "explained": explained_frac,
            }
        )

        added_ks = []
        current_basis = B_clock.copy()
        n_extras_needed = None
        for step in range(1, min(state["max_extra_freqs"], len(candidate_order)) + 1):
            k_new = candidate_order[step - 1]
            added_ks.append(k_new)
            freq_new = TWO_PI * k_new / P
            c = np.arange(P, dtype=np.float64).reshape(P, 1)
            cos_col = np.cos(freq_new * c)
            sin_col = np.sin(freq_new * c)
            current_basis = np.concatenate([current_basis, cos_col, sin_col], axis=1)

            W_fit, _ = _project_onto_basis(W_learned, current_basis)
            acc_now = _evaluate_decoder(pan, W_fit, vx, vy)
            energy_resid_now = float(((W_learned - W_fit) ** 2).sum())
            explained_now = 1.0 - energy_resid_now / max(energy_total, 1e-12)

            state["recovery_rows"].append(
                {
                    "run_id": cfg.display_id(),
                    "seed": cfg.seed,
                    "n_extras": step,
                    "added_ks": ",".join(str(k) for k in added_ks),
                    "val_acc": acc_now,
                    "explained": explained_now,
                }
            )

            if n_extras_needed is None and acc_now >= acc_learned - 0.01:
                n_extras_needed = step
                if step >= n_extras_needed + 3:
                    break

        state["summary_rows"].append(
            {
                "run_id": cfg.display_id(),
                "seed": cfg.seed,
                "grokked": True,
                "acc_learned": acc_learned,
                "acc_clock_only": acc_clock,
                "gap_clock": acc_learned - acc_clock,
                "clock_explained_frac": explained_frac,
                "n_extras_for_1pct": n_extras_needed if n_extras_needed else -1,
                "first_3_extras": ",".join(str(k) for k in candidate_order[:3]),
                "f_eff_channels": ",".join(f"{f:.4f}" for f in f_eff),
                "harmonic_H2_acc": harm["harmonic_fits"].get("H=2", {}).get("acc"),
                "harmonic_H3_acc": harm["harmonic_fits"].get("H=3", {}).get("acc"),
                "harmonic_H4_acc": harm["harmonic_fits"].get("H=4", {}).get("acc"),
                "gate_optimal_acc": gate_ub["gate_optimal_acc"],
            }
        )

    def finalize(self, reporter, state, out_dir):
        summary_df = pd.DataFrame(state["summary_rows"])
        recovery_df = pd.DataFrame(state["recovery_rows"])
        spectrum_df = pd.DataFrame(state["spectrum_rows"])

        summary_df.to_csv(os.path.join(out_dir, "decoder_analysis.csv"), index=False)
        recovery_df.to_csv(os.path.join(out_dir, "decoder_recovery_curve.csv"), index=False)
        spectrum_df.to_csv(os.path.join(out_dir, "decoder_residual_spectrum.csv"), index=False)

        print("\n── Decoder analysis summary ──")
        if len(summary_df):
            print(summary_df.to_string(index=False))
