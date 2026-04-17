#!/usr/bin/env python3
"""
paper_extract.py — extract everything needed for the PAN paper revision
from the results/ directory.

Run from the pan_lab repo root:
    python paper_extract.py

Writes a single markdown file paper_extract.md that bundles:
    - Per-experiment stdout-style summary tables (from runs.csv)
    - Per-run CSV contents for the small ones (decoder_swap, sifp16_inference)
    - Mixing-matrix analysis across every saved .pt model:
        * slot_census: Clock-pattern detection per seed
        * freq_init_ablation: same, split by fourier vs random init
        * tier3: the single canonical model's mix matrix
    - Convergence statistics: how many distinct Fourier frequencies per seed,
      which frequencies, Clock-compliance score
    - Cross-seed aggregate: do seeds converge to the same basis?
    - Ablation summaries
    - Held-out-primes per-prime table

Output is one .md file, human-readable, no plots. Designed to be pasted
into a chat / doc without attachments.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── Constants ──────────────────────────────────────────────────────────────
TWO_PI           = 2.0 * np.pi
SIFP16_QUANT_ERR = TWO_PI / 65536
CONV_THRESHOLD   = 0.01        # angular-error cutoff for "converged"
CLOCK_TOL_RATIO  = 0.20        # |w_i|/|w_j| must be within this of 1.0
RESULTS_DIR      = Path("results")
OUT_PATH         = Path("paper_extract.md")


# ── Helpers ────────────────────────────────────────────────────────────────
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  ! could not read {path}: {e}", file=sys.stderr)
        return None


def _wrap_phase(f: float) -> float:
    return float(f % TWO_PI)


def _angular_error_to_basis(learned: float, p: int, k_max: int = 50
                             ) -> Tuple[int, float]:
    """
    Given a learned frequency, find the nearest theoretical k*2pi/P
    for k in [1..k_max] and return (k, angular_error_in_radians).
    k_max=50 is generous — Nanda's 5 active freqs were k <= 52 for P=113.
    """
    f = _wrap_phase(learned)
    ks     = np.arange(1, k_max + 1)
    targets = (ks * TWO_PI / p) % TWO_PI
    diffs   = np.abs(f - targets)
    diffs   = np.minimum(diffs, TWO_PI - diffs)
    j       = int(np.argmin(diffs))
    return int(ks[j]), float(diffs[j])


# ── Data structures ────────────────────────────────────────────────────────
@dataclass
class ClockReport:
    run_id:             str
    seed:               int
    p:                  int
    K:                  int
    grok_step:          Optional[int]
    grokked:            bool
    final_val_acc:      float
    active_slots_enc0:  List[int]     # slot indices (1-based) with err<thresh
    active_slots_enc1:  List[int]
    active_freqs_enc0:  List[int]     # nearest theoretical k per active slot
    active_freqs_enc1:  List[int]
    distinct_freqs:     List[int]     # union across encoders
    clock_pairs:        List[Tuple[int, int, int, float]]
        # (out_channel_idx, enc0_slot, enc1_slot, avg_weight_magnitude)
    clock_score:        float          # fraction of channels that look like Clock pairs
    notes:              List[str] = field(default_factory=list)


def analyze_model(pt_path: Path, p: int, seed: int, run_id: str,
                  grok_step: Optional[int], grokked: bool,
                  final_val_acc: float) -> ClockReport:
    ckpt = torch.load(str(pt_path), weights_only=False, map_location="cpu")
    sd = ckpt["state_dict"]
    K  = ckpt.get("config", {}).get("k_freqs")
    if K is None:
        K = int(sd["phase_mix.weight"].shape[0])

    # ── Active slot identification ─────────────────────────────────────
    def _active(enc_idx: int) -> Tuple[List[int], List[int]]:
        freqs = sd[f"encoders.{enc_idx}.freq"].cpu().numpy()
        slots, nearest_ks = [], []
        for slot, f_raw in enumerate(freqs):
            k, err = _angular_error_to_basis(f_raw, p)
            if err < CONV_THRESHOLD:
                slots.append(slot + 1)      # 1-indexed to match stdout
                nearest_ks.append(k)
        return slots, nearest_ks

    a0_slots, a0_freqs = _active(0)
    a1_slots, a1_freqs = _active(1)
    distinct = sorted(set(a0_freqs) | set(a1_freqs))

    # ── Mixing-matrix structure ────────────────────────────────────────
    W = sd["phase_mix.weight"].cpu().numpy()   # (K, 2K)
    clock_pairs: List[Tuple[int, int, int, float]] = []
    clock_compliant = 0
    for j in range(W.shape[0]):
        row  = W[j]
        top2 = np.argsort(np.abs(row))[::-1][:2]
        if len(top2) < 2:
            continue
        idx_a, idx_b = int(top2[0]), int(top2[1])
        w_a, w_b     = row[idx_a], row[idx_b]
        # Must come from different encoders (one < K, one >= K)
        different_encoders = (idx_a < K) != (idx_b < K)
        # Must have comparable magnitudes
        ratio = abs(w_b) / max(abs(w_a), 1e-9)
        balanced = (1.0 - CLOCK_TOL_RATIO) <= ratio <= (1.0 + CLOCK_TOL_RATIO)
        # Must be non-trivial magnitude
        nontrivial = abs(w_a) > 0.3
        if different_encoders and balanced and nontrivial:
            clock_compliant += 1
            enc0_slot = (idx_a if idx_a < K else idx_b) % K + 1
            enc1_slot = (idx_b if idx_b >= K else idx_a) % K + 1
            clock_pairs.append((j, enc0_slot, enc1_slot,
                                 float((abs(w_a) + abs(w_b)) / 2)))

    clock_score = clock_compliant / W.shape[0]

    return ClockReport(
        run_id=run_id, seed=seed, p=p, K=K,
        grok_step=grok_step, grokked=grokked, final_val_acc=final_val_acc,
        active_slots_enc0=a0_slots, active_slots_enc1=a1_slots,
        active_freqs_enc0=a0_freqs, active_freqs_enc1=a1_freqs,
        distinct_freqs=distinct,
        clock_pairs=clock_pairs, clock_score=clock_score,
    )


# ── Section writers ────────────────────────────────────────────────────────
def section_summary_table(runs: pd.DataFrame, group_by: str) -> str:
    """Reproduce the stdout summary table from ExperimentReporter."""
    if runs is None or runs.empty:
        return "_no runs.csv present_\n"
    if group_by not in runs.columns:
        return f"_column {group_by!r} not present_\n"
    grp = runs.groupby(group_by)
    rows = []
    for key, g in grp:
        grokked    = int(g["grokked"].sum())
        n          = len(g)
        grokked_g  = g[g["grok_step"].astype(float) >= 0]
        mean_gs    = grokked_g["grok_step"].mean() if len(grokked_g) else float("nan")
        mean_peak  = g["peak_val_acc"].mean()
        mean_final = g["final_val_acc"].mean()
        mean_elapsed = g["elapsed_s"].mean()
        params     = g["param_count"].mean()
        mode_c     = int(g.get("mode_collapsed", pd.Series([False]*len(g))).sum())
        rows.append({
            group_by:        key,
            "n_runs":        n,
            "n_grokked":     grokked,
            "grok_rate":     grokked / n,
            "mean_grok_step": (f"{mean_gs:,.0f}" if not np.isnan(mean_gs) else "—"),
            "mean_peak_acc":  f"{mean_peak:.4f}",
            "mean_final_acc": f"{mean_final:.4f}",
            "mean_elapsed_s": f"{mean_elapsed:.1f}",
            "params":         int(params),
            "mode_collapse":  mode_c,
        })
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def section_clock_table(reports: List[ClockReport]) -> str:
    rows = []
    for r in reports:
        rows.append({
            "run_id":           r.run_id,
            "seed":             r.seed,
            "grokked":          r.grokked,
            "grok_step":        r.grok_step if r.grok_step is not None else "—",
            "final_val_acc":    f"{r.final_val_acc:.4f}",
            "n_active_e0":      len(r.active_slots_enc0),
            "n_active_e1":      len(r.active_slots_enc1),
            "distinct_freqs":   str(r.distinct_freqs),
            "n_distinct":       len(r.distinct_freqs),
            "n_clock_channels": len(r.clock_pairs),
            "clock_score":      f"{r.clock_score:.2f}",
        })
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def section_cross_seed_aggregate(reports: List[ClockReport]) -> str:
    """How often does each theoretical k get used across seeds?"""
    if not reports:
        return "_no reports_\n"
    grokked = [r for r in reports if r.grokked]
    n = len(grokked)
    if n == 0:
        return "_no grokked runs in this experiment_\n"

    freq_counts: Dict[int, int] = {}
    for r in grokked:
        for k in r.distinct_freqs:
            freq_counts[k] = freq_counts.get(k, 0) + 1

    rows = []
    for k, cnt in sorted(freq_counts.items(), key=lambda x: -x[1]):
        rows.append({
            "theoretical_k": k,
            "appears_in_N_seeds": cnt,
            "fraction": f"{cnt/n:.2f}",
        })
    table = pd.DataFrame(rows).to_markdown(index=False)

    n_freqs  = [len(r.distinct_freqs) for r in grokked]
    clock_ss = [r.clock_score          for r in grokked]
    summary = (
        f"\n**Summary across {n} grokked runs:**\n\n"
        f"- Mean distinct Fourier frequencies per run: "
        f"{np.mean(n_freqs):.2f}  (min={min(n_freqs)}, max={max(n_freqs)})\n"
        f"- Mean Clock-compliance score: {np.mean(clock_ss):.2f}  "
        f"(min={min(clock_ss):.2f}, max={max(clock_ss):.2f})\n"
    )
    return table + "\n" + summary


# ── Main write-out ─────────────────────────────────────────────────────────
def main():
    lines: List[str] = []
    P = lambda *a: lines.extend([*a, ""])      # paragraph
    H = lambda lvl, s: lines.append(("#" * lvl) + " " + s)

    H(1, "PAN paper extraction")
    P(f"Generated from `{RESULTS_DIR.resolve()}`.")
    P(f"Convergence threshold: angular error < {CONV_THRESHOLD} rad.")
    P(f"Clock-pair criterion: top-2 mixing-weight sources are from "
      f"different encoders, magnitudes within ±{int(CLOCK_TOL_RATIO*100)}%.")

    # ───────────────────────────────────────────────────────────────────
    H(2, "1. Compare — PAN vs transformer head-to-head")
    comp = _safe_read_csv(RESULTS_DIR / "compare" / "runs.csv")
    if comp is not None:
        P(comp.to_markdown(index=False))
    abl = _safe_read_csv(RESULTS_DIR / "compare" / "ablations.csv")
    if abl is not None:
        H(3, "Ablations")
        piv = abl.pivot_table(index="run_id", columns="intervention",
                               values="val_acc")
        P(piv.to_markdown())

    # ───────────────────────────────────────────────────────────────────
    H(2, "2. Tier3 — single-run mechanistic equivalence")
    t3_runs = _safe_read_csv(RESULTS_DIR / "tier3" / "runs.csv")
    if t3_runs is not None:
        P(t3_runs.to_markdown(index=False))

    t3_pt = next((RESULTS_DIR / "tier3").glob("model_tier3-*.pt"), None)
    if t3_pt is not None and t3_runs is not None and len(t3_runs):
        row = t3_runs.iloc[0]
        rep = analyze_model(
            t3_pt, p=int(row["p"]), seed=int(row["seed"]),
            run_id=row["run_id"],
            grok_step=(int(row["grok_step"]) if row["grok_step"] >= 0 else None),
            grokked=bool(row["grokked"]),
            final_val_acc=float(row["final_val_acc"]),
        )
        H(3, "Clock-circuit analysis of the grokked Tier 3 model")
        P(f"- Active slots in encoder 0: {rep.active_slots_enc0} "
          f"→ theoretical k = {rep.active_freqs_enc0}")
        P(f"- Active slots in encoder 1: {rep.active_slots_enc1} "
          f"→ theoretical k = {rep.active_freqs_enc1}")
        P(f"- Distinct Fourier frequencies across encoders: "
          f"{rep.distinct_freqs}")
        P(f"- Clock-compliance score: **{rep.clock_score:.2f}** "
          f"({len(rep.clock_pairs)} of {rep.K} channels)")
        if rep.clock_pairs:
            P("Clock pairs (output channel, enc0 slot, enc1 slot, "
              "mean |weight|):")
            lines.append("```")
            for j, a, b, w in rep.clock_pairs:
                lines.append(f"  out[{j}]  enc0.slot{a}  enc1.slot{b}  w={w:.3f}")
            lines.append("```")
            lines.append("")

    t3_abl = _safe_read_csv(RESULTS_DIR / "tier3" / "ablations.csv")
    if t3_abl is not None:
        H(3, "Tier 3 ablations")
        P(t3_abl.to_markdown(index=False))

    # ───────────────────────────────────────────────────────────────────
    H(2, "3. Slot census — 20 seeds at K=9")
    sc_runs = _safe_read_csv(RESULTS_DIR / "slot_census" / "runs.csv")
    if sc_runs is not None:
        P(section_summary_table(sc_runs, "seed"))

    # Analyze every .pt in slot_census/
    sc_pts = sorted((RESULTS_DIR / "slot_census").glob("model_census-*.pt"))
    sc_reports: List[ClockReport] = []
    for pt_path in sc_pts:
        # Parse seed from filename
        name_seed = pt_path.stem.split("-")[1]      # "s0", "s13", ...
        seed = int(name_seed.lstrip("s"))
        row_matches = sc_runs[sc_runs["seed"] == seed] if sc_runs is not None else pd.DataFrame()
        if not len(row_matches):
            continue
        row = row_matches.iloc[0]
        rep = analyze_model(
            pt_path, p=int(row["p"]), seed=seed, run_id=row["run_id"],
            grok_step=(int(row["grok_step"]) if row["grok_step"] >= 0 else None),
            grokked=bool(row["grokked"]),
            final_val_acc=float(row["final_val_acc"]),
        )
        sc_reports.append(rep)

    if sc_reports:
        H(3, "Clock-pattern table, per seed")
        P(section_clock_table(sorted(sc_reports, key=lambda r: r.seed)))
        H(3, "Cross-seed frequency usage")
        P(section_cross_seed_aggregate(sc_reports))

    # ───────────────────────────────────────────────────────────────────
    H(2, "4. Frequency-init ablation — Fourier vs random")
    fi_runs = _safe_read_csv(RESULTS_DIR / "freq_init_ablation" / "runs.csv")
    if fi_runs is not None:
        P(section_summary_table(fi_runs, "freq_init"))

    fi_pts = sorted((RESULTS_DIR / "freq_init_ablation").glob("model_*.pt"))
    fi_reports: List[Tuple[str, ClockReport]] = []   # (init, report)
    for pt_path in fi_pts:
        name = pt_path.stem.replace("model_", "")  # "fourier-s42-..."
        init = "fourier" if name.startswith("fourier") else "random"
        seed = int(name.split("-")[1].lstrip("s"))
        rm = fi_runs[(fi_runs["seed"] == seed) &
                     (fi_runs["freq_init"] == init)] if fi_runs is not None else pd.DataFrame()
        if not len(rm):
            continue
        row = rm.iloc[0]
        rep = analyze_model(
            pt_path, p=int(row["p"]), seed=seed, run_id=row["run_id"],
            grok_step=(int(row["grok_step"]) if row["grok_step"] >= 0 else None),
            grokked=bool(row["grokked"]),
            final_val_acc=float(row["final_val_acc"]),
        )
        fi_reports.append((init, rep))

    for init in ("fourier", "random"):
        sub = [r for i, r in fi_reports if i == init]
        if sub:
            H(3, f"Clock-pattern table — freq_init={init}")
            P(section_clock_table(sorted(sub, key=lambda r: r.seed)))
            H(4, f"Cross-seed frequency usage — freq_init={init}")
            P(section_cross_seed_aggregate(sub))

    # ───────────────────────────────────────────────────────────────────
    H(2, "5. Decoder swap — canonical-circuit test")
    ds = _safe_read_csv(RESULTS_DIR / "decoder_swap" / "decoder_swap.csv")
    if ds is not None:
        P(ds.to_markdown(index=False))

    # ───────────────────────────────────────────────────────────────────
    H(2, "6. SIFP-16 inference quantization")
    si = _safe_read_csv(RESULTS_DIR / "sifp16_inference" / "quant_eval.csv")
    if si is not None:
        P(si.to_markdown(index=False))

    # ───────────────────────────────────────────────────────────────────
    H(2, "7. K=8 anomaly — 10 seeds")
    k8_runs = _safe_read_csv(RESULTS_DIR / "k8_sweep" / "runs.csv")
    if k8_runs is not None:
        P(section_summary_table(k8_runs, "seed"))
        grokked = int(k8_runs["grokked"].sum())
        n       = len(k8_runs)
        P(f"**Headline: {grokked}/{n} seeds grokked at K=8.** "
          f"This is the key stat for Section 4 of the paper.")

    # ───────────────────────────────────────────────────────────────────
    H(2, "8. Held-out primes — reviewer robustness")
    hp = _safe_read_csv(RESULTS_DIR / "held_out_primes" / "runs.csv")
    if hp is not None:
        P(section_summary_table(hp, "p"))


    # ───────────────────────────────────────────────────────────────────
    H(2, "9. Decoder analysis — what the learned decoder contains")
    da_summary = _safe_read_csv(RESULTS_DIR / "decoder_analysis" / "decoder_analysis.csv")
    if da_summary is not None and not da_summary.empty:
        H(3, "Summary per seed")
        P(da_summary.to_markdown(index=False))

        grokked = da_summary[da_summary["grokked"] == True] if "grokked" in da_summary.columns else da_summary
        if len(grokked):
            import numpy as _np
            mean_expl = float(grokked["clock_explained_frac"].mean())
            mean_gap  = float(grokked["gap_clock"].mean())
            n_needed  = grokked["n_extras_for_1pct"].to_list()
            P(f"**Aggregate across {len(grokked)} grokked runs:**")
            P(f"- Mean Clock-explained energy fraction: **{mean_expl:.3f}**")
            P(f"- Mean accuracy gap (learned - Clock-only): **{mean_gap:.3f}**")
            P(f"- Extra frequencies needed to close the 1% gap: {n_needed}")

    da_recovery = _safe_read_csv(
        RESULTS_DIR / "decoder_analysis" / "decoder_recovery_curve.csv")
    if da_recovery is not None and not da_recovery.empty:
        H(3, "Recovery curve - val_acc vs basis-expansion depth")
        pivot = da_recovery.pivot_table(
            index="n_extras", columns="seed", values="val_acc")
        P(pivot.to_markdown())

    da_spec = _safe_read_csv(
        RESULTS_DIR / "decoder_analysis" / "decoder_residual_spectrum.csv")
    if da_spec is not None and not da_spec.empty:
        H(3, "Residual spectrum - top-5 unexplained FFT bins per seed")
        rows = []
        for seed in sorted(da_spec["seed"].unique()):
            sub = da_spec[da_spec["seed"] == seed].nlargest(5, "magnitude")
            rows.append({
                "seed":     seed,
                "top_ks":   ", ".join(str(int(k)) for k in sub["k"]),
                "top_mags": ", ".join(f"{m:.3f}" for m in sub["magnitude"]),
            })
        P(pd.DataFrame(rows).to_markdown(index=False))

    # ───────────────────────────────────────────────────────────────────
    H(2, "10. Manifest cross-check")
    for exp in sorted(os.listdir(RESULTS_DIR)):
        expdir = RESULTS_DIR / exp
        if not expdir.is_dir():
            continue
        mani = expdir / "manifest.json"
        if mani.exists():
            with open(mani) as f:
                m = json.load(f)
            prov = m.get("provenance", {})
            P(f"- **{exp}**: n_runs={m.get('n_runs','?')}  "
              f"torch={prov.get('torch','?')}  "
              f"device={prov.get('device','?')}  "
              f"git_sha={prov.get('git_sha','?')}  "
              f"timestamp={prov.get('timestamp','?')}")
        else:
            P(f"- **{exp}**: no manifest (likely did not run)")

    # ───────────────────────────────────────────────────────────────────
    out = "\n".join(lines) + "\n"
    OUT_PATH.write_text(out)
    sz_kb = OUT_PATH.stat().st_size / 1024
    print(f"✓ wrote {OUT_PATH}  ({sz_kb:.1f} KB, {len(lines):,} lines)")
    print(f"  tier3 models analyzed:        {1 if t3_pt else 0}")
    print(f"  slot_census models analyzed:  {len(sc_reports)}")
    print(f"  freq_init models analyzed:    {len(fi_reports)}")


if __name__ == "__main__":
    main()
