#!/usr/bin/env python3
"""
paper_v5_figures.py — generate the 8 seaborn evidence figures that back
the claims in docs/drafts/05_pan_paper_v5.md.

Run from the repo root:
    uv run --group dev python scripts/paper_v5_figures.py

Reads CSVs from data/20260418_paper_results/, writes PNGs to
docs/drafts/figs_v5/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_ROOT = Path("data/20260418_paper_results")
OUT_DIR = Path("docs/drafts/figs_v5")
C_PAN = "#c8322f"
C_TF = "#325a89"
C_FOURIER = "#5c7aa3"
C_RANDOM = "#c4743b"
PALETTE_INIT = {"fourier": C_FOURIER, "random": C_RANDOM}


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 150
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ─── F1 ────────────────────────────────────────────────────────────────────
def fig_param_efficiency() -> None:
    df = pd.read_csv(DATA_ROOT / "compare" / "runs.csv")
    df = df.assign(model=df["model_kind"].map({"pan": "PAN (K=9)", "transformer": "Transformer"}))
    order = ["PAN (K=9)", "Transformer"]
    palette = {"PAN (K=9)": C_PAN, "Transformer": C_TF}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))

    panels = [
        (axes[0], "param_count",  "Parameter count (log)", "params",  "log",  lambda v: f"{int(v):,}"),
        (axes[1], "elapsed_s",    "Wall-clock to grok",    "seconds", None,   lambda v: f"{v:.1f}s"),
        (axes[2], "grok_step",    "Grok step",             "step",    None,   lambda v: f"{int(v):,}"),
        (axes[3], "peak_val_acc", "Peak val_acc",          "val_acc", None,   lambda v: f"{v*100:.2f}%"),
    ]
    for ax, col, title, ylabel, scale, fmt in panels:
        sns.barplot(df, x="model", y=col, hue="model", order=order, hue_order=order,
                    palette=palette, legend=False, ax=ax)
        if scale:
            ax.set_yscale(scale)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        for i, v in enumerate(df.set_index("model").loc[order, col]):
            ax.text(i, v * 1.02 if scale == "log" else v, fmt(v),
                    ha="center", va="bottom", fontsize=9)
    axes[3].set_ylim(0.95, 1.0)

    fig.suptitle("§3.1  PAN vs 1-layer transformer on modular addition mod 113 (seed=42)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s31_param_efficiency.png", bbox_inches="tight")
    plt.close(fig)


# ─── F2 ────────────────────────────────────────────────────────────────────
def fig_ablations() -> None:
    df = pd.read_csv(DATA_ROOT / "tier3" / "ablations.csv").copy()
    label_map = {
        "baseline": "baseline",
        "zero_phase_mixing": "zero phase mixing",
        "randomize_frequencies": "randomize frequencies",
        "zero_ref_phases": "zero reference phases",
    }
    df["label"] = df["intervention"].map(label_map)
    df = df.sort_values("val_acc", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    sns.barplot(df, x="val_acc", y="label", color=C_PAN, ax=ax)
    ax.axvline(1.0, ls="--", color="gray", lw=1, alpha=0.6, label="baseline = 1.00")
    for i, v in enumerate(df["val_acc"]):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("val_acc after ablation")
    ax.set_ylabel("")
    ax.set_title("§3.2  Ablations: no single component is redundant (P=113, K=9, seed=42)")
    ax.legend(loc="upper right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s32_ablations.png", bbox_inches="tight")
    plt.close(fig)


# ─── F3 ────────────────────────────────────────────────────────────────────
def fig_decoder_fourier_dynamics() -> None:
    curves = pd.read_csv(DATA_ROOT / "tier3" / "curves.csv")
    metrics = pd.read_csv(DATA_ROOT / "tier3" / "metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(curves["step"], curves["val_acc"], label="val_acc", color="black", lw=1.8)
    ax.plot(metrics["step"], metrics["decoder_fourier_peak_mean"],
            label="decoder_fourier_peak_mean  (M8)", color=C_PAN, lw=1.4)
    ax.plot(metrics["step"], metrics["clock_compliance"],
            label="clock_compliance  (M2)", color="#2b7a3e", lw=1.4)
    gla = metrics.dropna(subset=["gate_linear_acc"])
    ax.plot(gla["step"], gla["gate_linear_acc"],
            label="gate_linear_acc  (M6)", color=C_TF, lw=1.4, marker="o", ms=3)

    grok_step = 16_500
    peak_step = 2_500
    ax.axvline(peak_step, color=C_PAN, ls=":", lw=1.2, alpha=0.7)
    ax.axvline(grok_step, color="black", ls="--", lw=1.2, alpha=0.7)
    ax.text(peak_step * 1.08, 1.09, "M8 peak 0.87 @ step 2,500",
            fontsize=8, color=C_PAN, va="bottom", ha="left")
    ax.text(grok_step * 1.08, 1.09, "grok @ step 16,500",
            fontsize=8, color="black", va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("metric value")
    ax.set_ylim(-0.02, 1.2)
    ax.set_xlim(10, 1.1e5)
    ax.set_title("§3.2 / §4.1  seed=42 dynamics: decoder Fourier structure peaks early, then decays",
                 pad=14)
    ax.legend(loc="center left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s32_decoder_fourier_dynamics.png", bbox_inches="tight")
    plt.close(fig)


# ─── F4 ────────────────────────────────────────────────────────────────────
def fig_slot_census() -> None:
    runs = pd.read_csv(DATA_ROOT / "k_census_n20_fourier" / "runs.csv")
    metrics = pd.read_csv(DATA_ROOT / "k_census_n20_fourier" / "metrics.csv")
    k9_grok = runs[(runs.k_freqs == 9) & runs.grokked]
    last = metrics.sort_values("step").groupby("run_id").tail(1)
    last = last[last.run_id.isin(k9_grok.run_id)].copy()
    assert len(last) == 14, f"expected 14 grokked K=9 seeds, got {len(last)}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    sns.violinplot(y=last["clock_compliance"], ax=axes[0], color=C_FOURIER,
                   inner=None, cut=0)
    sns.stripplot(y=last["clock_compliance"], ax=axes[0], color="black", size=5, alpha=0.75)
    axes[0].axhline(last["clock_compliance"].mean(), ls="--", color="red", lw=1.2,
                    label=f"mean = {last['clock_compliance'].mean():.2f}")
    axes[0].set_ylabel("clock_compliance")
    axes[0].set_xlabel("14 grokked K=9 Fourier-init seeds")
    axes[0].set_ylim(0, 1.05)
    n_perfect = int((last["clock_compliance"] == 1.0).sum())
    axes[0].set_title(f"(a) Clock-pair compliance\n{n_perfect}/14 at compliance = 1.00")
    axes[0].legend(loc="lower left")

    afc = last["active_freq_count"].astype(int)
    sns.countplot(x=afc, ax=axes[1], color=C_FOURIER, order=sorted(afc.unique()))
    axes[1].set_xlabel("active_freq_count")
    axes[1].set_ylabel("seeds")
    axes[1].set_title(f"(b) Distinct active Fourier frequencies\nmean = {afc.mean():.2f}")

    fig.suptitle("§3.3  n=20 slot census at P=113, K=9, Fourier init", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s33_slot_census.png", bbox_inches="tight")
    plt.close(fig)


# ─── F5 ────────────────────────────────────────────────────────────────────
def fig_k_sweep_reliability() -> None:
    fourier = pd.read_csv(DATA_ROOT / "k_census_n20_fourier" / "runs.csv")
    random = pd.read_csv(DATA_ROOT / "k_census_n20_random" / "runs.csv")
    k13_f = pd.read_csv(DATA_ROOT / "paper_k13_fourier" / "runs.csv")
    k13_r = pd.read_csv(DATA_ROOT / "paper_k13_random" / "runs.csv")
    k5_ext = pd.read_csv(DATA_ROOT / "paper_k5_extended" / "runs.csv")

    rows = []
    # Fourier K=6..12 from census
    for k, grp in fourier.groupby("k_freqs"):
        if 6 <= k <= 12:
            rows.append(("fourier", int(k), int(grp.grokked.sum()), len(grp)))
    # Fourier K=13 — authoritative from paper_k13_fourier
    rows.append(("fourier", 13, int(k13_f.grokked.sum()), len(k13_f)))

    # Random K=1..4, 6..12, 14 from census; K=5 pooled; K=13 from paper_k13_random
    for k, grp in random.groupby("k_freqs"):
        if k in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14):
            rows.append(("random", int(k), int(grp.grokked.sum()), len(grp)))
    # K=5 random pool: census + extended
    k5_cen = random[random.k_freqs == 5]
    pooled_grok = int(k5_cen.grokked.sum()) + int(k5_ext.grokked.sum())
    pooled_n = len(k5_cen) + len(k5_ext)
    rows.append(("random", 5, pooled_grok, pooled_n))
    # Random K=13 authoritative from paper_k13_random
    rows.append(("random", 13, int(k13_r.grokked.sum()), len(k13_r)))

    df = pd.DataFrame(rows, columns=["init", "K", "grok", "n"]).sort_values(["init", "K"])
    df["rate"] = df.grok / df.n
    df[["lo", "hi"]] = df.apply(lambda r: pd.Series(_wilson_ci(r.grok, r.n)), axis=1)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.axvspan(0.5, 4.5, color="#fbe2dc", alpha=0.55, label="insufficient (K ≤ 4)")
    ax.axvspan(4.5, 10.5, color="#fff3c4", alpha=0.55, label="transition (K = 5–10)")
    ax.axvspan(10.5, 15.5, color="#dfeedc", alpha=0.55, label="plateau (K ≥ 11)")

    for init in ("fourier", "random"):
        sub = df[df.init == init]
        ax.errorbar(sub.K, sub.rate,
                    yerr=[sub.rate - sub.lo, sub.hi - sub.rate],
                    fmt="o-", capsize=3, lw=1.6, ms=6,
                    color=PALETTE_INIT[init], label=f"{init} init")
        for _, r in sub.iterrows():
            ax.text(r.K, r.rate + 0.035, f"{int(r.grok)}/{int(r.n)}",
                    ha="center", va="bottom", fontsize=7, color=PALETTE_INIT[init])

    ax.set_xlabel("K  (phase slots)")
    ax.set_ylabel("grok rate")
    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(-0.03, 1.15)
    ax.set_xticks(range(1, 15))
    ax.set_title("§3.4 / §3.8  K-sweep reliability — three regimes at P=113\n"
                 "error bars = Wilson 95% CI; labels = (grokked / n)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s34_k_sweep_reliability.png", bbox_inches="tight")
    plt.close(fig)


# ─── F6 ────────────────────────────────────────────────────────────────────
def fig_cross_prime() -> None:
    rand = pd.read_csv(DATA_ROOT / "paper_cross_primes" / "runs.csv")
    four = pd.read_csv(DATA_ROOT / "primes_primary_k" / "runs.csv")
    rand = rand[~rand.label.str.startswith("p97-long")].copy()
    four = four[~four.label.str.startswith("p97-long")].copy()
    rand["init"] = "random"
    four["init"] = "fourier"
    df = pd.concat([four, rand], ignore_index=True)

    rel = (df.groupby(["p", "init"]).agg(grok=("grokked", "sum"), n=("grokked", "size"))
             .reset_index())
    rel["rate"] = rel.grok / rel.n

    gtimes = df[df.grokked].groupby(["p", "init"])["grok_step"].median().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    sns.barplot(rel, x="p", y="rate", hue="init",
                palette=PALETTE_INIT, ax=axes[0],
                hue_order=["fourier", "random"])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("grok rate (3 seeds per prime)")
    axes[0].set_xlabel("P")
    axes[0].set_title("(a) Grok rate per prime, K=10")
    for p, row in zip(axes[0].patches,
                      rel.sort_values(["init", "p"]).itertuples()):
        axes[0].text(p.get_x() + p.get_width() / 2, p.get_height() + 0.02,
                     f"{int(row.grok)}/{int(row.n)}", ha="center", va="bottom",
                     fontsize=7)
    axes[0].legend(title="init")

    sns.barplot(gtimes, x="p", y="grok_step", hue="init",
                palette=PALETTE_INIT, ax=axes[1],
                hue_order=["fourier", "random"])
    axes[1].set_ylabel("median grok_step  (grokked seeds only)")
    axes[1].set_xlabel("P")
    axes[1].set_title("(b) Median grok step per prime")
    axes[1].legend(title="init")

    fig.suptitle("§3.5  Cross-prime generalization at K=10  "
                 "(random: 24/24 grok; Fourier: 20/24 grok)", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s35_cross_prime.png", bbox_inches="tight")
    plt.close(fig)


# ─── F7 ────────────────────────────────────────────────────────────────────
def fig_decoder_comparison() -> None:
    da = pd.read_csv(DATA_ROOT / "decoder_analysis" / "decoder_analysis.csv")
    sw = pd.read_csv(DATA_ROOT / "decoder_swap" / "decoder_swap.csv")
    da = da[da.grokked].copy()  # seeds 42, 123, 789
    sw = sw.rename(columns={"val_acc_fourier_decoder": "acc_fourier_swap"})

    rows = []
    for _, r in da.iterrows():
        rows.append((r.seed, "learned decoder", r.acc_learned))
        rows.append((r.seed, "clock-only", r.acc_clock_only))
        rows.append((r.seed, "gate-optimal linear", r.gate_optimal_acc))
    for _, r in sw.iterrows():
        # only include Fourier-swap rows for seeds that also appear in `da`
        if r.seed in set(da.seed):
            rows.append((r.seed, "Fourier-swap decoder", r.acc_fourier_swap))

    df = pd.DataFrame(rows, columns=["seed", "decoder", "val_acc"])
    order = ["clock-only", "Fourier-swap decoder", "learned decoder", "gate-optimal linear"]
    palette = {
        "clock-only": "#b55a00",
        "Fourier-swap decoder": "#d9a73a",
        "learned decoder": C_PAN,
        "gate-optimal linear": "#2b7a3e",
    }

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.barplot(df, x="seed", y="val_acc", hue="decoder",
                order=sorted(df.seed.unique()), hue_order=order,
                palette=palette, ax=ax)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("seed (P=113, K=9, grokked)")
    ax.set_ylabel("val_acc")
    ax.set_title("§3.6 / §4.1  Decoder variants on three grokked seeds\n"
                 "learned & gate-optimal ≈ 0.99 ; Fourier-swap & clock-only collapse")
    ax.legend(title="decoder", loc="upper center", fontsize=8,
              ncols=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s36_decoder_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ─── F8 ────────────────────────────────────────────────────────────────────
def fig_sfp16_quant() -> None:
    df = pd.read_csv(DATA_ROOT / "sifp16_inference" / "quant_eval.csv").sort_values("seed")
    long = df.melt(id_vars=["seed", "delta"],
                   value_vars=["val_acc_fp32", "val_acc_sifp16"],
                   var_name="precision", value_name="val_acc")
    long["precision"] = long["precision"].map({"val_acc_fp32": "fp32",
                                                "val_acc_sifp16": "SFP-16"})
    seed_order = list(df["seed"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    palette = {"fp32": C_TF, "SFP-16": C_PAN}
    sns.barplot(long, x="seed", y="val_acc", hue="precision",
                order=seed_order, hue_order=["fp32", "SFP-16"],
                palette=palette, ax=ax)
    ax.set_ylim(0.8, 1.06)
    ax.set_ylabel("val_acc")
    ax.set_xlabel("seed (P=113, K=9)")
    ax.set_title("§3.7  SFP-16 phase quantization at inference  "
                 "(Δ annotated = SFP-16 − fp32)")
    for _, r in long.iterrows():
        x = seed_order.index(r.seed) + (-0.2 if r.precision == "fp32" else 0.2)
        ax.text(x, r.val_acc + 0.002, f"{r.val_acc*100:.2f}%",
                ha="center", va="bottom", fontsize=7)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, 1.04, f"Δ = {r.delta:+.3f}", ha="center", fontsize=8,
                color="dimgray")
    ax.legend(title="precision", loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_s37_sfp16_quant.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup()
    fig_param_efficiency()
    fig_ablations()
    fig_decoder_fourier_dynamics()
    fig_slot_census()
    fig_k_sweep_reliability()
    fig_cross_prime()
    fig_decoder_comparison()
    fig_sfp16_quant()
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
