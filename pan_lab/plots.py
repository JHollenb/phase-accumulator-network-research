"""
pan_lab.plots — publication-quality figures driven by the CSVs.

Design rule: every plot function takes a DataFrame (usually loaded from
runs.csv / curves.csv / checkpoints.csv) and a target path. Nothing here
depends on torch or in-memory model state — so plots can be regenerated
from an old experiment's CSVs without retraining.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pan_lab.config import SIFP16_QUANT_ERR, TWO_PI


# Default metrics to show in formation-curve / spectra / peak plots.
# Order matters — the grid panels are laid out in this order.
DEFAULT_FORMATION_METRICS: List[str] = [
    "enc0_snap_mean",                 # M1 encoder frequency snap
    "clock_compliance",               # M2
    "mix_row_eff_n_mean",             # M4 sparsification
    "active_freq_count",              # M5
    "decoder_fourier_peak_mean",      # M8 decoder clock projection
    "gate_linear_acc",                # M6 linear decodability
    "gate_decoder_gap",               # derived: M6 − M7.fp32_acc
    "sifp16_acc",                     # M7 quantized robustness
]

# ─────────────────────────────────────────────────────────────────────────────
# Colors kept consistent across figures. Avoid heavy styling — one
# accent per model kind, neutral grid, low-saturation.
C_PAN      = "#c8322f"
C_TF       = "#325a89"
C_CONVERGED = "#2b7a3e"
C_COLLAPSE = "#b55a00"
C_GRID     = "#d4d4d4"


# ─────────────────────────────────────────────────────────────────────────────
def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    curves_df: pd.DataFrame,
    runs_df:   pd.DataFrame,
    out_path:  str,
    title:     str = "Training curves",
    max_lines: int = 20,
) -> None:
    """
    One panel per metric (val_acc, val_loss), one line per run.
    Dotted vertical marker at each run's grokking step.
    """
    _ensure(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    run_ids = curves_df["run_id"].unique()
    if len(run_ids) > max_lines:
        run_ids = run_ids[:max_lines]

    grok_lookup = dict(zip(runs_df["run_id"], runs_df["grok_step"]))
    model_lookup = dict(zip(runs_df["run_id"], runs_df["model_kind"]))

    for rid in run_ids:
        sub = curves_df[curves_df["run_id"] == rid]
        color = C_PAN if model_lookup.get(rid) == "pan" else C_TF
        axes[0].plot(sub["step"], sub["val_acc"],  color=color, lw=1.2, alpha=0.65)
        axes[1].plot(sub["step"], sub["val_loss"], color=color, lw=1.2, alpha=0.65)
        g = grok_lookup.get(rid, -1)
        if g is not None and g >= 0:
            axes[0].axvline(g, color=color, lw=0.5, ls=":", alpha=0.3)

    axes[0].axhline(0.99, color="#888", lw=0.8, ls="--")
    axes[0].set(xlabel="step", ylabel="val accuracy", ylim=(-0.02, 1.04),
                title="Validation accuracy")
    axes[1].set(xlabel="step", ylabel="val loss", yscale="log",
                title="Validation loss")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_sweep_reliability(
    runs_df:  pd.DataFrame,
    group_by: str,
    out_path: str,
    title:    Optional[str] = None,
) -> None:
    """
    Bar chart: grok rate and mean-step vs the sweep axis (K, DW, WD, etc).
    Two subplots: (1) grok_rate with n_runs annotation, (2) mean_grok_step
    (log) for the configurations that grokked.
    """
    _ensure(out_path)
    if group_by not in runs_df.columns:
        raise KeyError(f"{group_by!r} not found in runs DataFrame")

    agg = (runs_df.groupby(group_by)
                   .agg(n_runs     = ("run_id",   "size"),
                        n_grokked  = ("grokked",  "sum"),
                        mean_grok_step = ("grok_step",
                            lambda s: float(np.nanmean([x for x in s if x >= 0]))
                                       if (s >= 0).any() else float("nan")),
                        mean_peak  = ("peak_val_acc", "mean"))
                   .reset_index())
    agg["grok_rate"] = agg["n_grokked"] / agg["n_runs"]
    agg = agg.sort_values(group_by)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(agg))

    # Panel 1: grok rate
    bars = axes[0].bar(x, agg["grok_rate"], color=C_PAN, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    axes[0].set(xlabel=group_by, ylabel="grok rate",
                title=f"Grok rate vs {group_by}", ylim=(0, 1.1))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agg[group_by], rotation=0)
    axes[0].axhline(1.0, color="#888", lw=0.6, ls="--")
    for i, (rate, n) in enumerate(zip(agg["grok_rate"], agg["n_runs"])):
        axes[0].text(i, rate + 0.03, f"{int(rate*n)}/{int(n)}",
                     ha="center", fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    # Panel 2: mean grok step (log)
    gs = agg["mean_grok_step"].to_numpy(dtype=float)
    mask_ok = ~np.isnan(gs)
    if mask_ok.any():
        axes[1].bar(x[mask_ok], gs[mask_ok], color=C_TF, alpha=0.85,
                     edgecolor="black", linewidth=0.5)
    axes[1].set(xlabel=group_by, ylabel="mean grok step",
                title=f"Mean grokking step vs {group_by}", yscale="log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agg[group_by])
    axes[1].grid(alpha=0.25, axis="y", which="both")

    fig.suptitle(title or f"Reliability sweep over {group_by}",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Patch to apply before running `make paper`.
#
# Drop-in replacement for pan_lab.plots.plot_freq_trajectories. The
# original plots `learned` (wrapped frequency) on the y-axis, which
# makes the Fourier convergence look like a collapse from 2π down to 0
# and hides which theoretical basis vector each k converged to. This
# version plots angular error (already computed and stored as `error`
# in checkpoints.csv) on a log y-axis, with the SIFP-16 quantization
# threshold drawn as a reference line.
#
# Apply with:
#     cat patch_plot_freq_trajectories.py >> pan_lab/pan_lab/plots.py
#     # then in pan_lab/pan_lab/plots.py rename the old function to
#     # _plot_freq_trajectories_raw and the new one to plot_freq_trajectories.
#
# Or just paste this function body over the existing
# plot_freq_trajectories in pan_lab/pan_lab/plots.py.

def plot_freq_err_trajectories(
    checkpoints_df,
    runs_df,
    out_path,
    title = "Angular error to theoretical Fourier basis",
):
    """
    Replace the raw-frequency plot with angular-error-on-log-scale.

    Each line is one k, showing its distance to the nearest theoretical
    basis vector over training. The SIFP-16 quantization floor is drawn
    at 2π/65536 ≈ 9.6e-5 rad. A dotted vertical at the grokking step
    lets you see whether convergence happens before, at, or after grok.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from pan_lab.config import SIFP16_QUANT_ERR

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if checkpoints_df.empty:
        return

    run_id = checkpoints_df["run_id"].iloc[0]
    grok_row = runs_df.loc[runs_df["run_id"] == run_id]
    grok = int(grok_row["grok_step"].iloc[0]) if len(grok_row) else -1
    encoders = sorted(checkpoints_df["encoder"].unique())

    fig, axes = plt.subplots(
        1, len(encoders),
        figsize=(5.5 * len(encoders), 4.5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, enc in zip(axes, encoders):
        sub = checkpoints_df[checkpoints_df["encoder"] == enc]
        ks  = sorted(sub["k"].unique())
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(ks)))
        for k, c in zip(ks, colors):
            row = sub[sub["k"] == k].sort_values("step")
            theory = row["theoretical"].iloc[0]
            # Clip zeros so log scale doesn't explode
            err = np.maximum(row["error"].to_numpy(), 1e-7)
            ax.plot(row["step"], err, color=c, lw=1.4,
                    label=f"k={k} (theory={theory:.3f})")
        ax.axhline(SIFP16_QUANT_ERR, color="#c8322f", lw=1,
                    ls="--", alpha=0.8,
                    label=f"SIFP-16 ({SIFP16_QUANT_ERR:.1e} rad)")
        if grok >= 0:
            ax.axvline(grok, color="black", lw=1.3, ls=":", alpha=0.7,
                        label=f"grok @ {grok:,}")
        ax.set(
            xlabel="step",
            ylabel="angular error (rad, log)",
            yscale="log",
            ylim=(1e-6, 2 * np.pi + 1),
            title=f"Encoder {enc}",
        )
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25, which="both")

    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_freq_trajectories(
    checkpoints_df: pd.DataFrame,
    runs_df:        pd.DataFrame,
    out_path:       str,
    title:          str = "Frequency trajectories",
) -> None:
    """
    One column per encoder. Each k is a line: learned freq over training
    steps, with dashed horizontal at the theoretical k*2pi/P. A dotted
    vertical at the grokking step.

    Handles a single run only (caller should subset checkpoints_df first).
    """
    _ensure(out_path)
    if checkpoints_df.empty:
        return

    run_id = checkpoints_df["run_id"].iloc[0]
    grok = int(runs_df.loc[runs_df["run_id"] == run_id, "grok_step"].iloc[0])
    encoders = sorted(checkpoints_df["encoder"].unique())

    fig, axes = plt.subplots(1, len(encoders),
                             figsize=(5.5 * len(encoders), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, enc in zip(axes, encoders):
        sub = checkpoints_df[checkpoints_df["encoder"] == enc]
        ks = sorted(sub["k"].unique())
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(ks)))
        for k, c in zip(ks, colors):
            row = sub[sub["k"] == k].sort_values("step")
            theory = row["theoretical"].iloc[0]
            ax.plot(row["step"], row["learned"], color=c, lw=1.4,
                    label=f"k={k} (theory={theory:.3f})")
            ax.axhline(theory, color=c, lw=0.8, ls="--", alpha=0.5)
        if grok >= 0:
            ax.axvline(grok, color="black", lw=1.3, ls=":", alpha=0.7,
                       label=f"grok @ {grok:,}")
        ax.set(xlabel="step", title=f"Encoder {enc}", ylabel="freq (rad/token)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_slot_census(
    slots_df: pd.DataFrame,
    out_path: str,
    title:    str = "Frequency-slot activation census",
) -> None:
    """
    For Experiment A. Heatmap: rows = (run_id, encoder), cols = k slot,
    cell color = 1 if converged to within SIFP-16 error else 0. Gives a
    one-glance answer to "do all seeds find the same slots?"
    """
    _ensure(out_path)
    if slots_df.empty:
        return

    # Build matrix
    piv = (slots_df
           .assign(row=lambda d: d["run_id"].astype(str) + "/e" + d["encoder"].astype(str))
           .pivot_table(index="row", columns="k", values="converged", aggfunc="max")
           .fillna(0)
           .astype(int))

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * piv.shape[1] + 3),
                                    max(4, 0.28 * piv.shape[0] + 1.5)))
    im = ax.imshow(piv.values, aspect="auto", cmap="Greens",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_xlabel("frequency slot k")
    ax.set_title(title, fontsize=11, weight="bold")

    # Column totals
    totals = piv.sum(axis=0)
    for i, t in enumerate(totals):
        ax.text(i, piv.shape[0] - 0.4, f"{t}/{piv.shape[0]}",
                ha="center", va="top", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_parameter_efficiency(
    runs_df: pd.DataFrame,
    out_path: str,
    title: str = "Parameter efficiency (grokked runs only)",
) -> None:
    """
    Scatter: params (log) vs grok_step (log), colored by model_kind.
    One marker per run. Grokked-only.
    """
    _ensure(out_path)
    g = runs_df[runs_df["grokked"]]
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for kind, color in [("pan", C_PAN), ("transformer", C_TF)]:
        sub = g[g["model_kind"] == kind]
        if sub.empty: continue
        ax.scatter(sub["param_count"], sub["grok_step"],
                   s=40, color=color, alpha=0.75, label=kind,
                   edgecolor="black", linewidth=0.5)
    ax.set(xscale="log", yscale="log",
           xlabel="parameter count (log)",
           ylabel="grokking step (log)",
           title=title)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation_bars(
    ablations_df: pd.DataFrame,
    out_path: str,
    title: str = "Ablations — accuracy after intervention",
) -> None:
    """Mean + min/max bars per intervention across runs."""
    _ensure(out_path)
    if ablations_df.empty:
        return

    agg = (ablations_df.groupby("intervention")["val_acc"]
           .agg(["mean", "min", "max"]).reset_index())
    # Preserve baseline-first ordering for readability
    pref = ["baseline", "zero_phase_mixing", "randomize_frequencies",
            "zero_ref_phases"]
    agg["_ord"] = agg["intervention"].apply(
        lambda x: pref.index(x) if x in pref else len(pref))
    agg = agg.sort_values("_ord").drop(columns="_ord")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], yerr=[agg["mean"] - agg["min"],
                                  agg["max"] - agg["mean"]],
           color=C_PAN, alpha=0.85, edgecolor="black", linewidth=0.5,
           capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["intervention"], rotation=15, ha="right")
    ax.set(ylabel="val accuracy", ylim=(0, 1.05), title=title)
    ax.axhline(0.99, color="#888", lw=0.8, ls="--")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Formation-curve / spectra / peak-timescale plots for metrics.csv
# ─────────────────────────────────────────────────────────────────────────────
def _filter_metrics_list(
    candidates: Sequence[str],
    available: Sequence[str],
) -> List[str]:
    """Drop any requested metric not present in `available`."""
    avail = set(available)
    return [m for m in candidates if m in avail]


def plot_metric_formation_curves(
    metrics_df: pd.DataFrame,
    runs_df:    pd.DataFrame,
    out_path:   str,
    metrics:    Optional[Sequence[str]] = None,
    title:      str = "Metric formation curves",
    max_lines:  int = 20,
) -> None:
    """
    Grid of small panels — one per mechanistic metric — showing how it
    evolves across training steps. One line per run; dotted vertical at
    each run's grok step.

    Expensive metrics (M6/M7/M9) are populated only at the
    `metrics_expensive_every` cadence, so they appear as sparse
    markers+lines; cheap metrics are dense.
    """
    _ensure(out_path)
    if metrics_df.empty:
        return

    panel_metrics = _filter_metrics_list(
        list(metrics) if metrics is not None else DEFAULT_FORMATION_METRICS,
        metrics_df.columns,
    )
    if not panel_metrics:
        return

    run_ids = list(metrics_df["run_id"].unique())
    if len(run_ids) > max_lines:
        run_ids = run_ids[:max_lines]
    grok_lookup = dict(zip(runs_df["run_id"], runs_df["grok_step"]))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(run_ids), 1)))

    n_cols = min(3, len(panel_metrics))
    n_rows = math.ceil(len(panel_metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)

    for i, metric in enumerate(panel_metrics):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.set_visible(True)
        for rid, color in zip(run_ids, colors):
            sub = metrics_df[metrics_df["run_id"] == rid][["step", metric]].dropna()
            if sub.empty:
                continue
            sub = sub.sort_values("step")
            ax.plot(sub["step"], sub[metric],
                    color=color, lw=1.2, alpha=0.75, marker=".", markersize=3)
            g = grok_lookup.get(rid, -1)
            try:
                g = int(g)
            except (TypeError, ValueError):
                g = -1
            if g is not None and g >= 0:
                ax.axvline(g, color=color, lw=0.5, ls=":", alpha=0.35)
        ax.set(xlabel="step", ylabel=metric, title=metric)
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metric_spectra(
    spectra_df: pd.DataFrame,
    out_path:   str,
    metrics:    Optional[Sequence[str]] = None,
    title:      str = "Metric spectra",
    aggregate:  str = "median",
) -> None:
    """
    Log-log overlay of the DFT of each selected metric's time series.
    x = timescale_steps (1/freq), y = spectral power.

    aggregate:
        "median"  — one line per metric, median power across runs, with
                    a shaded min/max envelope.
        "per_run" — one line per (run, metric).

    Excludes the DC bin (freq == 0).
    """
    _ensure(out_path)
    if spectra_df.empty:
        return

    available = list(spectra_df["metric"].unique())
    panel_metrics = _filter_metrics_list(
        list(metrics) if metrics is not None else DEFAULT_FORMATION_METRICS,
        available,
    )
    if not panel_metrics:
        return

    df = spectra_df[spectra_df["freq_cycles_per_step"] > 0].copy()
    df = df[df["metric"].isin(panel_metrics)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(panel_metrics)))

    if aggregate == "per_run":
        for (metric, rid), g in df.groupby(["metric", "run_id"], sort=False):
            idx   = panel_metrics.index(metric)
            color = colors[idx]
            g     = g.sort_values("timescale_steps")
            ax.plot(g["timescale_steps"], g["power"],
                    color=color, lw=0.9, alpha=0.55,
                    label=metric if rid == df["run_id"].iloc[0] else None)
    else:
        for metric, color in zip(panel_metrics, colors):
            g = df[df["metric"] == metric]
            agg = (g.groupby("freq_cycles_per_step")
                     .agg(power_med=("power", "median"),
                          power_min=("power", "min"),
                          power_max=("power", "max"),
                          timescale=("timescale_steps", "first"))
                     .reset_index()
                     .sort_values("timescale"))
            ax.plot(agg["timescale"], agg["power_med"],
                    color=color, lw=1.5, label=metric)
            ax.fill_between(agg["timescale"], agg["power_min"], agg["power_max"],
                            color=color, alpha=0.12)

    ax.set(xscale="log", yscale="log",
           xlabel="timescale (steps)",
           ylabel="spectral power",
           title=title)
    ax.invert_xaxis()   # slow (long timescale) on left → fast on right
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metric_peak_timescales(
    peaks_df: pd.DataFrame,
    out_path: str,
    metrics:  Optional[Sequence[str]] = None,
    title:    str = "Peak timescales",
) -> None:
    """
    Horizontal bar chart: one row per metric, x-axis = mean peak
    timescale across runs (log scale). Whiskers at min/max, annotated
    with n_runs.
    """
    _ensure(out_path)
    if peaks_df.empty:
        return

    available = list(peaks_df["metric"].unique())
    panel_metrics = _filter_metrics_list(
        list(metrics) if metrics is not None else DEFAULT_FORMATION_METRICS,
        available,
    )
    if not panel_metrics:
        return

    df = peaks_df[peaks_df["metric"].isin(panel_metrics)]
    if df.empty:
        return

    agg = (df.groupby("metric")["peak_timescale_steps"]
             .agg(["mean", "min", "max", "size"])
             .rename(columns={"size": "n_runs"}))
    # Only metrics present in panel_metrics; drop any all-NaN rows
    agg = agg.reindex([m for m in panel_metrics if m in agg.index]).dropna(subset=["mean"])
    if agg.empty:
        return

    agg = agg.sort_values("mean")
    y       = np.arange(len(agg))
    means   = agg["mean"].to_numpy()
    err_lo  = means - agg["min"].to_numpy()
    err_hi  = agg["max"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(agg) + 2))
    ax.barh(y, means,
            xerr=[err_lo, err_hi],
            color=C_PAN, alpha=0.85,
            edgecolor="black", linewidth=0.5,
            capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index)
    ax.set_xscale("log")
    ax.set(xlabel="peak timescale (steps, log)", title=title)
    ax.grid(alpha=0.25, axis="x", which="both")
    for yi, (m, n) in enumerate(zip(means, agg["n_runs"])):
        ax.text(m, yi, f"  n={int(n)}", va="center", fontsize=8, color="#333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
