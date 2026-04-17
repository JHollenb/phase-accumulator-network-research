from __future__ import annotations

import copy
import os
from typing import Any, Callable

import pandas as pd
import torch

from pan_lab.config import DEVICE
from pan_lab.models.quantize import apply_sifp16_to_pan
from pan_lab.plots import (
    plot_ablation_bars,
    plot_parameter_efficiency,
    plot_slot_census,
    plot_sweep_reliability,
    plot_training_curves,
)

AnalyzerFn = Callable[[Any, torch.Tensor, torch.Tensor, Any, dict[str, Any]], dict[str, Any] | None]
PlotFn = Callable[[Any, dict[str, Any], str, dict[str, Any]], None]

ANALYZER_REGISTRY: dict[str, AnalyzerFn] = {}
PLOT_REGISTRY: dict[str, PlotFn] = {}


def register_analyzer(name: str) -> Callable[[AnalyzerFn], AnalyzerFn]:
    def _wrap(fn: AnalyzerFn) -> AnalyzerFn:
        ANALYZER_REGISTRY[name] = fn
        return fn

    return _wrap


def register_plot(name: str) -> Callable[[PlotFn], PlotFn]:
    def _wrap(fn: PlotFn) -> PlotFn:
        PLOT_REGISTRY[name] = fn
        return fn

    return _wrap


def _ensure_plugin_state(state: dict[str, Any]) -> None:
    state.setdefault("plugin_rows", {})


def resolve_plugins(registry: dict[str, Any], declared_names: list[str] | None) -> list[tuple[str, Any]]:
    names = declared_names or []
    # deterministic + de-duplicated order
    ordered = sorted(set(str(n) for n in names))
    out: list[tuple[str, Any]] = []
    for name in ordered:
        fn = registry.get(name)
        if fn is None:
            print(f"  [plugins] unknown plugin {name!r}; skipping")
            continue
        out.append((name, fn))
    return out


def run_analyzers(
    analyzer_names: list[str] | None,
    result,
    vx: torch.Tensor,
    vy: torch.Tensor,
    cfg,
    state: dict[str, Any],
) -> None:
    _ensure_plugin_state(state)
    for name, fn in resolve_plugins(ANALYZER_REGISTRY, analyzer_names):
        payload = fn(result, vx, vy, cfg, state) or {}
        rows = payload.get("rows")
        if rows is None:
            continue
        if isinstance(rows, dict):
            rows = [rows]
        state["plugin_rows"].setdefault(name, []).extend(rows)


def write_plugin_rows(state: dict[str, Any], out_dir: str) -> None:
    for analyzer_name in sorted((state.get("plugin_rows") or {}).keys()):
        rows = state["plugin_rows"][analyzer_name]
        if not rows:
            continue
        path = os.path.join(out_dir, f"{analyzer_name}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)


def write_declared_plots(
    plot_specs: list[dict[str, Any]] | None,
    reporter,
    state: dict[str, Any],
    out_dir: str,
) -> None:
    for plot_spec in sorted((plot_specs or []), key=lambda x: str(x.get("type", ""))):
        plot_type = str(plot_spec.get("type", ""))
        fn = PLOT_REGISTRY.get(plot_type)
        if fn is None:
            print(f"  [plugins] unknown plot type {plot_type!r}; skipping")
            continue
        fn(reporter, state, out_dir, plot_spec)


@register_analyzer("decoder_swap")
def analyzer_decoder_swap(result, vx, vy, cfg, state):
    pan = result.model
    with torch.no_grad():
        theta = torch.arange(cfg.p, device=DEVICE).float().unsqueeze(-1)
        f = pan.encoders[0].freq.detach()
        decoder_fourier = torch.cos(theta * f.unsqueeze(0))
        n = decoder_fourier.norm(dim=1, keepdim=True).clamp(min=1e-8)
        decoder_fourier = decoder_fourier / n

        saved_w = pan.decoder.weight.data.clone()
        saved_b = pan.decoder.bias.data.clone()
        pan.decoder.weight.data = decoder_fourier
        pan.decoder.bias.data.zero_()
        logits = pan(vx)
        acc_swap = float((logits.argmax(-1) == vy).float().mean().item())
        pan.decoder.weight.data.copy_(saved_w)
        pan.decoder.bias.data.copy_(saved_b)

    acc_learned = result.history.val_acc[-1] if result.history.val_acc else 0.0
    return {
        "rows": {
            "run_id": cfg.display_id(),
            "seed": cfg.seed,
            "val_acc_learned_decoder": acc_learned,
            "val_acc_fourier_decoder": acc_swap,
            "delta": acc_swap - acc_learned,
        }
    }


@register_analyzer("sifp16_eval")
def analyzer_sifp16_eval(result, vx, vy, cfg, _state):
    qmodel = copy.deepcopy(result.model)
    qmodel.eval()
    apply_sifp16_to_pan(qmodel)
    with torch.no_grad():
        logits = qmodel(vx)
        acc_q = float((logits.argmax(-1) == vy).float().mean().item())
    acc_fp = result.history.val_acc[-1] if result.history.val_acc else 0.0
    return {
        "rows": {
            "run_id": cfg.display_id(),
            "seed": cfg.seed,
            "val_acc_fp32": acc_fp,
            "val_acc_sifp16": acc_q,
            "delta": acc_q - acc_fp,
        }
    }


@register_analyzer("decoder_harmonics")
def analyzer_decoder_harmonics(result, vx, vy, cfg, _state):
    from pan_lab.experiments.decoder_analysis import analyze_gate_space_upper_bound, analyze_harmonics

    pan = result.model
    harm = analyze_harmonics(pan, vx, vy, harmonic_order=4)
    gate_ub = analyze_gate_space_upper_bound(pan, vx, vy, p=cfg.p)
    return {
        "rows": {
            "run_id": cfg.display_id(),
            "seed": cfg.seed,
            "baseline_learned": harm["baseline_learned"],
            "clock_only_acc": harm["clock_only"]["acc"],
            "harmonic_H2_acc": harm["harmonic_fits"].get("H=2", {}).get("acc"),
            "harmonic_H3_acc": harm["harmonic_fits"].get("H=3", {}).get("acc"),
            "harmonic_H4_acc": harm["harmonic_fits"].get("H=4", {}).get("acc"),
            "gate_optimal_acc": gate_ub["gate_optimal_acc"],
            "gate_ols_acc": gate_ub["gate_ols_acc"],
            "gap_from_optimal": gate_ub["gap_from_optimal"],
        }
    }


@register_plot("reliability")
def plot_reliability(reporter, _state, out_dir: str, plot_spec: dict[str, Any]) -> None:
    plot_sweep_reliability(
        reporter.runs_df(),
        group_by=plot_spec.get("group_by", "k_freqs"),
        out_path=os.path.join(out_dir, plot_spec.get("filename") or "reliability.png"),
        title=plot_spec.get("title"),
    )


@register_plot("curves")
def plot_curves(reporter, _state, out_dir: str, plot_spec: dict[str, Any]) -> None:
    plot_training_curves(
        reporter.curves_df(),
        reporter.runs_df(),
        os.path.join(out_dir, plot_spec.get("filename") or "curves.png"),
        title=plot_spec.get("title", "Training curves"),
    )


@register_plot("parameter_efficiency")
def plot_param_eff(reporter, _state, out_dir: str, plot_spec: dict[str, Any]) -> None:
    plot_parameter_efficiency(
        reporter.runs_df(),
        os.path.join(out_dir, plot_spec.get("filename") or "parameter_efficiency.png"),
        title=plot_spec.get("title", "Parameter efficiency"),
    )


@register_plot("ablations")
def plot_ablations(reporter, _state, out_dir: str, plot_spec: dict[str, Any]) -> None:
    plot_ablation_bars(
        reporter.ablations_df(),
        os.path.join(out_dir, plot_spec.get("filename") or "ablations.png"),
        title=plot_spec.get("title", "Ablation analysis"),
    )


@register_plot("slot_census")
def plot_slots(reporter, _state, out_dir: str, plot_spec: dict[str, Any]) -> None:
    plot_slot_census(
        reporter.slots_df(),
        os.path.join(out_dir, plot_spec.get("filename") or "slot_census.png"),
        title=plot_spec.get("title", "Frequency-slot activation census"),
    )


@register_plot("decoder_analysis")
def plot_decoder_analysis_summary(reporter, state, out_dir: str, plot_spec: dict[str, Any]) -> None:
    rows = state.get("summary_rows") if isinstance(state, dict) else None
    if not rows:
        return

    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    if df.empty or "seed" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(df))
    ax.plot(x, df["acc_learned"], marker="o", label="learned decoder")
    if "acc_clock_only" in df.columns:
        ax.plot(x, df["acc_clock_only"], marker="o", label="clock-only decoder")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in df["seed"]])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("seed")
    ax.set_ylabel("validation accuracy")
    ax.set_title(plot_spec.get("title", "Decoder analysis summary"))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, plot_spec.get("filename") or "decoder_analysis.png"), dpi=140)
    plt.close(fig)
