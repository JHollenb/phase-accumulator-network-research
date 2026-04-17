"""
pan_lab.grid_sweep — single generic experiment that replaces the family of
near-identical sweep functions.

A sweep is fully described by a base RunConfig plus three things:
  - grid:    what varies across sub-runs
  - options: ablations/slots/save_model flags + named training hooks
  - plots:   declarative list of figures to render from the resulting CSVs

`run_grid_sweep` is registered as the experiment named "grid_sweep" and is
invoked by every YAML that previously had its own bespoke function.

Two grid forms are supported:

  grid (dict)  — Cartesian product over field values:
      grid: {seed: [42, 123], k_freqs: [3, 5]}   # → 4 sub-runs

  grid (list)  — explicit per-sub-run override dicts (use this when axes are
                 coupled, e.g. pan/transformer pairs whose weight_decay
                 differs by model_kind):
      grid:
        - {model_kind: pan, weight_decay: 0.01, label: pan}
        - {model_kind: transformer, weight_decay: 1.0, label: tf}
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from pan_lab.config import RunConfig
from pan_lab.hooks  import CheckpointLogger
from pan_lab.plots  import (
    plot_ablation_bars,
    plot_freq_err_trajectories,
    plot_freq_trajectories,
    plot_parameter_efficiency,
    plot_slot_census,
    plot_sweep_reliability,
    plot_training_curves,
)


# ─────────────────────────────────────────────────────────────────────────────
# Registries
# ─────────────────────────────────────────────────────────────────────────────
HOOK_REGISTRY: Dict[str, Callable[[], Any]] = {
    "checkpoint_logger": CheckpointLogger,
}

# Each entry: plot_fn, required ExperimentReporter getters, default filename.
# `None` filename means the renderer builds it (e.g. reliability_<group_by>.png).
PLOT_REGISTRY: Dict[str, Tuple[Callable, Tuple[str, ...], Optional[str]]] = {
    "training_curves":       (plot_training_curves,       ("curves_df", "runs_df"),       "curves.png"),
    "sweep_reliability":     (plot_sweep_reliability,     ("runs_df",),                   None),
    "ablation_bars":         (plot_ablation_bars,         ("ablations_df",),              "ablations.png"),
    "parameter_efficiency":  (plot_parameter_efficiency,  ("runs_df",),                   "param_efficiency.png"),
    "slot_census":           (plot_slot_census,           ("slots_df",),                  "slot_census.png"),
    "freq_trajectories":     (plot_freq_trajectories,     ("checkpoints_df", "runs_df"),  "freq_trajectories.png"),
    "freq_err_trajectories": (plot_freq_err_trajectories, ("checkpoints_df", "runs_df"),  "freq_err_trajectories.png"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Label auto-generation
# ─────────────────────────────────────────────────────────────────────────────
# Short prefix per RunConfig field, used when building labels from grid
# entries. Fields not listed here fall back to `<field>=<value>`.
_LABEL_PREFIX: Dict[str, str] = {
    "k_freqs":          "K",
    "seed":             "s",
    "diversity_weight": "DW",
    "weight_decay":     "WD",
    "p":                "P",
    "d_model":          "d",
    "n_heads":          "h",
    "d_mlp":            "m",
    "freq_init":        "",
    "task_kind":        "",
    "model_kind":       "",
    "n_steps":          "steps",
    "train_frac":       "tf",
}


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        # Trim trailing zeros but keep enough precision to distinguish values.
        s = f"{v:g}"
        return s
    return str(v)


def _auto_label(overrides: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, val in overrides.items():
        if key == "label":
            continue
        prefix = _LABEL_PREFIX.get(key, f"{key}=")
        parts.append(f"{prefix}{_format_value(val)}")
    return "-".join(parts) if parts else "run"


# ─────────────────────────────────────────────────────────────────────────────
# Grid expansion
# ─────────────────────────────────────────────────────────────────────────────
def _expand_grid(
    base: RunConfig,
    grid: Optional[Any],
) -> List[RunConfig]:
    """
    Turn `grid` into a concrete list of RunConfigs.

    - None/empty:        single run (just base)
    - dict[field, list]: Cartesian product across fields
    - list[dict]:        one sub-run per dict (each dict is a full
                         override set; missing fields inherit from base)
    """
    if not grid:
        return [base]

    if isinstance(grid, dict):
        keys   = list(grid.keys())
        values = [list(grid[k]) for k in keys]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    elif isinstance(grid, list):
        combos = [dict(entry) for entry in grid]
    else:
        raise TypeError(
            f"grid must be a dict (Cartesian) or list-of-dicts (explicit); "
            f"got {type(grid).__name__}"
        )

    cfgs: List[RunConfig] = []
    for entry in combos:
        label = entry.pop("label", None) or _auto_label(entry)
        cfgs.append(base.with_overrides(label=label, **entry))
    return cfgs


# ─────────────────────────────────────────────────────────────────────────────
# Plot rendering
# ─────────────────────────────────────────────────────────────────────────────
def _render_plots(plots: List[Any], rep, out_dir: str) -> None:
    """
    Dispatch a list of plot specs through PLOT_REGISTRY.

    Each spec is either a string (plot type with defaults) or a dict with
    `type:` plus any extra kwargs forwarded to the plot function (e.g.
    `group_by`, `title`, `filename`).
    """
    for spec in plots:
        if isinstance(spec, str):
            spec = {"type": spec}
        kind = spec.get("type")
        if kind not in PLOT_REGISTRY:
            raise KeyError(
                f"Unknown plot type: {kind!r}. "
                f"Available: {sorted(PLOT_REGISTRY)}"
            )
        fn, df_names, default_filename = PLOT_REGISTRY[kind]

        dfs = [getattr(rep, n)() for n in df_names]
        if any(df.empty for df in dfs):
            print(f"  [plots] skipping {kind}: missing data")
            continue

        kwargs = {k: v for k, v in spec.items() if k not in ("type", "filename")}

        # Build output filename
        filename = spec.get("filename")
        if filename is None:
            if kind == "sweep_reliability":
                gb = kwargs.get("group_by", "value")
                filename = f"reliability_{gb}.png"
            else:
                filename = default_filename or f"{kind}.png"
        out_path = os.path.join(out_dir, filename)

        # out_path is passed as a kwarg so plot functions with different
        # positional parameter orderings (e.g. plot_sweep_reliability,
        # whose `group_by` comes before `out_path`) don't collide.
        fn(*dfs, out_path=out_path, **kwargs)
        print(f"  [plots] wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_grid_sweep(
    base:     RunConfig,
    out_dir:  str,
    *,
    grid:     Optional[Any]         = None,
    options:  Optional[Dict[str, Any]] = None,
    plots:    Optional[List[Any]]   = None,
    dry_run:  bool                  = False,
    name:     Optional[str]         = None,
    **_,
):
    """
    Run a grid of training configs and render any plots declared in the YAML.

    The `name` is used for the ExperimentReporter's display/manifest name.
    It defaults to the basename of `out_dir` (so `results/k8_sweep` →
    `k8_sweep`) which matches the prior per-experiment naming.
    """
    from pan_lab.experiments import _run_cfgs

    options = options or {}
    plots   = plots   or []

    cfgs = _expand_grid(base, grid)

    hook_names   = list(options.get("hooks", []) or [])
    hook_factory = None
    if hook_names:
        for h in hook_names:
            if h not in HOOK_REGISTRY:
                raise KeyError(
                    f"Unknown hook: {h!r}. Available: {sorted(HOOK_REGISTRY)}"
                )
        hook_factory = lambda cfg: [HOOK_REGISTRY[h]() for h in hook_names]

    reporter_name = name or os.path.basename(os.path.normpath(out_dir)) or "grid_sweep"

    rep = _run_cfgs(
        cfgs,
        reporter_name,
        out_dir,
        dry_run,
        hook_factory = hook_factory,
        ablations    = options.get("ablations", True),
        slots        = options.get("slots",     False),
    )

    if not dry_run and plots:
        _render_plots(plots, rep, out_dir)

    return rep
