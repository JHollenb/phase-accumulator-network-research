"""
pan_lab.reporting — CSV persistence and cross-run aggregation via pandas.

Every experiment writes two CSVs:

    runs.csv   — one row per training run
        cols: run_id, experiment, label, p, k_freqs, model_kind,
              seed, weight_decay, diversity_weight, n_steps,
              grok_step, final_val_acc, peak_val_acc, elapsed_s,
              param_count, mode_collapsed

    curves.csv — one row per eval step per run
        cols: run_id, step, train_loss, val_loss, val_acc

Optional:
    slots.csv       — frequency-slot census (Experiment A)
    ablations.csv   — ablation acc per intervention per run
    checkpoints.csv — per-(run, step, encoder, k) frequency snapshot

Design rationale: pandas is the right layer for cross-run slicing
(groupby experiment, pivot on (K, seed), etc.). Every DataFrame is
also written as CSV so anybody downstream can open it in Excel.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from pan_lab.analysis import (
    ablation_test,
    detect_mode_collapse,
    slot_activation_census,
)
from pan_lab.trainer import TrainResult


# ─────────────────────────────────────────────────────────────────────────────
def run_row(result: TrainResult, experiment: str) -> dict:
    """Flatten one TrainResult into a dict ready for DataFrame insertion."""
    h        = result.history
    peak_acc = max(h.val_acc) if h.val_acc else 0.0
    final    = h.val_acc[-1]  if h.val_acc else 0.0
    cfg      = result.cfg

    row = {
        "run_id":           cfg.display_id(),
        "experiment":       experiment,
        "label":            cfg.label,
        "p":                cfg.p,
        "task_kind":        cfg.task_kind,
        "model_kind":       cfg.model_kind,
        "k_freqs":          cfg.k_freqs,
        "d_model":          cfg.d_model,
        "seed":             cfg.seed,
        "weight_decay":     cfg.weight_decay,
        "diversity_weight": cfg.diversity_weight,
        "freq_init":        cfg.freq_init,
        "n_steps_planned":  cfg.n_steps,
        "n_steps_actual":   h.steps[-1] if h.steps else 0,
        "grok_step":        h.grok_step if h.grok_step is not None else -1,
        "grokked":          h.grok_step is not None,
        "final_val_acc":    final,
        "peak_val_acc":     peak_acc,
        "final_train_loss": h.train_loss[-1] if h.train_loss else float("nan"),
        "final_val_loss":   h.val_loss[-1]   if h.val_loss   else float("nan"),
        "elapsed_s":        result.elapsed_s,
        "param_count":      result.param_count,
    }

    # Structural flag: only meaningful for PAN
    from pan_lab.models.pan import PhaseAccumulatorNetwork
    if isinstance(result.model, PhaseAccumulatorNetwork):
        row["mode_collapsed"] = bool(detect_mode_collapse(result.model))
    else:
        row["mode_collapsed"] = False

    return row


def curve_rows(result: TrainResult) -> Iterable[dict]:
    run_id = result.cfg.display_id()
    h = result.history
    for step, tl, vl, va in zip(h.steps, h.train_loss, h.val_loss, h.val_acc):
        yield {
            "run_id":     run_id,
            "step":       step,
            "train_loss": tl,
            "val_loss":   vl,
            "val_acc":    va,
        }


def checkpoint_rows(result: TrainResult) -> Iterable[dict]:
    """
    Long-format frequency checkpoints: one row per
    (run, step, encoder, k). Ready for group/pivot.
    """
    run_id = result.cfg.display_id()
    h      = result.history
    if not h.freq_checkpoints:
        return
    model = result.model
    for step, info in h.freq_checkpoints.items():
        for i in range(model.n_inputs):
            for k in range(model.k_freqs):
                yield {
                    "run_id":      run_id,
                    "step":        int(step),
                    "encoder":     i,
                    "k":           k + 1,
                    "theoretical": float(info["theoretical"][k]),
                    "learned":     float(info[f"learned_{i}"][k]),
                    "error":       float(info[f"error_{i}"][k]),
                }


# ─────────────────────────────────────────────────────────────────────────────
class ExperimentReporter:
    """
    Collects TrainResults and writes tidy CSVs + a manifest.

    Usage:
        rep = ExperimentReporter(name="k8_sweep", out_dir="results/k8")
        for seed in seeds:
            result = train(...)
            rep.add_run(result, ablations=True)
        rep.write_all()
        summary = rep.summary()
    """

    def __init__(self, name: str, out_dir: str):
        self.name     = name
        self.out_dir  = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self._runs:        List[dict] = []
        self._curves:      List[dict] = []
        self._checkpoints: List[dict] = []
        self._ablations:   List[dict] = []
        self._slots:       List[dict] = []

        self._provenance: Optional[dict] = None

    # ──────────────────────────────────────────────────────────────
    def add_run(
        self,
        result:   TrainResult,
        val_x:    Optional[torch.Tensor] = None,
        val_y:    Optional[torch.Tensor] = None,
        ablations: bool = False,
        slots:    bool = False,
    ) -> None:
        self._runs.append(run_row(result, experiment=self.name))
        self._curves.extend(curve_rows(result))
        self._checkpoints.extend(checkpoint_rows(result))

        if self._provenance is None:
            self._provenance = result.provenance

        from pan_lab.models.pan import PhaseAccumulatorNetwork
        is_pan = isinstance(result.model, PhaseAccumulatorNetwork)

        if ablations and val_x is not None and val_y is not None:
            abl = ablation_test(result.model, val_x, val_y, verbose=False)
            rid = result.cfg.display_id()
            for k, v in abl.items():
                self._ablations.append({"run_id": rid, "intervention": k, "val_acc": float(v)})

        if slots and is_pan:
            df = slot_activation_census([result.model])
            df = df.assign(run_id=result.cfg.display_id(),
                           seed=result.cfg.seed)
            self._slots.extend(df.to_dict("records"))

    # ──────────────────────────────────────────────────────────────
    def runs_df(self)        -> pd.DataFrame: return pd.DataFrame(self._runs)
    def curves_df(self)      -> pd.DataFrame: return pd.DataFrame(self._curves)
    def checkpoints_df(self) -> pd.DataFrame: return pd.DataFrame(self._checkpoints)
    def ablations_df(self)   -> pd.DataFrame: return pd.DataFrame(self._ablations)
    def slots_df(self)       -> pd.DataFrame: return pd.DataFrame(self._slots)

    # ──────────────────────────────────────────────────────────────
    def write_all(self) -> dict:
        """
        Write every non-empty DataFrame to the output dir.
        Returns a dict mapping kind -> filepath.
        """
        paths: dict = {}

        def _w(df: pd.DataFrame, name: str):
            if len(df) == 0:
                return
            p = os.path.join(self.out_dir, name)
            df.to_csv(p, index=False)
            paths[name] = p

        _w(self.runs_df(),        "runs.csv")
        _w(self.curves_df(),      "curves.csv")
        _w(self.checkpoints_df(), "checkpoints.csv")
        _w(self.ablations_df(),   "ablations.csv")
        _w(self.slots_df(),       "slots.csv")

        manifest = {
            "experiment":  self.name,
            "n_runs":      len(self._runs),
            "provenance":  self._provenance,
            "files":       paths,
        }
        mp = os.path.join(self.out_dir, "manifest.json")
        with open(mp, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        paths["manifest.json"] = mp

        return paths

    # ──────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """
        High-level per-experiment summary. Groups runs by the most
        important axis (K for parameter sweeps, DW for DW sweeps, P
        for prime sweeps, etc.) and reports grok rate, mean step,
        mean accuracy. The caller picks the groupby column.

        Returns an empty DataFrame when no runs exist yet.
        """
        df = self.runs_df()
        if df.empty:
            return df

        # Auto-pick group-by columns: anything that varies across runs.
        cand_cols = [
            "k_freqs", "weight_decay", "diversity_weight", "p",
            "model_kind", "d_model", "freq_init", "task_kind",
        ]
        varying = [c for c in cand_cols if c in df.columns and df[c].nunique() > 1]
        if not varying:
            # Single configuration — summarize by seed.
            varying = ["seed"]

        agg = (df.groupby(varying)
                 .agg(n_runs         = ("run_id",       "size"),
                      n_grokked      = ("grokked",      "sum"),
                      mean_grok_step = ("grok_step",
                          lambda s: float(np.nanmean([x for x in s if x >= 0])) if (s >= 0).any() else float("nan")),
                      mean_peak_acc  = ("peak_val_acc", "mean"),
                      mean_final_acc = ("final_val_acc", "mean"),
                      mean_param_ct  = ("param_count",   "mean"),
                      mode_collapse  = ("mode_collapsed", "sum"),
                      mean_elapsed_s = ("elapsed_s",    "mean"))
                 .reset_index()
               )
        agg["grok_rate"] = agg["n_grokked"] / agg["n_runs"]
        return agg

    def print_summary(self) -> None:
        s = self.summary()
        if s.empty:
            print(f"[{self.name}] no runs collected")
            return
        print(f"\n══ {self.name} — summary ══")
        with pd.option_context("display.max_columns", None,
                                "display.width", 200):
            print(s.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1  —  pan_lab/reporting.py
#
# Add this function anywhere in the file (top level). It's the writer
# the experiment runners will call when cfg.save_model is True.
# ─────────────────────────────────────────────────────────────────────────────

def save_model_weights(result, out_dir: str) -> str:
    """
    Write trained model state_dict to <out_dir>/model_<display_id>.pt.

    The payload includes everything needed to reconstruct the model
    without the original process:
      state_dict   — torch-serializable weights
      arch         — 'PAN' or 'TransformerBaseline'
      config       — flat dict of RunConfig fields
      grok_step    — int or None
      param_count  — total trainable parameters

    Load back with:
        import torch
        from pan_lab.models.pan import PhaseAccumulatorNetwork
        ckpt = torch.load('results/tier3/model_tier3-xxx.pt',
                          weights_only=False)
        cfg  = ckpt['config']
        pan  = PhaseAccumulatorNetwork(cfg['p'], cfg['k_freqs'],
                                        n_inputs=2 if cfg['task_kind'] != 'mod_two_step' else 3,
                                        freq_init=cfg['freq_init'])
        pan.load_state_dict(ckpt['state_dict'])
    """
    import os
    import torch
    from pan_lab.models.pan import PhaseAccumulatorNetwork

    os.makedirs(out_dir, exist_ok=True)
    m = result.model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod

    arch = ("PAN" if isinstance(m, PhaseAccumulatorNetwork)
            else type(m).__name__)

    payload = {
        "state_dict":  m.state_dict(),
        "arch":        arch,
        "config":      result.cfg.as_dict(),
        "grok_step":   result.history.grok_step,
        "param_count": result.param_count,
    }
    path = os.path.join(out_dir, f"model_{result.cfg.display_id()}.pt")
    torch.save(payload, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1  —  pan_lab/reporting.py
#
# Add this function anywhere in the file (top level). It's the writer
# the experiment runners will call when cfg.save_model is True.
# ─────────────────────────────────────────────────────────────────────────────

def save_model_weights(result, out_dir: str) -> str:
    """
    Write trained model state_dict to <out_dir>/model_<display_id>.pt.

    The payload includes everything needed to reconstruct the model
    without the original process:
      state_dict   — torch-serializable weights
      arch         — 'PAN' or 'TransformerBaseline'
      config       — flat dict of RunConfig fields
      grok_step    — int or None
      param_count  — total trainable parameters

    Load back with:
        import torch
        from pan_lab.models.pan import PhaseAccumulatorNetwork
        ckpt = torch.load('results/tier3/model_tier3-xxx.pt',
                          weights_only=False)
        cfg  = ckpt['config']
        pan  = PhaseAccumulatorNetwork(cfg['p'], cfg['k_freqs'],
                                        n_inputs=2 if cfg['task_kind'] != 'mod_two_step' else 3,
                                        freq_init=cfg['freq_init'])
        pan.load_state_dict(ckpt['state_dict'])
    """
    import os
    import torch
    from pan_lab.models.pan import PhaseAccumulatorNetwork

    os.makedirs(out_dir, exist_ok=True)
    m = result.model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod

    arch = ("PAN" if isinstance(m, PhaseAccumulatorNetwork)
            else type(m).__name__)

    payload = {
        "state_dict":  m.state_dict(),
        "arch":        arch,
        "config":      result.cfg.as_dict(),
        "grok_step":   result.history.grok_step,
        "param_count": result.param_count,
    }
    path = os.path.join(out_dir, f"model_{result.cfg.display_id()}.pt")
    torch.save(payload, path)
    return path

# ─── PASTE INTO pan_lab/reporting.py (end of file) ───
_reporting_patch = '''
def save_model_weights(result, out_dir: str) -> str:
    """Write trained model state_dict + config to <out_dir>/model_<id>.pt."""
    import os
    import torch
    from pan_lab.models.pan import PhaseAccumulatorNetwork

    os.makedirs(out_dir, exist_ok=True)
    m = result.model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod

    arch = ("PAN" if isinstance(m, PhaseAccumulatorNetwork)
            else type(m).__name__)

    payload = {
        "state_dict":  m.state_dict(),
        "arch":        arch,
        "config":      result.cfg.as_dict(),
        "grok_step":   result.history.grok_step,
        "param_count": result.param_count,
    }
    path = os.path.join(out_dir, f"model_{result.cfg.display_id()}.pt")
    torch.save(payload, path)
    return path
'''

