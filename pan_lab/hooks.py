"""
pan_lab.hooks — training-loop callbacks.

The trainer is generic; anything specific to a particular experiment
(checkpoint snapshots, extra metric logging, custom early-stop criteria)
plugs in through a Hook. Each hook sees every eval step and can accumulate
state on the history object.

Hook API (duck-typed, no base class required):
    def on_step_start(self, step, model, cfg):        ...
    def on_eval     (self, step, model, cfg, history, val_loss, val_acc): ...
    def on_end      (self, model, cfg, history):      ...

All methods are optional — the trainer checks hasattr before calling.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class CheckpointLogger:
    """
    Tier 3 mechanistic logger. On every eval step, records:
      - learned frequency dict (via model.get_learned_frequencies())
      - Fourier concentration of decoder weights

    Data goes onto history.freq_checkpoints[step] and
    history.fourier_concentration.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def on_eval(self, step, model, cfg, history, val_loss, val_acc):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        if not hasattr(raw, "get_learned_frequencies"):
            return
        history.freq_checkpoints[step] = raw.get_learned_frequencies()

        # Fourier concentration of the decoder weight matrix.
        # High concentration = decoder is projecting onto a sparse set of
        # frequency components, i.e. a canonical Fourier decoder.
        W = raw.decoder.weight.detach().float()          # (P, K)
        from pan_lab.analysis import fourier_concentration
        conc = fourier_concentration(W, top_k=min(self.top_k, W.shape[0]))
        history.fourier_conc_steps.append(step)
        history.fourier_conc_values.append(conc)


class CSVStreamLogger:
    """
    Stream every eval step to a CSV file as training progresses.

    Why stream rather than dump-at-end: when a 200K-step run crashes at
    step 180K, you still want the curve up to 180K. File is flushed after
    every append.
    """

    def __init__(self, path: str, run_id: str):
        import csv
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path   = path
        self.run_id = run_id
        self._new   = not os.path.exists(path)
        self._fh    = open(path, "a", newline="")
        self._w     = csv.writer(self._fh)
        if self._new:
            self._w.writerow(
                ["run_id", "step", "train_loss", "val_loss", "val_acc",
                 "elapsed_s", "grokked"]
            )
            self._fh.flush()

    def on_eval(self, step, model, cfg, history, val_loss, val_acc):
        import time
        elapsed = getattr(history, "_t0", None)
        elapsed = 0.0 if elapsed is None else time.time() - elapsed
        grokked = (history.grok_step is not None)
        train_loss = history.train_loss[-1] if history.train_loss else float("nan")
        self._w.writerow(
            [self.run_id, step, f"{train_loss:.6f}", f"{val_loss:.6f}",
             f"{val_acc:.6f}", f"{elapsed:.1f}", int(grokked)]
        )
        self._fh.flush()

    def on_end(self, model, cfg, history):
        self._fh.close()
