"""
pan_lab.trainer — the training loop.

One loop for all models and all tasks. Everything task/model-specific
either goes through the model interface (forward, mix_features) or
through hooks.

Key differences from the monolithic pan.py:

1. Diversity regularization runs through model.mix_features(), which
   keeps autograd edges into encoder frequencies AND mixing weights.
   In the old code, phi_a/phi_b were computed under torch.no_grad(),
   so the penalty only ever reached phase_mix.weight — the encoders
   got no regularization signal. This is a real bug.

2. Hooks replace the hard-coded `if record_checkpoints:` branch.
   CheckpointLogger and CSVStreamLogger are both hooks; adding a new
   per-step probe is a ~10-line class.

3. The trainer returns a TrainResult that bundles the model, history,
   and provenance — everything needed to reconstruct or re-analyze.
"""
from __future__ import annotations

import dataclasses
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pan_lab.config import DEVICE, RunConfig, capture_provenance


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainHistory:
    steps:      list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss:   list = field(default_factory=list)
    val_acc:    list = field(default_factory=list)
    grok_step:  Optional[int] = None

    # Tier 3: mechanistic checkpoints — populated by CheckpointLogger.
    freq_checkpoints:    dict = field(default_factory=dict)
    fourier_conc_steps:  list = field(default_factory=list)
    fourier_conc_values: list = field(default_factory=list)

    # Internal: start time of the run (used by CSVStreamLogger).
    _t0: Optional[float] = None


@dataclass
class TrainResult:
    model:      nn.Module
    history:    TrainHistory
    cfg:        RunConfig
    provenance: dict
    elapsed_s:  float
    param_count: int


# ─────────────────────────────────────────────────────────────────────────────
def _set_seed(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    if not use_compile:
        return model
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile unavailable; running in eager mode")
        return model
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.recompile_limit = 64
        return torch.compile(model, backend="aot_eager")
    except Exception as e:
        warnings.warn(f"torch.compile failed ({e}); running in eager mode")
        return model


def _call_hooks(hooks: Iterable, method: str, *args, **kwargs) -> None:
    """Invoke `method` on every hook that defines it."""
    for h in hooks:
        fn = getattr(h, method, None)
        if fn is not None:
            fn(*args, **kwargs)


def _subsample_val(val_x, val_y, n: Optional[int]):
    if n is None or n >= len(val_x):
        return val_x, val_y
    idx = torch.randperm(len(val_x), device=val_x.device)[:n]
    return val_x[idx], val_y[idx]


# ─────────────────────────────────────────────────────────────────────────────
def train(
    model:    nn.Module,
    cfg:      RunConfig,
    train_x:  torch.Tensor,
    train_y:  torch.Tensor,
    val_x:    torch.Tensor,
    val_y:    torch.Tensor,
    hooks:    Optional[List] = None,
    verbose:  bool = True,
) -> TrainResult:
    """
    Generic training loop for PAN or transformer on any modular task.

    Parameters
    ----------
    model :   the already-built model on the correct device
    cfg :     RunConfig
    train_x/y, val_x/y : tensors already on device
    hooks :   list of hook objects; see pan_lab.hooks
    """
    hooks = hooks or []

    if verbose:
        print(f"  [{cfg.display_id()}] start  "
              f"p={cfg.p} k={cfg.k_freqs} task={cfg.task_kind} "
              f"model={cfg.model_kind} seed={cfg.seed} steps={cfg.n_steps:,} "
              f"wd={cfg.weight_decay} dw={cfg.diversity_weight}")

    if cfg.dry_run:
        if verbose:
            print("  [dry-run] skipping training")
        return TrainResult(
            model=model, history=TrainHistory(), cfg=cfg,
            provenance=capture_provenance(), elapsed_s=0.0,
            param_count=sum(p.numel() for p in model.parameters()),
        )

    _set_seed(cfg.seed)
    model = _maybe_compile(model, cfg.use_compile)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    eval_x, eval_y = _subsample_val(val_x, val_y, cfg.val_samples)

    history = TrainHistory()
    history._t0 = time.time()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    has_pan_mix = hasattr(raw_model, "mix_features")

    _call_hooks(hooks, "on_run_start", cfg=cfg, model=model)

    for step in range(cfg.n_steps):
        _call_hooks(hooks, "on_step_start", step=step, model=model, cfg=cfg)

        model.train()
        idx = torch.randperm(len(train_x), device=train_x.device)[:cfg.batch_size]
        xb  = train_x[idx]
        yb  = train_y[idx]

        logits = model(xb)
        loss   = F.cross_entropy(logits, yb)

        # ── Diversity regularization (PAN only, when dw > 0) ──────────────
        # Bug fix: we run mix_features under autograd so the off-diagonal
        # Gram penalty back-propagates into encoder.freq as well as
        # phase_mix.weight. The old code wrapped this in torch.no_grad().
        if cfg.diversity_weight > 0 and has_pan_mix:
            mix_out  = raw_model.mix_features(xb)              # (B, K)
            mix_norm = mix_out - mix_out.mean(0, keepdim=True)
            norms    = mix_norm.norm(dim=0, keepdim=True).clamp(min=1e-6)
            mix_norm = mix_norm / norms
            gram     = mix_norm.T @ mix_norm / mix_out.shape[0]  # (K, K)
            eye      = torch.eye(gram.shape[0], device=gram.device)
            div_loss = (gram - eye).pow(2).sum() / gram.shape[0]
            loss     = loss + cfg.diversity_weight * div_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Eval ─────────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                vlogits = model(eval_x)
                vloss   = F.cross_entropy(vlogits, eval_y).item()
                vacc    = (vlogits.argmax(-1) == eval_y).float().mean().item()

            history.steps.append(step)
            history.train_loss.append(float(loss.item()))
            history.val_loss.append(vloss)
            history.val_acc.append(vacc)

            _call_hooks(
                hooks, "on_eval",
                step=step, model=model, cfg=cfg,
                history=history, val_loss=vloss, val_acc=vacc,
            )

            if vacc >= cfg.grok_threshold and history.grok_step is None:
                history.grok_step = step
                elapsed = time.time() - history._t0
                if verbose:
                    print(f"  [{cfg.display_id()}] GROK @ step={step:,} "
                          f"val_acc={vacc:.3f} ({elapsed:.0f}s)")
                if cfg.early_stop:
                    break

            if verbose and step % (cfg.log_every * 5) == 0:
                elapsed = time.time() - history._t0
                print(f"  [{cfg.display_id()}] step={step:6d} "
                      f"train_loss={loss.item():.3f} val_acc={vacc:.3f} "
                      f"({elapsed:.0f}s)")

    elapsed = time.time() - history._t0
    _call_hooks(hooks, "on_end", model=model, cfg=cfg, history=history)

    if verbose:
        print(f"  [{cfg.display_id()}] end  grok={history.grok_step or 'no'} "
              f"final_acc={history.val_acc[-1] if history.val_acc else 0:.3f} "
              f"elapsed={elapsed:.0f}s")

    return TrainResult(
        model       = raw_model,
        history     = history,
        cfg         = cfg,
        provenance  = capture_provenance(),
        elapsed_s   = elapsed,
        param_count = sum(p.numel() for p in raw_model.parameters()),
    )
