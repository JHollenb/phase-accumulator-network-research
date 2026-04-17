"""
pan_lab.config — run configuration and global constants.

Every run is described by a RunConfig. RunConfigs are YAML-serializable and
hashable for caching/dedup. Provenance (git SHA, torch version, device,
hostname) is captured at run start and written alongside results.

Design notes
------------
- Configs are flat dataclasses, not nested. Nested configs become annoying
  to override on the CLI and harder to aggregate in pandas.
- One field per lever. No config inheritance. Easier to reason about.
- `seed` is always explicit. No implicit determinism.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import torch


# ── Constants ────────────────────────────────────────────────────────────────
PHASE_SCALE   = 65536
PHASE_SCALE_F = 65536.0
TWO_PI        = 2.0 * math.pi

# SIFP-16 phase quantization error — the yardstick for "converged to the
# Fourier basis to within hardware precision".
SIFP16_QUANT_ERR = TWO_PI / PHASE_SCALE


# ── Device selection ─────────────────────────────────────────────────────────
def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()


# ── RunConfig ────────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    """
    Declarative description of one training run.

    Everything the trainer needs is in this object. Nothing else is
    implicit. Two runs with identical RunConfigs produce identical
    results on the same device.

    Grouping convention used in field names (alphabetical within group):
      - task:   p, task_kind, train_frac
      - model:  model_kind, k_freqs, d_model, n_heads, d_mlp
      - optim:  batch_size, lr, n_steps, weight_decay, diversity_weight
      - eval:   early_stop, grok_threshold, log_every, val_samples
      - misc:   seed, label, use_compile, record_checkpoints
                output_dir, dry_run, save_model
    """
    # — task —
    p:            int   = 113
    task_kind:    str   = "mod_add"     # mod_add | mod_mul | mod_two_step
    train_frac:   float = 0.4

    # — model —
    model_kind:   str   = "pan"         # pan | transformer
    k_freqs:      int   = 9             # PAN only
    d_model:      int   = 128           # Transformer only
    n_heads:      int   = 4             # Transformer only
    d_mlp:        int   = 512           # Transformer only
    freq_init:    str   = "fourier"     # fourier | random — PAN encoder init

    # — optim —
    batch_size:       int   = 256
    lr:               float = 1e-3
    n_steps:          int   = 50_000
    weight_decay:     float = 0.01      # PAN default. TF uses 1.0.
    diversity_weight: float = 0.01      # off-diag Gram penalty (PAN)

    # — eval —
    early_stop:     bool  = True
    grok_threshold: float = 0.99
    log_every:      int   = 200
    val_samples:    Optional[int] = None

    # — misc —
    seed:               int  = 42
    label:              str  = "run"
    use_compile:        bool = False    # OFF by default — compile changes
                                        # MPS float accumulation order and
                                        # hurts reproducibility.
    record_checkpoints: bool = False    # Tier 3 mechanistic logging
    output_dir:         str  = "."
    dry_run:            bool = False
    save_model:         bool = False

    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        return asdict(self)

    def short_id(self) -> str:
        """
        Deterministic 10-char hash of the config.
        Useful as a filename / run ID without manually tracking labels.
        """
        raw = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha1(raw).hexdigest()[:10]

    def display_id(self) -> str:
        """Human-friendly run ID: label + short_id."""
        return f"{self.label}-{self.short_id()}"

    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Safely build from a YAML dict — ignores extra keys with a warning."""
        known   = {f.name for f in dataclasses.fields(cls)}
        kwargs  = {k: v for k, v in d.items() if k in known}
        unknown = set(d) - known
        if unknown:
            import warnings
            warnings.warn(f"Unknown RunConfig fields ignored: {sorted(unknown)}")
        return cls(**kwargs)

    def with_overrides(self, **kwargs) -> "RunConfig":
        """Return a copy with fields replaced. Used for sweeps."""
        return dataclasses.replace(self, **kwargs)


# ── Provenance capture ───────────────────────────────────────────────────────
def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def capture_provenance() -> dict:
    """
    Return a dict of environment metadata that pins a run to a moment in time.
    Written to manifest.json alongside every result set.
    """
    return {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "hostname":    socket.gethostname(),
        "platform":    platform.platform(),
        "python":      sys.version.split()[0],
        "torch":       torch.__version__,
        "cuda":        torch.version.cuda if torch.cuda.is_available() else None,
        "device":      DEVICE,
        "git_sha":     _git_sha(),
        "cwd":         os.getcwd(),
        "argv":        sys.argv,
    }
