"""
pan_lab.data — modular arithmetic datasets.

Currently supports:
    mod_add:      (a, b) -> (a + b) mod P
    mod_mul:      (a, b) -> (a * b) mod P          (§5.3 in the paper)
    mod_two_step: (a, b, c) -> (a + b) * c mod P   (composition test)

Splits are deterministic functions of `seed` so the same config always
produces the same train/val partition across machines.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch

from pan_lab.config import DEVICE

TaskKind = Literal["mod_add", "mod_mul", "mod_two_step"]


def _all_pairs(p: int) -> np.ndarray:
    """Return every (a,b) pair for a,b in [0,P). Shape (P*P, 2)."""
    a = np.repeat(np.arange(p, dtype=np.int64), p)
    b = np.tile  (np.arange(p, dtype=np.int64), p)
    return np.stack([a, b], axis=1)


def _all_triples(p: int) -> np.ndarray:
    """Return every (a,b,c) triple. Shape (P^3, 3)."""
    rng = np.arange(p, dtype=np.int64)
    a   = np.repeat(np.repeat(rng, p), p)
    b   = np.tile  (np.repeat(rng, p), p)
    c   = np.tile  (np.tile  (rng, p), p)
    return np.stack([a, b, c], axis=1)


def make_modular_dataset(
    p:          int,
    task_kind:  TaskKind = "mod_add",
    train_frac: float    = 0.4,
    seed:       int      = 42,
    device:     str      = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate every (inputs, label) pair for the task, then shuffle and split.

    Match Nanda's grokking protocol: relatively small train fraction so the
    network must generalize rather than memorize.

    Returns
    -------
    train_x : (N_train, D) int64 — D=2 for mod_add/mod_mul, D=3 for two_step
    train_y : (N_train,)  int64 — target in [0, P)
    val_x   : (N_val,   D) int64
    val_y   : (N_val,)    int64
    """
    rng = np.random.default_rng(seed)

    if task_kind == "mod_add":
        inputs = _all_pairs(p)
        labels = (inputs[:, 0] + inputs[:, 1]) % p
    elif task_kind == "mod_mul":
        inputs = _all_pairs(p)
        labels = (inputs[:, 0] * inputs[:, 1]) % p
    elif task_kind == "mod_two_step":
        inputs = _all_triples(p)
        labels = ((inputs[:, 0] + inputs[:, 1]) * inputs[:, 2]) % p
    else:
        raise ValueError(f"Unknown task_kind: {task_kind!r}")

    perm   = rng.permutation(len(inputs))
    inputs = inputs[perm]
    labels = labels[perm]

    n_train = int(train_frac * len(inputs))
    tx = torch.tensor(inputs[:n_train], dtype=torch.long, device=device)
    ty = torch.tensor(labels[:n_train], dtype=torch.long, device=device)
    vx = torch.tensor(inputs[n_train:], dtype=torch.long, device=device)
    vy = torch.tensor(labels[n_train:], dtype=torch.long, device=device)

    return tx, ty, vx, vy
