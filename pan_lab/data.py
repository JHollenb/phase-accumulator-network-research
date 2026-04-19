"""
pan_lab.data — modular-arithmetic and Walsh (F_2^n) datasets.

Modular arithmetic (PAN side):
    mod_add:      (a, b) -> (a + b) mod P
    mod_mul:      (a, b) -> (a * b) mod P          (§5.3 in the paper)
    mod_two_step: (a, b, c) -> (a + b) * c mod P   (composition test)

Walsh / F_2^n (WAN side):
    walsh_parity:       x -> XOR over all bits of x
    walsh_bit:          x -> x_i  for a fixed (deterministic) bit i
    walsh_xor_subset:   x -> XOR over a fixed bit subset of x
    walsh_popcount_mod: x -> (sum_i x_i) mod `mod_base`
    walsh_xor:          (a, b) -> a XOR b  (label in [0, 2^n_bits))
    walsh_rotl:         x -> rotate-left(x, r)

Splits are deterministic functions of `seed` so the same config always
produces the same train/val partition across machines.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch

from pan_lab.config import DEVICE

TaskKind = Literal[
    "mod_add", "mod_mul", "mod_two_step",
    "walsh_parity", "walsh_bit", "walsh_xor_subset",
    "walsh_popcount_mod", "walsh_xor", "walsh_rotl",
]


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
    # Walsh-specific knobs (ignored for modular tasks)
    n_bits:     int      = 0,
    mod_base:   int      = 4,
    xor_mask:   int      = 0,
    bit_index:  int      = 0,
    rot_amount: int      = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate every (inputs, label) pair for the task, then shuffle and split.

    Match Nanda's grokking protocol: relatively small train fraction so the
    network must generalize rather than memorize.

    Returns
    -------
    train_x : (N_train, D) int64 — D depends on task
    train_y : (N_train,)  int64 — target class index
    val_x   : (N_val,   D) int64
    val_y   : (N_val,)    int64
    """
    if task_kind.startswith("walsh_"):
        return _make_walsh_dataset(
            n_bits     = n_bits,
            task_kind  = task_kind,
            train_frac = train_frac,
            seed       = seed,
            device     = device,
            mod_base   = mod_base,
            xor_mask   = xor_mask,
            bit_index  = bit_index,
            rot_amount = rot_amount,
        )

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


# ─────────────────────────────────────────────────────────────────────────────
# Walsh (F_2^n) dataset
# ─────────────────────────────────────────────────────────────────────────────
def _parity(x: np.ndarray) -> np.ndarray:
    """XOR-parity of the low bits of each integer in x. Returns 0/1."""
    out = x.copy()
    bits = int(np.ceil(np.log2(max(int(out.max()) + 1, 2))))
    for shift in (1, 2, 4, 8, 16):
        if shift < (1 << bits):
            out = out ^ (out >> shift)
    return (out & 1).astype(np.int64)


def _popcount(x: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(n_bits):
        out = out + ((x >> i) & 1)
    return out.astype(np.int64)


def _rotl(x: np.ndarray, r: int, n_bits: int) -> np.ndarray:
    r = r % n_bits
    mask = (1 << n_bits) - 1
    return (((x << r) | (x >> (n_bits - r))) & mask).astype(np.int64)


def _make_walsh_dataset(
    n_bits:     int,
    task_kind:  str,
    train_frac: float,
    seed:       int,
    device:     str,
    mod_base:   int,
    xor_mask:   int,
    bit_index:  int,
    rot_amount: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the (inputs, label) table for a Walsh task, then seeded-shuffle
    and split. `n_bits` drives everything — total rows are 2^n_bits for
    single-input tasks and 4^n_bits for walsh_xor.

    For very large n_bits (> 14 on the 2-input task) the enumeration will
    be big. The same train-frac protocol as PAN still applies — we prefer
    small train_frac so that generalization is what's measured.
    """
    if n_bits <= 0:
        raise ValueError("walsh tasks require n_bits > 0")

    rng = np.random.default_rng(seed)
    N   = 1 << n_bits

    if task_kind == "walsh_parity":
        x      = np.arange(N, dtype=np.int64)
        labels = _parity(x)
        inputs = x.reshape(-1, 1)

    elif task_kind == "walsh_bit":
        # Deterministic "fixed, unknown" bit so the seed doesn't pick it.
        i      = int(bit_index) % n_bits
        x      = np.arange(N, dtype=np.int64)
        labels = ((x >> i) & 1).astype(np.int64)
        inputs = x.reshape(-1, 1)

    elif task_kind == "walsh_xor_subset":
        # XOR of the bits selected by xor_mask. If xor_mask == 0,
        # default to the lower half of the bits (stable across seeds).
        mask = int(xor_mask)
        if mask == 0:
            mask = (1 << (n_bits // 2 or 1)) - 1
        mask  = mask & ((1 << n_bits) - 1)
        x     = np.arange(N, dtype=np.int64)
        sel   = x & mask
        labels = _parity(sel)
        inputs = x.reshape(-1, 1)

    elif task_kind == "walsh_popcount_mod":
        m      = int(mod_base)
        x      = np.arange(N, dtype=np.int64)
        labels = (_popcount(x, n_bits) % m).astype(np.int64)
        inputs = x.reshape(-1, 1)

    elif task_kind == "walsh_xor":
        a      = np.repeat(np.arange(N, dtype=np.int64), N)
        b      = np.tile  (np.arange(N, dtype=np.int64), N)
        labels = (a ^ b).astype(np.int64)
        inputs = np.stack([a, b], axis=1)

    elif task_kind == "walsh_rotl":
        x      = np.arange(N, dtype=np.int64)
        labels = _rotl(x, int(rot_amount), n_bits)
        inputs = x.reshape(-1, 1)

    else:
        raise ValueError(f"Unknown walsh task_kind: {task_kind!r}")

    perm   = rng.permutation(len(inputs))
    inputs = inputs[perm]
    labels = labels[perm]

    n_train = max(int(train_frac * len(inputs)), 1)
    tx = torch.tensor(inputs[:n_train], dtype=torch.long, device=device)
    ty = torch.tensor(labels[:n_train], dtype=torch.long, device=device)
    vx = torch.tensor(inputs[n_train:], dtype=torch.long, device=device)
    vy = torch.tensor(labels[n_train:], dtype=torch.long, device=device)
    return tx, ty, vx, vy


def make_dataset_from_cfg(cfg, device: str = DEVICE):
    """
    Build the dataset described by `cfg`. Central entry point so every
    experiment dispatches the same way — modular tasks use `cfg.p`,
    walsh tasks use `cfg.n_bits` plus any relevant task-specific fields.
    """
    return make_modular_dataset(
        p          = cfg.p,
        task_kind  = cfg.task_kind,
        train_frac = cfg.train_frac,
        seed       = cfg.seed,
        device     = device,
        n_bits     = getattr(cfg, "n_bits",     0),
        mod_base   = getattr(cfg, "mod_base",   4),
        xor_mask   = getattr(cfg, "xor_mask",   0),
        bit_index  = getattr(cfg, "bit_index",  0),
        rot_amount = getattr(cfg, "rot_amount", 3),
    )


def walsh_task_shape(cfg) -> Tuple[int, int]:
    """
    Return (n_inputs, n_classes) for the Walsh task described by cfg.

    Centralised so make_model and any analysis code agree on output
    cardinality without re-deriving it from task_kind everywhere.
    """
    kind = cfg.task_kind
    n_bits = cfg.n_bits
    if kind == "walsh_parity":       return (1, 2)
    if kind == "walsh_bit":          return (1, 2)
    if kind == "walsh_xor_subset":   return (1, 2)
    if kind == "walsh_popcount_mod": return (1, int(cfg.mod_base))
    if kind == "walsh_xor":          return (2, 1 << n_bits)
    if kind == "walsh_rotl":         return (1, 1 << n_bits)
    raise ValueError(f"Not a walsh task: {kind!r}")
