"""Tests for pan_lab.data: correctness and determinism of dataset generation."""
from __future__ import annotations

import pytest
import torch

from pan_lab.data import make_modular_dataset


def test_mod_add_labels_are_correct():
    tx, ty, vx, vy = make_modular_dataset(p=11, task_kind="mod_add",
                                           seed=0, device="cpu")
    assert torch.all(ty == (tx[:, 0] + tx[:, 1]) % 11)
    assert torch.all(vy == (vx[:, 0] + vx[:, 1]) % 11)


def test_mod_mul_labels_are_correct():
    tx, ty, vx, vy = make_modular_dataset(p=11, task_kind="mod_mul",
                                           seed=0, device="cpu")
    assert torch.all(ty == (tx[:, 0] * tx[:, 1]) % 11)


def test_two_step_labels_and_shape():
    tx, ty, vx, vy = make_modular_dataset(p=7, task_kind="mod_two_step",
                                           seed=0, device="cpu")
    # P^3 total triples
    assert len(tx) + len(vx) == 7 ** 3
    assert tx.shape[1] == 3 and vx.shape[1] == 3
    assert torch.all(ty == ((tx[:, 0] + tx[:, 1]) * tx[:, 2]) % 7)


def test_all_pairs_cover_complete_cartesian_product():
    """Every (a,b) in [0,P) x [0,P) must appear exactly once across train+val."""
    tx, _, vx, _ = make_modular_dataset(p=7, task_kind="mod_add",
                                         seed=0, device="cpu")
    all_pairs = torch.cat([tx, vx], dim=0).tolist()
    expected = {(a, b) for a in range(7) for b in range(7)}
    assert {tuple(p) for p in all_pairs} == expected


def test_train_val_split_fraction():
    tx, _, vx, _ = make_modular_dataset(p=11, train_frac=0.4, seed=0,
                                         device="cpu")
    total = len(tx) + len(vx)
    assert len(tx) == int(0.4 * total)


def test_split_is_deterministic_across_calls():
    a = make_modular_dataset(p=11, seed=42, device="cpu")
    b = make_modular_dataset(p=11, seed=42, device="cpu")
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_different_seeds_produce_different_splits():
    a_tx, _, _, _ = make_modular_dataset(p=11, seed=1, device="cpu")
    b_tx, _, _, _ = make_modular_dataset(p=11, seed=2, device="cpu")
    assert not torch.equal(a_tx, b_tx)
