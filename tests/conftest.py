"""
Shared pytest fixtures for pan_lab tests.

Every test uses a tiny config (P=11, K=3, 200 steps) so the whole suite
runs in under a minute on CPU. No test depends on grokking actually
happening — only on the mechanics of the code path.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from pan_lab.config import RunConfig
from pan_lab.data   import make_modular_dataset


@pytest.fixture
def tiny_cfg() -> RunConfig:
    """Smallest useful config. P=11 -> 121 pairs, K=3, 200 steps."""
    return RunConfig(
        p           = 11,
        k_freqs     = 3,
        n_steps     = 200,
        batch_size  = 64,
        log_every   = 50,
        seed        = 42,
        weight_decay = 0.01,
        diversity_weight = 0.01,
        val_samples = None,
        use_compile = False,
        early_stop  = False,
        label       = "test",
    )


@pytest.fixture
def tmp_outdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="panlab_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tiny_data():
    tx, ty, vx, vy = make_modular_dataset(
        p=11, task_kind="mod_add", train_frac=0.4, seed=42, device="cpu",
    )
    return tx, ty, vx, vy
