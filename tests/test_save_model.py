"""
Regression test for RunConfig.save_model.

This field existed in config.py and was referenced in YAMLs and the
README for three days without ever producing a .pt file on disk,
because the experiment runners never honored it. This test forces
save_model=True through a real _run_cfgs call and asserts the .pt
file exists with the expected structure.

Add this to tests/ as test_save_model.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from pan_lab.config      import RunConfig
from pan_lab.experiments import _run_cfgs


def test_save_model_writes_pt_file(tmp_outdir):
    cfg = RunConfig(
        p=11, k_freqs=3, n_steps=50, batch_size=32, log_every=25,
        seed=42, weight_decay=0.01, diversity_weight=0.01,
        val_samples=None, use_compile=False, early_stop=False,
        label="save_test",
        save_model=True,
    )

    rep = _run_cfgs(
        [cfg], "save_test", str(tmp_outdir),
        dry_run=False, ablations=False, slots=False,
    )

    # Exactly one .pt should exist
    pts = list(Path(tmp_outdir).glob("model_*.pt"))
    assert len(pts) == 1, f"expected 1 .pt file, got {[p.name for p in pts]}"

    # Payload schema
    ckpt = torch.load(str(pts[0]), weights_only=False)
    assert "state_dict"  in ckpt
    assert "arch"        in ckpt
    assert "config"      in ckpt
    assert "grok_step"   in ckpt
    assert "param_count" in ckpt
    assert ckpt["arch"] == "PAN"
    assert ckpt["config"]["p"]       == 11
    assert ckpt["config"]["k_freqs"] == 3

    # state_dict actually round-trips into a model
    from pan_lab.models.pan import PhaseAccumulatorNetwork
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    m.load_state_dict(ckpt["state_dict"])


def test_save_model_false_writes_no_pt_file(tmp_outdir):
    cfg = RunConfig(
        p=11, k_freqs=3, n_steps=50, batch_size=32, log_every=25,
        seed=42, weight_decay=0.01, diversity_weight=0.01,
        val_samples=None, use_compile=False, early_stop=False,
        label="nosave_test",
        save_model=False,
    )
    _run_cfgs([cfg], "nosave_test", str(tmp_outdir),
              dry_run=False, ablations=False, slots=False)
    pts = list(Path(tmp_outdir).glob("model_*.pt"))
    assert len(pts) == 0
