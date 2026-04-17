"""Tests for pan_lab.trainer — training mechanics, not convergence."""
from __future__ import annotations

import pytest
import torch

from pan_lab import RunConfig, make_model, train
from pan_lab.data import make_modular_dataset


def _fresh(cfg):
    tx, ty, vx, vy = make_modular_dataset(
        p=cfg.p, task_kind=cfg.task_kind, seed=cfg.seed, device="cpu")
    m = make_model(cfg)
    return m, tx, ty, vx, vy


def test_train_returns_populated_history(tiny_cfg):
    m, tx, ty, vx, vy = _fresh(tiny_cfg)
    r = train(m, tiny_cfg, tx, ty, vx, vy, verbose=False)
    assert r.model is m
    assert r.param_count > 0
    assert len(r.history.steps) > 0
    assert len(r.history.val_acc) == len(r.history.steps)
    assert len(r.history.train_loss) == len(r.history.steps)


def test_train_dry_run_does_nothing(tiny_cfg):
    cfg = tiny_cfg.with_overrides(dry_run=True)
    m, tx, ty, vx, vy = _fresh(cfg)
    r = train(m, cfg, tx, ty, vx, vy, verbose=False)
    assert r.elapsed_s == 0.0
    assert len(r.history.steps) == 0
    assert r.history.grok_step is None


def test_same_seed_same_config_same_history(tiny_cfg):
    """Reproducibility: two identical runs produce identical val_acc curves."""
    cfg = tiny_cfg
    a_model, a_tx, a_ty, a_vx, a_vy = _fresh(cfg)
    a = train(a_model, cfg, a_tx, a_ty, a_vx, a_vy, verbose=False)

    b_model, b_tx, b_ty, b_vx, b_vy = _fresh(cfg)
    b = train(b_model, cfg, b_tx, b_ty, b_vx, b_vy, verbose=False)

    assert a.history.steps == b.history.steps
    assert a.history.val_acc == pytest.approx(b.history.val_acc, abs=1e-6)


def test_different_seeds_differ(tiny_cfg):
    a_cfg = tiny_cfg.with_overrides(seed=1)
    b_cfg = tiny_cfg.with_overrides(seed=2)
    a_model, a_tx, a_ty, a_vx, a_vy = _fresh(a_cfg)
    b_model, b_tx, b_ty, b_vx, b_vy = _fresh(b_cfg)
    a = train(a_model, a_cfg, a_tx, a_ty, a_vx, a_vy, verbose=False)
    b = train(b_model, b_cfg, b_tx, b_ty, b_vx, b_vy, verbose=False)
    # Train_loss should differ somewhere
    assert a.history.train_loss != b.history.train_loss


def test_early_stop_triggers_on_high_accuracy(tiny_cfg, monkeypatch):
    """
    Patch the val-accuracy path so we can force a grok detection without
    actually waiting for the model to converge. This tests the mechanism,
    not whether a tiny PAN can grok mod-11 in 200 steps.
    """
    cfg = tiny_cfg.with_overrides(early_stop=True, grok_threshold=0.5,
                                   n_steps=5000, log_every=10)
    m, tx, ty, vx, vy = _fresh(cfg)

    # Replace val_y with labels the model WILL match often enough — use
    # a single fixed label so even a random model has 1/11 accuracy, then
    # lower grok_threshold to 0.05.
    cfg = cfg.with_overrides(grok_threshold=0.05)
    vy_uniform = torch.zeros_like(vy)
    r = train(m, cfg, tx, ty, vx, vy_uniform, verbose=False)

    # With an untrained model and grok_threshold=0.05, the first eval
    # should trigger grok detection and early-stop within a few steps.
    assert r.history.grok_step is not None
    assert r.history.steps[-1] < cfg.n_steps - 1


def test_diversity_reg_actually_updates_encoder_freqs(tiny_cfg):
    """
    End-to-end version of the gradient-flow test. After a few steps
    with DW>0, the encoder frequencies should have *moved* from their
    Fourier init.
    """
    cfg = tiny_cfg.with_overrides(n_steps=50, diversity_weight=0.5)
    m, tx, ty, vx, vy = _fresh(cfg)
    init_freqs = torch.stack([enc.freq.data.clone() for enc in m.encoders])
    train(m, cfg, tx, ty, vx, vy, verbose=False)
    final_freqs = torch.stack([enc.freq.data for enc in m.encoders])
    assert (init_freqs - final_freqs).abs().sum() > 1e-4


def test_transformer_trains_without_errors(tiny_cfg):
    cfg = tiny_cfg.with_overrides(
        model_kind="transformer", d_model=16, n_heads=2, d_mlp=32,
        weight_decay=1.0,
    )
    m, tx, ty, vx, vy = _fresh(cfg)
    r = train(m, cfg, tx, ty, vx, vy, verbose=False)
    assert len(r.history.val_acc) > 0
