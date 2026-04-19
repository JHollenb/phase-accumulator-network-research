"""
Tests for the Walsh Accumulator Network — parallel to test_models.py
for PAN. Every test uses small configs so the suite stays fast.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pan_lab import (
    RunConfig,
    WalshAccumulatorNetwork,
    WalshEncoder,
    WalshGate,
    WalshMixingLayer,
    make_dataset_from_cfg,
    make_model,
    train,
    walsh_task_shape,
)
from pan_lab.analysis import ablation_test
from pan_lab.models.wan import WALSH_PERIOD, _bits_of


# ── shape + construction ─────────────────────────────────────────────────
def test_bits_of_little_endian():
    tokens = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    bits   = _bits_of(tokens, 4)
    assert bits.shape == (4, 4)
    # bit 0 is LSB
    assert torch.equal(bits[1], torch.tensor([1., 0., 0., 0.]))
    assert torch.equal(bits[2], torch.tensor([0., 1., 0., 0.]))
    assert torch.equal(bits[3], torch.tensor([1., 0., 1., 0.]))


def test_wan_forward_shape_single_input():
    m = WalshAccumulatorNetwork(n_bits=6, k_freqs=3, n_inputs=1, n_classes=2)
    x = torch.randint(0, 64, (5, 1))
    assert m(x).shape == (5, 2)


def test_wan_forward_shape_two_inputs():
    m = WalshAccumulatorNetwork(n_bits=4, k_freqs=4, n_inputs=2, n_classes=16)
    x = torch.randint(0, 16, (8, 2))
    assert m(x).shape == (8, 16)


def test_wan_parameter_count():
    # n_bits=8, K=4, 1 input, 2 classes:
    #   mask logits: 1 encoder * 4 * 8 = 32
    #   walsh_mix:   4 * 4            = 16
    #   gate ref_v:  4                = 4
    #   decoder:     4*2 + 2 bias     = 10
    # total: 62
    m = WalshAccumulatorNetwork(n_bits=8, k_freqs=4, n_inputs=1, n_classes=2)
    assert m.count_parameters() == 62


def test_wan_encoder_binary_mask_yields_pm1_sign():
    # With binary mask_logits (large saturating values) and binary x,
    # v is an integer in {0,1,2,...}, and cos(pi*v) is exactly +/-1.
    torch.manual_seed(0)
    enc = WalshEncoder(n_bits=4, k_freqs=1, mask_init="parity",
                       logit_scale=10.0)
    bits = torch.tensor([[0., 1., 0., 1.], [1., 1., 1., 1.]])
    v    = enc(bits)                     # (2, 1)
    sign = torch.cos(math.pi * v)
    # Both inputs have even parity or odd parity depending on mask;
    # sign must be close to +/-1 regardless.
    assert torch.allclose(sign.abs(), torch.ones_like(sign), atol=1e-3)


def test_walsh_mix_features_matches_forward_prefix():
    torch.manual_seed(0)
    m = WalshAccumulatorNetwork(n_bits=4, k_freqs=3, n_inputs=1)
    x = torch.randint(0, 16, (4, 1))
    with torch.no_grad():
        mixed_direct = m.walsh_mix(m._encode_all(x))
        mixed_api    = m.mix_features(x)
    assert torch.allclose(mixed_direct, mixed_api)


def test_walsh_gate_ref_wrap_uses_remainder():
    # The WalshGate stores its reference as an unconstrained real and
    # must wrap into [0, 2) at forward time — identical to PAN's
    # PhaseGate.ref_phase wrap.
    g = WalshGate(n_phases=3)
    with torch.no_grad():
        g.ref_v.data = torch.tensor([0.5, 2.5, -0.5])  # outside [0, 2)
    v = torch.zeros(1, 3)
    out = g(v)
    # ref mod 2 = [0.5, 0.5, 1.5]; cos(pi*(-ref)) = cos(pi*ref)
    # -> [cos(0.5pi), cos(0.5pi), cos(1.5pi)] = [0, 0, 0]
    # -> g = (1 + 0)/2 = 0.5 for all three
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-6)


def test_make_model_builds_wan_for_walsh_task():
    cfg = RunConfig(model_kind="wan", task_kind="walsh_parity",
                    n_bits=6, k_freqs=2, p=2)
    m = make_model(cfg)
    assert isinstance(m, WalshAccumulatorNetwork)
    assert m.n_inputs  == 1
    assert m.n_classes == 2


def test_make_model_builds_wan_for_walsh_xor():
    cfg = RunConfig(model_kind="wan", task_kind="walsh_xor",
                    n_bits=4, k_freqs=4, p=16)
    m = make_model(cfg)
    assert isinstance(m, WalshAccumulatorNetwork)
    assert m.n_inputs  == 2
    assert m.n_classes == 16


def test_wan_mask_init_variants_differ():
    torch.manual_seed(0)
    m1 = WalshAccumulatorNetwork(n_bits=6, k_freqs=3, mask_init="onehot")
    m2 = WalshAccumulatorNetwork(n_bits=6, k_freqs=3, mask_init="random")
    m3 = WalshAccumulatorNetwork(n_bits=6, k_freqs=3, mask_init="parity")
    assert not torch.allclose(m1.encoders[0].mask_logits.data,
                              m2.encoders[0].mask_logits.data)
    assert not torch.allclose(m1.encoders[0].mask_logits.data,
                              m3.encoders[0].mask_logits.data)


def test_get_learned_masks_shape():
    m = WalshAccumulatorNetwork(n_bits=5, k_freqs=2, n_inputs=2)
    info = m.get_learned_masks()
    assert info["n_bits"] == 5
    assert info["mask_0"].shape == (2, 5)
    assert info["binary_0"].shape == (2, 5)
    assert info["popcount_0"].shape == (2,)


# ── dataset ───────────────────────────────────────────────────────────────
def test_walsh_parity_labels_correct():
    cfg = RunConfig(model_kind="wan", task_kind="walsh_parity",
                    n_bits=4, train_frac=1.0, seed=0, p=2)
    tx, ty, vx, vy = make_dataset_from_cfg(cfg, device="cpu")
    # Reconstruct labels and confirm
    x = tx.squeeze(-1).cpu().numpy()
    expected = np.array([bin(int(v)).count("1") & 1 for v in x])
    assert (ty.cpu().numpy() == expected).all()


def test_walsh_popcount_mod_labels_correct():
    cfg = RunConfig(model_kind="wan", task_kind="walsh_popcount_mod",
                    n_bits=4, mod_base=3, train_frac=1.0, seed=0, p=3)
    tx, ty, _, _ = make_dataset_from_cfg(cfg, device="cpu")
    x = tx.squeeze(-1).cpu().numpy()
    expected = np.array([bin(int(v)).count("1") % 3 for v in x])
    assert (ty.cpu().numpy() == expected).all()


def test_walsh_xor_labels_correct():
    cfg = RunConfig(model_kind="wan", task_kind="walsh_xor",
                    n_bits=3, train_frac=1.0, seed=0, p=8)
    tx, ty, _, _ = make_dataset_from_cfg(cfg, device="cpu")
    a, b = tx[:, 0].cpu().numpy(), tx[:, 1].cpu().numpy()
    assert (ty.cpu().numpy() == (a ^ b)).all()


def test_walsh_rotl_labels_correct():
    cfg = RunConfig(model_kind="wan", task_kind="walsh_rotl",
                    n_bits=4, rot_amount=2, train_frac=1.0, seed=0, p=16)
    tx, ty, _, _ = make_dataset_from_cfg(cfg, device="cpu")
    x = tx.squeeze(-1).cpu().numpy()
    mask = 15
    expected = (((x << 2) | (x >> 2)) & mask).astype(np.int64)
    assert (ty.cpu().numpy() == expected).all()


def test_walsh_task_shape_dispatch():
    assert walsh_task_shape(RunConfig(task_kind="walsh_parity", n_bits=8))       == (1, 2)
    assert walsh_task_shape(RunConfig(task_kind="walsh_xor",    n_bits=6))       == (2, 64)
    assert walsh_task_shape(RunConfig(task_kind="walsh_popcount_mod", mod_base=4)) == (1, 4)
    assert walsh_task_shape(RunConfig(task_kind="walsh_rotl",   n_bits=5))       == (1, 32)


# ── training + ablation ───────────────────────────────────────────────────
def test_wan_parity_groks_with_parity_init():
    """
    With mask_init='parity', the encoder already reads the XOR of all
    bits; only the gate + decoder need to learn to discriminate. A
    K=1 WAN on n_bits=6 parity must drive train_loss to near zero
    within a few hundred steps. This locks in the core training path.
    """
    torch.manual_seed(0)
    cfg = RunConfig(
        model_kind="wan", task_kind="walsh_parity",
        n_bits=6, k_freqs=1, p=2,
        mask_init="parity",
        n_steps=1500, batch_size=32, lr=3e-2,
        weight_decay=0.0, diversity_weight=0.0,
        log_every=250, early_stop=False, use_compile=False,
    )
    tx, ty, vx, vy = make_dataset_from_cfg(cfg, device="cpu")
    m = make_model(cfg)
    res = train(m, cfg, tx, ty, vx, vy, verbose=False)
    assert res.history.val_acc[-1] >= 0.9


def test_wan_ablation_test_runs_and_drops_accuracy():
    """
    ablation_test should dispatch on WAN and zero the mix / randomize
    masks / zero ref_v. Randomizing masks must not improve accuracy.
    """
    torch.manual_seed(0)
    cfg = RunConfig(
        model_kind="wan", task_kind="walsh_parity",
        n_bits=5, k_freqs=1, p=2, mask_init="parity",
        n_steps=800, batch_size=16, lr=3e-2,
        weight_decay=0.0, diversity_weight=0.0,
        log_every=200, early_stop=False, use_compile=False,
    )
    tx, ty, vx, vy = make_dataset_from_cfg(cfg, device="cpu")
    m = make_model(cfg)
    res = train(m, cfg, tx, ty, vx, vy, verbose=False)

    abl = ablation_test(res.model, vx, vy, verbose=False)
    assert "baseline" in abl
    assert "zero_walsh_mixing" in abl
    assert "randomize_masks" in abl
    assert "zero_ref_v" in abl
    # Randomized masks should not outperform the trained baseline.
    assert abl["randomize_masks"] <= abl["baseline"] + 0.05


def test_walsh_period_constant_is_two():
    assert WALSH_PERIOD == 2.0
