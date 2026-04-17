"""
Test that the models are constructed correctly and produce tensors with
the expected shapes. These tests don't train — they only call forward.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pan_lab import (
    PhaseAccumulatorNetwork,
    TransformerBaseline,
    make_model,
    RunConfig,
)
from pan_lab.config import TWO_PI


def test_pan_forward_shape_2input():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (5, 2))
    logits = m(x)
    assert logits.shape == (5, 11)


def test_pan_forward_shape_3input():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=3)
    x = torch.randint(0, 11, (5, 3))
    logits = m(x)
    assert logits.shape == (5, 11)


def test_transformer_forward_shape():
    m = TransformerBaseline(p=11, d_model=16, n_heads=2, d_mlp=32, n_inputs=2)
    x = torch.randint(0, 11, (5, 2))
    logits = m(x)
    assert logits.shape == (5, 11)


def test_pan_parameter_count_matches_paper_arithmetic():
    # P=113, K=9, N=2:
    #   2 encoders  * 9 freqs                        = 18
    #   phase_mix  = 9 * 18 (no bias in PhaseMixingLayer) = 162
    #   gate ref_phase                               = 9
    #   decoder weight = 9 * 113                     = 1017
    #   decoder bias   = 113                         = 113
    # total = 1319 (matches the paper claim)
    m = PhaseAccumulatorNetwork(p=113, k_freqs=9, n_inputs=2)
    assert m.count_parameters() == 1319


def test_pan_encoder_output_in_range():
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (32, 2))
    phi = m.encoders[0](x[:, 0])
    assert phi.min().item() >= 0
    assert phi.max().item() < TWO_PI + 1e-6     # modulo output in [0, 2pi)


def test_pan_mix_features_same_as_forward_prefix():
    """
    mix_features() should produce the same tensor as the mixing-layer
    stage of forward(). This is the call path diversity reg uses.
    """
    torch.manual_seed(0)
    m = PhaseAccumulatorNetwork(p=11, k_freqs=3, n_inputs=2)
    x = torch.randint(0, 11, (4, 2))
    with torch.no_grad():
        phis   = [enc(x[:, i]) for i, enc in enumerate(m.encoders)]
        concat = torch.cat(phis, dim=-1)
        mixed_direct = m.phase_mix(concat)
        mixed_api    = m.mix_features(x)
    assert torch.allclose(mixed_direct, mixed_api)


def test_make_model_builds_pan_for_mod_add():
    cfg = RunConfig(p=11, k_freqs=3, model_kind="pan", task_kind="mod_add")
    m = make_model(cfg)
    assert isinstance(m, PhaseAccumulatorNetwork)
    assert m.n_inputs == 2


def test_make_model_builds_pan_for_two_step():
    cfg = RunConfig(p=11, k_freqs=3, model_kind="pan",
                    task_kind="mod_two_step")
    m = make_model(cfg)
    assert isinstance(m, PhaseAccumulatorNetwork)
    assert m.n_inputs == 3


def test_make_model_builds_transformer():
    cfg = RunConfig(p=11, model_kind="transformer", d_model=16,
                    n_heads=2, d_mlp=32)
    m = make_model(cfg)
    assert isinstance(m, TransformerBaseline)


def test_random_freq_init_differs_from_fourier():
    torch.manual_seed(0)
    m1 = PhaseAccumulatorNetwork(p=11, k_freqs=3, freq_init="fourier")
    m2 = PhaseAccumulatorNetwork(p=11, k_freqs=3, freq_init="random")
    assert not torch.allclose(
        m1.encoders[0].freq.data, m2.encoders[0].freq.data
    )
