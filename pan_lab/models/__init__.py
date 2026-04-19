"""Model registry for pan_lab."""
from pan_lab.models.pan         import (
    PhaseEncoder,
    PhaseMixingLayer,
    PhaseGate,
    PhaseAccumulatorNetwork,
)
from pan_lab.models.wan         import (
    WalshEncoder,
    WalshMixingLayer,
    WalshGate,
    WalshAccumulatorNetwork,
)
from pan_lab.models.transformer import TransformerBaseline
from pan_lab.models.quantize    import quantize_phase_sifp16


def make_model(cfg):
    """
    Build the model described by cfg.model_kind.

    Centralized so experiments and sweeps never have to reach into the
    specific model classes directly.

    Seeds the RNG before construction so that weight initialization is
    a deterministic function of cfg.seed. Without this, two runs with
    the same cfg could produce different random init states depending
    on whatever code ran before make_model (e.g. another model built
    in the same process), and end up with different final val_acc
    curves despite identical configs.
    """
    import random
    import numpy as np
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.model_kind == "pan":
        n_inputs = 3 if cfg.task_kind == "mod_two_step" else 2
        return PhaseAccumulatorNetwork(
            p         = cfg.p,
            k_freqs   = cfg.k_freqs,
            n_inputs  = n_inputs,
            freq_init = cfg.freq_init,
        )
    if cfg.model_kind == "wan":
        from pan_lab.data import walsh_task_shape
        n_inputs, n_classes = walsh_task_shape(cfg)
        return WalshAccumulatorNetwork(
            n_bits    = cfg.n_bits,
            k_freqs   = cfg.k_freqs,
            n_inputs  = n_inputs,
            n_classes = n_classes,
            mask_init = cfg.mask_init,
        )
    if cfg.model_kind == "transformer":
        # Transformer baseline over 2^n_bits tokens for Walsh tasks.
        if cfg.task_kind.startswith("walsh_"):
            from pan_lab.data import walsh_task_shape
            n_inputs, n_classes = walsh_task_shape(cfg)
            return TransformerBaseline(
                p        = 1 << cfg.n_bits,
                d_model  = cfg.d_model,
                n_heads  = cfg.n_heads,
                d_mlp    = cfg.d_mlp,
                n_inputs = n_inputs,
                n_classes = n_classes,
            )
        return TransformerBaseline(
            p        = cfg.p,
            d_model  = cfg.d_model,
            n_heads  = cfg.n_heads,
            d_mlp    = cfg.d_mlp,
            n_inputs = 3 if cfg.task_kind == "mod_two_step" else 2,
        )
    raise ValueError(f"Unknown model_kind: {cfg.model_kind!r}")


__all__ = [
    "PhaseEncoder",
    "PhaseMixingLayer",
    "PhaseGate",
    "PhaseAccumulatorNetwork",
    "WalshEncoder",
    "WalshMixingLayer",
    "WalshGate",
    "WalshAccumulatorNetwork",
    "TransformerBaseline",
    "quantize_phase_sifp16",
    "make_model",
]
