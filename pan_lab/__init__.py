"""
pan_lab — Phase Accumulator Network research library.

Public API:

    # Models
    from pan_lab import PhaseAccumulatorNetwork, TransformerBaseline
    from pan_lab.models import make_model

    # Data
    from pan_lab import make_modular_dataset

    # Training
    from pan_lab import RunConfig, train, TrainResult

    # Analysis
    from pan_lab import analyze_pan, ablation_test, fourier_concentration

    # Experiments (YAML-driven)
    from pan_lab.experiments import run_experiment, EXPERIMENT_REGISTRY

The library separates *what* you run (experiments/*.yaml) from *how* it
runs (pan_lab.trainer) from *the primitives* (models, data). Any experiment
that doesn't exist today can be added in three steps: write the YAML, add
a function in pan_lab/experiments.py, register it.
"""
from pan_lab.config import RunConfig, DEVICE, TWO_PI, PHASE_SCALE
from pan_lab.data import make_modular_dataset
from pan_lab.models import (
    PhaseAccumulatorNetwork,
    PhaseEncoder,
    PhaseMixingLayer,
    PhaseGate,
    TransformerBaseline,
    make_model,
)
from pan_lab.trainer import train, TrainResult, TrainHistory
from pan_lab.analysis import (
    analyze_pan,
    ablation_test,
    fourier_concentration,
    compute_frequency_errors,
    detect_mode_collapse,
    slot_activation_census,
)

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "DEVICE",
    "TWO_PI",
    "PHASE_SCALE",
    "make_modular_dataset",
    "PhaseAccumulatorNetwork",
    "PhaseEncoder",
    "PhaseMixingLayer",
    "PhaseGate",
    "TransformerBaseline",
    "make_model",
    "train",
    "TrainResult",
    "TrainHistory",
    "analyze_pan",
    "ablation_test",
    "fourier_concentration",
    "compute_frequency_errors",
    "detect_mode_collapse",
    "slot_activation_census",
]
