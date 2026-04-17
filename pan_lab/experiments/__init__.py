"""pan_lab.experiments — named experiments, driven by YAML configs."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import yaml

from pan_lab.config import RunConfig
from pan_lab.reporting import ExperimentReporter

from .base import _print_plan, _run_cfgs
from .compare import CompareExperiment
from .decoder_analysis import DecoderAnalysisExperiment
from .decoder_swap import DecoderSwapExperiment
from .dw_sweep import DWSweepExperiment
from .freq_init_ablation import FreqInitAblationExperiment
from .held_out_primes import HeldOutPrimesExperiment
from .k8_sweep import K8SweepExperiment
from .k_sweep import KSweepExperiment
from .mod_mul import ModMulExperiment
from .mod_two_step import ModTwoStepExperiment
from .primes import PrimesExperiment
from .sifp16_inference import SIFP16InferenceExperiment
from .slot_census import SlotCensusExperiment
from .tf_sweep import TFSweepExperiment
from .tier3 import Tier3Experiment
from .wd_sweep import WDSweepExperiment

EXPERIMENT_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def _wrap(fn):
        EXPERIMENT_REGISTRY[name] = fn
        return fn

    return _wrap


def _register_default_experiments() -> None:
    for exp in [
        CompareExperiment(),
        KSweepExperiment(),
        DWSweepExperiment(),
        WDSweepExperiment(),
        K8SweepExperiment(),
        PrimesExperiment(),
        HeldOutPrimesExperiment(),
        Tier3Experiment(),
        SlotCensusExperiment(),
        FreqInitAblationExperiment(),
        SIFP16InferenceExperiment(),
        DecoderSwapExperiment(),
        ModMulExperiment(),
        ModTwoStepExperiment(),
        TFSweepExperiment(),
        DecoderAnalysisExperiment(),
    ]:
        EXPERIMENT_REGISTRY[exp.name] = exp


_register_default_experiments()


def load_experiment_yaml(path: str) -> tuple:
    with open(path, "r") as f:
        spec = yaml.safe_load(f)

    name = spec["experiment"]
    out_dir = spec.get("out_dir", f"results/{name}")
    dry_run = bool(spec.get("dry_run", False))

    base_dict = spec.get("base", {})
    base_cfg = RunConfig.from_dict(base_dict)

    exp_args = spec.get("experiment_args", {}) or {}
    return name, base_cfg, out_dir, dry_run, exp_args


def run_experiment(
    name: str,
    base: RunConfig,
    out_dir: str,
    dry_run: bool = False,
    **exp_args,
) -> ExperimentReporter:
    if name not in EXPERIMENT_REGISTRY:
        raise KeyError(f"Unknown experiment: {name!r}. Available: {sorted(EXPERIMENT_REGISTRY)}")
    fn = EXPERIMENT_REGISTRY[name]
    return fn(base=base, out_dir=out_dir, dry_run=dry_run, **exp_args)


def run_from_yaml(path: str, force_dry_run: Optional[bool] = None):
    name, base, out_dir, dry_run, exp_args = load_experiment_yaml(path)
    if force_dry_run is not None:
        dry_run = force_dry_run
    print(f"\n▶ loading {path}: experiment={name!r} out={out_dir} dry_run={dry_run}")
    return run_experiment(name, base, out_dir, dry_run, **exp_args)


__all__ = [
    "EXPERIMENT_REGISTRY",
    "register",
    "_print_plan",
    "_run_cfgs",
    "load_experiment_yaml",
    "run_experiment",
    "run_from_yaml",
]
