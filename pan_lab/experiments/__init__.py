"""pan_lab.experiments — named experiments, driven by YAML configs."""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Dict, Optional

import yaml

from pan_lab.config import RunConfig
from pan_lab.reporting import ExperimentReporter

from .base import _print_plan, _run_cfgs
from .compare import CompareExperiment
from .decoder_analysis import DecoderAnalysisExperiment
from .factory import build_factory_runner, can_run_with_factory
from .decoder_swap import DecoderSwapExperiment
from .dw_sweep import DWSweepExperiment
from .freq_init_ablation import FreqInitAblationExperiment
from .held_out_primes import HeldOutPrimesExperiment
from .k8_sweep import K8SweepExperiment
from .k_sweep import KSweepExperiment
from .mod_mul import ModMulExperiment
from .mod_two_step import ModTwoStepExperiment
from .plugins import ANALYZER_REGISTRY, PLOT_REGISTRY
from .primes import PrimesExperiment
from .sifp16_inference import SIFP16InferenceExperiment
from .slot_census import SlotCensusExperiment
from .tf_sweep import TFSweepExperiment
from .tier3 import Tier3Experiment
from .wd_sweep import WDSweepExperiment

EXPERIMENT_REGISTRY: Dict[str, Callable] = {}
_RUN_CONFIG_FIELDS = {f.name for f in dataclasses.fields(RunConfig)}
_CAPTURE_FIELDS = {"ablations", "slots", "checkpoints", "stream_curves"}
_PLOT_FIELDS = {"type", "group_by", "title", "filename"}
_V1_ALLOWED_KEYS = {"experiment", "out_dir", "dry_run", "base", "experiment_args", "schema_version"}
_V2_ALLOWED_KEYS = _V1_ALLOWED_KEYS | {"grid", "capture", "analyses", "plots"}
_PLOT_DEFAULT_FILENAMES = {
    "reliability": "reliability.png",
    "curves": "curves.png",
    "parameter_efficiency": "parameter_efficiency.png",
    "ablations": "ablations.png",
    "slot_census": "slot_census.png",
    "decoder_analysis": "decoder_analysis.png",
}


class ExperimentSpecValidationError(ValueError):
    """Raised when an experiment YAML spec fails strict schema validation."""


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


def _ensure_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentSpecValidationError(f"{field_name!r} must be a mapping, got {type(value).__name__}")
    return value


def _strict_validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise ExperimentSpecValidationError(f"Experiment YAML root must be a mapping, got {type(spec).__name__}")

    schema_version = int(spec.get("schema_version", 1))
    allowed = _V2_ALLOWED_KEYS if schema_version >= 2 else _V1_ALLOWED_KEYS
    unknown_top = sorted(set(spec) - allowed)
    if unknown_top:
        raise ExperimentSpecValidationError(f"Unknown top-level YAML keys: {unknown_top}")

    name = spec.get("experiment")
    if not isinstance(name, str) or not name.strip():
        raise ExperimentSpecValidationError("Missing or invalid 'experiment' field")
    if name not in EXPERIMENT_REGISTRY:
        raise ExperimentSpecValidationError(
            f"Unknown experiment {name!r}. Available: {sorted(EXPERIMENT_REGISTRY)}"
        )

    base = _ensure_mapping(spec.get("base", {}), "base")
    unknown_base = sorted(set(base) - _RUN_CONFIG_FIELDS)
    if unknown_base:
        raise ExperimentSpecValidationError(f"Unknown base RunConfig keys: {unknown_base}")

    if schema_version < 2:
        return

    _ensure_mapping(spec.get("grid", {}), "grid")
    capture = _ensure_mapping(spec.get("capture", {}), "capture")
    unknown_capture = sorted(set(capture) - _CAPTURE_FIELDS)
    if unknown_capture:
        raise ExperimentSpecValidationError(f"Unknown capture keys: {unknown_capture}")

    analyses = spec.get("analyses", []) or []
    if not isinstance(analyses, list) or any(not isinstance(x, str) for x in analyses):
        raise ExperimentSpecValidationError("'analyses' must be a list[str]")
    unknown_analyses = sorted(set(analyses) - set(ANALYZER_REGISTRY))
    if unknown_analyses:
        raise ExperimentSpecValidationError(
            f"Unknown analyzer plugin names: {unknown_analyses}. "
            f"Available: {sorted(ANALYZER_REGISTRY)}"
        )

    plots = spec.get("plots", []) or []
    if not isinstance(plots, list):
        raise ExperimentSpecValidationError("'plots' must be a list[dict]")
    for idx, plot in enumerate(plots):
        if not isinstance(plot, dict):
            raise ExperimentSpecValidationError(f"plots[{idx}] must be a mapping")
        unknown_plot = sorted(set(plot) - _PLOT_FIELDS)
        if unknown_plot:
            raise ExperimentSpecValidationError(f"plots[{idx}] has unknown keys: {unknown_plot}")
        plot_type = plot.get("type")
        if not isinstance(plot_type, str) or not plot_type:
            raise ExperimentSpecValidationError(f"plots[{idx}].type must be a non-empty string")
        if plot_type not in PLOT_REGISTRY:
            raise ExperimentSpecValidationError(
                f"Unknown plot plugin name {plot_type!r}. Available: {sorted(PLOT_REGISTRY)}"
            )


def _effective_capture(name: str, exp_args: dict[str, Any]) -> dict[str, bool]:
    capture = {"ablations": False, "slots": False, "checkpoints": False, "stream_curves": False}
    if can_run_with_factory(name):
        spec = build_factory_runner(name).spec
        capture.update(spec.default_capture.__dict__)
    else:
        exp = EXPERIMENT_REGISTRY[name]
        capture["ablations"] = bool(getattr(exp, "collect_ablations", False))
        capture["slots"] = bool(getattr(exp, "collect_slots", False))
    capture.update({k: bool(v) for k, v in (exp_args.get("capture") or {}).items() if k in capture})
    return capture


def _expanded_cfgs(name: str, base: RunConfig, exp_args: dict[str, Any]) -> list[RunConfig]:
    clean_args = dict(exp_args)
    clean_args.pop("_schema_version", None)
    if can_run_with_factory(name):
        runner = build_factory_runner(name)
        grid = {k: v for k, v in clean_args.items() if k not in {"capture", "analyses", "plots"}}
        return runner.spec.expand_configs(base=base, grid=grid)

    exp = EXPERIMENT_REGISTRY[name]
    if hasattr(exp, "build_configs"):
        return exp.build_configs(base=base, **clean_args)
    return [base]


def build_execution_manifest(name: str, base: RunConfig, out_dir: str, exp_args: dict[str, Any]) -> dict[str, Any]:
    cfgs = _expanded_cfgs(name, base, exp_args)
    capture = _effective_capture(name, exp_args)
    analyses = sorted(set((exp_args.get("analyses") or [])))
    plots = list(exp_args.get("plots") or [])

    expected_files = {"runs.csv", "curves.csv", "manifest.json"}
    if capture["ablations"]:
        expected_files.add("ablations.csv")
    if capture["slots"]:
        expected_files.add("slots.csv")
    if capture["checkpoints"]:
        expected_files.add("checkpoints.csv")
    if capture["stream_curves"]:
        expected_files.add("curves_stream.csv")
    for analyzer in analyses:
        expected_files.add(f"{analyzer}.csv")
    for plot in plots:
        ptype = str(plot["type"])
        expected_files.add(str(plot.get("filename") or _PLOT_DEFAULT_FILENAMES.get(ptype, f"{ptype}.png")))
    if any(cfg.save_model for cfg in cfgs):
        expected_files.add("model_<run_id>.pt")
    if name == "decoder_analysis":
        expected_files.update(
            {"decoder_analysis.csv", "decoder_recovery_curve.csv", "decoder_residual_spectrum.csv"}
        )

    return {
        "experiment": name,
        "out_dir": out_dir,
        "expanded_runs": [
            {"run_id": cfg.display_id(), "config": cfg.as_dict()}
            for cfg in cfgs
        ],
        "capture": capture,
        "analyses": analyses,
        "plots": plots,
        "expected_outputs": sorted(expected_files),
    }


def _print_dry_run_manifest(manifest: dict[str, Any]) -> None:
    print("\n── execution manifest ──")
    print(f"  expanded runs: {len(manifest['expanded_runs'])}")
    for row in manifest["expanded_runs"]:
        print(f"  - {row['run_id']}: {json.dumps(row['config'], sort_keys=True)}")
    print(f"  capture modules: {json.dumps(manifest['capture'], sort_keys=True)}")
    print(f"  analyzers: {manifest['analyses'] or ['(none)']}")
    print(f"  plots: {manifest['plots'] or ['(none)']}")
    print("  expected outputs:")
    for path in manifest["expected_outputs"]:
        print(f"    - {path}")


def _normalize_experiment_args(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy + v2 experiment YAML fields into exp_args."""
    exp_args = dict(spec.get("experiment_args", {}) or {})

    schema_version = int(spec.get("schema_version", 1))
    exp_args["_schema_version"] = schema_version
    if schema_version >= 2:
        grid = spec.get("grid", {}) or {}
        capture = spec.get("capture", {}) or {}
        analyses = spec.get("analyses", []) or []
        plots = spec.get("plots", []) or []

        # `grid` is the declarative replacement for legacy `experiment_args`.
        exp_args.update(grid)
        exp_args["capture"] = capture
        exp_args["analyses"] = analyses
        exp_args["plots"] = plots

    return exp_args


def load_experiment_yaml(path: str) -> tuple:
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    _strict_validate_spec(spec)

    name = spec["experiment"]
    out_dir = spec.get("out_dir", f"results/{name}")
    dry_run = bool(spec.get("dry_run", False))

    base_dict = spec.get("base", {})
    base_cfg = RunConfig.from_dict(base_dict)

    exp_args = _normalize_experiment_args(spec)
    return name, base_cfg, out_dir, dry_run, exp_args


def run_experiment(
    name: str,
    base: RunConfig,
    out_dir: str,
    dry_run: bool = False,
    **exp_args,
) -> ExperimentReporter:
    schema_version = int(exp_args.get("_schema_version", 1))
    if dry_run:
        _print_dry_run_manifest(build_execution_manifest(name, base, out_dir, exp_args))
    exp_args = dict(exp_args)
    exp_args.pop("_schema_version", None)

    if schema_version >= 2 and can_run_with_factory(name):
        runner = build_factory_runner(name)
        grid = {
            k: v
            for k, v in exp_args.items()
            if k not in {"capture", "analyses", "plots"}
        }
        return runner.run(
            base=base,
            out_dir=out_dir,
            dry_run=dry_run,
            grid=grid,
            capture=exp_args.get("capture"),
            analyses=exp_args.get("analyses"),
            plots=exp_args.get("plots"),
        )

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
    "ExperimentSpecValidationError",
    "register",
    "_print_plan",
    "_run_cfgs",
    "load_experiment_yaml",
    "build_execution_manifest",
    "run_experiment",
    "run_from_yaml",
]
