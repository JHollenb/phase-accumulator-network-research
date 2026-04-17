from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

from pan_lab.config import RunConfig
from pan_lab.experiments.base import _print_plan, _train_cfg
from pan_lab.experiments.plugins import run_analyzers, write_declared_plots, write_plugin_rows
from pan_lab.hooks import CSVStreamLogger, CheckpointLogger
from pan_lab.reporting import ExperimentReporter, save_model_weights


@dataclass
class ReporterPolicy:
    ablations: bool = False
    slots: bool = False
    checkpoints: bool = False
    stream_curves: bool = False


@dataclass
class ExperimentFactory:
    name: str
    default_grid: dict[str, list[Any]]
    fixed_overrides: dict[str, Any]
    run_label: Callable[[dict[str, Any]], str]
    grid_aliases: dict[str, str] = field(default_factory=dict)
    default_capture: ReporterPolicy = field(default_factory=ReporterPolicy)
    default_analyses: list[str] = field(default_factory=list)
    default_plots: list[dict[str, Any]] = field(default_factory=list)
    hook_injectors: list[Callable[[RunConfig], list[Any]]] = field(default_factory=list)

    def expand_configs(self, base: RunConfig, grid: dict[str, Any] | None = None) -> list[RunConfig]:
        merged = dict(self.default_grid)
        for k, v in (grid or {}).items():
            axis = self.grid_aliases.get(k, k)
            merged[axis] = v if isinstance(v, list) else [v]

        axes = sorted(merged)
        cfgs: list[RunConfig] = []
        for values in product(*(merged[a] for a in axes)):
            point = dict(zip(axes, values))
            overrides = dict(self.fixed_overrides)
            overrides.update(point)
            overrides["label"] = self.run_label(overrides)
            cfgs.append(base.with_overrides(**overrides))
        return cfgs


class FactoryBackedRunner:
    def __init__(self, spec: ExperimentFactory):
        self.spec = spec

    def _policy(self, capture: dict[str, Any] | None) -> ReporterPolicy:
        policy = ReporterPolicy(**self.spec.default_capture.__dict__)
        for k, v in (capture or {}).items():
            if hasattr(policy, k):
                setattr(policy, k, bool(v))
        return policy

    def _hook_factory(self, policy: ReporterPolicy, out_dir: str) -> Callable[[RunConfig], list[Any]]:
        stream_path = os.path.join(out_dir, "curves_stream.csv")
        if policy.stream_curves and os.path.exists(stream_path):
            os.remove(stream_path)

        def _hooks(cfg: RunConfig) -> list[Any]:
            hooks = []
            for injector in self.spec.hook_injectors:
                hooks.extend(injector(cfg))
            if policy.checkpoints:
                hooks.append(CheckpointLogger())
            if policy.stream_curves:
                hooks.append(CSVStreamLogger(stream_path, run_id=cfg.display_id()))
            return hooks

        return _hooks

    def run(
        self,
        base: RunConfig,
        out_dir: str,
        dry_run: bool = False,
        grid: dict[str, Any] | None = None,
        capture: dict[str, Any] | None = None,
        analyses: list[str] | None = None,
        plots: list[dict[str, Any]] | None = None,
        **_,
    ) -> ExperimentReporter:
        cfgs = self.spec.expand_configs(base=base, grid=grid)
        rep = ExperimentReporter(name=self.spec.name, out_dir=out_dir)

        if dry_run:
            _print_plan(cfgs, self.spec.name)
            return rep

        os.makedirs(out_dir, exist_ok=True)
        policy = self._policy(capture)
        hook_factory = self._hook_factory(policy, out_dir)
        state: dict[str, Any] = {}
        analyzer_names = analyses if analyses is not None else self.spec.default_analyses

        for cfg in cfgs:
            result, vx, vy = _train_cfg(cfg, hook_factory=hook_factory)
            rep.add_run(
                result,
                val_x=vx,
                val_y=vy,
                ablations=policy.ablations and cfg.model_kind == "pan",
                slots=policy.slots and cfg.model_kind == "pan",
            )
            run_analyzers(analyzer_names, result, vx, vy, cfg, state)
            if cfg.save_model:
                save_model_weights(result, out_dir)

        rep.write_all()
        write_plugin_rows(state, out_dir)
        write_declared_plots(plots if plots is not None else self.spec.default_plots, rep, state, out_dir)
        rep.print_summary()
        return rep


def _k_sweep_spec() -> ExperimentFactory:
    return ExperimentFactory(
        name="k_sweep",
        default_grid={"k_freqs": list(range(1, 16)), "seed": [42, 123, 456]},
        grid_aliases={"ks": "k_freqs", "seeds": "seed"},
        fixed_overrides={"model_kind": "pan", "weight_decay": 0.01},
        run_label=lambda row: f"K{row['k_freqs']}-s{row['seed']}",
        default_plots=[
            {"type": "reliability", "group_by": "k_freqs", "filename": "reliability.png"},
            {"type": "curves", "filename": "curves.png", "title": "K sweep curves"},
        ],
    )


def _wd_sweep_spec() -> ExperimentFactory:
    return ExperimentFactory(
        name="wd_sweep",
        default_grid={"weight_decay": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1], "seed": [42, 123, 456], "k_freqs": [9]},
        grid_aliases={"wds": "weight_decay", "seeds": "seed"},
        fixed_overrides={"model_kind": "pan"},
        run_label=lambda row: f"WD{row['weight_decay']}-s{row['seed']}",
        default_plots=[
            {"type": "reliability", "group_by": "weight_decay", "filename": "reliability.png"},
        ],
    )


def _dw_sweep_spec() -> ExperimentFactory:
    return ExperimentFactory(
        name="dw_sweep",
        default_grid={"diversity_weight": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1], "seed": [42, 123, 456, 789, 999], "k_freqs": [9]},
        grid_aliases={"dws": "diversity_weight", "seeds": "seed"},
        fixed_overrides={"model_kind": "pan", "weight_decay": 0.01},
        run_label=lambda row: f"DW{row['diversity_weight']}-s{row['seed']}",
        default_plots=[
            {"type": "reliability", "group_by": "diversity_weight", "filename": "reliability.png"},
        ],
    )


def _k8_sweep_spec() -> ExperimentFactory:
    return ExperimentFactory(
        name="k8_sweep",
        default_grid={"seed": [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]},
        grid_aliases={"seeds": "seed"},
        fixed_overrides={"model_kind": "pan", "k_freqs": 8, "weight_decay": 0.01, "early_stop": False},
        run_label=lambda row: f"K8-s{row['seed']}",
        default_plots=[
            {"type": "curves", "filename": "curves.png", "title": "K=8 — all seeds"},
        ],
    )


FACTORY_SPECS: dict[str, Callable[[], ExperimentFactory]] = {
    "k_sweep": _k_sweep_spec,
    "wd_sweep": _wd_sweep_spec,
    "dw_sweep": _dw_sweep_spec,
    "k8_sweep": _k8_sweep_spec,
}


def can_run_with_factory(name: str) -> bool:
    return name in FACTORY_SPECS


def build_factory_runner(name: str) -> FactoryBackedRunner:
    if name not in FACTORY_SPECS:
        raise KeyError(name)
    return FactoryBackedRunner(FACTORY_SPECS[name]())
