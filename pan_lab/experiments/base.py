from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from pan_lab.config import DEVICE, RunConfig
from pan_lab.data import make_modular_dataset
from pan_lab.hooks import CSVStreamLogger
from pan_lab.models import make_model
from pan_lab.reporting import ExperimentReporter, save_model_weights
from pan_lab.experiments.plugins import run_analyzers, write_declared_plots, write_plugin_rows
from pan_lab.trainer import train


def _move(tensors, device):
    return tuple(t.to(device) for t in tensors)


def _print_plan(cfgs: List[RunConfig], name: str) -> None:
    print(f"\n══ {name} — dry-run plan ({len(cfgs)} sub-runs) ══")
    print(f"  device: {DEVICE}")
    for c in cfgs:
        print(
            f"  - {c.display_id():<22} "
            f"p={c.p} k={c.k_freqs} task={c.task_kind} model={c.model_kind} "
            f"seed={c.seed} steps={c.n_steps:,} wd={c.weight_decay} "
            f"dw={c.diversity_weight} freq_init={c.freq_init}"
        )
    total_steps = sum(c.n_steps for c in cfgs)
    print(f"  total planned steps: {total_steps:,}")


def _train_cfg(cfg: RunConfig, hook_factory=None):
    tx, ty, vx, vy = make_modular_dataset(
        p=cfg.p, task_kind=cfg.task_kind, train_frac=cfg.train_frac, seed=cfg.seed
    )
    model = make_model(cfg).to(DEVICE)
    hooks = list(hook_factory(cfg)) if hook_factory else []
    result = train(model, cfg, tx, ty, vx, vy, hooks=hooks, verbose=True)
    return result, vx, vy


def _run_cfgs(
    cfgs,
    name,
    out_dir,
    dry_run,
    hook_factory=None,
    ablations=True,
    slots=False,
):
    rep = ExperimentReporter(name=name, out_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if dry_run:
        _print_plan(cfgs, name)
        return rep

    stream_path = os.path.join(out_dir, "curves_stream.csv")
    if os.path.exists(stream_path):
        os.remove(stream_path)

    for cfg in cfgs:
        per_run_factory = None
        if hook_factory:
            per_run_factory = hook_factory
        else:
            per_run_factory = lambda _: []

        def _hooks(c):
            hooks = list(per_run_factory(c))
            hooks.append(CSVStreamLogger(stream_path, run_id=c.display_id()))
            return hooks

        result, vx, vy = _train_cfg(cfg, hook_factory=_hooks)
        rep.add_run(
            result,
            val_x=vx,
            val_y=vy,
            ablations=ablations and cfg.model_kind == "pan",
            slots=slots and cfg.model_kind == "pan",
        )

        if cfg.save_model:
            path = save_model_weights(result, out_dir)
            print(f"  saved model weights: {path}")

    rep.write_all()
    rep.print_summary()
    return rep


def build_pan_seed_cfgs(
    base: RunConfig,
    seeds: list[int],
    label_prefix: str,
    **overrides,
) -> list[RunConfig]:
    return [
        base.with_overrides(
            model_kind="pan",
            seed=s,
            weight_decay=0.01,
            label=f"{label_prefix}-s{s}",
            **overrides,
        )
        for s in seeds
    ]


class BaseExperiment(ABC):
    name: str
    collect_ablations: bool = False
    collect_slots: bool = False

    def __call__(self, base: RunConfig, out_dir: str, dry_run: bool = False, **exp_args):
        cfgs = self.build_configs(base, **exp_args)
        if dry_run:
            self.print_plan(cfgs)
            return ExperimentReporter(self.name, out_dir)

        reporter = ExperimentReporter(self.name, out_dir)
        state = self.init_state(base=base, out_dir=out_dir, exp_args=exp_args) or {}
        analyzer_names = exp_args.get("analyses")
        plot_specs = exp_args.get("plots")

        for cfg in cfgs:
            result, vx, vy = self.run_one(cfg, state)
            self.handle_result(reporter, result, vx, vy, cfg, state)
            run_analyzers(analyzer_names, result, vx, vy, cfg, state)

        reporter.write_all()
        self.finalize(reporter, state, out_dir)
        write_plugin_rows(state, out_dir)
        write_declared_plots(plot_specs, reporter, state, out_dir)
        return reporter

    @abstractmethod
    def build_configs(self, base: RunConfig, **exp_args):
        ...

    def run_one(self, cfg: RunConfig, state):
        return _train_cfg(cfg)

    def init_state(self, **_):
        return None

    def handle_result(self, reporter, result, vx, vy, cfg: RunConfig, state):
        reporter.add_run(
            result,
            val_x=vx,
            val_y=vy,
            ablations=self.collect_ablations,
            slots=self.collect_slots,
        )

    def finalize(self, reporter: ExperimentReporter, state, out_dir: str):
        pass

    def print_plan(self, cfgs):
        _print_plan(cfgs, self.name)
