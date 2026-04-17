from __future__ import annotations

import os

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import (
    plot_ablation_bars,
    plot_parameter_efficiency,
    plot_training_curves,
)


class CompareExperiment(BaseExperiment):
    name = "compare"

    def build_configs(self, base: RunConfig, **exp_args):
        pan_cfg = base.with_overrides(model_kind="pan", label="pan", weight_decay=0.01)
        tf_cfg = base.with_overrides(
            model_kind="transformer", label="tf", weight_decay=1.0
        )
        return [pan_cfg, tf_cfg]

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=cfg.model_kind == "pan")

    def finalize(self, reporter, state, out_dir):
        plot_training_curves(
            reporter.curves_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "curves.png"),
            title="PAN vs Transformer",
        )
        plot_parameter_efficiency(
            reporter.runs_df(), os.path.join(out_dir, "param_efficiency.png")
        )
        plot_ablation_bars(
            reporter.ablations_df(), os.path.join(out_dir, "ablations.png")
        )
