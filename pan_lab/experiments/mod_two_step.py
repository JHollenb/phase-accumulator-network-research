from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.plots import plot_training_curves


class ModTwoStepExperiment(BaseExperiment):
    name = "mod_two_step"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        return build_pan_seed_cfgs(
            base,
            seeds=seeds,
            default_seeds=[42, 123, 456],
            overrides={"task_kind": "mod_two_step", "weight_decay": 0.01},
            label_prefix="2step-",
        )

    def init_state(self, **kwargs):
        return {"base": kwargs["base"]}

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=True)

    def finalize(self, reporter, state, out_dir):
        plot_training_curves(
            reporter.curves_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "curves.png"),
            title=f"Two-step (a+b)*c mod {state['base'].p}",
        )
