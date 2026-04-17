from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.plots import plot_training_curves


class ModMulExperiment(BaseExperiment):
    name = "mod_mul"
    collect_ablations = True

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456]
        return build_pan_seed_cfgs(base, seeds, label_prefix="mul", task_kind="mod_mul")

    def init_state(self, **kwargs):
        return {"base": kwargs["base"]}

    def finalize(self, reporter, state, out_dir):
        plot_training_curves(
            reporter.curves_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "curves.png"),
            title=f"Modular multiplication  —  P={state['base'].p}",
        )
