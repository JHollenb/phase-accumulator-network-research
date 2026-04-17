from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.plots import plot_training_curves


class K8SweepExperiment(BaseExperiment):
    name = "k8_sweep"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456, 789, 999, 1234, 2345, 3456, 4567, 5678]
        return build_pan_seed_cfgs(base, seeds, label_prefix="K8", k_freqs=8, early_stop=False)

    def finalize(self, reporter, state, out_dir):
        plot_training_curves(
            reporter.curves_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "curves.png"),
            title="K=8 — all seeds",
        )
