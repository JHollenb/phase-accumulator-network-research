from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.plots import plot_sweep_reliability


class FreqInitAblationExperiment(BaseExperiment):
    name = "freq_init_ablation"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        cfgs = []
        for init in ("fourier", "random"):
            cfgs.extend(
                build_pan_seed_cfgs(
                    base,
                    seeds=seeds,
                    default_seeds=[42, 123, 456, 789, 999],
                    overrides={"freq_init": init, "weight_decay": 0.01},
                    label_prefix=f"{init}-",
                )
            )
        return cfgs

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=False, slots=True)

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="freq_init",
            out_path=os.path.join(out_dir, "reliability.png"),
            title="Fourier vs random frequency init",
        )
