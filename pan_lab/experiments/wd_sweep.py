from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import plot_sweep_reliability


class WDSweepExperiment(BaseExperiment):
    name = "wd_sweep"

    def build_configs(
        self,
        base: RunConfig,
        wds: Optional[list[float]] = None,
        seeds: Optional[list[int]] = None,
        k_freqs: int = 9,
        **_,
    ):
        wds = wds or [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        seeds = seeds or [42, 123, 456]
        return [
            base.with_overrides(
                model_kind="pan",
                k_freqs=k_freqs,
                weight_decay=wd,
                seed=s,
                label=f"WD{wd}-s{s}",
            )
            for wd in wds
            for s in seeds
        ]

    def init_state(self, **kwargs):
        return {"k_freqs": kwargs["exp_args"].get("k_freqs", 9)}

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=False)

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="weight_decay",
            out_path=os.path.join(out_dir, "reliability.png"),
            title=f"Weight-decay sweep  —  K={state['k_freqs']}",
        )
