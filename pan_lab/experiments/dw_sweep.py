from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import plot_sweep_reliability


class DWSweepExperiment(BaseExperiment):
    name = "dw_sweep"

    def build_configs(
        self,
        base: RunConfig,
        dws: Optional[list[float]] = None,
        seeds: Optional[list[int]] = None,
        k_freqs: int = 9,
        **_,
    ):
        dws = dws or [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
        seeds = seeds or [42, 123, 456, 789, 999]
        return [
            base.with_overrides(
                model_kind="pan",
                k_freqs=k_freqs,
                weight_decay=0.01,
                diversity_weight=dw,
                seed=s,
                label=f"DW{dw}-s{s}",
            )
            for dw in dws
            for s in seeds
        ]

    def init_state(self, **kwargs):
        return {"k_freqs": kwargs["exp_args"].get("k_freqs", 9)}

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=False)

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="diversity_weight",
            out_path=os.path.join(out_dir, "reliability.png"),
            title=f"Diversity-weight sweep  —  K={state['k_freqs']}",
        )
