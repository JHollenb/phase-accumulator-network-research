from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import plot_sweep_reliability, plot_training_curves


class KSweepExperiment(BaseExperiment):
    name = "k_sweep"

    def build_configs(
        self,
        base: RunConfig,
        ks: Optional[list[int]] = None,
        seeds: Optional[list[int]] = None,
        **_,
    ):
        ks = ks or list(range(1, 16))
        seeds = seeds or [42, 123, 456]
        return [
            base.with_overrides(
                model_kind="pan",
                k_freqs=k,
                seed=s,
                weight_decay=0.01,
                label=f"K{k}-s{s}",
            )
            for k in ks
            for s in seeds
        ]

    def init_state(self, **kwargs):
        return {"base": kwargs["base"]}

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="k_freqs",
            out_path=os.path.join(out_dir, "reliability.png"),
            title=f"K sweep  —  P={state['base'].p}",
        )
        plot_training_curves(
            reporter.curves_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "curves.png"),
            title="K sweep curves",
        )
