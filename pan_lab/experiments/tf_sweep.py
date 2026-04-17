from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import plot_parameter_efficiency, plot_sweep_reliability


class TFSweepExperiment(BaseExperiment):
    name = "tf_sweep"

    def build_configs(
        self,
        base: RunConfig,
        d_models: Optional[list[int]] = None,
        seeds: Optional[list[int]] = None,
        **_,
    ):
        d_models = d_models or [8, 16, 32, 64, 128]
        seeds = seeds or [42, 123, 456]
        cfgs = []
        for d in d_models:
            n_heads = max(1, d // 16)
            d_mlp = 4 * d
            for s in seeds:
                cfgs.append(
                    base.with_overrides(
                        model_kind="transformer",
                        d_model=d,
                        n_heads=n_heads,
                        d_mlp=d_mlp,
                        weight_decay=1.0,
                        seed=s,
                        label=f"TF-d{d}-s{s}",
                    )
                )
        return cfgs

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=False)

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="d_model",
            out_path=os.path.join(out_dir, "reliability.png"),
            title="Transformer d_model sweep",
        )
        plot_parameter_efficiency(reporter.runs_df(), os.path.join(out_dir, "param_efficiency.png"))
