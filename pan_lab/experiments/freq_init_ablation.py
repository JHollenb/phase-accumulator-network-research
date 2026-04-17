from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import plot_sweep_reliability


class FreqInitAblationExperiment(BaseExperiment):
    name = "freq_init_ablation"
    collect_slots = True

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456, 789, 999]
        cfgs = []
        for init in ("fourier", "random"):
            for s in seeds:
                cfgs.append(
                    base.with_overrides(
                        model_kind="pan",
                        seed=s,
                        freq_init=init,
                        weight_decay=0.01,
                        label=f"{init}-s{s}",
                    )
                )
        return cfgs

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="freq_init",
            out_path=os.path.join(out_dir, "reliability.png"),
            title="Fourier vs random frequency init",
        )
