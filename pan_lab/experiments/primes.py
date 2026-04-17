from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment
from pan_lab.plots import plot_sweep_reliability


class PrimesExperiment(BaseExperiment):
    name = "primes"

    def build_configs(self, base: RunConfig, primes: Optional[list[int]] = None, **_):
        primes = primes or [43, 67, 89, 113, 127]
        return [
            base.with_overrides(
                model_kind="pan", p=p, k_freqs=base.k_freqs, weight_decay=0.01, label=f"P{p}"
            )
            for p in primes
        ]

    def init_state(self, **kwargs):
        return {"base": kwargs["base"]}

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=True)

    def finalize(self, reporter, state, out_dir):
        plot_sweep_reliability(
            reporter.runs_df(),
            group_by="p",
            out_path=os.path.join(out_dir, "reliability.png"),
            title=f"Cross-prime generalization  —  K={state['base'].k_freqs}",
        )
