from __future__ import annotations

from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment


class HeldOutPrimesExperiment(BaseExperiment):
    name = "held_out_primes"

    def build_configs(self, base: RunConfig, primes: Optional[list[int]] = None, **_):
        primes = primes or [59, 71, 97]
        return [
            base.with_overrides(
                model_kind="pan", p=p, k_freqs=base.k_freqs, weight_decay=0.01, label=f"P{p}-held"
            )
            for p in primes
        ]

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=True)
