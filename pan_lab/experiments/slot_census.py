from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.plots import plot_slot_census


class SlotCensusExperiment(BaseExperiment):
    name = "slot_census"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        return build_pan_seed_cfgs(
            base,
            seeds=seeds,
            default_seeds=list(range(20)),
            overrides={"weight_decay": 0.01},
            label_prefix="census-",
        )

    def init_state(self, **kwargs):
        return {"base": kwargs["base"]}

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=False, slots=True)

    def finalize(self, reporter, state, out_dir):
        base = state["base"]
        plot_slot_census(
            reporter.slots_df(),
            os.path.join(out_dir, "slot_census.png"),
            title=f"Slot census  —  P={base.p}  K={base.k_freqs}",
        )
