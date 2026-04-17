from __future__ import annotations

import os
from typing import Optional

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.plots import plot_slot_census


class SlotCensusExperiment(BaseExperiment):
    name = "slot_census"
    collect_slots = True

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or list(range(20))
        return build_pan_seed_cfgs(base, seeds, label_prefix="census")

    def init_state(self, **kwargs):
        return {"base": kwargs["base"]}

    def finalize(self, reporter, state, out_dir):
        base = state["base"]
        plot_slot_census(
            reporter.slots_df(),
            os.path.join(out_dir, "slot_census.png"),
            title=f"Slot census  —  P={base.p}  K={base.k_freqs}",
        )
