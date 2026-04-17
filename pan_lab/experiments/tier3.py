from __future__ import annotations

import os

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, _train_cfg
from pan_lab.hooks import CheckpointLogger
from pan_lab.plots import (
    plot_ablation_bars,
    plot_freq_err_trajectories,
    plot_freq_trajectories,
    plot_training_curves,
)


class Tier3Experiment(BaseExperiment):
    name = "tier3"

    def build_configs(self, base: RunConfig, **_):
        return [
            base.with_overrides(
                model_kind="pan",
                weight_decay=0.01,
                early_stop=False,
                record_checkpoints=True,
                label="tier3",
            )
        ]

    def init_state(self, **kwargs):
        return {"cfg": kwargs["base"]}

    def run_one(self, cfg: RunConfig, state):
        return _train_cfg(cfg, hook_factory=lambda _: [CheckpointLogger()])

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=True, slots=True)

    def finalize(self, reporter, state, out_dir):
        if reporter.checkpoints_df().empty:
            return
        cfg = state["cfg"]
        plot_freq_err_trajectories(
            reporter.checkpoints_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "freq_err_trajectories.png"),
            title=f"Frequency error trajectories  —  P={cfg.p}  K={cfg.k_freqs}",
        )
        plot_freq_trajectories(
            reporter.checkpoints_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "freq_trajectories.png"),
            title=f"Frequency trajectories  —  P={cfg.p}  K={cfg.k_freqs}",
        )
        plot_training_curves(
            reporter.curves_df(),
            reporter.runs_df(),
            os.path.join(out_dir, "curves.png"),
            title="Tier 3 training curve",
        )
        plot_ablation_bars(reporter.ablations_df(), os.path.join(out_dir, "ablations.png"))
