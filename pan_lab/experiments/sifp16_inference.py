from __future__ import annotations

import copy
import os
from typing import Optional

import pandas as pd
import torch

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, _train_cfg
from pan_lab.models.quantize import apply_sifp16_to_pan
from pan_lab.reporting import ExperimentReporter


class SIFP16InferenceExperiment(BaseExperiment):
    name = "sifp16_inference"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456]
        return [
            base.with_overrides(model_kind="pan", seed=s, weight_decay=0.01, label=f"sifp-s{s}")
            for s in seeds
        ]

    def init_state(self, **_):
        return {"quant_rows": []}

    def run_one(self, cfg: RunConfig, state):
        return _train_cfg(cfg)

    def handle_result(self, reporter: ExperimentReporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=True)

        qmodel = copy.deepcopy(result.model)
        qmodel.eval()
        apply_sifp16_to_pan(qmodel)
        with torch.no_grad():
            logits = qmodel(vx)
            acc_q = float((logits.argmax(-1) == vy).float().mean().item())
        acc_fp = result.history.val_acc[-1] if result.history.val_acc else 0.0
        state["quant_rows"].append(
            {
                "run_id": cfg.display_id(),
                "seed": cfg.seed,
                "val_acc_fp32": acc_fp,
                "val_acc_sifp16": acc_q,
                "delta": acc_q - acc_fp,
            }
        )

    def finalize(self, reporter, state, out_dir):
        quant_df = pd.DataFrame(state["quant_rows"])
        quant_df.to_csv(os.path.join(out_dir, "quant_eval.csv"), index=False)
        print("\n── SIFP-16 quantization eval ──")
        if len(quant_df):
            print(quant_df.to_string(index=False))
