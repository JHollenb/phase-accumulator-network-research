from __future__ import annotations

import copy
import os
from typing import Optional

import pandas as pd
import torch

from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment, build_pan_seed_cfgs
from pan_lab.models.quantize import apply_sifp16_to_pan
from pan_lab.reporting import ExperimentReporter


class SIFP16InferenceExperiment(BaseExperiment):
    name = "sifp16_inference"
    collect_ablations = True

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456]
        return build_pan_seed_cfgs(base, seeds, label_prefix="sifp")

    def init_state(self, **_):
        return {"quant_rows": []}

    def handle_result(self, reporter: ExperimentReporter, result, vx, vy, cfg, state):
        super().handle_result(reporter, result, vx, vy, cfg, state)

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
