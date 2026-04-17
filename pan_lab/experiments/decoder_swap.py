from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch

from pan_lab.config import DEVICE, RunConfig
from pan_lab.experiments.base import BaseExperiment, _train_cfg


class DecoderSwapExperiment(BaseExperiment):
    name = "decoder_swap"

    def build_configs(self, base: RunConfig, seeds: Optional[list[int]] = None, **_):
        seeds = seeds or [42, 123, 456]
        return [
            base.with_overrides(model_kind="pan", seed=s, weight_decay=0.01, label=f"swap-s{s}")
            for s in seeds
        ]

    def init_state(self, **_):
        return {"swap_rows": []}

    def run_one(self, cfg: RunConfig, state):
        return _train_cfg(cfg)

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=False)
        pan = result.model
        with torch.no_grad():
            theta = torch.arange(cfg.p, device=DEVICE).float().unsqueeze(-1)
            f = pan.encoders[0].freq.detach()
            decoder_fourier = torch.cos(theta * f.unsqueeze(0))
            n = decoder_fourier.norm(dim=1, keepdim=True).clamp(min=1e-8)
            decoder_fourier = decoder_fourier / n

            saved_w = pan.decoder.weight.data.clone()
            saved_b = pan.decoder.bias.data.clone()
            pan.decoder.weight.data = decoder_fourier
            pan.decoder.bias.data.zero_()
            logits = pan(vx)
            acc_swap = float((logits.argmax(-1) == vy).float().mean().item())
            pan.decoder.weight.data.copy_(saved_w)
            pan.decoder.bias.data.copy_(saved_b)

        acc_learned = result.history.val_acc[-1] if result.history.val_acc else 0.0
        state["swap_rows"].append(
            {
                "run_id": cfg.display_id(),
                "seed": cfg.seed,
                "val_acc_learned_decoder": acc_learned,
                "val_acc_fourier_decoder": acc_swap,
                "delta": acc_swap - acc_learned,
            }
        )

    def finalize(self, reporter, state, out_dir):
        swap_df = pd.DataFrame(state["swap_rows"])
        swap_df.to_csv(os.path.join(out_dir, "decoder_swap.csv"), index=False)
        print("\n── Decoder-swap eval ──")
        if len(swap_df):
            print(swap_df.to_string(index=False))
