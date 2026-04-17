"""
pan_lab.models.transformer — Nanda-style one-layer transformer.

Baseline for comparison against PAN. Hyperparameters default to Nanda et al.
2023 §3.1 (d=128, 4 heads, d_mlp=512). Generalized to accept N-input tasks
(N=2 for mod_add, N=3 for two-step).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        p:        int,
        d_model:  int = 128,
        n_heads:  int = 4,
        d_mlp:    int = 512,
        n_inputs: int = 2,
    ):
        super().__init__()
        self.p        = p
        self.d_model  = d_model
        self.n_inputs = n_inputs
        self.seq_len  = n_inputs + 1      # inputs + "="

        # Tokens are [0..P-1] for data, P for the "=" separator.
        self.tok_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(self.seq_len, d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp       = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )
        self.unembed = nn.Linear(d_model, p, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs : (B, N) long
        returns: (B, P) logits
        """
        B   = inputs.shape[0]
        eq  = torch.full((B, 1), self.p, dtype=torch.long, device=inputs.device)
        seq = torch.cat([inputs, eq], dim=1)                    # (B, N+1)
        pos = torch.arange(self.seq_len, device=inputs.device).unsqueeze(0)
        x   = self.tok_embed(seq) + self.pos_embed(pos)          # (B, N+1, D)

        mask = torch.triu(
            torch.ones(self.seq_len, self.seq_len, device=inputs.device),
            diagonal=1,
        ).bool()
        x_attn, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + x_attn
        x = x + self.mlp(x)
        return self.unembed(x[:, -1, :])                         # (B, P)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
