"""Auxiliary conditioning modules: noise level and self-conditioning.

Both modules are small, live *outside* the base checkpoint, and are **zero-initialised
at their output projection**. That last property matters more than it looks:

    at initialisation, DiffusionQwen(x) is bit-identical to the unmodified
    autoregressive Qwen3.5 forward pass.

So any change in behaviour is attributable to training, and the AR control path
(research question 8) is exactly the same weights, not an approximation of them.

SELF-CONDITIONING REPRESENTATION
--------------------------------
Iteration k of the sampler produces a distribution over clean tokens. We summarise
it as an *expected token embedding* and project it back into the residual stream:

    p_hat  in R^{B x C x V}        (softmax over the vocabulary)
    e_hat  = p_hat @ E             (E = the model's tied input embedding matrix)
    delta  = W_out(SiLU(W_in(RMSNorm(e_hat))))       W_out zero-initialised
    inputs_embeds[canvas] += delta

Chosen over the alternatives (a learned second embedding table, or concatenation
along the hidden dim) because Qwen3.5 ties input embeddings to the output head, so
`E` is already the natural shared token space and no new vocabulary-sized parameter
is introduced. With 248,320 tokens, a full `p_hat @ E` matmul is expensive, so the
expectation is taken over the top-k tokens by default; `top_k=0` uses the exact
full-vocabulary expectation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_features(t: torch.Tensor, dim: int, max_period: float = 1.0e4) -> torch.Tensor:
    """Standard transformer/diffusion sinusoidal embedding of a continuous t in [0, 1].

    Args:
        t: [B] float.
        dim: output width (rounded down to even).
    Returns:
        [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepConditioner(nn.Module):
    """Maps a scalar noise level to an additive bias on the canvas hidden states."""

    def __init__(self, hidden_size: int, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """t: [B] -> [B, 1, H] additive term, broadcast over canvas positions."""
        feats = sinusoidal_features(t, self.feature_dim).to(self.mlp[0].weight.dtype)
        return self.mlp(feats)[:, None, :].to(dtype)


class SelfConditioner(nn.Module):
    """Projects the previous denoising prediction back into the residual stream."""

    def __init__(self, hidden_size: int, top_k: int = 32, eps: float = 1e-6):
        super().__init__()
        self.top_k = top_k
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def expected_embedding(
        self, logits: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        """E_{p_hat}[embedding(token)] for each canvas position.

        Args:
            logits: [B, C, V]
            embedding: [V, H] the tied input embedding matrix.
        Returns:
            [B, C, H]
        """
        if self.top_k and self.top_k < logits.shape[-1]:
            vals, idx = logits.float().topk(self.top_k, dim=-1)  # [B, C, k]
            probs = vals.softmax(dim=-1)
            vecs = embedding[idx]  # [B, C, k, H]
            return (probs.unsqueeze(-1).to(vecs.dtype) * vecs).sum(dim=-2)
        probs = logits.float().softmax(dim=-1)
        return probs.to(embedding.dtype) @ embedding

    def forward(self, logits: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """[B, C, V] logits -> [B, C, H] additive term for the canvas."""
        e_hat = self.expected_embedding(logits, embedding).to(self.norm.weight.dtype)
        return self.proj(self.norm(e_hat))
