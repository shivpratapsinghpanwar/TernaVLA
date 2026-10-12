"""Exact layer shapes of the candidate ternary backbones and the per-step token budget.

Shapes are copied from each model's published ``config.json`` (fetched 2026-10-12). Latency of a
dense matrix product does not depend on the weight values, so the benchmarks build random
ternary weights with these shapes instead of downloading checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LMShape:
    """One decoder-only language model, described by what its matrix products cost."""

    key: str            # short id used in result files
    tier: str           # "S", "M-lite" or "M"
    hf_id: str          # where the published config lives
    licence: str
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    act: str            # "relu2" (BitNet) or "silu" (Llama / MiniCPM)

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def qkv_out(self) -> int:
        return (self.heads + 2 * self.kv_heads) * self.head_dim

    def linears(self) -> list[tuple[str, int, int]]:
        """(name, K in, N out) of every ternary matrix in one layer; q/k/v and gate/up are fused."""
        return [
            ("qkv", self.hidden, self.qkv_out),
            ("o", self.q_dim, self.hidden),
            ("gate_up", self.hidden, 2 * self.intermediate),
            ("down", self.intermediate, self.hidden),
        ]

    @property
    def params_per_layer(self) -> int:
        return sum(k * n for _, k, n in self.linears())

    @property
    def ternary_params(self) -> int:
        """Parameters in the transformer blocks: the part that is ternary and costs compute."""
        return self.layers * self.params_per_layer

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.hidden

    def gop_per_token(self) -> float:
        """Multiply-adds of the block matrices for one token, counted as 2 operations each."""
        return 2.0 * self.ternary_params / 1e9


BITNET_2B4T = LMShape(
    key="bitnet_2b4t", tier="M", hf_id="microsoft/bitnet-b1.58-2B-4T", licence="MIT",
    layers=30, hidden=2560, intermediate=6912, heads=20, kv_heads=5, head_dim=128,
    vocab=128256, act="relu2",
)

BITCPM_1B = LMShape(
    key="bitcpm_cann_1b", tier="M-lite", hf_id="openbmb/BitCPM-CANN-1B", licence="Apache-2.0",
    layers=28, hidden=2048, intermediate=6144, heads=16, kv_heads=2, head_dim=128,
    vocab=73448, act="silu",
)

BITCPM_05B = LMShape(
    key="bitcpm_cann_0p5b", tier="S", hf_id="openbmb/BitCPM-CANN-0.5B", licence="Apache-2.0",
    layers=24, hidden=1024, intermediate=4096, heads=16, kv_heads=2, head_dim=64,
    vocab=73448, act="silu",
)

MODELS: dict[str, LMShape] = {m.key: m for m in (BITCPM_05B, BITCPM_1B, BITNET_2B4T)}


@dataclass(frozen=True)
class TokenBudget:
    """Tokens in one forward pass of the action model (the agreed interface, PLAN section 3)."""

    image: int
    instruction: int = 12   # a short command plus the chat template around it
    state: int = 1
    prefix: int = 1         # the already-committed actions, for asynchronous execution
    queries: int = 13       # 13 placeholders x 4 steps = 52 steps at 50 Hz

    @property
    def total(self) -> int:
        return self.image + self.instruction + self.state + self.prefix + self.queries


# S pools the 256 vision patches 4x4 into 16 tokens; M and M-lite pool 2x2 into 64.
BUDGETS: dict[str, TokenBudget] = {
    "S": TokenBudget(image=16),
    "M-lite": TokenBudget(image=64),
    "M": TokenBudget(image=64),
}
# Ablation agreed with the interface: one query per step (50) instead of 13.
ABLATION_ONE_PER_STEP = TokenBudget(image=64, queries=50)

# SigLIP2-base-patch16-256 vision tower (ViT-B/16 defaults; the published config only sets
# image_size, so the transformers defaults apply).
VISION = dict(image_size=256, patch_size=16, hidden=768, intermediate=3072, layers=12, heads=12)
