# 2026-10-12 · Experiment zero: CPU cost of every tier

## What

Timed one full TernaVLA action step for each tier on the target laptop CPU (Intel Core
i5-10210U, 4 cores / 8 threads, AVX2 without VNNI), using random ternary weights at the real
layer shapes. Full write-up: [EXP0.md](../EXP0.md).

## Why

Matrix-product latency does not depend on the weight values, so the CPU budget can be known
before any training. If a tier could not produce a 1.04 s action chunk in reasonable time on a
laptop, training it would be wasted GPU time.

## Result

Seconds per 1.04 s action chunk, ONNX Runtime, 4 / 8 threads:

| Tier | Tokens | 4 threads | 8 threads |
|---|---|---|---|
| S (BitCPM-CANN-0.5B) | 43 | 0.54 | 0.44 |
| M-lite (BitCPM-CANN-1B) | 91 | 2.05 | 1.48 |
| M (BitNet 2B4T) | 91 | 3.00 | 2.42 |
| M, one query per step (50 queries) | 128 | 4.21 | 3.27 |

- S already runs faster than real time on this laptop; M does not yet.
- The fastest language-model path was 2-bit `MatMulNBits` (block 128) at 4 threads for every
  tier; at 8 threads plain fp32 was fastest for M-lite and M.
- Grouping 4 steps per query token (13 queries instead of 50) saves about 30 % per chunk on M,
  which supports the action-interface choice.

## Next

Check whether a dedicated ternary runtime beats ONNX Runtime on the real M checkpoint:
[bitnet.cpp measurement](2026-10-13-bitnet-cpp-and-smolvla.md).
