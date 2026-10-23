# 2026-10-12 · Plan and the five starting decisions

## What

A survey of runtimes, ternary backbones, BitVLA's released assets and published action
interfaces, followed by the research plan in [PLAN.md](../PLAN.md) and five decisions that fix
its scope.

## Why

BitVLA (arXiv 2506.07530) already shows that a ternary VLA can work, but only with GPU-scale
training and GPU-only inference. Before writing training code we needed to know what is
genuinely missing, which backbones are usable under open licences, and what a CPU can afford
per control step.

Survey findings that shaped the plan:

- No CPU latency has been published for any BitNet-based VLA; BitVLA's ternary kernels are
  CUDA-only.
- BitVLA's checkpoints and code are public under MIT, so its VLM can serve as a backbone.
- One query token per 4 action steps, decoded by a float L1 head, beats dense per-step tokens
  in published ablations; quantising the action head to 4 bits destroys performance.
- Usable ternary backbones: BitCPM-CANN-0.5B (Apache-2.0) and BitNet b1.58 2B4T (MIT).
  Rejected for quality or licence: TriLM, OLMo-Bitnet, 1bitLLM, Bonsai, Falcon-E, ParetoQ.

## Result

| # | Decision |
|---|---|
| D1 | Build both tiers: S on BitCPM-CANN-0.5B and M on BitNet 2B4T. |
| D2 | Work in a separate public repository presented as active research (this one). |
| D3 | Action interface, as written into PLAN.md: 13 query tokens × 4 steps × 17 joints, causal attention, float L1 head. |
| D4 | Headline benchmark: full LIBERO four-suite average, with CPU latency and memory in the same table. |
| D5 | Use WSL2 (Ubuntu 22.04) for Linux-only tools such as bitnet.cpp and LIBERO evaluation. |

## Next

Measure the CPU cost of every tier before spending any GPU time:
[experiment zero](2026-10-12-exp0-cpu-cost.md).
