# TernaVLA research plan

Status: active research, agreed 2026-10-12. Nothing is trained yet; experiment zero (CPU cost) is
done, see [EXP0.md](EXP0.md).

## Goal

A ternary (1.58-bit weights) vision-language-action model that controls a humanoid's upper body
from one head camera and a spoken-style instruction, and runs on edge CPUs: an x86 laptop first,
then a Jetson Orin Nano. Two sizes share one recipe:

| Tier | Backbone | Licence | Target hardware |
|---|---|---|---|
| S | openbmb/BitCPM-CANN-0.5B (ternary latents, Llama-style, SiLU) | Apache-2.0 | laptop CPU, Raspberry-Pi-class boards |
| M | microsoft/bitnet-b1.58-2B-4T via TernaVLM (ternary-LoRA VLM) | MIT | i7-class CPU, Jetson Orin Nano |

Vision for both: frozen SigLIP2-base-patch16-256 (Apache-2.0), int8 on CPU.

## What is new, honestly

BitVLA (arXiv 2506.07530) already showed a ternary VLA on LIBERO (96.0 %) with GPU-scale
training. TernaVLA's contributions are different:

1. adaptation on a free-tier budget (Kaggle T4s), not pretraining-scale quantisation-aware training;
2. the first CPU latency numbers for a ternary VLA, published with machine state;
3. a humanoid joint-space action contract (17 upper-body joints) trained from simulator ground truth;
4. a one-pass action interface sized for CPU prefill.

## Action interface

Each step reads one 256 × 256 image, the instruction, the 17 current joint offsets from the home
pose, and the actions already committed for the next few control ticks. One forward pass outputs
52 steps × 17 joint targets at 50 Hz (1.04 s).

```
[image ×64] [state] [prefix] [instruction ...] [q1] [q2] ... [q13]
                                                 |    |          |
                                              head  head  ...  head
                                                 |    |          |
                                          steps 1-4  5-8  ...  49-52   (each step = 17 joint targets)
```

- 13 placeholder query tokens at the end, each decoded by a shared fp32 MLP-ResNet head into
  4 steps × 17 joints. Causal attention, so standard runtimes need no custom mask.
- L1 loss on normalised offsets from the home pose.
- State token: 2-layer MLP over the joint offsets.
- Prefix token: the already-committed actions; training simulates inference delay so the next
  chunk is computed while the current one plays (real-time chunking by action conditioning).
- The head stays in floating point: every published ≤ 4-bit VLA keeps it there.

Ablations: one query per step (50 tokens), 8 steps per query (2 s chunks), a 2-step flow head
instead of L1, a calibrated int8 head.

## Training ladder (Kaggle 2 × T4)

1. Frozen backbone, cached hidden states, train the head only (hours).
2. Plus projector, state and prefix modules.
3. Plus ternary-LoRA with a learning-rate sweep, using the ternary flip rate as the diagnostic.
4. Fallback: keep the low-rank branch in fp16 at inference if ternary flips stay slow.

S is small enough for full ternary fine-tuning on one T4.

## Data

- Simulator ground truth (MuJoCo): scripted look_at, point_at, reach, wave, nod and idle skills
  on the public Tiangong 2.0 Pro model, with per-frame randomisation of camera pose, heights,
  lighting and distractors, rendered through the same camera model as the real head camera.
- LIBERO through LeRobot as the external yardstick (four-suite average).
- A small set of real demonstrations on a lab humanoid for co-training (data stays private).

## Milestones

| Week | Milestone |
|---|---|
| 0 | CPU cost of every tier measured (done); bitnet.cpp under WSL2 |
| 1 to 2 | Head-only probe on M; S alignment and fine-tune; LIBERO-Spatial ≥ 90 % gate |
| 3 to 5 | Humanoid simulator data; S and M on the 17-joint suite; CPU runtime with real weights |
| 6 + | Lab humanoid with real demonstrations; Jetson Orin Nano |

## Related repositories

- [Humanoid_Perception_for_Edge](https://github.com/shivpratapsinghpanwar/Humanoid_Perception_for_Edge):
  robot models, simulator, camera model, data engine and the VLA census.
- [ternary-vlm](https://github.com/shivpratapsinghpanwar/ternary-vlm): TernaVLM, the ternary-LoRA
  vision-language model that the M tier starts from.
