# 2026-10-24 · Vision: a full-stack ternary robot package, and what is actually new

## What

The end goal, stated: a robot package whose three components are all ternary (1.58-bit
weights) and run on edge CPUs and Jetson Orin Nano without a discrete GPU.

| Component | Role | Project |
|---|---|---|
| Ternary VLM | perception: what is where, what the instruction refers to | [TernaVLM](https://github.com/shivpratapsinghpanwar/ternary-vlm) |
| Ternary VLA | action: camera + instruction + joint state → 17-joint motion | TernaVLA (this repository) |
| Edge runtime | runs both on a laptop CPU, then Orin Nano | bitnet.cpp and ONNX Runtime 2-bit paths measured in [experiment zero](2026-10-12-exp0-cpu-cost.md) |

A different humanoid joins by describing its body in one file, resolved by joint name against
the 17-joint contract (design: Humanoid_Perception_for_Edge, "plug-and-play humanoid bodies").

## Why

Before claiming "first of its kind", we checked what exists. Three parallel literature searches
(ternary and low-bit VLAs; ternary VLMs and edge runtimes; integrated on-device robot stacks)
covered work up to 2026-10-24, extending the 2026-10-12 survey.

## Result

### Already done by others: not claimable

| Claim | Prior work |
|---|---|
| First ternary VLA | BitVLA, arXiv 2506.07530 (June 2025; CoRL 2026), ternary language model and 1.58-bit SigLIP vision encoder; LIBERO and a real Franka arm |
| First on-device perception + policy stack | Figure Helix (Feb 2025), 1X Redwood (Jun 2025), Gemini Robotics On-Device (Jun 2025) |
| First low-bit edge VLA | GR00T N1.7 in NVFP4 on Jetson Thor (~40 ms); LiteVLA-Edge, 4-bit GGUF on AGX Orin (arXiv 2603.03380); FoldQuantVLA, QCRL (Sep–Oct 2026) |
| First VLA without a discrete GPU | Intel Embodied Intelligence SDK, pi0.5 on an integrated GPU via OpenVINO (2026) |
| First config-file robot onboarding | GR00T `modality.json`, LeRobot robot configs, StarVLA YAML |

### Still open as of 2026-10-24

1. **CPU latency for a ternary VLA.** No ternary or 1-bit VLA has a published CPU number.
   vla.cpp (arXiv 2606.08094) runs BitVLA only through CUDA kernels; every CPU row of its
   benchmark is N/A. Its Orin Nano Super figure, 335 ms per step on the GPU, is the edge
   baseline to cite. Our experiment zero timed random ternary weights at the real shapes, so it
   is "CPU latency of a ternary VLA-sized backbone" until a trained policy is timed.
2. **A ternary VLA controlling a humanoid in joint space.** Every low-bit VLA found is evaluated
   on arms (Franka, ALOHA, SO-101, ViperX, UR3). Narrow: 4-bit GR00T variants already run on
   Orin and Thor, so a 4-bit humanoid result is close for others; the ternary case is open.
3. **A ternary VLM for robot perception on edge CPUs.** The only robot-oriented ternary model is
   BitVLA, which outputs actions. bitnet.cpp has no vision support; the only ternary vision
   model shown on a CPU is a 258M document-OCR model at seconds per page. A ternary vision
   encoder (BitSigLIP-L, MIT) exists but no CPU runtime executes it as ternary.
4. **All of it in one package.** No public stack combines ternary perception and action, a
   CPU / Orin Nano runtime without a discrete GPU, and one-file humanoid onboarding.

### The claim we will make

> To our knowledge, no public stack yet combines natively ternary perception and action models
> with a CPU / Orin-Nano-class runtime and one-file humanoid onboarding. This project aims to be
> the first, building on BitVLA and on config-driven cross-embodiment frameworks such as GR00T,
> LeRobot and StarVLA.

It stays forward-looking ("aims to") until the S tier passes its LIBERO-Spatial gate and a
trained policy is timed on the CPU.

### Not verified

Licences of BitVLA's Hugging Face weights (the repository is MIT) and of vla.cpp (Apache-2.0
in the repository, MIT on its model card); why vla.cpp has no ternary CPU path; whether the
1-bit VLA quantisation paper HBVLA (arXiv 2602.13710) runs real 1-bit kernels; the action spaces
of HBVLA, FoldQuantVLA and QCRL; closed industry work not visible on the web.

## Next

- Re-run this check before any public claim; search terms beyond "ternary" and "1-bit"
  (for example "1.58-bit", "BitNet robot") may surface work these searches missed.
- The evidence that turns "aims to" into a result: TernaVLA S ≥ 90 % on LIBERO-Spatial, a
  trained policy timed on the laptop CPU, then whole-stack latency and memory on Orin Nano.
