# Experiment zero: what one TernaVLA action step costs on a laptop CPU

Measured 2026-10-12 on an Intel Core i5-10210U (4 cores / 8 threads, AVX2 without VNNI,
15 W, Windows 11), on mains power. Raw data: [`results/exp0_aiml-shivpratap_20261012T1256Z.json`](../results/exp0_aiml-shivpratap_20261012T1256Z.json).
Reproduce: `uv sync && uv run python scripts/exp0.py --threads 4 8`.

## Why measure before training

No CPU latency has been published for any BitNet-based vision-language-action model. BitVLA
and its runtime report GPU numbers only. Before spending GPU time on training, we time the exact
computation each TernaVLA tier will run, at real layer shapes, with random ternary weights.
Dense matrix-product latency does not depend on the weight values, so no checkpoint is needed.

## What is timed

| Part | What | Size |
|---|---|---|
| Vision | SigLIP2-base-patch16-256 vision tower, 256 patches | ≈ 93 M params |
| Language model | every ternary projection of every layer (q/k/v, o, gate/up, down) plus norms and activation; attention scores omitted (< 1 % at ≤ 128 tokens) | see tiers |
| Projector + head | pooled patches to LM width; 13-query MLP-ResNet action head, fp32 | 3 to 20 M params |

| Tier | Backbone (shapes from its published config) | Licence | Ternary params | Tokens per step |
|---|---|---|---|---|
| S | openbmb/BitCPM-CANN-0.5B | Apache-2.0 | 0.36 B | 43 (16 image + 12 text + state + prefix + 13 queries) |
| M-lite | openbmb/BitCPM-CANN-1B | Apache-2.0 | 1.32 B | 91 (64 image + 12 text + state + prefix + 13 queries) |
| M | microsoft/bitnet-b1.58-2B-4T | MIT | 2.08 B | 91; ablation with one query per step: 128 |

Language-model backends:

- `torch-fp32`: PyTorch `F.linear` with fp32 ternary values.
- `ort-fp32`: ONNX Runtime 1.31 `MatMul`, fp32 weights.
- `ort-nbits{8,4,2}`: ONNX Runtime `com.microsoft.MatMulNBits` with `accuracy_level=4` (int8 activations inside the kernel). For 2 bits the ternary weights are stored exactly as codes {1, 2, 3} around the default zero point 2; a unit test checks the operator really computes the ternary product (`tests/test_bench.py`).
- `-b128`: block size 128 instead of 32.
- `torch-int8` (`torch._int_mm`) was dropped: on this CPU it runs a reference kernel (131 s for one S pass versus 0.28 s in fp32).

Method: 2 warm-up runs excluded, p50 of 10. The language model is built 4 layers deep and scaled to full depth; a full-depth 2B run agreed within 1 % (4 threads: 4.01 s measured versus 3.97 s scaled; 8 threads: 3.27 s versus 3.27 s).

## Results

Seconds per action chunk (vision + language model on its fastest backend + projector and head, p50). A chunk is 52 steps at 50 Hz = 1.04 s of motion.

| Tier | 4 threads | 8 threads | Fastest LM backend | Real-time on this laptop |
|---|---|---|---|---|
| S | 0.54 | **0.44** | 2-bit, block 128 | yes |
| M-lite | 2.05 | 1.48 | 2-bit b128 (4t), fp32 (8t) | only with 2 s chunks |
| M | 3.00 | 2.42 | 2-bit b128 (4t), fp32 (8t) | no (measurement tier) |
| M, one query per step | 4.21 | 3.27 | as above | no |

Reference point on the same laptop: **SmolVLA-base** (450 M parameters, Apache-2.0) takes
**9.62 s** per 50-action chunk in fp32 at 4 threads (p50, n 20, LeRobot's own inference path;
[census row](https://github.com/shivpratapsinghpanwar/Humanoid_Perception_for_Edge/blob/main/docs/BENCHMARKS.md)).
Not like for like: SmolVLA encodes three camera views at 512 px and runs 10 flow-matching steps,
and TernaVLA is not trained yet. Its checkpoint ships in bfloat16, which this CPU only emulates:
uncast it took 77 s per chunk.

Language-model pass only, seconds, 4 threads:

| Backend | S (43 tok) | M-lite (91) | M (91) | M (128) |
|---|---|---|---|---|
| torch-fp32 | 0.31 | 2.81 | 4.54 | 5.29 |
| ort-fp32 | 0.21 | 1.91 | 2.91 | 4.30 |
| ort-nbits8 | 0.31 | 3.30 | 5.43 | 7.17 |
| ort-nbits4 | 0.25 | 1.89 | 3.10 | 4.35 |
| ort-nbits2 (block 32) | 0.25 | 2.67 | 3.97 | 5.63 |
| ort-nbits2 (block 128) | **0.21** | **1.72** | **2.66** | **3.87** |

Vision tower, seconds: ONNX Runtime int8 0.31 (4t) / 0.25 (8t); ONNX fp32 0.43 / 0.35; PyTorch fp32 0.52 / 0.42.
Projector + head: 19 to 32 ms.

Machine state: fp32 SGEMM throughput at 4 threads was 196 GFLOPS at the start and 149 at the end (heat and battery charging), so expect about ±15 % run-to-run spread. One core sustains about 2.8 GHz; four loaded cores sit near 1.4 GHz under the 15 W limit.

## What it means

1. **The small tier is real-time on a 2019 laptop CPU today**, before any custom kernel: 0.44 s for a 1.04 s chunk.
2. **On this chip the 2-bit format buys memory, not speed.** The best 2-bit path is at most about 10 % faster than fp32 at 4 threads and slower than fp32 at 8 threads, because AVX2 without VNNI has no fast int8 dot product and the chip is power-limited. Weights shrink 16× (2B model: ≈ 0.55 GB instead of 8.3 GB), which is what matters on 4 to 8 GB boards.
3. **Tokens are the main lever.** 13 query tokens instead of one per step saves 30 % on M. This confirms the interface choice.
4. **Block size matters for ONNX Runtime's 2-bit kernel**: block 128 takes 17 to 36 % less time than block 32.
5. **8-bit weights are the slowest path** in ONNX Runtime on this CPU; avoid them here.

## Not yet measured

- bitnet.cpp's I2_S kernel (fastest published AVX2 ternary kernel), under WSL2: `scripts/wsl_bitnet_cpp.sh`.
- llama.cpp with a ReLU² patch and an exact Q8_0 repack.
- An i7 with VNNI, and a Jetson Orin Nano (GPU int2 kernels via vla.cpp).
- Real weights end to end: the numbers above are the computation, not yet a trained model.
