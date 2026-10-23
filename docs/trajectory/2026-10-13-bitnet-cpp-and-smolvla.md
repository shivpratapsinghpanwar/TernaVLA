# 2026-10-13 · bitnet.cpp on the real 2B model, and a SmolVLA reference point

## What

1. Ran bitnet.cpp (I2_S kernels) on the real BitNet b1.58 2B4T checkpoint under WSL2, 4 threads.
2. Timed SmolVLA-base, a full-precision VLA, on the same laptop as an outside reference.

## Why

Experiment zero used ONNX Runtime with random weights. bitnet.cpp is the reference runtime for
BitNet models, so it is the fair number for the M tier. SmolVLA is the smallest widely used
open VLA and shows what a non-ternary model costs on the same CPU.

## Result

bitnet.cpp, median prefill time, 4 threads:

| Tokens | Seconds |
|---|---|
| 43 | 0.89 |
| 91 | 2.10 |
| 128 | 2.54 |

- 21–34 % faster than ONNX Runtime 2-bit; an M action chunk is about 2.4 s.
- WSL2 exposes only 6 of the 8 logical CPUs, so 8-thread runs under WSL2 are oversubscribed and
  are not reported.
- Raw data: [`results/bitnet_cpp_2b4t_i2s_20261013T0613Z.json`](../../results/bitnet_cpp_2b4t_i2s_20261013T0613Z.json).

SmolVLA-base: 9.62 s per 50-action chunk (p50, fp32, 4 threads). Not like for like: SmolVLA
encodes three 512 px camera views and runs 10 flow-matching steps. The checkpoint ships in bf16,
which this CPU only emulates; without casting to fp32 a chunk takes about 77 s.

## Next

The CPU side is measured. What remains is training, which needs GPU time:
[cloud GPU compute](2026-10-23-cloud-gpu-compute.md).
