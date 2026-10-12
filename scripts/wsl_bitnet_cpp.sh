#!/usr/bin/env bash
# Experiment zero, step 2: bitnet.cpp prefill of BitNet b1.58 2B4T on this laptop (inside WSL2).
#
# bitnet.cpp's I2_S kernel is the fastest published AVX2 ternary kernel; it has no vision and its
# Windows build currently fails to link, so it is measured under WSL2 (same CPU, Linux build).
# Run from an Ubuntu (WSL2) terminal, laptop on mains:
#
#     bash /mnt/d/TernaVLA/scripts/wsl_bitnet_cpp.sh
#
# It asks for your sudo password once (to install clang 18 and cmake), builds in ~/ternavla-wsl,
# downloads the official 1.19 GB GGUF and writes results to D:\TernaVLA\results\.
set -euo pipefail

WORK="$HOME/ternavla-wsl"
RESULTS="/mnt/d/TernaVLA/results"
MODEL_DIR="$WORK/BitNet/models/BitNet-b1.58-2B-4T"
TOKENS="43,91,128"     # S budget, M budget, M one-query-per-step ablation
THREADS="4,8"

mkdir -p "$WORK" "$RESULTS"
cd "$WORK"

if ! command -v clang-18 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y wget lsb-release software-properties-common gnupg cmake git python3-venv
  wget -qO llvm.sh https://apt.llvm.org/llvm.sh
  sudo bash llvm.sh 18
fi
export CC=clang-18 CXX=clang++-18

if [ ! -d BitNet ]; then
  git clone --recursive https://github.com/microsoft/BitNet.git
fi
cd BitNet
python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt "huggingface_hub[cli]"

if [ ! -f "$MODEL_DIR/ggml-model-i2_s.gguf" ]; then
  huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir "$MODEL_DIR"
fi
python setup_env.py -md "$MODEL_DIR" -q i2_s

BENCH="$(find build -name llama-bench -type f | head -n 1)"
STAMP="$(date -u +%Y%m%dT%H%MZ)"
OUT="$RESULTS/bitnet_cpp_2b4t_i2s_${STAMP}.json"
{
  echo "{"
  echo "  \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\","
  echo "  \"nproc\": $(nproc),"
  echo "  \"kernel\": \"i2_s\", \"tokens\": \"$TOKENS\", \"threads\": \"$THREADS\","
  echo "  \"bitnet_commit\": \"$(git rev-parse --short HEAD)\","
  echo "  \"llama_bench\": "
  "$BENCH" -m "$MODEL_DIR/ggml-model-i2_s.gguf" -p "$TOKENS" -n 0 -t "$THREADS" -r 5 -o json
  echo "}"
} > "$OUT"
echo "saved $OUT"
