"""Experiment zero: what does one TernaVLA action step cost on this CPU, before any training?

Times, at the real shapes of each tier (random weights; latency does not depend on values):
  vision   SigLIP2-base-patch16-256 tower: torch fp32, ONNX Runtime fp32 and int8
  LM       the layer stack of S / M-lite / M at its token budget, six backends
  head     projector + 13-query action head, fp32
and composes per-chunk totals from the fastest measured backend of each part.

    python scripts/exp0.py --threads 4 8
    python scripts/exp0.py --quick          # smoke run, fewer repeats, S only

Results: results/exp0_<host>_<utc-date>.json (raw rows + machine state + clock samples).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ternavla.bench import machine  # noqa: E402
from ternavla.bench.lm_graph import Periphery, TorchStack, build_onnx, ort_session  # noqa: E402
from ternavla.bench.timing import measure  # noqa: E402
from ternavla.bench.vision import (export_onnx, ort_vision_session, quantize_int8,  # noqa: E402
                                   siglip_vision)
from ternavla.shapes import ABLATION_ONE_PER_STEP, BUDGETS, MODELS  # noqa: E402

OUT = ROOT / "output"
RESULTS = ROOT / "results"

# torch-int8 (torch._int_mm) is excluded by default: on this AVX2 CPU it fell back to a reference
# kernel (S, 43 tokens: 131 s per pass vs 0.28 s fp32), so it measures nothing useful.
LM_BACKENDS = ["torch-fp32", "ort-fp32", "ort-nbits8", "ort-nbits4", "ort-nbits2", "ort-nbits2-b128"]


def gemm_gflops(threads: int, seconds: float = 6.0, n: int = 2048) -> float:
    """Sustained fp32 SGEMM throughput: the honest 'how fast is the CPU right now' number.

    The Windows package clock counter averages idle cores in and under-reads badly, so the run
    records this calibration at the start and the end instead."""
    import torch

    torch.set_num_threads(threads)
    a, b = torch.randn(n, n), torch.randn(n, n)
    torch.mm(a, b)
    t0, reps = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        torch.mm(a, b)
        reps += 1
    return round(2 * n**3 * reps / (time.perf_counter() - t0) / 1e9, 1)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def time_lm(shape, backend: str, tokens: int, threads: int, depth: int, n: int, warmup: int,
            block_size: int = 32) -> dict:
    import torch

    torch.set_num_threads(threads)
    x = np.random.default_rng(0).standard_normal((tokens, shape.hidden)).astype(np.float32)
    row = dict(part="lm", model=shape.key, tier=shape.tier, backend=backend, tokens=tokens,
               threads=threads, depth=depth, layers=shape.layers,
               block_size=block_size if backend.startswith("ort-nbits") else None)
    t_build = time.perf_counter()
    if backend.startswith("torch"):
        stack = TorchStack(shape, depth, backend.split("-")[1])
        xt = torch.from_numpy(x)
        fn = lambda: stack(xt)  # noqa: E731
        holder = stack
    else:
        parts = backend.split("-")          # ort-fp32 | ort-nbits2 | ort-nbits2-b128
        if len(parts) == 3:
            block_size = int(parts[2][1:])
            row["block_size"] = block_size
        model = build_onnx(shape, depth, parts[1], block_size=block_size)
        sess = ort_session(model, threads)
        del model
        gc.collect()
        fn = lambda: sess.run(None, {"x": x})  # noqa: E731
        holder = sess
    row["build_s"] = round(time.perf_counter() - t_build, 2)
    row["rss_mb"] = round(machine.pid_rss_mb(), 1)
    t = measure(fn, n=n, warmup=warmup)
    row["timing_depth"] = t.to_dict()
    row["full_ms_p50"] = round(t.p50_ms * shape.layers / depth, 2)
    row["full_ms_p95"] = round(t.p95_ms * shape.layers / depth, 2)
    row["gops_effective"] = round(shape.gop_per_token() * tokens * depth / shape.layers
                                  / (t.p50_ms / 1e3), 1)
    del holder, fn
    gc.collect()
    return row


def time_vision(threads: int, n: int, warmup: int) -> list[dict]:
    import torch

    torch.set_num_threads(threads)
    rows = []
    model = siglip_vision()
    px = torch.randn(1, 3, 256, 256)
    for impl in ("eager", "sdpa"):
        model.config._attn_implementation = impl
        try:
            with torch.inference_mode():
                t = measure(lambda: model(pixel_values=px), n=n, warmup=warmup)
            rows.append(dict(part="vision", backend=f"torch-fp32-{impl}", threads=threads,
                             timing=t.to_dict(), ms_p50=t.p50_ms))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(part="vision", backend=f"torch-fp32-{impl}", threads=threads,
                             error=repr(e)))
    model.config._attn_implementation = "eager"
    fp32 = OUT / "siglip2_base_256_fp32.onnx"
    int8 = OUT / "siglip2_base_256_int8.onnx"
    try:
        if not fp32.exists():
            export_onnx(model, fp32)
        if not int8.exists():
            quantize_int8(fp32, int8)
        xin = px.numpy()
        for name, path in (("ort-fp32", fp32), ("ort-int8", int8)):
            sess = ort_vision_session(path, threads)
            t = measure(lambda: sess.run(None, {"pixel_values": xin}), n=n, warmup=warmup)
            rows.append(dict(part="vision", backend=name, threads=threads, timing=t.to_dict(),
                             ms_p50=t.p50_ms))
            del sess
    except Exception as e:  # noqa: BLE001
        rows.append(dict(part="vision", backend="ort", threads=threads, error=repr(e),
                         trace=traceback.format_exc()[-1500:]))
    return rows


def time_periphery(shape, threads: int, n: int, warmup: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    pool = 4 if shape.tier == "S" else 2
    per = Periphery(shape.hidden, pool=pool)
    patches = torch.randn(per.image_tokens, per.in_dim)
    h = torch.randn(per.queries, shape.hidden)
    t = measure(lambda: (per.project(patches), per.head(h)), n=n, warmup=warmup)
    return dict(part="periphery", model=shape.key, tier=shape.tier, threads=threads,
                head_params=per.params, timing=t.to_dict(), ms_p50=t.p50_ms)


def compose(rows: list[dict]) -> list[dict]:
    """Per tier and thread count: fastest vision + fastest LM backend + periphery."""
    out = []
    for threads in sorted({r["threads"] for r in rows}):
        vis = [r for r in rows if r["part"] == "vision" and r["threads"] == threads and "ms_p50" in r]
        if not vis:
            continue
        v = min(vis, key=lambda r: r["ms_p50"])
        for tier_rows in _group(rows, threads):
            lm = min(tier_rows["lm"], key=lambda r: r["full_ms_p50"])
            per = tier_rows["periphery"]
            total = v["ms_p50"] + lm["full_ms_p50"] + (per["ms_p50"] if per else 0.0)
            out.append(dict(tier=lm["tier"], model=lm["model"], tokens=lm["tokens"], threads=threads,
                            vision_backend=v["backend"], vision_ms=v["ms_p50"],
                            lm_backend=lm["backend"], lm_ms=lm["full_ms_p50"],
                            periphery_ms=per["ms_p50"] if per else None,
                            chunk_ms=round(total, 1)))
    return out


def _group(rows, threads):
    keys = sorted({(r["model"], r["tokens"]) for r in rows if r["part"] == "lm" and r["threads"] == threads
                   and "full_ms_p50" in r})
    for model, tokens in keys:
        lm = [r for r in rows if r["part"] == "lm" and r["threads"] == threads and r["model"] == model
              and r["tokens"] == tokens and "full_ms_p50" in r]
        per = next((r for r in rows if r["part"] == "periphery" and r["threads"] == threads
                    and r["model"] == model), None)
        yield {"lm": lm, "periphery": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--depth", type=int, default=4, help="layers built per backend; time is scaled")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--backends", nargs="+", default=LM_BACKENDS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-fulldepth-check", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n, args.warmup, args.models, args.threads = 3, 1, ["bitcpm_cann_0p5b"], [4]

    state = machine.capture()
    log(f"{state.cpu} | AC power: {state.on_ac_power} | battery {state.battery_percent}% | "
        f"versions {state.versions}")
    if not state.publishable:
        log("WARNING: on battery - results will be marked unpublishable")

    calib_start = {t: gemm_gflops(t) for t in sorted(set(args.threads) | {1})}
    log(f"fp32 SGEMM calibration at start (GFLOPS by threads): {calib_start}")

    rows: list[dict] = []
    with machine.ClockSampler() as clock:
        for threads in args.threads:
            log(f"vision, {threads} threads")
            rows += time_vision(threads, args.n, args.warmup)
            for r in rows[-4:]:
                log(f"  {r.get('backend')}: {r.get('ms_p50', r.get('error'))} ms")
            for key in args.models:
                shape = MODELS[key]
                budgets = [BUDGETS[shape.tier].total]
                if shape.tier == "M":
                    budgets.append(ABLATION_ONE_PER_STEP.total)
                rows.append(time_periphery(shape, threads, args.n, args.warmup))
                for tokens in budgets:
                    for backend in args.backends:
                        try:
                            r = time_lm(shape, backend, tokens, threads, min(args.depth, shape.layers),
                                        args.n, args.warmup)
                            log(f"  {shape.tier:6s} {key:18s} T={tokens:3d} {backend:11s} "
                                f"{r['full_ms_p50']:9.1f} ms/pass  ({r['gops_effective']} GOPS)")
                        except Exception as e:  # noqa: BLE001
                            r = dict(part="lm", model=key, tier=shape.tier, backend=backend,
                                     tokens=tokens, threads=threads, error=repr(e),
                                     trace=traceback.format_exc()[-1500:])
                            log(f"  {shape.tier:6s} {key:18s} T={tokens:3d} {backend:11s} ERROR {e!r}")
                        rows.append(r)
            if not args.no_fulldepth_check and "bitnet_2b4t" in args.models and not args.quick:
                shape = MODELS["bitnet_2b4t"]
                r = time_lm(shape, "ort-nbits2", BUDGETS["M"].total, threads, shape.layers,
                            max(3, args.n // 2), 1)
                r["note"] = "full-depth check of the depth-scaling assumption"
                log(f"  full-depth M ort-nbits2: {r['full_ms_p50']} ms/pass")
                rows.append(r)
        clock_summary = clock.summary()

    calib_end = {t: gemm_gflops(t) for t in sorted(set(args.threads) | {1})}
    log(f"fp32 SGEMM calibration at end (GFLOPS by threads): {calib_end}")
    end_state = machine.capture()
    summary = compose(rows)
    log("per-chunk totals (p50):")
    for s in summary:
        log(f"  {s['tier']:6s} T={s['tokens']:3d} {s['threads']}t  vision {s['vision_ms']:.0f} "
            f"({s['vision_backend']}) + LM {s['lm_ms']:.0f} ({s['lm_backend']}) + head "
            f"{s['periphery_ms']:.0f} = {s['chunk_ms']:.0f} ms")
    log(f"package clock counter under load (averages idle cores; indicative only): {clock_summary}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = RESULTS / f"exp0_{state.hostname.lower()}_{stamp}{'_quick' if args.quick else ''}.json"
    publishable = state.publishable and end_state.publishable
    out.write_text(json.dumps(dict(
        experiment="exp0", args=vars(args), publishable=publishable,
        machine_start=state.to_dict(), machine_end=end_state.to_dict(),
        sgemm_gflops_start=calib_start, sgemm_gflops_end=calib_end,
        package_clock_counter=clock_summary,
        rows=rows, summary=summary), indent=2))
    log(f"saved {out}  (publishable: {publishable})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
