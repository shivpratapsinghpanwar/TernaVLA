"""Is the CPU running at full speed right now? fp32 GEMM throughput on 1 and 4 threads plus the
effective clock sampled while loaded. An i5-10210U core peaks at 32 fp32 FLOP per cycle (two
AVX2 FMA ports), so GFLOPS / 32 / 0.85 roughly gives the clock in GHz for a large SGEMM."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ternavla.bench.machine import capture, effective_mhz  # noqa: E402

n = 2048
a, b = torch.randn(n, n), torch.randn(n, n)
st = capture()
print(f"AC {st.on_ac_power}, battery {st.battery_percent}%")
for threads in (1, 4, 8):
    torch.set_num_threads(threads)
    torch.mm(a, b)
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            m = effective_mhz()
            if m:
                samples.append(m)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0, reps = time.perf_counter(), 0
    while time.perf_counter() - t0 < 12:
        torch.mm(a, b)
        reps += 1
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    gflops = 2 * n**3 * reps / dt / 1e9
    print(f"{threads} thread(s): {gflops:6.1f} GFLOPS fp32 "
          f"({gflops / threads / 32:.2f} GHz-equivalent per core at 100% FMA use); "
          f"counter clock samples MHz {sorted(samples)}")
