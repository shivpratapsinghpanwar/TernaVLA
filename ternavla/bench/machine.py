"""What the machine was doing during a measurement, so numbers can be judged and reproduced.

A run on battery is recorded but marked unpublishable: laptops throttle hard when discharging
(the first SmolVLA run here ran at 0.8 GHz). The effective clock is sampled from Windows
performance counters *while the benchmark is loading the CPU*, because an idle reading says
nothing about the clock under load.
"""

from __future__ import annotations

import os
import platform
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import psutil

_PERF = r"\Processor Information(_Total)\% Processor Performance"
_FREQ = r"\Processor Information(_Total)\Processor Frequency"


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20)
            name = out.stdout.strip()
            if name:
                return name
        except Exception:
            pass
    return platform.processor()


def effective_mhz() -> float | None:
    """Base frequency x % processor performance, averaged over the whole package (Windows)."""
    if platform.system() != "Windows":
        f = psutil.cpu_freq()
        return f.current if f else None
    try:
        out = subprocess.run(["typeperf", _FREQ, _PERF, "-sc", "1"],
                             capture_output=True, text=True, timeout=20).stdout
        row = [ln for ln in out.splitlines() if ln.startswith('"') and "/" in ln][-1]
        cells = [c.strip('"') for c in row.split(",")]
        base, perf = float(cells[1]), float(cells[2])
        return round(base * perf / 100.0, 1)
    except Exception:
        return None


class ClockSampler:
    """Samples the effective clock every few seconds on a side thread while a benchmark runs."""

    def __init__(self, every_s: float = 4.0):
        self.every_s = every_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            mhz = effective_mhz()
            if mhz is not None:
                self.samples.append(mhz)

    def __enter__(self) -> "ClockSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def summary(self) -> dict:
        s = sorted(self.samples)
        if not s:
            return {"n": 0}
        return {"n": len(s), "min_mhz": s[0], "median_mhz": s[len(s) // 2], "max_mhz": s[-1]}


@dataclass
class MachineState:
    hostname: str
    os: str
    cpu: str
    cores_physical: int
    cores_logical: int
    ram_gb: float
    on_ac_power: bool | None
    battery_percent: float | None
    python: str
    versions: dict = field(default_factory=dict)
    timestamp_utc: str = ""

    @property
    def publishable(self) -> bool:
        return self.on_ac_power is not False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["publishable"] = self.publishable
        return d


def capture() -> MachineState:
    import numpy
    versions = {"numpy": numpy.__version__}
    for mod in ("torch", "onnxruntime", "onnx", "transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    battery = psutil.sensors_battery()
    return MachineState(
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        cpu=_cpu_name(),
        cores_physical=psutil.cpu_count(logical=False) or 0,
        cores_logical=psutil.cpu_count(logical=True) or 0,
        ram_gb=round(psutil.virtual_memory().total / 1e9, 2),
        on_ac_power=None if battery is None else bool(battery.power_plugged),
        battery_percent=None if battery is None else float(battery.percent),
        python=platform.python_version(),
        versions=versions,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def pid_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6
