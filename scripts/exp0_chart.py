"""Chart for experiment zero: per-chunk latency by tier, and the LM backend comparison.

    python scripts/exp0_chart.py [results/exp0_*.json]   -> docs/media/exp0_chunk_latency.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TIER_ORDER = ["S", "M-lite", "M"]
COLORS = {"vision": "#4C78A8", "lm": "#F58518", "head": "#54A24B"}


def latest() -> Path:
    files = sorted((ROOT / "results").glob("exp0_*.json"))
    files = [f for f in files if "_quick" not in f.name]
    if not files:
        raise SystemExit("no exp0 results")
    return files[-1]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest()
    data = json.loads(path.read_text())
    summary, rows = data["summary"], data["rows"]
    threads = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: stacked per-chunk totals (design token budget per tier), 4 and 8 threads side by side.
    def design_rows(t):
        out = []
        for tier in TIER_ORDER:
            cands = [s for s in summary if s["tier"] == tier and s["threads"] == t]
            if cands:
                out.append(min(cands, key=lambda r: r["tokens"]))
        return out

    bars = design_rows(threads)
    labels = [f"{s['tier']}\n{s['tokens']} tokens" for s in bars]
    width = 0.38
    for k, t in enumerate((4, 8)):
        rows_t = design_rows(t)
        if not rows_t:
            continue
        xs = [i + (k - 0.5) * width for i in range(len(rows_t))]
        v = [s["vision_ms"] / 1e3 for s in rows_t]
        lm = [s["lm_ms"] / 1e3 for s in rows_t]
        hd = [(s["periphery_ms"] or 0) / 1e3 for s in rows_t]
        hatch = None if t == 4 else "//"
        ax1.bar(xs, v, width, color=COLORS["vision"], hatch=hatch, edgecolor="white",
                label="vision (SigLIP2-base, int8)" if k == 0 else None)
        ax1.bar(xs, lm, width, bottom=v, color=COLORS["lm"], hatch=hatch, edgecolor="white",
                label="ternary LM (fastest backend)" if k == 0 else None)
        ax1.bar(xs, hd, width, bottom=[a + b for a, b in zip(v, lm)], color=COLORS["head"], hatch=hatch,
                edgecolor="white", label="projector + head" if k == 0 else None)
        for xi, s in zip(xs, rows_t):
            ax1.text(xi, s["chunk_ms"] / 1e3 + 0.05, f"{s['chunk_ms'] / 1e3:.2f}\n{t}t", ha="center",
                     fontsize=8)
    ax1.set_ylim(0, max(s["chunk_ms"] for s in summary if s in bars) / 1e3 * 1.18)
    ax1.axhline(1.04, color="#666", ls="--", lw=1)
    ax1.text(-0.45, 1.09, "chunk length 1.04 s", ha="left", fontsize=8, color="#444")
    ax1.set_xticks(list(range(len(bars))), labels)
    ax1.set_ylabel("seconds per action chunk (p50)")
    ax1.set_title("One action step on an i5-10210U (plain: 4 threads, hatched: 8)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: LM backends for each tier at 4 threads (ms per pass, full depth).
    backends = ["torch-fp32", "ort-fp32", "ort-nbits8", "ort-nbits4", "ort-nbits2", "ort-nbits2-b128"]
    width = 0.13
    for j, b in enumerate(backends):
        ys = []
        for s in bars:
            r = next((r for r in rows if r["part"] == "lm" and r["tier"] == s["tier"]
                      and r["tokens"] == s["tokens"] and r["threads"] == threads
                      and r["backend"] == b and "full_ms_p50" in r and r.get("depth", 0) < 30), None)
            ys.append(r["full_ms_p50"] / 1e3 if r else 0)
        ax2.bar([i + (j - 2.5) * width for i in range(len(bars))], ys, width, label=b)
    ax2.set_xticks(list(range(len(bars))), labels)
    ax2.set_ylabel("seconds per LM pass (p50)")
    ax2.set_title("Language-model pass by CPU backend, 4 threads", fontsize=10)
    ax2.legend(fontsize=7, ncol=2, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("TernaVLA experiment zero: random weights at real shapes, laptop on mains", fontsize=10)
    fig.tight_layout()
    out = ROOT / "docs" / "media" / "exp0_chunk_latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
