"""Build repository figures and an offline interactive figure from a public exp0 result.

    python scripts/build_research_visuals.py [results/exp0_*.json]

No model is loaded. Only allowlisted public benchmark fields enter the interactive artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/exp0_aiml-shivpratap_20261012T1256Z.json"
MEDIA = ROOT / "docs/media"
VISUALS = ROOT / "docs/visuals"
TIERS = ("S", "M-lite", "M")
COLORS = ("#61d4c4", "#bca7ff", "#ffb775")


def public_data(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("experiment") != "exp0" or data.get("publishable") is not True:
        raise ValueError("Select a publishable exp0 result; quick/unpublishable runs are not presentation evidence")
    if data.get("args", {}).get("quick"):
        raise ValueError("Quick runs are smoke checks, not presentation evidence")
    summary_keys = ("tier", "tokens", "threads", "vision_backend", "vision_ms", "lm_backend",
                    "lm_ms", "periphery_ms", "chunk_ms")
    summary = [{key: row[key] for key in summary_keys} for row in data["summary"]]
    for tier in TIERS:
        for threads in (4, 8):
            if not any(s["tier"] == tier and s["threads"] == threads for s in summary):
                raise ValueError(f"Missing design result for {tier}, {threads} threads")
    for row in summary:
        if row["periphery_ms"] is None or abs(sum(row[k] for k in ("vision_ms", "lm_ms", "periphery_ms")) - row["chunk_ms"]) > 0.11:
            raise ValueError("Component timings do not support the displayed total")
    lm_keys = ("tier", "tokens", "threads", "backend", "depth", "layers", "full_ms_p50")
    lm = [{key: row[key] for key in lm_keys} for row in data["rows"]
          if row["part"] == "lm" and "full_ms_p50" in row]
    machine = data["machine_start"]
    return dict(schema=1, source=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                date=machine["timestamp_utc"][:10], cpu=machine["cpu"],
                summary=summary, lm=lm, horizon_s=1.04,
                measurement="Synthetic component timings; untrained policy; total is sum of component p50s")


def design(data: dict, tier: str, threads: int) -> dict:
    return min((r for r in data["summary"] if r["tier"] == tier and r["threads"] == threads), key=lambda r: r["tokens"])


def banner(data: dict) -> str:
    seconds = design(data, "S", 8)["chunk_ms"] / 1000
    cells = []
    for row in range(5):
        for col in range(7):
            value = (-1, 0, 1)[(row * 2 + col) % 3]
            color = {0: "#314555", 1: "#61d4c4", -1: "#bca7ff"}[value]
            x, y = 808 + col * 48, 49 + row * 44
            cells.append(f'<rect x="{x}" y="{y}" width="38" height="34" rx="7" fill="{color}" opacity=".14"/><text x="{x+19}" y="{y+23}" text-anchor="middle" font-size="16" fill="{color}">{value}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 300" role="img" aria-labelledby="title desc">
<title id="title">TernaVLA — small weights, embodied intelligence</title><desc id="desc">Research into ternary language backbones for humanoid action on edge CPUs. Experiment zero is measured; a trained ternary policy is pending.</desc>
<defs><linearGradient id="bg"><stop stop-color="#0c1824"/><stop offset="1" stop-color="#122a36"/></linearGradient></defs>
<rect width="1200" height="300" rx="20" fill="url(#bg)"/>
<g font-family="Segoe UI,Arial,sans-serif"><rect x="42" y="34" width="175" height="26" rx="13" fill="#173d3e"/><text x="56" y="52" fill="#91eadc" font-size="12" letter-spacing="1.5">RESEARCH IN PROGRESS</text>
<text x="40" y="126" fill="#f1f5f9" font-size="62" font-weight="700" letter-spacing="-2">Terna<tspan fill="#61d4c4">VLA</tspan></text>
<text x="43" y="168" fill="#d9e3ec" font-size="24">Small weights. Embodied intelligence.</text>
<text x="43" y="206" fill="#9eb3c3" font-size="16">Ternary backbones · 17-joint action · CPU-first evidence</text>
<text x="43" y="264" fill="#61d4c4" font-size="16" font-weight="600">{seconds:.2f} s</text><text x="103" y="264" fill="#a6bbc9" font-size="14">synthetic S estimate / 8 threads · trained policy pending</text>
{''.join(cells)}</g></svg>'''


def architecture() -> str:
    boxes = [(34, "01 / SEE", "Camera", "256 × 256 RGB", "SigLIP2 · INT8", "#61d4c4"),
             (275, "02 / CONDITION", "Context tokens", "16 or 64 image tokens", "+ instruction, state, prefix", "#61d4c4"),
             (516, "03 / INFER", "Ternary LM", "Weights in {−1, 0, +1}", "13 causal action queries", "#bca7ff"),
             (757, "04 / DECODE", "Action head", "Floating-point MLP", "13 × 4 × 17 outputs", "#ffb775")]
    content = []
    for x, label, title, detail, detail2, color in boxes:
        content.append(f'<rect x="{x}" y="94" width="211" height="146" rx="14" fill="#122330" stroke="#2b414f"/><text x="{x+18}" y="123" font-size="11" fill="{color}" letter-spacing="1.1">{label}</text><text x="{x+18}" y="158" font-size="22" fill="#f2f6f9" font-weight="600">{title}</text><text x="{x+18}" y="189" font-size="13" fill="#afc0cd">{html.escape(detail)}</text><text x="{x+18}" y="211" font-size="13" fill="#afc0cd">{html.escape(detail2)}</text>')
        if x < 757:
            content.append(f'<path d="M {x+216} 165 h 18" stroke="#6f8b9d" stroke-width="2" marker-end="url(#arrow)"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 330" role="img" aria-labelledby="title desc">
<title id="title">Proposed TernaVLA action interface</title><desc id="desc">INT8 vision and context feed a ternary language model, followed by a floating-point head producing 52 steps for 17 joints. This is a proposed architecture, not a trained end-to-end result.</desc>
<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0 L5 2.5 L0 5" fill="#6f8b9d"/></marker></defs>
<rect width="1000" height="330" rx="18" fill="#0d1925"/>
<g font-family="Segoe UI,Arial,sans-serif"><text x="34" y="42" fill="#eef4f7" font-size="22" font-weight="600">One image. One forward pass. A motion chunk.</text><text x="34" y="69" fill="#9bb0bf" font-size="13">PROPOSED INTERFACE · current recipe uses INT8 vision and a floating-point action head</text>
{''.join(content)}<text x="34" y="285" fill="#61d4c4" font-size="20" font-weight="600">52 steps × 17 joints at 50 Hz = 1.04 s of motion</text><text x="34" y="311" fill="#9bb0bf" font-size="13">Model capability and runtime still require validation with trained weights.</text></g></svg>'''


def chart(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "#e4edf4",
                         "axes.labelcolor": "#afc0cf", "xtick.color": "#c4d2df", "ytick.color": "#a5b8c7",
                         "axes.edgecolor": "#314453", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, facecolor="#0d1925")
    for ax, threads in zip(axes, (4, 8)):
        ax.set_facecolor("#0d1925")
        rows = [design(data, tier, threads) for tier in TIERS]
        bottoms = [0.0] * len(rows)
        for key, label, color in zip(("vision_ms", "lm_ms", "periphery_ms"), ("Vision", "Language model", "Projector + head"), COLORS):
            values = [r[key] / 1000 for r in rows]
            ax.barh(range(3), values, left=bottoms, color=color, height=.45, label=label)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        for i, r in enumerate(rows):
            ax.text(r["chunk_ms"] / 1000 + .07, i, f"{r['chunk_ms']/1000:.2f} s", va="center", fontsize=11)
        ax.set_yticks(range(3), [f"{t}  /  {r['tokens']} tokens" for t, r in zip(TIERS, rows)])
        ax.set_xlim(0, 3.55)
        ax.set_ylim(2.6, -.8)
        ax.axvline(1.04, color="#f4d49b", ls=(0, (4, 4)), lw=1)
        ax.text(1.08, -.52, "1.04 s motion horizon", fontsize=9, color="#f4d49b")
        ax.set_title(f"{threads} CPU threads", loc="left", fontweight="bold", pad=20)
        ax.set_xlabel("Sum of component p50 timings (seconds)", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=True)
        ax.grid(axis="x", color="#233745", alpha=.5)
        ax.set_axisbelow(True)
    fig.suptitle("A promising CPU budget. A trained policy still to prove.", x=.04, y=.98, ha="left", fontsize=17, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .01), fontsize=10)
    fig.text(.04, .91, f"Experiment zero · {data['date']} · i5-10210U · random ternary weights at real layer shapes", color="#a5b8c7", fontsize=10)
    fig.subplots_adjust(left=.14, right=.98, top=.74, bottom=.24, wspace=.38)
    fig.savefig(MEDIA / "research_benchmark.png", dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(MEDIA / "research_benchmark.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    data = public_data(args.source)
    MEDIA.mkdir(parents=True, exist_ok=True)
    VISUALS.mkdir(parents=True, exist_ok=True)
    (MEDIA / "research_header.svg").write_text(banner(data), encoding="utf-8")
    (MEDIA / "research_architecture.svg").write_text(architecture(), encoding="utf-8")
    chart(data)
    template = (VISUALS / "benchmark.template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    if template.count("__BENCHMARK_DATA__") != 1:
        raise ValueError("Interactive template needs exactly one data placeholder")
    (VISUALS / "benchmark.html").write_text(template.replace("__BENCHMARK_DATA__", payload), encoding="utf-8")
    print(f"Built four static figures and the offline explorer from {args.source.name}")
    print(f"Source SHA256: {data['sha256']}")


if __name__ == "__main__":
    main()
