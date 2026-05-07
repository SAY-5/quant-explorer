"""Pareto-frontier computation + Markdown table emission.

A configuration is on the Pareto frontier if no other configuration
dominates it on all three axes — smaller size, lower latency, and higher
accuracy. Strict domination on at least one axis is required (otherwise a
tie wouldn't kick anything off).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    """A single config's summary for Pareto comparison."""

    name: str
    size_kb: float
    p50_lat_ms_b1: float
    top1_acc: float

    def dominates(self, other: ParetoPoint) -> bool:
        """True if ``self`` is at least as good on every axis and strictly
        better on at least one.

        Better means: smaller size, smaller latency, larger accuracy.
        """
        not_worse = (
            self.size_kb <= other.size_kb
            and self.p50_lat_ms_b1 <= other.p50_lat_ms_b1
            and self.top1_acc >= other.top1_acc
        )
        if not not_worse:
            return False
        strictly_better = (
            self.size_kb < other.size_kb
            or self.p50_lat_ms_b1 < other.p50_lat_ms_b1
            or self.top1_acc > other.top1_acc
        )
        return strictly_better


def pareto_frontier(points: list[ParetoPoint]) -> set[str]:
    """Return the set of config names on the Pareto frontier."""
    frontier: set[str] = set()
    for candidate in points:
        if any(other.dominates(candidate) for other in points if other.name != candidate.name):
            continue
        frontier.add(candidate.name)
    return frontier


def _fmt_pp(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}pp"


def render_pareto_markdown(
    rows: list[dict[str, float | int | str]],
    baseline_name: str = "fp32_baseline",
) -> str:
    """Render the report Markdown.

    ``rows`` are ordered as the caller wants them displayed; each row must
    contain ``name``, ``size_kb``, ``p50_lat_ms_b1``, ``top1_acc``,
    ``mem_peak_mb``.
    """
    by_name = {r["name"]: r for r in rows}
    if baseline_name not in by_name:
        raise ValueError(f"baseline {baseline_name!r} not in rows")
    baseline = by_name[baseline_name]
    base_size = float(baseline["size_kb"])
    base_lat = float(baseline["p50_lat_ms_b1"])
    base_acc = float(baseline["top1_acc"])

    points = [
        ParetoPoint(
            name=str(r["name"]),
            size_kb=float(r["size_kb"]),
            p50_lat_ms_b1=float(r["p50_lat_ms_b1"]),
            top1_acc=float(r["top1_acc"]),
        )
        for r in rows
    ]
    frontier = pareto_frontier(points)

    lines = [
        "# Quantization tradeoff Pareto",
        "",
        "| config | size_kb | size_ratio | p50_lat_ms_b1 | latency_speedup | top1_acc | acc_drop_pp | mem_peak_mb | pareto_optimal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for r in rows:
        name = str(r["name"])
        size_kb = float(r["size_kb"])
        lat = float(r["p50_lat_ms_b1"])
        acc = float(r["top1_acc"])
        mem = float(r["mem_peak_mb"])
        size_ratio = size_kb / base_size if base_size > 0 else 0.0
        speedup = base_lat / lat if lat > 0 else 0.0
        acc_delta_pp = (acc - base_acc) * 100.0
        opt = "yes" if name in frontier else "no"
        lines.append(
            f"| {name} | {size_kb:.0f} | {size_ratio:.2f}x | "
            f"{lat:.2f} | {speedup:.2f}x | {acc * 100:.1f}% | "
            f"{_fmt_pp(acc_delta_pp)} | {mem:.1f} | {opt} |"
        )

    # Pick recommendations from the frontier.
    frontier_pts = [p for p in points if p.name in frontier]
    if frontier_pts:
        smallest = min(frontier_pts, key=lambda p: p.size_kb)
        most_accurate = max(frontier_pts, key=lambda p: p.top1_acc)
        fastest = min(frontier_pts, key=lambda p: p.p50_lat_ms_b1)
        lines.extend(
            [
                "",
                "Pareto frontier picks:",
                f"- minimum size: `{smallest.name}` ({smallest.size_kb / base_size:.2f}x of FP32)",
                f"- highest accuracy: `{most_accurate.name}` (top-1 {most_accurate.top1_acc * 100:.1f}%)",
                f"- lowest latency: `{fastest.name}` (p50 {fastest.p50_lat_ms_b1:.2f}ms at batch 1)",
            ]
        )
    return "\n".join(lines) + "\n"
