"""Pareto-frontier algorithm: decision table over synthetic configs."""

from __future__ import annotations

import pytest

from quant_explorer.report.pareto import ParetoPoint, pareto_frontier, render_pareto_markdown


def _pt(name: str, size: float, lat: float, acc: float) -> ParetoPoint:
    return ParetoPoint(name=name, size_kb=size, p50_lat_ms_b1=lat, top1_acc=acc)


def test_singleton_is_on_frontier() -> None:
    assert pareto_frontier([_pt("a", 100, 1.0, 0.8)]) == {"a"}


def test_strict_domination_excludes() -> None:
    a = _pt("a", 100, 1.0, 0.8)  # smaller, faster, more accurate -> dominates b
    b = _pt("b", 200, 2.0, 0.7)
    assert pareto_frontier([a, b]) == {"a"}


def test_no_one_dominates_when_tradeoffs() -> None:
    # smallest, slowest, lowest acc
    a = _pt("a", 100, 5.0, 0.7)
    # biggest, fastest, mid acc
    b = _pt("b", 500, 1.0, 0.78)
    # mid size, mid lat, highest acc
    c = _pt("c", 300, 3.0, 0.82)
    assert pareto_frontier([a, b, c]) == {"a", "b", "c"}


def test_equal_points_neither_dominates() -> None:
    a = _pt("a", 100, 1.0, 0.8)
    b = _pt("b", 100, 1.0, 0.8)
    # Strict domination requires *some* improvement. Both stay.
    assert pareto_frontier([a, b]) == {"a", "b"}


def test_dominated_in_two_axes_but_better_in_one_stays() -> None:
    a = _pt("a", 100, 5.0, 0.6)
    b = _pt("b", 200, 1.0, 0.6)  # bigger but faster — both stay
    assert pareto_frontier([a, b]) == {"a", "b"}


def test_realistic_quant_table() -> None:
    """The expected shape of the real report.

    fp32 has highest accuracy, biggest size, mid latency.
    dynamic_int8 reduces size a bit, mid lat, near-baseline accuracy.
    static_per_tensor: small, fast, lower accuracy.
    static_per_channel: small, fast, accuracy almost matches baseline.
    """
    fp32 = _pt("fp32", 600, 3.0, 0.785)  # strictly highest acc
    dyn = _pt("dyn", 540, 2.9, 0.78)
    spt = _pt("spt", 170, 2.0, 0.74)
    spc = _pt("spc", 175, 2.0, 0.77)
    frontier = pareto_frontier([fp32, dyn, spt, spc])
    # fp32 has strictly highest acc -> nothing dominates it.
    assert "fp32" in frontier
    # spc has highest acc among small/fast configs.
    assert "spc" in frontier
    # spt is the smallest -> on frontier
    assert "spt" in frontier


def test_render_markdown_contains_required_columns() -> None:
    rows = [
        {
            "name": "fp32_baseline",
            "size_kb": 600.0,
            "p50_lat_ms_b1": 3.0,
            "top1_acc": 0.78,
            "mem_peak_mb": 200.0,
        },
        {
            "name": "dynamic_int8",
            "size_kb": 540.0,
            "p50_lat_ms_b1": 2.9,
            "top1_acc": 0.78,
            "mem_peak_mb": 195.0,
        },
    ]
    md = render_pareto_markdown(rows)
    for header in [
        "size_kb",
        "size_ratio",
        "p50_lat_ms_b1",
        "latency_speedup",
        "top1_acc",
        "acc_drop_pp",
        "pareto_optimal",
    ]:
        assert header in md
    assert "fp32_baseline" in md
    assert "dynamic_int8" in md


def test_render_markdown_requires_baseline() -> None:
    rows = [
        {
            "name": "dynamic_int8",
            "size_kb": 540.0,
            "p50_lat_ms_b1": 2.9,
            "top1_acc": 0.78,
            "mem_peak_mb": 195.0,
        }
    ]
    with pytest.raises(ValueError):
        render_pareto_markdown(rows)
