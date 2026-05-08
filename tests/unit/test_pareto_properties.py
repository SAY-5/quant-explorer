"""Hypothesis property tests for the Pareto-frontier algorithm.

Properties:

1. Frontier excludes any config strictly dominated on all three axes (size,
   latency, accuracy). I.e. if there exists ``other`` strictly better on every
   axis, ``self`` MUST NOT be on the frontier.
2. Frontier is non-empty for non-empty input.
3. Frontier is a subset of input names.
4. Idempotence — running ``pareto_frontier`` on (the points whose names are)
   the frontier returns the same set.
5. A point that is the unique min-size, unique min-latency, OR unique max-acc
   must always be on the frontier.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_explorer.report.pareto import ParetoPoint, pareto_frontier

# Bounded ranges keep the search space tractable without losing generality.
_size = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_lat = st.floats(min_value=0.001, max_value=1_000.0, allow_nan=False, allow_infinity=False)
_acc = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _point_strategy() -> st.SearchStrategy[tuple[float, float, float]]:
    return st.tuples(_size, _lat, _acc)


def _build_points(triples: list[tuple[float, float, float]]) -> list[ParetoPoint]:
    return [
        ParetoPoint(name=f"c{i}", size_kb=s, p50_lat_ms_b1=lat, top1_acc=a)
        for i, (s, lat, a) in enumerate(triples)
    ]


@settings(max_examples=200, deadline=None)
@given(st.lists(_point_strategy(), min_size=1, max_size=12))
def test_strictly_dominated_points_are_excluded(
    triples: list[tuple[float, float, float]],
) -> None:
    """If any other point is *strictly* better on every axis, p is excluded."""
    points = _build_points(triples)
    frontier = pareto_frontier(points)

    for p in points:
        strictly_dominated = any(
            other.name != p.name
            and other.size_kb < p.size_kb
            and other.p50_lat_ms_b1 < p.p50_lat_ms_b1
            and other.top1_acc > p.top1_acc
            for other in points
        )
        if strictly_dominated:
            assert (
                p.name not in frontier
            ), f"{p.name} is strictly dominated on every axis but stayed on the frontier"


@settings(max_examples=100, deadline=None)
@given(st.lists(_point_strategy(), min_size=1, max_size=12))
def test_frontier_is_nonempty_and_subset_of_input(
    triples: list[tuple[float, float, float]],
) -> None:
    points = _build_points(triples)
    names = {p.name for p in points}
    frontier = pareto_frontier(points)
    assert frontier, "frontier should never be empty for non-empty input"
    assert frontier.issubset(names)


@settings(max_examples=100, deadline=None)
@given(st.lists(_point_strategy(), min_size=1, max_size=10))
def test_frontier_is_idempotent(triples: list[tuple[float, float, float]]) -> None:
    """Re-running on just the frontier members should reproduce the same set."""
    points = _build_points(triples)
    frontier = pareto_frontier(points)
    frontier_points = [p for p in points if p.name in frontier]
    assert pareto_frontier(frontier_points) == frontier


@settings(max_examples=100, deadline=None)
@given(st.lists(_point_strategy(), min_size=2, max_size=10))
def test_unique_extremum_on_any_axis_is_on_frontier(
    triples: list[tuple[float, float, float]],
) -> None:
    """If a point is the *unique* min-size, min-latency, or max-acc, it cannot
    be dominated (domination requires being at least as good on every axis,
    which fails on that one axis)."""
    points = _build_points(triples)
    frontier = pareto_frontier(points)

    sizes = sorted(p.size_kb for p in points)
    if len(sizes) >= 2 and sizes[0] < sizes[1]:
        unique_smallest = min(points, key=lambda p: p.size_kb)
        assert unique_smallest.name in frontier

    lats = sorted(p.p50_lat_ms_b1 for p in points)
    if len(lats) >= 2 and lats[0] < lats[1]:
        unique_fastest = min(points, key=lambda p: p.p50_lat_ms_b1)
        assert unique_fastest.name in frontier

    accs = sorted((p.top1_acc for p in points), reverse=True)
    if len(accs) >= 2 and accs[0] > accs[1]:
        unique_best = max(points, key=lambda p: p.top1_acc)
        assert unique_best.name in frontier


@settings(max_examples=100, deadline=None)
@given(st.lists(_point_strategy(), min_size=1, max_size=8))
def test_adding_dominated_point_does_not_change_frontier(
    triples: list[tuple[float, float, float]],
) -> None:
    """Appending a strictly-dominated point must not shift the frontier."""
    if not triples:
        return
    points = _build_points(triples)
    frontier_before = pareto_frontier(points)
    if not frontier_before:
        return

    # Pick any frontier point and synthesise a strictly-worse copy.
    anchor = next(p for p in points if p.name in frontier_before)
    worse = ParetoPoint(
        name="worse",
        size_kb=anchor.size_kb + 1.0,
        p50_lat_ms_b1=anchor.p50_lat_ms_b1 + 0.1,
        top1_acc=max(0.0, anchor.top1_acc - 0.01),
    )
    frontier_after = pareto_frontier([*points, worse])
    assert "worse" not in frontier_after
    assert frontier_after == frontier_before
