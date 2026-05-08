"""Unit + property tests for the full-results JSON emitter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_explorer.report.json_emit import emit_full_results


def test_emit_writes_pretty_json(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    payload = {"b": 2, "a": 1, "c": [3, 2, 1]}
    emit_full_results(payload, out)
    text = out.read_text()
    assert text.endswith("\n")
    data = json.loads(text)
    assert data == payload
    # sort_keys=True is in effect.
    assert text.index('"a"') < text.index('"b"') < text.index('"c"')


def test_emit_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "results.json"
    emit_full_results({"x": 1}, out)
    assert out.exists()


@dataclass
class _AsDictHolder:
    a: int
    b: int

    def as_dict(self) -> dict[str, int]:
        return {"a": self.a, "b": self.b}


@dataclass
class _DunderDictHolder:
    name: str
    n: int


def test_emit_uses_as_dict_hook(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    emit_full_results({"holder": _AsDictHolder(1, 2)}, out)
    assert json.loads(out.read_text()) == {"holder": {"a": 1, "b": 2}}


def test_emit_falls_back_to_dunder_dict(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    emit_full_results({"holder": _DunderDictHolder("k", 7)}, out)
    assert json.loads(out.read_text()) == {"holder": {"name": "k", "n": 7}}


class _Unserializable:
    __slots__ = ()


def test_emit_raises_on_truly_unserializable(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    with pytest.raises(TypeError):
        emit_full_results({"x": _Unserializable()}, out)


_json_payload = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=6,
    ),
    values=st.recursive(
        st.one_of(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=10),
            st.booleans(),
            st.none(),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(
                keys=st.text(
                    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                    min_size=1,
                    max_size=4,
                ),
                values=children,
                max_size=4,
            ),
        ),
        max_leaves=12,
    ),
    max_size=6,
)


@settings(max_examples=50, deadline=None)
@given(payload=_json_payload)
def test_emit_round_trips_arbitrary_json(
    payload: dict[str, object], tmp_path_factory: pytest.TempPathFactory
) -> None:
    # ``tmp_path_factory`` is a session-scoped fixture and is safe to use
    # inside a Hypothesis-driven test (``tmp_path`` is function-scoped and
    # would be reset on every example).
    tmp = tmp_path_factory.mktemp("emit")
    out = tmp / "rand.json"
    emit_full_results(payload, out)
    parsed = json.loads(out.read_text())
    assert parsed == payload
