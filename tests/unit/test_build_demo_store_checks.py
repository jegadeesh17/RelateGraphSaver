"""AC-135, AC-136: the demo builder's real-API lookup check reports unreadable arguments.

``_check_real_api_lookups`` reads recorded tool calls from the store. A lookup whose arguments
cannot be decoded into a dict means the thread did not record a usable run, so the builder
reports it as an unreachable PyPI instead of crashing with a stray ``TypeError`` or
``AttributeError``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from examples.build_demo_store import (
    MISSING_PACKAGE,
    PyPIUnreachableError,
    _check_real_api_lookups,
)

from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.serde import RelateStateSerializer

_SERDE = RelateStateSerializer()


class _StubProxy:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute_read(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self._rows


class _StubSaver:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.proxy = _StubProxy(rows)
        self.serde = _SERDE


def _saver(rows: list[dict[str, Any]]) -> RelateSaver:
    return cast(RelateSaver, _StubSaver(rows))


def _row(args: Any, *, error: str | None = None, result: Any = b"ok") -> dict[str, Any]:
    return {"args": args, "error": error, "result": result}


def _good_rows() -> list[dict[str, Any]]:
    return [
        _row(_SERDE.dumps({"name": "requests"})),
        _row(_SERDE.dumps({"name": MISSING_PACKAGE}), error="404 Not Found", result=None),
    ]


def test_ac136_valid_lookups_pass() -> None:
    """Control: well-formed bytes arguments pass the check."""
    _check_real_api_lookups(_saver(_good_rows()))


@pytest.mark.parametrize(
    "bad_args",
    [None, "not-a-blob", _SERDE.dumps(["relategraphsaver"]), 42],
    ids=["none", "str-not-a-blob", "serialized-list", "int"],
)
def test_ac136_unreadable_arguments_report_unreachable(bad_args: Any) -> None:
    """AC-136: a missing, corrupt or non-dict argument blob raises PyPIUnreachableError."""
    rows = [_row(bad_args), *_good_rows()]
    with pytest.raises(PyPIUnreachableError, match="unreadable arguments"):
        _check_real_api_lookups(_saver(rows))


def test_ac136_str_form_blob_decodes() -> None:
    """AC-135: a blob the driver returns in a str form is decoded, not rejected."""
    # A json-typed envelope is valid UTF-8 end to end, so a driver may hand it back as text.
    text_blob = b'json{"name": "requests"}'.decode("utf-8")
    assert _SERDE.loads(text_blob.encode("utf-8")) == {"name": "requests"}
    rows = [_row(text_blob), *_good_rows()[1:]]
    _check_real_api_lookups(_saver(rows))
