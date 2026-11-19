"""Unit tests for ``diff_runs`` on hand-built snapshots (AC-104, AC-105, AC-106)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_checkpoint_relate.models import RunSnapshot, ToolCallRecord
from langgraph_checkpoint_relate.replay import diff_runs


def _call(
    name: str = "fetch_source",
    ordinal: int = 1,
    args: Any = None,
    status: str = "ok",
    call_id: str = "call_a",
    detail: str | None = "text",
) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=name,
        ordinal=ordinal,
        args={"url": "u"} if args is None else args,
        status=status,
        call_id=call_id,
        detail=detail,
    )


def _snapshot(
    *,
    calls: tuple[ToolCallRecord, ...] = (),
    messages: tuple[Any, ...] = (),
    channels: dict[str, Any] | None = None,
    checkpoint_id: str = "cp-1",
    step: int = 3,
) -> RunSnapshot:
    return RunSnapshot(
        thread_id="t",
        checkpoint_id=checkpoint_id,
        step=step,
        messages=messages,
        channels=channels or {},
        tool_calls=calls,
    )


def test_ac104_different_call_ids_same_names_order_and_outcomes_give_no_change() -> None:
    recorded = _snapshot(
        calls=(_call(call_id="call_1"), _call(name="other", call_id="call_2", status="error"))
    )
    replayed = _snapshot(
        calls=(
            _call(call_id="call_9", detail="different text"),
            _call(name="other", call_id="call_8", status="error", detail="other error"),
        )
    )

    diff = diff_runs(recorded, replayed)

    assert diff.tool_calls == ()
    assert diff.is_empty


def test_ac104_ordinal_is_per_tool_name() -> None:
    recorded = _snapshot(calls=(_call(ordinal=1), _call(ordinal=2, call_id="call_b")))
    replayed = _snapshot(calls=(_call(ordinal=1, call_id="x"), _call(ordinal=2, call_id="y")))

    assert diff_runs(recorded, replayed).is_empty


def test_ac105_added_removed_and_changed_calls_are_reported_in_order() -> None:
    recorded = _snapshot(
        calls=(
            _call(name="a", ordinal=1, status="error"),
            _call(name="b", ordinal=1),
            _call(name="c", ordinal=1, args={"x": 1}),
        )
    )
    replayed = _snapshot(
        calls=(
            _call(name="a", ordinal=1, status="ok"),
            _call(name="c", ordinal=1, args={"x": 2}),
            _call(name="d", ordinal=1),
        )
    )

    diff = diff_runs(recorded, replayed)

    assert [(c.tool_name, c.ordinal, c.kind) for c in diff.tool_calls] == [
        ("a", 1, "changed"),
        ("b", 1, "removed"),
        ("c", 1, "changed"),
        ("d", 1, "added"),
    ]
    changed_status = diff.tool_calls[0]
    assert changed_status.before is not None and changed_status.before.status == "error"
    assert changed_status.after is not None and changed_status.after.status == "ok"
    removed = diff.tool_calls[1]
    assert removed.before is not None and removed.after is None
    changed_args = diff.tool_calls[2]
    assert changed_args.before is not None and changed_args.before.args == {"x": 1}
    assert changed_args.after is not None and changed_args.after.args == {"x": 2}
    added = diff.tool_calls[3]
    assert added.before is None and added.after is not None
    assert not diff.is_empty


def test_ac105_message_and_channel_changes_are_reported() -> None:
    recorded = _snapshot(
        messages=(HumanMessage(content="old"), AIMessage(content="done")),
        channels={"plan": ["a"], "gone": 1},
    )
    replayed = _snapshot(
        messages=(HumanMessage(content="new"),),
        channels={"plan": ["b"], "fresh": 2},
    )

    diff = diff_runs(recorded, replayed)

    assert [(m.channel, m.kind) for m in diff.messages] == [
        ("messages[0]", "changed"),
        ("messages[1]", "removed"),
    ]
    assert diff.messages[0].before == "human: old"
    assert diff.messages[0].after == "human: new"
    assert [(c.channel, c.kind) for c in diff.channels] == [
        ("fresh", "added"),
        ("gone", "removed"),
        ("plan", "changed"),
    ]
    assert diff.channels[2].before == "['a']"
    assert diff.channels[2].after == "['b']"


def test_ac105_extra_replayed_message_is_added() -> None:
    diff = diff_runs(_snapshot(), _snapshot(messages=(AIMessage(content="hi"),)))

    assert [(m.channel, m.kind, m.before, m.after) for m in diff.messages] == [
        ("messages[0]", "added", None, "ai: hi")
    ]


def test_ac106_ids_timestamps_internal_channels_and_late_text_are_ignored() -> None:
    shared = "x" * 120
    recorded = _snapshot(
        messages=(HumanMessage(content=shared + "tail one", id="m-1"),),
        channels={"__start__": 1, "branch:to:agent": 2, "created_at": None},
        checkpoint_id="cp-a",
        step=1,
    )
    replayed = _snapshot(
        messages=(HumanMessage(content=shared + "tail two", id="m-2"),),
        channels={"__start__": 9, "branch:to:tools": 3, "created_at": None},
        checkpoint_id="cp-b",
        step=7,
    )

    diff = diff_runs(recorded, replayed)

    assert diff.is_empty
    assert diff == diff_runs(replayed, recorded)


def test_ac106_a_difference_inside_the_first_120_characters_is_reported() -> None:
    recorded = _snapshot(messages=(HumanMessage(content="a" * 119 + "x"),))
    replayed = _snapshot(messages=(HumanMessage(content="a" * 119 + "y"),))

    assert len(diff_runs(recorded, replayed).messages) == 1


def test_ac106_message_text_over_the_preview_limit_ends_with_three_dots() -> None:
    long_text = "a" * 130
    exact_text = "b" * 120

    diff = diff_runs(
        _snapshot(messages=(HumanMessage(content=long_text), HumanMessage(content=exact_text))),
        _snapshot(messages=(HumanMessage(content="c"), HumanMessage(content="d"))),
    )

    assert diff.messages[0].before == "human: " + "a" * 120 + "..."
    assert diff.messages[1].before == "human: " + exact_text
