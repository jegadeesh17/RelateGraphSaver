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


def test_ac106_ids_timestamps_and_internal_channels_are_ignored() -> None:
    recorded = _snapshot(
        messages=(HumanMessage(content="x" * 150, id="m-1"),),
        channels={"__start__": 1, "branch:to:agent": 2, "created_at": None},
        checkpoint_id="cp-a",
        step=1,
    )
    replayed = _snapshot(
        messages=(HumanMessage(content="x" * 150, id="m-2"),),
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


def test_ac159_messages_sharing_their_first_120_characters_are_reported_as_changed() -> None:
    shared = "a" * 120

    for longer in (shared + "x", shared + "y" * 50):
        recorded = _snapshot(messages=(HumanMessage(content=shared),))
        replayed = _snapshot(messages=(HumanMessage(content=longer),))

        forward = diff_runs(recorded, replayed)
        backward = diff_runs(replayed, recorded)

        assert [(m.channel, m.kind) for m in forward.messages] == [("messages[0]", "changed")]
        assert forward.messages[0].before == "human: " + shared
        assert forward.messages[0].after == "human: " + shared + "..."
        assert [(m.channel, m.kind) for m in backward.messages] == [("messages[0]", "changed")]


def test_ac159_a_difference_after_the_limit_shows_the_same_truncated_text_on_both_sides() -> None:
    shared = "a" * 120

    diff = diff_runs(
        _snapshot(messages=(HumanMessage(content=shared + "tail one"),)),
        _snapshot(messages=(HumanMessage(content=shared + "tail two"),)),
    )

    assert [(m.channel, m.kind, m.before, m.after) for m in diff.messages] == [
        ("messages[0]", "changed", "human: " + shared + "...", "human: " + shared + "...")
    ]


def test_ac159_equal_content_with_different_tool_call_arguments_is_changed() -> None:
    recorded = _snapshot(
        messages=(
            AIMessage(
                content="looking it up",
                tool_calls=[{"name": "fetch_source", "args": {"url": "a"}, "id": "call_1"}],
            ),
        )
    )
    replayed = _snapshot(
        messages=(
            AIMessage(
                content="looking it up",
                tool_calls=[{"name": "fetch_source", "args": {"url": "b"}, "id": "call_1"}],
            ),
        )
    )

    diff = diff_runs(recorded, replayed)

    assert [(m.channel, m.kind) for m in diff.messages] == [("messages[0]", "changed")]
    assert diff.messages[0].before == "ai: looking it up"
    assert diff.messages[0].after == "ai: looking it up"


def test_ac159_equal_content_with_a_different_tool_call_name_is_changed() -> None:
    recorded = _snapshot(
        messages=(AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "call_1"}]),)
    )
    replayed = _snapshot(
        messages=(AIMessage(content="", tool_calls=[{"name": "b", "args": {}, "id": "call_1"}]),)
    )

    assert [m.kind for m in diff_runs(recorded, replayed).messages] == ["changed"]


def test_ac106_ai_messages_differing_only_in_tool_call_ids_and_message_ids_are_equal() -> None:
    recorded = _snapshot(
        messages=(
            AIMessage(
                content="looking it up",
                id="m-1",
                tool_calls=[{"name": "fetch_source", "args": {"url": "a"}, "id": "call_1"}],
            ),
        )
    )
    replayed = _snapshot(
        messages=(
            AIMessage(
                content="looking it up",
                id="m-2",
                tool_calls=[{"name": "fetch_source", "args": {"url": "a"}, "id": "call_9"}],
            ),
        )
    )

    assert diff_runs(recorded, replayed).is_empty
    assert diff_runs(replayed, recorded).is_empty


def test_ac159_list_content_differing_after_the_limit_is_changed() -> None:
    shared = "a" * 130
    recorded = _snapshot(messages=(AIMessage(content=[{"type": "text", "text": shared + " one"}]),))
    replayed = _snapshot(messages=(AIMessage(content=[{"type": "text", "text": shared + " two"}]),))

    diff = diff_runs(recorded, replayed)

    assert [(m.channel, m.kind) for m in diff.messages] == [("messages[0]", "changed")]
    assert diff.messages[0].before is not None and diff.messages[0].before.endswith("...")
    assert diff.messages[0].after is not None and diff.messages[0].after.endswith("...")


def test_ac159_identical_list_content_is_equal() -> None:
    parts = [{"type": "text", "text": "a" * 200}]

    recorded = _snapshot(messages=(AIMessage(content=list(parts)),))
    replayed = _snapshot(messages=(AIMessage(content=list(parts)),))

    assert diff_runs(recorded, replayed).is_empty


def test_ac106_tool_call_arguments_in_a_different_key_order_are_equal() -> None:
    recorded = _snapshot(
        messages=(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fetch", "args": {"url": "a", "opts": {"x": 1, "y": 2}}, "id": "c1"}
                ],
            ),
        )
    )
    replayed = _snapshot(
        messages=(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fetch", "args": {"opts": {"y": 2, "x": 1}, "url": "a"}, "id": "c2"}
                ],
            ),
        )
    )

    assert diff_runs(recorded, replayed).is_empty
    assert diff_runs(replayed, recorded).is_empty


def test_ac106_tool_call_arguments_equal_by_value_but_with_different_reprs_are_equal() -> None:
    class Handle:
        def __init__(self, value: int) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Handle) and other.value == self.value

        __hash__ = None  # type: ignore[assignment]

    first, second = Handle(1), Handle(1)
    assert repr(first) != repr(second)
    recorded = _snapshot(
        messages=(
            AIMessage(content="", tool_calls=[{"name": "f", "args": {"h": first}, "id": "c1"}]),
        )
    )
    replayed = _snapshot(
        messages=(
            AIMessage(content="", tool_calls=[{"name": "f", "args": {"h": second}, "id": "c2"}]),
        )
    )

    assert diff_runs(recorded, replayed).is_empty


def test_ac106_content_block_ids_are_ignored_for_tool_use_blocks() -> None:
    def blocks(block_id: str) -> list[str | dict[Any, Any]]:
        return [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": block_id, "name": "x", "input": {"q": 1, "r": [1, 2]}},
        ]

    recorded = _snapshot(messages=(AIMessage(content=blocks("toolu_A")),))
    replayed = _snapshot(messages=(AIMessage(content=blocks("toolu_B")),))

    assert diff_runs(recorded, replayed).is_empty
    assert diff_runs(replayed, recorded).is_empty


def test_ac106_content_block_ids_are_ignored_for_text_blocks() -> None:
    recorded = _snapshot(
        messages=(AIMessage(content=[{"type": "text", "text": "hi", "id": "lc_111"}]),)
    )
    replayed = _snapshot(
        messages=(AIMessage(content=[{"type": "text", "text": "hi", "id": "lc_222"}]),)
    )

    assert diff_runs(recorded, replayed).is_empty


def test_ac159_list_content_with_different_tool_use_input_is_changed_despite_same_ids() -> None:
    def blocks(query: str) -> list[str | dict[Any, Any]]:
        return [{"type": "tool_use", "id": "toolu_A", "name": "x", "input": {"q": query}}]

    diff = diff_runs(
        _snapshot(messages=(AIMessage(content=blocks("one")),)),
        _snapshot(messages=(AIMessage(content=blocks("two")),)),
    )

    assert [(m.channel, m.kind) for m in diff.messages] == [("messages[0]", "changed")]


def test_ac159_list_content_with_different_text_is_changed_despite_different_block_ids() -> None:
    diff = diff_runs(
        _snapshot(messages=(AIMessage(content=[{"type": "text", "text": "a", "id": "lc_1"}]),)),
        _snapshot(messages=(AIMessage(content=[{"type": "text", "text": "b", "id": "lc_2"}]),)),
    )

    assert [(m.channel, m.kind) for m in diff.messages] == [("messages[0]", "changed")]


def test_ac159_a_tool_call_argument_named_id_still_counts_as_a_change() -> None:
    def message(value: int) -> AIMessage:
        return AIMessage(
            content="", tool_calls=[{"name": "get", "args": {"id": value}, "id": "c1"}]
        )

    diff = diff_runs(_snapshot(messages=(message(1),)), _snapshot(messages=(message(2),)))

    assert [(m.channel, m.kind) for m in diff.messages] == [("messages[0]", "changed")]


def test_ac106_the_display_marker_appears_only_on_values_longer_than_the_limit() -> None:
    shared = "a" * 120

    diff = diff_runs(
        _snapshot(messages=(HumanMessage(content=shared), HumanMessage(content="one"))),
        _snapshot(messages=(HumanMessage(content=shared), HumanMessage(content="two"))),
    )

    assert [(m.channel, m.before, m.after) for m in diff.messages] == [
        ("messages[1]", "human: one", "human: two")
    ]


def test_ac106_auxiliary_channel_list_of_messages_differing_only_in_id_is_empty_diff() -> None:
    # AC-106, ADR-0024: list of messages in auxiliary channel normalized per AC-106.
    recorded = _snapshot(channels={"scratchpad": [AIMessage(content="draft", id="msg-1")]})
    replayed = _snapshot(channels={"scratchpad": [AIMessage(content="draft", id="msg-2")]})

    diff = diff_runs(recorded, replayed)

    assert diff.is_empty


def test_ac106_dict_channel_with_different_key_order_gives_empty_diff() -> None:
    # AC-106, ADR-0024: dict key order is ignored under ==.
    recorded = _snapshot(channels={"config": {"first": 1, "second": 2}})
    replayed = _snapshot(channels={"config": {"second": 2, "first": 1}})

    diff = diff_runs(recorded, replayed)

    assert diff.is_empty


def test_ac106_changed_value_in_non_message_channel_reports_repr_before_and_after() -> None:
    # AC-106, ADR-0024: changed value in a non-message channel is reported with repr(value).
    recorded = _snapshot(channels={"counter": 10, "meta": {"status": "pending"}})
    replayed = _snapshot(channels={"counter": 20, "meta": {"status": "done"}})

    diff = diff_runs(recorded, replayed)

    assert [(c.channel, c.kind, c.before, c.after) for c in diff.channels] == [
        ("counter", "changed", "10", "20"),
        ("meta", "changed", "{'status': 'pending'}", "{'status': 'done'}"),
    ]


def test_ac106_plain_dict_id_keys_in_channel_are_not_stripped() -> None:
    # ADR-0024: Plain-dict id keys outside messages are not stripped.
    recorded = _snapshot(channels={"entity": {"id": "ent-1", "name": "sample"}})
    replayed = _snapshot(channels={"entity": {"id": "ent-2", "name": "sample"}})

    diff = diff_runs(recorded, replayed)

    assert [(c.channel, c.kind, c.before, c.after) for c in diff.channels] == [
        (
            "entity",
            "changed",
            "{'id': 'ent-1', 'name': 'sample'}",
            "{'id': 'ent-2', 'name': 'sample'}",
        ),
    ]
