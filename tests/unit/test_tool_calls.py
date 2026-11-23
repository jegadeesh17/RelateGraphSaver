"""Unit tests for tool call extraction, statement building and the applied-call record.

Covers AC-46..AC-49 (ADR-0005) and the applied-call record (ADR-0011).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from langgraph_checkpoint_relate.serde import default_serializer
from langgraph_checkpoint_relate.tool_calls import (
    EMPTY_ERROR_TEXT,
    AppliedToolCalls,
    ToolEvents,
    ToolInvocation,
    ToolOutcome,
    build_tool_call_statements,
    extract_tool_events,
)


class _FakeSerde:
    """Serializer that records calls and can fail for chosen objects."""

    def __init__(self, fail_on: Any = None) -> None:
        self.fail_on = fail_on

    def dumps(self, obj: Any) -> bytes:
        if self.fail_on is not None and obj == self.fail_on:
            raise ValueError("cannot serialize")
        return repr(obj).encode()


class _ExplodingAIMessage(AIMessage):
    """AI message whose tool_calls attribute access raises."""

    def __getattribute__(self, name: str) -> Any:
        if name == "tool_calls":
            raise RuntimeError("boom")
        return super().__getattribute__(name)


def _ai(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def _call(call_id: str, name: str = "search", args: Any = None) -> dict[str, Any]:
    return {"id": call_id, "name": name, "args": {"q": "x"} if args is None else args}


def _build(events: ToolEvents, serde: Any = None) -> list[tuple[str, dict[str, Any]]]:
    return build_tool_call_statements(
        events,
        thread_id="t1",
        checkpoint_ns="",
        checkpoint_id="cp1",
        serde=serde or _FakeSerde(),
        created_at="2026-01-01T00:00:00+00:00",
    )


# --- AC-46: no messages ---------------------------------------------------------------


def test_no_messages_yields_empty_events_and_statements_ac_46() -> None:
    values = [{"a": 1}, "text", 3, None, [], (), {"messages": []}]
    events = extract_tool_events(values)
    assert events == ToolEvents((), ())
    assert _build(events) == []


def test_empty_values_iterable_ac_46() -> None:
    assert extract_tool_events([]) == ToolEvents((), ())


def test_non_message_lists_and_nested_values_not_searched_ac_46() -> None:
    nested = {"messages": [_ai(_call("c1"))]}
    events = extract_tool_events([nested, [[_ai(_call("c2"))]], [HumanMessage("hi")]])
    assert events == ToolEvents((), ())


# --- discovery ------------------------------------------------------------------------


def test_single_message_value_is_one_message_list() -> None:
    events = extract_tool_events([_ai(_call("c1", "fetch", {"url": "u"}))])
    assert events.invocations == (ToolInvocation("c1", "fetch", {"url": "u"}),)
    assert events.outcomes == ()


def test_list_and_tuple_values_contribute_messages() -> None:
    events = extract_tool_events(
        [
            [_ai(_call("c1")), ToolMessage(content="ok", tool_call_id="c1")],
            (_ai(_call("c2")),),
        ]
    )
    assert [i.call_id for i in events.invocations] == ["c1", "c2"]
    assert events.outcomes == (ToolOutcome("c1", False, "ok"),)


def test_ai_message_chunk_tool_calls_captured() -> None:
    chunk = AIMessageChunk(content="", tool_calls=[{"id": "c9", "name": "lookup", "args": {}}])
    events = extract_tool_events([chunk])
    assert [(i.call_id, i.tool_name) for i in events.invocations] == [("c9", "lookup")]


def test_invalid_tool_calls_ignored() -> None:
    msg = AIMessage(
        content="",
        invalid_tool_calls=[
            {"id": "bad", "name": "n", "args": "{oops", "error": "e", "type": "invalid_tool_call"}
        ],
    )
    assert extract_tool_events([msg]) == ToolEvents((), ())


def test_missing_args_defaults_to_empty_dict() -> None:
    msg = _ai(_call("c1"))
    del msg.tool_calls[0]["args"]  # type: ignore[misc]
    events = extract_tool_events([msg])
    assert events.invocations == (ToolInvocation("c1", "search", {}),)


def test_tool_message_status_error_flag() -> None:
    err = ToolMessage(content="bad", tool_call_id="c1", status="error")
    ok = ToolMessage(content="fine", tool_call_id="c2")
    events = extract_tool_events([[err, ok]])
    assert events.outcomes == (
        ToolOutcome("c1", True, "bad"),
        ToolOutcome("c2", False, "fine"),
    )


def test_tool_message_with_empty_call_id_skipped() -> None:
    events = extract_tool_events([ToolMessage(content="x", tool_call_id="")])
    assert events.outcomes == ()


# --- AC-47: malformed entries ---------------------------------------------------------


def test_tool_call_without_id_or_name_or_nonstring_skipped_ac_47() -> None:
    msg = _ai(_call("good"))
    # tool_calls are validated TypedDicts; bypass validation with malformed raw entries.
    msg.tool_calls.extend(
        [
            {"name": "no_id", "args": {}},  # type: ignore[typeddict-item]
            {"id": "no_name", "args": {}},  # type: ignore[typeddict-item]
            {"id": "", "name": "empty_id", "args": {}},
            {"id": "e", "name": "", "args": {}},
            {"id": 123, "name": "int_id", "args": {}},  # type: ignore[typeddict-item]
            {"id": "n", "name": None, "args": {}},  # type: ignore[typeddict-item]
        ]
    )
    events = extract_tool_events([msg])
    assert [i.call_id for i in events.invocations] == ["good"]


def test_non_message_objects_in_list_skipped_ac_47() -> None:
    values = [[{"role": "ai"}, "str", None, 5, _ai(_call("c1")), object()]]
    events = extract_tool_events(values)
    assert [i.call_id for i in events.invocations] == ["c1"]


def test_message_whose_attribute_access_raises_is_skipped_ac_47() -> None:
    bad = _ExplodingAIMessage(content="")
    good = _ai(_call("c1"))
    events = extract_tool_events([[bad, good], bad])
    assert [i.call_id for i in events.invocations] == ["c1"]


def test_tool_message_attribute_error_does_not_hide_later_messages_ac_47() -> None:
    class _BadTool(ToolMessage):
        def __getattribute__(self, name: str) -> Any:
            if name == "tool_call_id":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

    bad = _BadTool(content="x", tool_call_id="c0")
    good = ToolMessage(content="ok", tool_call_id="c1")
    events = extract_tool_events([[bad, good]])
    assert events.outcomes == (ToolOutcome("c1", False, "ok"),)


def test_unreadable_value_and_non_iterable_input_never_raise_ac_47() -> None:
    class _BadSeq(list):  # type: ignore[type-arg]
        def __iter__(self) -> Any:
            raise RuntimeError("boom")

    events = extract_tool_events([_BadSeq([_ai(_call("x"))]), [_ai(_call("c1"))]])
    assert [i.call_id for i in events.invocations] == ["c1"]
    assert extract_tool_events(None) == ToolEvents((), ())  # type: ignore[arg-type]


def test_failing_values_iterator_keeps_events_found_so_far_ac_47() -> None:
    def gen() -> Any:
        yield [_ai(_call("c1"))]
        raise RuntimeError("boom")

    events = extract_tool_events(gen())
    assert [i.call_id for i in events.invocations] == ["c1"]


# --- AC-48: duplicates ----------------------------------------------------------------


def test_duplicate_call_id_collapsed_keeping_first_ac_48() -> None:
    first = _ai(_call("c1", "first", {"a": 1}))
    second = _ai(_call("c1", "second", {"a": 2}), _call("c2"))
    out1 = ToolMessage(content="one", tool_call_id="c1")
    out2 = ToolMessage(content="two", tool_call_id="c1", status="error")
    events = extract_tool_events([[first, out1], [second, out2]])
    assert events.invocations == (
        ToolInvocation("c1", "first", {"a": 1}),
        ToolInvocation("c2", "search", {"q": "x"}),
    )
    assert events.outcomes == (ToolOutcome("c1", False, "one"),)
    statements = _build(events)
    assert sum("MERGE (tc:ToolCall" in q for q, _ in statements) == 2


def test_invocation_statement_is_merge_on_call_id_ac_48() -> None:
    events = ToolEvents((ToolInvocation("c1", "search", {"q": 1}),), ())
    query, params = _build(events)[0]
    assert query.startswith("MERGE (tc:ToolCall {call_id: $call_id})\nON CREATE SET ")
    assert "MERGE (c)-[:INVOKED]->(tc)" in query
    assert params == {
        "call_id": "c1",
        "tool_name": "search",
        "args_blob": repr({"q": 1}).encode(),
        "created_at": "2026-01-01T00:00:00+00:00",
        "thread_id": "t1",
        "checkpoint_ns": "",
        "checkpoint_id": "cp1",
    }


# --- AC-49: outcomes cannot create nodes ----------------------------------------------


def test_outcome_statements_use_match_not_merge_ac_49() -> None:
    events = ToolEvents(
        (),
        (ToolOutcome("c1", False, "ok"), ToolOutcome("c2", True, "bad")),
    )
    statements = _build(events)
    assert len(statements) == 2
    for query, _ in statements:
        assert query.startswith("MATCH (tc:ToolCall {call_id: $call_id})")
        assert "MERGE" not in query and "CREATE" not in query
        assert "WHERE tc.result_blob IS NULL AND tc.error IS NULL" in query


def test_outcome_parameters_ac_49() -> None:
    events = ToolEvents((), (ToolOutcome("c1", False, "ok"), ToolOutcome("c2", True, "bad")))
    (ok_q, ok_p), (err_q, err_p) = _build(events)
    assert ok_q.endswith("SET tc.result_blob = $result_blob")
    assert ok_p == {"call_id": "c1", "result_blob": repr("ok").encode()}
    assert err_q.endswith("SET tc.error = $error")
    assert err_p == {"call_id": "c2", "error": "bad"}


def test_invocation_statements_precede_outcome_statements() -> None:
    events = extract_tool_events(
        [
            [
                ToolMessage(content="ok", tool_call_id="c1"),
                _ai(_call("c1")),
                ToolMessage(content="ok", tool_call_id="c2"),
                _ai(_call("c2")),
            ]
        ]
    )
    kinds = ["inv" if q.startswith("MERGE") else "out" for q, _ in _build(events)]
    assert kinds == ["inv", "inv", "out", "out"]


# --- error text -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("boom", "boom"),
        (["a", {"type": "text", "text": "b"}, {"type": "image"}, 5], "a\nb"),
        ([], EMPTY_ERROR_TEXT),
        ("", EMPTY_ERROR_TEXT),
        ([{"type": "image"}], EMPTY_ERROR_TEXT),
        ({"k": "v"}, "{'k': 'v'}"),
    ],
)
def test_error_text_rules(content: Any, expected: str) -> None:
    events = ToolEvents((), (ToolOutcome("c1", True, content),))
    ((_, params),) = _build(events)
    assert params["error"] == expected


# --- serializer failures --------------------------------------------------------------


def test_serializer_failure_skips_only_that_invocation() -> None:
    events = ToolEvents(
        (
            ToolInvocation("c1", "a", "BAD"),
            ToolInvocation("c2", "b", {"ok": 1}),
        ),
        (),
    )
    statements = _build(events, _FakeSerde(fail_on="BAD"))
    assert [p["call_id"] for _, p in statements] == ["c2"]


def test_serializer_failure_skips_only_that_outcome() -> None:
    events = ToolEvents(
        (),
        (ToolOutcome("c1", False, "BAD"), ToolOutcome("c2", False, "fine")),
    )
    statements = _build(events, _FakeSerde(fail_on="BAD"))
    assert [p["call_id"] for _, p in statements] == ["c2"]


def test_error_outcome_does_not_need_serializer() -> None:
    events = ToolEvents((), (ToolOutcome("c1", True, "BAD"),))
    statements = _build(events, _FakeSerde(fail_on="BAD"))
    assert [p["error"] for _, p in statements] == ["BAD"]


def test_real_serializer_blobs_roundtrip() -> None:
    events = ToolEvents(
        (ToolInvocation("c1", "search", {"q": "x"}),),
        (ToolOutcome("c1", False, "result"),),
    )
    inv, out = _build(events, default_serializer)
    assert default_serializer.loads(inv[1]["args_blob"]) == {"q": "x"}
    assert default_serializer.loads(out[1]["result_blob"]) == "result"


def _events(invocation_ids: tuple[str, ...] = (), outcome_ids: tuple[str, ...] = ()) -> ToolEvents:
    return ToolEvents(
        tuple(ToolInvocation(cid, "t", {}) for cid in invocation_ids),
        tuple(ToolOutcome(cid, False, "ok") for cid in outcome_ids),
    )


def _ids(events: ToolEvents) -> tuple[list[str], list[str]]:
    return (
        [inv.call_id for inv in events.invocations],
        [out.call_id for out in events.outcomes],
    )


def test_applied_record_starts_empty_and_plan_keeps_everything_new() -> None:
    """ADR-0011: a fresh record keeps all events in input order and does not modify itself."""
    record = AppliedToolCalls()

    plan = record.plan(_events(("a", "b"), ("a",)), thread_id="t1")

    assert _ids(plan.events) == (["a", "b"], ["a"])
    assert plan.confirmable_outcomes == frozenset({"a"})
    assert len(record) == 0


def test_applied_record_marked_invocations_and_outcomes_are_dropped() -> None:
    """ADR-0011: after mark_applied, the same events yield an empty plan."""
    record = AppliedToolCalls()
    events = _events(("a",), ("a",))
    record.mark_applied(record.plan(events, thread_id="t1"), thread_id="t1")

    plan = record.plan(events, thread_id="t1")

    assert _ids(plan.events) == ([], [])
    assert plan.confirmable_outcomes == frozenset()
    assert len(record) == 1


def test_applied_record_outcome_without_recorded_invocation_is_not_confirmable() -> None:
    """ADR-0011: an outcome racing ahead of its invocation is sent but never marked."""
    record = AppliedToolCalls()
    events = _events((), ("orphan",))

    plan = record.plan(events, thread_id="t1")
    record.mark_applied(plan, thread_id="t1")

    assert _ids(plan.events) == ([], ["orphan"])
    assert plan.confirmable_outcomes == frozenset()
    assert len(record) == 0
    assert _ids(record.plan(events, thread_id="t1").events) == ([], ["orphan"])


def test_applied_record_outcome_confirmable_when_invocation_already_recorded() -> None:
    """ADR-0011: a later outcome for an already-recorded call is confirmable and then dropped."""
    record = AppliedToolCalls()
    record.mark_applied(record.plan(_events(("a",)), thread_id="t1"), thread_id="t1")

    plan = record.plan(_events(("a",), ("a",)), thread_id="t1")

    assert _ids(plan.events) == ([], ["a"])
    assert plan.confirmable_outcomes == frozenset({"a"})
    record.mark_applied(plan, thread_id="t1")
    assert _ids(record.plan(_events(("a",), ("a",)), thread_id="t1").events) == ([], [])


def test_applied_record_unconfirmed_outcome_is_resent_with_its_invocation() -> None:
    """ADR-0011: an unconfirmed outcome stays pending and becomes confirmable with its call."""
    record = AppliedToolCalls()
    record.mark_applied(record.plan(_events((), ("a",)), thread_id="t1"), thread_id="t1")

    plan = record.plan(_events(("a",), ("a",)), thread_id="t1")

    assert _ids(plan.events) == (["a"], ["a"])
    assert plan.confirmable_outcomes == frozenset({"a"})


def test_applied_record_plan_does_not_mark() -> None:
    """ADR-0011: planning twice without mark_applied returns the same plan."""
    record = AppliedToolCalls()
    events = _events(("a",), ("a",))

    first = record.plan(events, thread_id="t1")
    second = record.plan(events, thread_id="t1")

    assert first == second
    assert len(record) == 0


def test_applied_record_is_keyed_by_thread() -> None:
    """ADR-0011: the same call id in another thread is still new (the edge is per thread)."""
    record = AppliedToolCalls()
    record.mark_applied(record.plan(_events(("a",), ("a",)), thread_id="t1"), thread_id="t1")

    plan = record.plan(_events(("a",), ("a",)), thread_id="t2")

    assert _ids(plan.events) == (["a"], ["a"])
    record.mark_applied(plan, thread_id="t2")
    assert len(record) == 2


def test_applied_record_len_counts_distinct_keys_once() -> None:
    """ADR-0011: a call whose invocation and outcome are both marked counts as one key."""
    record = AppliedToolCalls()
    record.mark_applied(record.plan(_events(("a", "b"), ("a",)), thread_id="t1"), thread_id="t1")

    assert len(record) == 2


def test_applied_record_concurrent_marking_is_consistent() -> None:
    """ADR-0011: concurrent plan and mark calls from many threads lose no entries."""
    record = AppliedToolCalls()
    ids = tuple(f"c{i}" for i in range(50))
    errors: list[BaseException] = []

    def worker(offset: int) -> None:
        try:
            for i in range(offset, offset + 50):
                events = _events((ids[i % 50],), (ids[i % 50],))
                record.mark_applied(record.plan(events, thread_id="t1"), thread_id="t1")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(record) == 50
    assert _ids(record.plan(_events(ids, ids), thread_id="t1").events) == ([], [])
