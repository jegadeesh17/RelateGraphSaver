"""Integration tests for tool call capture in RelateSaver against a real Relate database.

Validates (ADR-0005):
- put_writes() and put() record tool calls from LangChain messages (AC-41, AC-42).
- Later ToolMessage results set result_blob or error on the call (AC-43, AC-44).
- Values without messages create nothing (AC-46).
- Malformed entries are skipped and capture failures never affect the save (AC-47).
- Re-saving the same call never duplicates the node or the INVOKED edge (AC-48).
- A result with no recorded call creates nothing (AC-49).

Also validates ADR-0011: a saver sends only tool call statements it has not already committed,
issues no capture transaction when nothing is new, and retries after a failed capture.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph_checkpoint_relate.saver import RelateSaver


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


@pytest.fixture
def saver() -> Generator[RelateSaver]:
    """A saver backed by a real in-memory Relate database."""
    instance = RelateSaver(db_path=":memory:")
    try:
        yield instance
    finally:
        instance.close()


def _config(thread_id: str = "tc-thread", checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _put(
    saver: RelateSaver,
    thread_id: str,
    checkpoint_id: str,
    channel_values: dict[str, Any],
    step: int = 0,
) -> RunnableConfig:
    checkpoint: Any = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = channel_values
    metadata: Any = {"source": "loop", "step": step, "parents": {}}
    return saver.put(_config(thread_id), checkpoint, metadata, {})


def _ai(*calls: tuple[str, str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": cid, "name": name, "args": args} for cid, name, args in calls],
    )


def _tool_nodes(saver: RelateSaver) -> dict[str, dict[str, Any]]:
    rows = saver.proxy.execute_read(
        "MATCH (tc:ToolCall) "
        "RETURN tc.call_id AS call_id, tc.tool_name AS tool_name, "
        "tc.args_blob AS args_blob, tc.result_blob AS result_blob, tc.error AS error"
    )
    return {row["call_id"]: row for row in rows}


def _edges(saver: RelateSaver) -> list[tuple[str, str]]:
    rows = saver.proxy.execute_read(
        "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) "
        "RETURN c.checkpoint_id AS checkpoint_id, tc.call_id AS call_id"
    )
    return sorted((row["checkpoint_id"], row["call_id"]) for row in rows)


def _count_checkpoints(saver: RelateSaver, checkpoint_id: str) -> int:
    rows = saver.proxy.execute_read(
        "MATCH (c:Checkpoint {checkpoint_id: $cid}) RETURN count(c) AS n", {"cid": checkpoint_id}
    )
    return int(rows[0]["n"])


def test_ac41_put_writes_creates_nodes_linked_from_config_checkpoint(saver: RelateSaver) -> None:
    """AC-41: two tool calls in put_writes() become two linked ToolCall nodes."""
    id_a, id_b = _call_id(), _call_id()
    _put(saver, "t41", "chk-41", {})
    message = _ai((id_a, "search", {"q": "x"}), (id_b, "lookup", {"k": 1}))

    saver.put_writes(_config("t41", "chk-41"), [("messages", [message])], "task-1")

    nodes = _tool_nodes(saver)
    assert set(nodes) == {id_a, id_b}
    assert nodes[id_a]["tool_name"] == "search"
    assert nodes[id_b]["tool_name"] == "lookup"
    assert saver.serde.loads(bytes(nodes[id_a]["args_blob"])) == {"q": "x"}
    assert saver.serde.loads(bytes(nodes[id_b]["args_blob"])) == {"k": 1}
    assert _edges(saver) == sorted([("chk-41", id_a), ("chk-41", id_b)])


def test_ac42_put_links_calls_from_stored_checkpoint(saver: RelateSaver) -> None:
    """AC-42: tool calls in put() channel_values link from the stored checkpoint."""
    id_a, id_b = _call_id(), _call_id()
    message = _ai((id_a, "search", {"q": "y"}), (id_b, "fetch", {}))

    _put(saver, "t42", "chk-42", {"messages": [HumanMessage(content="hi"), message]})

    nodes = _tool_nodes(saver)
    assert set(nodes) == {id_a, id_b}
    assert nodes[id_b]["tool_name"] == "fetch"
    assert _edges(saver) == sorted([("chk-42", id_a), ("chk-42", id_b)])


def test_ac43_tool_message_success_sets_result_blob(saver: RelateSaver) -> None:
    """AC-43: a successful ToolMessage sets result_blob and leaves error null."""
    cid = _call_id()
    _put(saver, "t43", "chk-43", {})
    saver.put_writes(
        _config("t43", "chk-43"), [("messages", [_ai((cid, "search", {"q": "z"}))])], "task-1"
    )

    saver.put_writes(
        _config("t43", "chk-43"),
        [("messages", [ToolMessage(content="result text", tool_call_id=cid)])],
        "task-2",
    )

    node = _tool_nodes(saver)[cid]
    assert node["result_blob"] is not None
    assert saver.serde.loads(bytes(node["result_blob"])) == "result text"
    assert node["error"] is None


def test_ac44_tool_message_error_sets_error_text(saver: RelateSaver) -> None:
    """AC-44: a ToolMessage with status='error' sets error and no result_blob."""
    cid = _call_id()
    _put(saver, "t44", "chk-44", {})
    saver.put_writes(_config("t44", "chk-44"), [("messages", [_ai((cid, "search", {}))])], "task-1")

    saver.put_writes(
        _config("t44", "chk-44"),
        [
            (
                "messages",
                [ToolMessage(content="boom: upstream down", tool_call_id=cid, status="error")],
            )
        ],
        "task-2",
    )

    node = _tool_nodes(saver)[cid]
    assert node["error"] == "boom: upstream down"
    assert node["result_blob"] is None


def test_ac44_first_outcome_wins(saver: RelateSaver) -> None:
    """AC-44: a later outcome for the same call changes nothing."""
    cid = _call_id()
    _put(saver, "t44b", "chk-44b", {})
    cfg = _config("t44b", "chk-44b")
    saver.put_writes(cfg, [("messages", [_ai((cid, "search", {}))])], "task-1")
    saver.put_writes(
        cfg, [("messages", [ToolMessage(content="bad", tool_call_id=cid, status="error")])], "t2"
    )
    saver.put_writes(cfg, [("messages", [ToolMessage(content="ok", tool_call_id=cid)])], "t3")

    node = _tool_nodes(saver)[cid]
    assert node["error"] == "bad"
    assert node["result_blob"] is None


def test_ac46_no_messages_creates_no_tool_calls(saver: RelateSaver) -> None:
    """AC-46: writes and checkpoints without messages create no nodes or edges."""
    _put(saver, "t46", "chk-46", {"counter": 3, "notes": ["a", "b"], "messages": []})
    saver.put_writes(
        _config("t46", "chk-46"),
        [("counter", 4), ("messages", [HumanMessage(content="plain")]), ("blob", None)],
        "task-1",
    )

    assert _tool_nodes(saver) == {}
    assert _edges(saver) == []


def test_ac47_malformed_entries_skipped_and_valid_persisted(saver: RelateSaver) -> None:
    """AC-47: malformed entries are skipped; valid calls and other writes persist."""
    good = _call_id()
    bad_message = AIMessage(content="", tool_calls=[{"id": "", "name": "x", "args": {}}])
    _put(saver, "t47", "chk-47", {})

    saver.put_writes(
        _config("t47", "chk-47"),
        [
            ("messages", [bad_message, {"not": "a message"}, "text", None, _ai((good, "ok", {}))]),
            ("counter", 7),
        ],
        "task-1",
    )

    assert set(_tool_nodes(saver)) == {good}
    assert _edges(saver) == [("chk-47", good)]
    writes = saver.proxy.execute_read(
        "MATCH (w:Write {checkpoint_id: 'chk-47'}) RETURN w.channel AS channel"
    )
    assert sorted(r["channel"] for r in writes) == ["counter", "messages"]


def test_ac47_capture_failure_does_not_fail_put_writes(
    saver: RelateSaver, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-47: a failing capture batch is logged without content; the writes stay stored."""
    secret = "SECRET-ARGUMENT-VALUE"
    _put(saver, "t47w", "chk-47w", {})
    original = saver.proxy.execute_batch
    calls: list[int] = []

    def flaky(statements: list[tuple[str, dict[str, Any]]], timeout: float | None = None) -> Any:
        calls.append(len(statements))
        if len(calls) == 2:
            raise RuntimeError(f"capture exploded {secret}")
        return original(statements, timeout)

    monkeypatch.setattr(saver.proxy, "execute_batch", flaky)

    with caplog.at_level(logging.WARNING):
        saver.put_writes(
            _config("t47w", "chk-47w"),
            [("messages", [_ai((_call_id(), "t", {"a": secret}))]), ("counter", 1)],
            "task-1",
        )

    assert len(calls) == 2
    assert _tool_nodes(saver) == {}
    stored = saver.proxy.execute_read(
        "MATCH (w:Write {checkpoint_id: 'chk-47w'}) RETURN count(w) AS n"
    )
    assert stored[0]["n"] == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("chk-47w" in r.getMessage() and "RuntimeError" in r.getMessage() for r in warnings)
    assert all(secret not in r.getMessage() for r in caplog.records)


def test_ac47_capture_failure_does_not_fail_put(
    saver: RelateSaver, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-47: a failing capture batch leaves put() returning normally with the checkpoint stored."""

    def broken(statements: list[tuple[str, dict[str, Any]]], timeout: float | None = None) -> Any:
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(saver.proxy, "execute_batch", broken)

    with caplog.at_level(logging.WARNING):
        result = _put(saver, "t47p", "chk-47p", {"messages": [_ai((_call_id(), "t", {}))]})

    assert result["configurable"]["checkpoint_id"] == "chk-47p"
    assert _count_checkpoints(saver, "chk-47p") == 1
    assert _tool_nodes(saver) == {}
    assert any("chk-47p" in r.getMessage() for r in caplog.records)


def test_ac47_non_dict_channel_values_do_nothing(saver: RelateSaver) -> None:
    """AC-47: put() with non-dict channel_values stores the checkpoint and captures nothing."""
    checkpoint: Any = empty_checkpoint()
    checkpoint["id"] = "chk-47n"
    checkpoint["channel_values"] = [_ai((_call_id(), "t", {}))]
    metadata: Any = {"source": "loop", "step": 0, "parents": {}}

    saver.put(_config("t47n"), checkpoint, metadata, {})

    assert _count_checkpoints(saver, "chk-47n") == 1
    assert _tool_nodes(saver) == {}


def test_ac48_resaving_same_call_keeps_one_node_and_one_edge(saver: RelateSaver) -> None:
    """AC-48: put_writes() then put() for the same call leaves one node and one edge."""
    cid = _call_id()
    message = _ai((cid, "search", {"q": "dup"}))
    _put(saver, "t48", "chk-48a", {})

    saver.put_writes(_config("t48", "chk-48a"), [("messages", [message])], "task-1")
    _put(saver, "t48", "chk-48b", {"messages": [message]}, step=1)
    _put(saver, "t48", "chk-48c", {"messages": [message]}, step=2)

    assert list(_tool_nodes(saver)) == [cid]
    assert _edges(saver) == [("chk-48a", cid)]


def test_ac48_other_thread_gets_its_own_edge(saver: RelateSaver) -> None:
    """AC-48: the edge is per thread; a second thread reusing the id shares the node."""
    cid = _call_id()
    message = _ai((cid, "search", {}))

    _put(saver, "t48x", "chk-48x", {"messages": [message]})
    _put(saver, "t48y", "chk-48y", {"messages": [message]})

    assert list(_tool_nodes(saver)) == [cid]
    assert _edges(saver) == sorted([("chk-48x", cid), ("chk-48y", cid)])


def test_ac49_orphan_tool_message_creates_nothing(saver: RelateSaver) -> None:
    """AC-49: a ToolMessage matching no recorded call creates no node and no edge."""
    _put(saver, "t49", "chk-49", {"messages": [ToolMessage(content="x", tool_call_id=_call_id())]})
    saver.put_writes(
        _config("t49", "chk-49"),
        [("messages", [ToolMessage(content="e", tool_call_id=_call_id(), status="error")])],
        "task-1",
    )

    assert _tool_nodes(saver) == {}
    assert _edges(saver) == []


def test_tool_call_and_result_in_one_put_land_together(saver: RelateSaver) -> None:
    """AC-42, AC-43: a call and its result in the same channel_values are both recorded."""
    cid = _call_id()

    _put(
        saver,
        "t42b",
        "chk-42b",
        {"messages": [_ai((cid, "search", {})), ToolMessage(content="done", tool_call_id=cid)]},
    )

    node = _tool_nodes(saver)[cid]
    assert saver.serde.loads(bytes(node["result_blob"])) == "done"
    assert _edges(saver) == [("chk-42b", cid)]


def _count_batches(saver: RelateSaver, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap execute_batch and return a list that records the size of every submitted batch."""
    original = saver.proxy.execute_batch
    sizes: list[int] = []

    def counting(statements: list[tuple[str, dict[str, Any]]], timeout: float | None = None) -> Any:
        sizes.append(len(statements))
        return original(statements, timeout)

    monkeypatch.setattr(saver.proxy, "execute_batch", counting)
    return sizes


def test_applied_save_with_nothing_new_makes_only_the_checkpoint_submission(
    saver: RelateSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0011: a put() whose calls and results are all committed issues no capture batch."""
    cid = _call_id()
    history: list[Any] = [_ai((cid, "search", {})), ToolMessage(content="r", tool_call_id=cid)]
    _put(saver, "t11a", "chk-11a", {"messages": list(history)})
    submissions: list[str] = []
    original_write = saver.proxy.execute_write
    original_batch = saver.proxy.execute_batch

    def write(*args: Any, **kwargs: Any) -> Any:
        submissions.append("write")
        return original_write(*args, **kwargs)

    def batch(*args: Any, **kwargs: Any) -> Any:
        submissions.append("batch")
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(saver.proxy, "execute_write", write)
    monkeypatch.setattr(saver.proxy, "execute_batch", batch)

    _put(saver, "t11a", "chk-11b", {"messages": list(history)}, step=1)

    assert submissions == ["write"]
    assert list(_tool_nodes(saver)) == [cid]


def test_applied_only_new_statements_are_sent(
    saver: RelateSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0011: a growing history sends each call once and each result once."""
    first, second = _call_id(), _call_id()
    ai1, ai2 = _ai((first, "t", {})), _ai((second, "t", {}))
    tm1 = ToolMessage(content="r1", tool_call_id=first)
    sizes = _count_batches(saver, monkeypatch)

    _put(saver, "t11b", "chk-1", {"messages": [ai1]})
    _put(saver, "t11b", "chk-2", {"messages": [ai1, tm1]}, step=1)
    _put(saver, "t11b", "chk-3", {"messages": [ai1, tm1, ai2]}, step=2)
    _put(saver, "t11b", "chk-4", {"messages": [ai1, tm1, ai2]}, step=3)

    assert sizes == [1, 1, 1]
    assert _edges(saver) == sorted([("chk-1", first), ("chk-3", second)])
    assert saver.serde.loads(bytes(_tool_nodes(saver)[first]["result_blob"])) == "r1"


def test_applied_failed_capture_is_retried_by_next_save(
    saver: RelateSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0011: a capture failure marks nothing, so the next save sends the same call again."""
    cid = _call_id()
    message = _ai((cid, "t", {}))
    original = saver.proxy.execute_batch
    attempts: list[int] = []

    def flaky(statements: list[tuple[str, dict[str, Any]]], timeout: float | None = None) -> Any:
        attempts.append(len(statements))
        if len(attempts) == 1:
            raise RuntimeError("capture exploded")
        return original(statements, timeout)

    monkeypatch.setattr(saver.proxy, "execute_batch", flaky)

    _put(saver, "t11c", "chk-c1", {"messages": [message]})
    assert _tool_nodes(saver) == {}
    _put(saver, "t11c", "chk-c2", {"messages": [message]}, step=1)

    assert attempts == [1, 1]
    assert list(_tool_nodes(saver)) == [cid]
    assert _edges(saver) == [("chk-c2", cid)]


def test_applied_outcome_before_invocation_is_resent_until_confirmed(
    saver: RelateSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0011: a result seen before its call is sent again until the call is committed."""
    cid = _call_id()
    result = ToolMessage(content="late", tool_call_id=cid)
    _put(saver, "t11d", "chk-d1", {})
    sizes = _count_batches(saver, monkeypatch)

    saver.put_writes(_config("t11d", "chk-d1"), [("messages", [result])], "task-1")
    saver.put_writes(_config("t11d", "chk-d1"), [("messages", [result])], "task-1")
    _put(saver, "t11d", "chk-d2", {"messages": [_ai((cid, "t", {})), result]}, step=1)
    _put(saver, "t11d", "chk-d3", {"messages": [_ai((cid, "t", {})), result]}, step=2)

    # put_writes: write batch + unconfirmed outcome batch, twice; put d2: call + result; d3: nothing.
    assert sizes == [1, 1, 1, 1, 2]
    assert saver.serde.loads(bytes(_tool_nodes(saver)[cid]["result_blob"])) == "late"
    assert _edges(saver) == [("chk-d2", cid)]


def test_applied_put_writes_and_put_share_the_record(
    saver: RelateSaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0011: a call recorded by put_writes() is not resent by a later put()."""
    cid = _call_id()
    message = _ai((cid, "t", {}))
    _put(saver, "t11e", "chk-e1", {})
    saver.put_writes(_config("t11e", "chk-e1"), [("messages", [message])], "task-1")
    sizes = _count_batches(saver, monkeypatch)

    _put(saver, "t11e", "chk-e2", {"messages": [message]}, step=1)

    assert sizes == []
    assert _edges(saver) == [("chk-e1", cid)]


def test_applied_new_saver_starts_empty_and_resends_idempotently() -> None:
    """ADR-0011: a second saver on the same proxy re-sends once and leaves the graph unchanged."""
    first = RelateSaver(db_path=":memory:")
    try:
        second = RelateSaver(proxy=first.proxy)
        cid = _call_id()
        message = _ai((cid, "t", {}))
        _put(first, "t11f", "chk-f1", {"messages": [message]})
        assert len(second._applied_tool_calls) == 0

        _put(second, "t11f", "chk-f2", {"messages": [message]}, step=1)

        assert len(first._applied_tool_calls) == 1
        assert len(second._applied_tool_calls) == 1
        assert list(_tool_nodes(first)) == [cid]
        assert _edges(first) == [("chk-f1", cid)]
    finally:
        first.close()


def test_applied_fork_starts_with_an_empty_record(saver: RelateSaver) -> None:
    """ADR-0011: the saver returned by fork_in_memory does not share the source's record."""
    cid = _call_id()
    _put(saver, "t11g", "chk-g1", {"messages": [_ai((cid, "t", {}))]})
    assert len(saver._applied_tool_calls) == 1

    forked = saver.fork_in_memory("t11g", "chk-g1")
    try:
        assert forked._applied_tool_calls is not saver._applied_tool_calls
        assert len(forked._applied_tool_calls) == 0
    finally:
        forked.close()
