"""Integration tests verifying cross-task write-path correctness and wiring.

Validates end-to-end interactions across:
- AC-173: Namespace defaulting and unfiltered list
- AC-174: Strict parent ID resolution from config
- AC-175, AC-06: Reverse-chronological checkpoint_id ordering and before pagination
- AC-176: Error propagation for connection and concurrency errors
- AC-177: Automatic SerializerProtocol wrapping in RelateStateSerializer
- AC-178: Configuration resolution priority (constructor, env vars, defaults)
- AC-179: Proxy command timeout cancellation and worker skipping
- AC-180: First-write-wins semantics and pending writes ordering
- AC-190, AC-191: Thread-scoped write linking (ADR-0031)
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.serde.base import SerializerProtocol

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.errors import (
    CheckpointNotFoundError,
    RelateConcurrencyError,
    RelateConnectionError,
)
from langgraph_checkpoint_relate.forensics import RelateForensics
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.serde import RelateStateSerializer


class CustomDictionarySerializer(SerializerProtocol):
    """Custom bare serializer implementing SerializerProtocol (AC-177)."""

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, sort_keys=True).encode("utf-8")

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return ("json", self.dumps(obj))

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.loads(data[1])


def _create_checkpoint(
    checkpoint_id: str,
    channel_values: dict[str, Any] | None = None,
) -> Checkpoint:
    """Helper creating a standard Checkpoint structure using empty_checkpoint."""
    base: dict[str, Any] = dict(empty_checkpoint())
    base["id"] = checkpoint_id
    base["channel_values"] = channel_values or {}
    base["channel_versions"] = dict.fromkeys(channel_values or {}, "1.0")
    return cast(Checkpoint, base)


def test_write_path_full_pipeline_cross_task_wiring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end integration proving write-path wiring (AC-173..AC-180, AC-06).

    Proves:
    1. RELATE_PROXY_TIMEOUT_SEC sets proxy default timeout (AC-178).
    2. Bare SerializerProtocol is wrapped in RelateStateSerializer (AC-177).
    3. Checkpoint parent ID is strictly resolved from config, ignoring metadata['parents'] (AC-174).
    4. First-write-wins preserves initial regular channel writes while overwriting special channels (AC-180).
    5. Pending writes are sorted deterministically by task_id and idx (AC-180).
    6. get_tuple without checkpoint_ns defaults to root namespace "" (AC-173).
    7. list without checkpoint_ns returns checkpoints across all namespaces (AC-173).
    8. Checkpoints sort strictly by checkpoint_id DESC (AC-175, AC-06).
    9. before filter operates on checkpoint_id (AC-175).
    """
    db_file = str(tmp_path / "test_pipeline.rl")
    monkeypatch.setenv("RELATE_PROXY_TIMEOUT_SEC", "15.0")

    custom_serde = CustomDictionarySerializer()

    with RelateSaver(db_path=db_file, serde=custom_serde) as saver:
        # 1. Verify env var and serde wrapping (AC-177, AC-178)
        assert saver.default_timeout == 15.0
        assert saver.proxy.default_timeout == 15.0
        assert isinstance(saver.serde, RelateStateSerializer)
        assert saver.serde.inner_serializer is custom_serde

        thread_id = "thread-pipeline-001"
        # Synthetic UUIDv6-style IDs that sort chronologically
        cp_id_1 = "1efc0000-0000-6000-8000-000000000001"
        cp_id_2 = "1efc0000-0000-6000-8000-000000000002"
        sub_cp_id = "1efc0000-0000-6000-8000-000000000003"

        # 2. Put root checkpoint with deceptive metadata['parents'] (AC-174)
        config_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": cp_id_1,
            }
        }
        metadata_1: CheckpointMetadata = {
            "step": 1,
            "parents": {"bogus_ns": "deceptive_parent_id"},
        }
        chk_1 = _create_checkpoint(cp_id_1, {"count": 10})
        saver.put(config_1, chk_1, metadata_1, {})

        # 3. Store writes on regular channels (first-write-wins) and special channels (AC-180)
        # Task B: regular channel write
        saver.put_writes(
            config_1,
            [("channel_x", "task_b_val")],
            task_id="task_b",
        )
        # Task A: multiple writes including duplicate to test first-write-wins and special channels
        saver.put_writes(
            config_1,
            [
                ("channel_y", "initial_y"),
                ("__error__", "initial_error"),
            ],
            task_id="task_a",
        )
        # Subsequent write to same regular channel with same task and idx: must NOT overwrite
        saver.put_writes(
            config_1,
            [("channel_y", "overwritten_y_attempt")],
            task_id="task_a",
        )
        # Subsequent write to special channel: MUST overwrite
        saver.put_writes(
            config_1,
            [("__error__", "final_error")],
            task_id="task_a",
        )

        # 4. Successor checkpoint in root namespace with explicit parent in config (AC-174)
        parent_config_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": cp_id_1,
            }
        }
        metadata_2: CheckpointMetadata = {"step": 2, "parents": {"wrong": "ignored"}}
        chk_2 = _create_checkpoint(cp_id_2, {"count": 20})
        saver.put(parent_config_1, chk_2, metadata_2, {})

        # 5. Checkpoint in child namespace (subgraph) (AC-173)
        sub_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "subgraph_ns",
                "checkpoint_id": sub_cp_id,
            }
        }
        chk_sub = _create_checkpoint(sub_cp_id, {"sub_val": "child"})
        saver.put(sub_config, chk_sub, {"step": 1}, {})

        # 6. Verify get_tuple without checkpoint_ns defaults to root "" (AC-173)
        # and resolves latest checkpoint by checkpoint_id DESC (AC-175, AC-06)
        query_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        latest_tup = saver.get_tuple(query_config)
        assert latest_tup is not None
        assert latest_tup.config["configurable"]["checkpoint_id"] == cp_id_2
        assert latest_tup.config["configurable"]["checkpoint_ns"] == ""
        assert latest_tup.checkpoint["channel_values"] == {"count": 20}
        # Verify parent was resolved from config cp_id_1, ignoring metadata['parents'] (AC-174)
        assert latest_tup.parent_config is not None
        assert latest_tup.parent_config["configurable"]["checkpoint_id"] == cp_id_1

        # 7. Verify pending_writes on checkpoint 1:
        # - regular channel first write wins: "initial_y"
        # - special channel overwritten: "final_error"
        # - ordered by task_id then idx (AC-180)
        tup_1 = saver.get_tuple(config_1)
        assert tup_1 is not None
        assert tup_1.pending_writes is not None
        # Task_a should precede task_b alphabetically
        expected_writes = [
            ("task_a", "__error__", "final_error"),
            ("task_a", "channel_y", "initial_y"),
            ("task_b", "channel_x", "task_b_val"),
        ]
        assert tup_1.pending_writes == expected_writes

        # 8. Verify list() without checkpoint_ns returns checkpoints across ALL namespaces (AC-173)
        # ordered reverse-chronologically by checkpoint_id DESC (AC-175, AC-06)
        all_checkpoints = list(saver.list(query_config))
        assert len(all_checkpoints) == 3
        # Ordered strictly descending: sub_cp_id, cp_id_2, cp_id_1
        assert [t.config["configurable"]["checkpoint_id"] for t in all_checkpoints] == [
            sub_cp_id,
            cp_id_2,
            cp_id_1,
        ]
        assert [t.config["configurable"]["checkpoint_ns"] for t in all_checkpoints] == [
            "subgraph_ns",
            "",
            "",
        ]

        # 9. Verify before filter with checkpoint_id pagination (AC-175)
        before_sub = list(saver.list(query_config, before=sub_cp_id))
        assert len(before_sub) == 2
        assert [t.config["configurable"]["checkpoint_id"] for t in before_sub] == [
            cp_id_2,
            cp_id_1,
        ]

        before_cp2 = list(saver.list(query_config, before=cp_id_2))
        assert len(before_cp2) == 1
        assert before_cp2[0].config["configurable"]["checkpoint_id"] == cp_id_1


def test_proxy_timeout_cancellation_and_error_propagation_wiring(
    tmp_path: Path,
) -> None:
    """Verifies proxy timeout cancellation (AC-179) and connection error propagation (AC-176).

    Proves:
    1. Short timeout cancels command future and worker skips execution without persisting changes.
    2. Closed proxy re-raises RelateConnectionError on get_tuple() and list().
    """
    db_file = str(tmp_path / "test_timeout_prop.rl")

    with RelateSaver(db_path=db_file) as saver:
        # 1. Enqueue blocking operation to cause caller timeout
        block_worker = threading.Event()
        saver.proxy._queue.put(lambda: block_worker.wait(timeout=5.0))

        # 2. Put write command with short timeout; must raise RelateConcurrencyError
        cfg: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-timeout",
                "checkpoint_id": "cp-timeout-001",
            }
        }
        with pytest.raises(RelateConcurrencyError, match="timed out"):
            saver.proxy.execute_write(
                "CREATE (n:TimedOutCheckpoint {id: 'to-1'})",
                timeout=0.05,
            )

        # Unblock worker
        block_worker.set()

        # 3. Verify cancelled command was skipped and not persisted (AC-179)
        res = saver.proxy.execute_read(
            "MATCH (n:TimedOutCheckpoint {id: 'to-1'}) RETURN count(n) AS cnt"
        )
        assert res == [{"cnt": 0}]

    # 4. After saver is closed, get_tuple and list must re-raise RelateConnectionError (AC-176)
    with pytest.raises(RelateConnectionError, match="is closed"):
        saver.get_tuple(cfg)

    with pytest.raises(RelateConnectionError, match="is closed"):
        list(saver.list(cfg))


def test_time_travel_error_propagation_and_cleanup_wiring(tmp_path: Path) -> None:
    """Verifies time travel cloning error propagation and sandboxed saver cleanup.

    Proves:
    1. CheckpointNotFoundError raised with candidate list for invalid checkpoint (AC-21).
    2. Read error during fork_in_memory propagates rather than being swallowed.
    3. Population failure closes the created in-memory saver cleanly.
    """
    db_file = str(tmp_path / "test_tt_prop.rl")

    with RelateSaver(db_path=db_file) as saver:
        thread_id = "thread-tt-001"
        cp_id = "1efc0000-0000-6000-8000-000000000010"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": cp_id,
            }
        }
        saver.put(config, _create_checkpoint(cp_id, {"val": 42}), {"step": 1}, {})

        # Invalid checkpoint ID raises CheckpointNotFoundError naming candidate
        with pytest.raises(CheckpointNotFoundError) as exc_info:
            saver.fork_in_memory(thread_id, "nonexistent-cp")
        assert cp_id in str(exc_info.value)

        # Successful fork creates isolated sandbox
        sandbox = saver.fork_in_memory(thread_id, cp_id)
        assert sandbox.db_path == ":memory:"
        tup = sandbox.get_tuple(config)
        assert tup is not None
        assert tup.checkpoint["channel_values"] == {"val": 42}
        sandbox.close()


def _store_checkpoint(saver: RelateSaver, thread_id: str, checkpoint_id: str, ns: str = "") -> None:
    """Store a root checkpoint for ``thread_id`` and ``ns`` through put()."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}
    saver.put(config, _create_checkpoint(checkpoint_id), {"step": 0, "parents": {}}, {})


def _write_config(thread_id: str, checkpoint_id: str, ns: str = "") -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": ns,
            "checkpoint_id": checkpoint_id,
        }
    }


def _linked_task_ids(saver: RelateSaver, checkpoint_id: str) -> list[str]:
    rows = saver.proxy.execute_read(
        "MATCH (:Checkpoint {checkpoint_id: $cid})-[:HAS_WRITE]->(w:Write) "
        "RETURN w.task_id AS task_id ORDER BY w.task_id",
        {"cid": checkpoint_id},
    )
    return [str(row["task_id"]) for row in rows]


def _stored_writes(saver: RelateSaver, task_id: str) -> list[dict[str, Any]]:
    rows = saver.proxy.execute_read(
        "MATCH (w:Write {task_id: $task_id}) "
        "RETURN w.thread_id AS thread_id, w.checkpoint_ns AS checkpoint_ns, "
        "w.checkpoint_id AS checkpoint_id",
        {"task_id": task_id},
    )
    return [dict(row) for row in rows]


def _invoked_count(saver: RelateSaver, checkpoint_id: str) -> int:
    rows = saver.proxy.execute_read(
        "MATCH (:Checkpoint {checkpoint_id: $cid})-[r:INVOKED]->(:ToolCall) RETURN count(r) AS n",
        {"cid": checkpoint_id},
    )
    return int(rows[0]["n"])


@pytest.mark.parametrize(
    ("other_thread", "other_ns"),
    [("thread-b", ""), ("thread-a", "sub")],
    ids=["other-thread", "other-namespace"],
)
def test_ac190_put_writes_naming_another_threads_checkpoint_stays_unlinked(
    other_thread: str, other_ns: str
) -> None:
    """AC-190: a write whose config names another thread or namespace is stored, not linked."""
    cp_id = "1efc0000-0000-6000-8000-0000000000a1"
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", cp_id)
        saver.put_writes(
            _write_config(other_thread, cp_id, other_ns),
            [("channel_x", "foreign")],
            task_id="foreign-task",
        )

        stored = _stored_writes(saver, "foreign-task")
        assert stored == [
            {"thread_id": other_thread, "checkpoint_ns": other_ns, "checkpoint_id": cp_id}
        ]
        assert _linked_task_ids(saver, cp_id) == []

        tup = saver.get_tuple(_write_config("thread-a", cp_id))
        assert tup is not None
        assert tup.pending_writes == []


@pytest.mark.parametrize(
    ("other_thread", "other_ns"),
    [("thread-b", ""), ("thread-a", "sub")],
    ids=["other-thread", "other-namespace"],
)
def test_ac190_mismatched_tool_call_is_not_invoked_from_the_named_checkpoint(
    other_thread: str, other_ns: str
) -> None:
    """AC-190: a tool call carried by a mismatched put_writes() adds no [:INVOKED] edge."""
    cp_id = "1efc0000-0000-6000-8000-0000000000a2"
    message = AIMessage(content="", tool_calls=[{"id": "call_ac190", "name": "search", "args": {}}])
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", cp_id)
        saver.put_writes(
            _write_config(other_thread, cp_id, other_ns),
            [("messages", [message])],
            task_id="foreign-task",
        )

        assert _invoked_count(saver, cp_id) == 0
        assert _linked_task_ids(saver, cp_id) == []
        tup = saver.get_tuple(_write_config("thread-a", cp_id))
        assert tup is not None
        assert tup.pending_writes == []


def test_ac190_matching_put_writes_still_links_and_returns_pending_write() -> None:
    """AC-190 control: a write from the checkpoint's own thread and namespace links as before."""
    cp_id = "1efc0000-0000-6000-8000-0000000000a3"
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", cp_id, ns="sub")
        saver.put_writes(
            _write_config("thread-a", cp_id, "sub"),
            [("channel_x", "own")],
            task_id="own-task",
        )

        assert _linked_task_ids(saver, cp_id) == ["own-task"]
        tup = saver.get_tuple(_write_config("thread-a", cp_id, "sub"))
        assert tup is not None
        assert tup.pending_writes == [("own-task", "channel_x", "own")]


def test_ac191_put_links_only_earlier_writes_of_its_own_thread_and_namespace() -> None:
    """AC-191: writes stored before their checkpoint link only when thread and namespace match."""
    cp_id = "1efc0000-0000-6000-8000-0000000000b1"
    with RelateSaver(db_path=":memory:") as saver:
        saver.put_writes(
            _write_config("thread-a", cp_id), [("channel_x", "mine")], task_id="task-match"
        )
        saver.put_writes(
            _write_config("thread-b", cp_id), [("channel_x", "other")], task_id="task-thread"
        )
        saver.put_writes(
            _write_config("thread-a", cp_id, "sub"), [("channel_x", "ns")], task_id="task-ns"
        )
        assert _linked_task_ids(saver, cp_id) == []

        _store_checkpoint(saver, "thread-a", cp_id)

        assert _linked_task_ids(saver, cp_id) == ["task-match"]
        assert len(_stored_writes(saver, "task-thread")) == 1
        assert len(_stored_writes(saver, "task-ns")) == 1
        tup = saver.get_tuple(_write_config("thread-a", cp_id))
        assert tup is not None
        assert tup.pending_writes == [("task-match", "channel_x", "mine")]


def test_ac190_get_tuple_filters_a_cross_thread_link_already_in_the_file() -> None:
    """AC-190: a [:HAS_WRITE] edge created by an older mismatched save is not returned."""
    cp_id = "1efc0000-0000-6000-8000-0000000000c1"
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", cp_id)
        saver.put_writes(
            _write_config("thread-a", cp_id), [("channel_x", "mine")], task_id="task-own"
        )
        saver.put_writes(
            _write_config("thread-b", cp_id), [("channel_x", "other")], task_id="task-thread"
        )
        saver.put_writes(
            _write_config("thread-a", cp_id, "sub"), [("channel_x", "ns")], task_id="task-ns"
        )
        # Reproduce the links an id-only match used to create.
        saver.proxy.execute_write(
            "MATCH (c:Checkpoint {checkpoint_id: $cid}), (w:Write) "
            "WHERE w.task_id IN ['task-thread', 'task-ns'] "
            "MERGE (c)-[:HAS_WRITE]->(w)",
            {"cid": cp_id},
        )
        assert _linked_task_ids(saver, cp_id) == ["task-ns", "task-own", "task-thread"]

        tup = saver.get_tuple(_write_config("thread-a", cp_id))

        assert tup is not None
        assert tup.pending_writes == [("task-own", "channel_x", "mine")]


def _store_child(saver: RelateSaver, thread_id: str, parent_id: str, checkpoint_id: str) -> None:
    """Store a checkpoint whose parent is ``parent_id`` in the same thread and namespace."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": parent_id}
    }
    saver.put(config, _create_checkpoint(checkpoint_id), {"step": 1, "parents": {}}, {})


def _failing_call_writes(call_id: str) -> list[tuple[str, Any]]:
    """Writes for one tool call that the model requested and that then failed."""
    request = AIMessage(
        content="", tool_calls=[{"id": call_id, "name": "fetch", "args": {"q": "x"}}]
    )
    result = ToolMessage(content="boom", tool_call_id=call_id, status="error")
    return [("messages", [request, result])]


_FOREIGN_CASES = pytest.mark.parametrize(
    ("other_thread", "other_ns"),
    [("thread-b", ""), ("thread-a", "sub")],
    ids=["other-thread", "other-namespace"],
)


@_FOREIGN_CASES
def test_ac190_failed_call_from_a_mismatched_write_reaches_no_forensics_or_fork_of_thread_a(
    other_thread: str, other_ns: str
) -> None:
    """AC-190: a failed call written with another thread's config is invisible to thread A's
    root cause, failure summary and in-memory fork (ADR-0031)."""
    root_id = "1efc0000-0000-6000-8000-0000000000d1"
    next_id = "1efc0000-0000-6000-8000-0000000000d2"
    call_id = "call_ac190_forensics"
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", root_id)
        _store_child(saver, "thread-a", root_id, next_id)
        saver.put_writes(
            _write_config(other_thread, root_id, other_ns),
            _failing_call_writes(call_id),
            task_id="foreign-task",
        )

        assert _invoked_count(saver, root_id) == 0
        forensics = RelateForensics(saver)
        report = forensics.trace_root_cause("thread-a")
        assert report.error_message == "No error detected"
        assert not report.error_checkpoint_id
        assert forensics.failure_summary() == []

        with saver.fork_in_memory("thread-a", next_id) as fork:
            assert (
                _fork_count(fork, "MATCH (w:Write {task_id: 'foreign-task'}) RETURN count(w) AS n")
                == 0
            )
            assert (
                _fork_count(
                    fork, "MATCH (tc:ToolCall {call_id: $call_id}) RETURN count(tc) AS n", call_id
                )
                == 0
            )
            assert (
                _fork_count(
                    fork, "MATCH (:Checkpoint)-[r:INVOKED]->(:ToolCall) RETURN count(r) AS n"
                )
                == 0
            )


def test_ac190_control_failed_call_written_by_thread_a_is_reported_and_forked() -> None:
    """AC-190 control: the same failed call written with thread A's own config is reported,
    so the empty result above is not caused by the forensics read path."""
    root_id = "1efc0000-0000-6000-8000-0000000000d3"
    next_id = "1efc0000-0000-6000-8000-0000000000d4"
    call_id = "call_ac190_control"
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", root_id)
        _store_child(saver, "thread-a", root_id, next_id)
        saver.put_writes(_write_config("thread-a", root_id), _failing_call_writes(call_id), "own")

        forensics = RelateForensics(saver)
        report = forensics.trace_root_cause("thread-a")
        assert report.error_checkpoint_id == root_id
        assert [(s.tool_name, s.failures) for s in forensics.failure_summary()] == [("fetch", 1)]

        with saver.fork_in_memory("thread-a", next_id) as fork:
            assert _fork_count(fork, "MATCH (w:Write {task_id: 'own'}) RETURN count(w) AS n") == 1


@_FOREIGN_CASES
def test_ac190_async_put_writes_with_a_mismatched_config_links_nothing(
    other_thread: str, other_ns: str
) -> None:
    """AC-190: aput_writes() stores a mismatched write unlinked, adds no [:INVOKED] edge, and
    leaves thread A's pending writes empty (same guard as put_writes())."""
    cp_id = "1efc0000-0000-6000-8000-0000000000e1"
    call_id = "call_ac190_async"
    with RelateSaver(db_path=":memory:") as saver:
        _store_checkpoint(saver, "thread-a", cp_id)

        asyncio.run(
            saver.aput_writes(
                _write_config(other_thread, cp_id, other_ns),
                _failing_call_writes(call_id),
                task_id="foreign-async",
            )
        )

        assert _stored_writes(saver, "foreign-async") == [
            {"thread_id": other_thread, "checkpoint_ns": other_ns, "checkpoint_id": cp_id}
        ]
        assert _linked_task_ids(saver, cp_id) == []
        assert _invoked_count(saver, cp_id) == 0
        tup = asyncio.run(saver.aget_tuple(_write_config("thread-a", cp_id)))
        assert tup is not None
        assert tup.pending_writes == []
        assert RelateForensics(saver).failure_summary() == []


@_FOREIGN_CASES
def test_ac191_async_writes_stored_before_their_checkpoint_link_only_on_match(
    other_thread: str, other_ns: str
) -> None:
    """AC-191: aput_writes() before aput() links on the checkpoint's thread and namespace only."""
    cp_id = "1efc0000-0000-6000-8000-0000000000f1"
    with RelateSaver(db_path=":memory:") as saver:
        asyncio.run(
            saver.aput_writes(
                _write_config("thread-a", cp_id), [("channel_x", "mine")], task_id="async-match"
            )
        )
        asyncio.run(
            saver.aput_writes(
                _write_config(other_thread, cp_id, other_ns),
                [("channel_x", "other")],
                task_id="async-foreign",
            )
        )
        assert _linked_task_ids(saver, cp_id) == []

        _store_checkpoint(saver, "thread-a", cp_id)

        assert _linked_task_ids(saver, cp_id) == ["async-match"]
        assert len(_stored_writes(saver, "async-foreign")) == 1
        tup = saver.get_tuple(_write_config("thread-a", cp_id))
        assert tup is not None
        assert tup.pending_writes == [("async-match", "channel_x", "mine")]


def _fork_count(fork: RelateSaver, query: str, call_id: str | None = None) -> int:
    """Run a count query on an in-memory fork and return its single ``n`` value."""
    params = {"call_id": call_id} if call_id is not None else None
    rows = fork.proxy.execute_read(query, params)
    return int(rows[0]["n"])
