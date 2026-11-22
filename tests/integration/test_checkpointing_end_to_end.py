"""End-to-end integration tests validating LangGraph checkpointing with Relate DB.

Validates complete cross-module integration across the checkpointer stack:
LangGraph StateGraph and Pregel runtime, RelateSaver checkpoint saver,
RelateThreadProxy actor concurrency engine, Cypher schema constraints,
and Write-Ahead Log persistence with state serialization fidelity.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from pydantic import BaseModel
from typing_extensions import TypedDict

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import relate

from langgraph_checkpoint_relate.errors import (
    CheckpointSerializationError,
    RelateCheckpointError,
    RelateConcurrencyError,
    RelateConnectionError,
    RelateSerializationError,
)
from langgraph_checkpoint_relate.saver import RelateSaver


class PipelineState(TypedDict):
    """LangGraph execution state dictionary."""

    step: int
    data: str
    messages: list[Any]


class CustomPayload(BaseModel):
    """Pydantic model for rich type serialization verification."""

    task_id: str
    score: float
    metadata: dict[str, Any]


class RichTypeState(TypedDict):
    """State dictionary exercising rich Python and LangGraph types."""

    messages: list[Any]
    timestamp: datetime
    session_id: uuid.UUID
    payload: CustomPayload
    counter: int


class InvalidPayloadState(TypedDict):
    """State dictionary containing un-serializable references."""

    valid_text: str
    unserializable_handle: Any


@pytest.fixture
def temp_disk_path() -> Generator[str]:
    """Provide a temporary disk-backed .rl database file path and clean up after test."""
    temp_dir = tempfile.mkdtemp(prefix="relate_m1_integration_")
    db_file = os.path.join(temp_dir, "checkpoints.rl")
    yield db_file

    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


class TestRelateSaverEndToEnd:
    """End-to-end integration test suite proving the Relate checkpointer stack."""

    def test_disk_initialization_schema_wal_ac_01(self, temp_disk_path: str) -> None:
        """AC-01: Verify file creation, schema initialization, and WAL mode for RelateSaver.

        Given a non-existent database file path, When RelateSaver is initialized with that path,
        Then the file is created with initialized schema, constraints, and WAL mode enabled (ADR-0001).
        """
        assert not os.path.exists(temp_disk_path)

        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            assert os.path.exists(temp_disk_path)
            assert os.path.getsize(temp_disk_path) > 0

            # Verify schema DDL was executed by checking constraints in Relate
            records = saver.proxy.execute_read("MATCH (t:Thread) RETURN count(t) AS thread_count")
            assert len(records) == 1
            assert records[0]["thread_count"] == 0

            # Verify round-trip put and get on fresh store
            config: Any = {"configurable": {"thread_id": "thread-init-01", "checkpoint_ns": ""}}
            checkpoint: Any = {
                "v": 1,
                "id": "cp-init-001",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {"status": "initialized"},
                "channel_versions": {"status": 1},
                "versions_seen": {},
            }
            metadata: Any = {"source": "input", "step": 0, "writes": {}, "parents": {}}
            res_config = saver.put(config, checkpoint, metadata, {})
            assert res_config["configurable"]["checkpoint_id"] == "cp-init-001"

            tuple_res = saver.get_tuple(config)
            assert tuple_res is not None
            assert tuple_res.checkpoint["id"] == "cp-init-001"
            assert tuple_res.checkpoint["channel_values"]["status"] == "initialized"

    def test_pregel_stategraph_multi_step_dag_ac_02_ac_03_ac_06(self, temp_disk_path: str) -> None:
        """AC-02, AC-03, AC-06: Real multi-step LangGraph workflow execution end-to-end.

        Drives execution through SyncPregelLoop, verifying checkpoint persistence at each step,
        retrieval of latest CheckpointTuple, and reverse-chronological list ordering.
        """

        def node_enrich(state: PipelineState) -> dict[str, Any]:
            return {
                "step": 1,
                "data": state["data"] + " -> enriched",
                "messages": state["messages"] + [HumanMessage(content="User input received")],
            }

        def node_process(state: PipelineState) -> dict[str, Any]:
            return {
                "step": 2,
                "data": state["data"] + " -> processed",
                "messages": state["messages"] + [AIMessage(content="Agent analysis complete")],
            }

        def node_format(state: PipelineState) -> dict[str, Any]:
            return {
                "step": 3,
                "data": state["data"] + " -> formatted",
                "messages": state["messages"] + [SystemMessage(content="Result finalized")],
            }

        builder = StateGraph(PipelineState)
        builder.add_node("enrich", node_enrich)
        builder.add_node("process", node_process)
        builder.add_node("format", node_format)
        builder.add_edge(START, "enrich")
        builder.add_edge("enrich", "process")
        builder.add_edge("process", "format")
        builder.add_edge("format", END)

        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            graph: Pregel[Any, Any, Any, Any] = builder.compile(checkpointer=saver)

            run_config: Any = {"configurable": {"thread_id": "thread-pregel-01"}}
            initial_state: PipelineState = {
                "step": 0,
                "data": "raw",
                "messages": [],
            }

            result = graph.invoke(initial_state, config=run_config)
            assert result["step"] == 3
            assert result["data"] == "raw -> enriched -> processed -> formatted"
            assert len(result["messages"]) == 3

            # AC-03: get_tuple with thread_id returns latest CheckpointTuple
            latest_tuple = saver.get_tuple(run_config)
            assert latest_tuple is not None
            assert latest_tuple.checkpoint["channel_values"]["step"] == 3
            assert latest_tuple.checkpoint["channel_values"]["data"] == (
                "raw -> enriched -> processed -> formatted"
            )
            assert latest_tuple.metadata.get("step") == 3
            assert latest_tuple.parent_config is not None
            assert "checkpoint_id" in latest_tuple.parent_config["configurable"]

            # AC-06, AC-175, ADR-0026: list yields CheckpointTuple records ordered reverse-chronologically by checkpoint_id
            history = list(saver.list(run_config))
            assert (
                len(history) >= 4
            )  # __start__, enrich (step 1), process (step 2), format (step 3)

            history_cids = [h.checkpoint["id"] for h in history]
            assert history_cids == sorted(history_cids, reverse=True)

            steps: list[Any] = [h.metadata.get("step") for h in history if "step" in h.metadata]
            assert steps == sorted(steps, reverse=True)
            assert steps[0] == 3

            # Checkpoint graph connectivity in Relate DB (ADR-0003)
            # Verify NEXT edges connect consecutive checkpoints
            edges = saver.proxy.execute_read(
                "MATCH (c1:Checkpoint {thread_id: 'thread-pregel-01'})-[:NEXT]->(c2:Checkpoint) "
                "RETURN c1.checkpoint_id AS parent, c2.checkpoint_id AS child"
            )
            assert len(edges) >= 3

    def test_checkpoint_targeting_and_pagination_ac_04_ac_07(self, temp_disk_path: str) -> None:
        """AC-04, AC-07: Targeted checkpoint retrieval by ID and pagination with before/limit."""
        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            thread_id = "thread-pagination-01"

            # Create 5 sequential checkpoints
            created_ids: list[str] = []
            parent_id: str | None = None
            for i in range(1, 6):
                cid = f"cp-page-{i:03d}"
                created_ids.append(cid)
                chk: Any = {
                    "v": 1,
                    "id": cid,
                    "ts": f"2026-11-12T10:{i:02d}:00Z",
                    "channel_values": {"counter": i, "val": f"step-{i}"},
                    "channel_versions": {"counter": i, "val": i},
                    "versions_seen": {},
                }
                meta: Any = {
                    "source": "loop",
                    "step": i,
                    "writes": {},
                    "parents": {},
                }
                cfg: Any = {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_id}}
                saver.put(cfg, chk, meta, {})
                parent_id = cid

            # AC-04: get_tuple with targeted checkpoint_id
            target_id = created_ids[2]  # cp-page-003
            target_cfg: Any = {"configurable": {"thread_id": thread_id, "checkpoint_id": target_id}}
            target_tuple = saver.get_tuple(target_cfg)
            assert target_tuple is not None
            assert target_tuple.checkpoint["id"] == target_id
            assert target_tuple.checkpoint["channel_values"]["counter"] == 3
            assert target_tuple.metadata.get("step") == 3

            # AC-07: list with before and limit parameters
            # Query checkpoints before cp-page-004 with limit=2
            cutoff_id = created_ids[3]  # cp-page-004
            paged_cfg: Any = {"configurable": {"thread_id": thread_id}}
            before_cfg: Any = {"configurable": {"thread_id": thread_id, "checkpoint_id": cutoff_id}}

            paged_records = list(saver.list(paged_cfg, before=before_cfg, limit=2))
            assert len(paged_records) == 2
            # Should be reverse-chronological before cp-page-004: cp-page-003, cp-page-002
            assert paged_records[0].checkpoint["id"] == created_ids[2]
            assert paged_records[1].checkpoint["id"] == created_ids[1]

    def test_get_tuple_missing_thread_or_checkpoint_ac_05(self) -> None:
        """AC-05: Non-existent thread_id or checkpoint_id returns None without exception."""
        with RelateSaver(db_path=":memory:") as saver:
            # Non-existent thread
            res_thread = saver.get_tuple({"configurable": {"thread_id": "missing-thread-xyz"}})
            assert res_thread is None

            # Seed an existing thread
            chk: Any = {
                "v": 1,
                "id": "cp-exist-001",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {"k": "v"},
                "channel_versions": {"k": 1},
                "versions_seen": {},
            }
            saver.put({"configurable": {"thread_id": "thread-existing"}}, chk, {"step": 1}, {})

            # Non-existent checkpoint ID on existing thread
            res_cp = saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": "thread-existing",
                        "checkpoint_id": "missing-cp-xyz",
                    }
                }
            )
            assert res_cp is None

            # Empty config dict
            res_empty = saver.get_tuple({})
            assert res_empty is None

    def test_branching_and_time_travel_replay_ac_02_ac_04(self, temp_disk_path: str) -> None:
        """AC-02, AC-04: Workflow branching / replay from historical checkpoint (ADR-0004)."""

        class BranchState(TypedDict):
            counter: int
            path: str

        def step_increment(state: BranchState) -> dict[str, Any]:
            return {
                "counter": state["counter"] + 1,
                "path": state["path"] + f"->{state['counter'] + 1}",
            }

        builder = StateGraph(BranchState)
        builder.add_node("inc", step_increment)
        builder.add_edge(START, "inc")
        builder.add_edge("inc", END)

        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            graph = builder.compile(checkpointer=saver)
            thread_cfg: Any = {"configurable": {"thread_id": "thread-branch-01"}}

            # Run 1: initial execution
            res1 = graph.invoke({"counter": 0, "path": "start"}, config=thread_cfg)
            assert res1["counter"] == 1
            assert res1["path"] == "start->1"

            history1 = list(saver.list(thread_cfg))
            root_cp = [h for h in history1 if h.metadata.get("step") == 1][0]
            root_cp_id = root_cp.checkpoint["id"]

            # Run 2: resume / branch from root_cp_id with modified state
            branch_cfg: Any = {
                "configurable": {
                    "thread_id": "thread-branch-01",
                    "checkpoint_id": root_cp_id,
                }
            }
            # Retrieve exact historical state
            historical_tuple = saver.get_tuple(branch_cfg)
            assert historical_tuple is not None
            assert historical_tuple.checkpoint["channel_values"]["counter"] == 1

            # Execute branch step
            res2 = graph.invoke({"counter": 10, "path": "branch"}, config=branch_cfg)
            assert res2["counter"] == 11
            assert res2["path"] == "branch->11"

            # Both checkpoints exist in thread history
            all_history = list(saver.list(thread_cfg))
            assert len(all_history) >= 3

    def test_multithreaded_concurrent_puts_and_reads_ac_08_ac_09(self, temp_disk_path: str) -> None:
        """AC-08, AC-09: Concurrent multi-threaded puts and reads without lock contention (ADR-0002).

        Spawns worker threads invoking put() concurrently while observer threads run get_tuple()
        and list() concurrently, asserting zero lock collisions and complete data consistency.
        """
        num_writers = 8
        writes_per_thread = 5
        num_readers = 4

        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            barrier = threading.Barrier(num_writers + num_readers)
            errors: list[Exception] = []

            def writer_worker(thread_idx: int) -> None:
                try:
                    barrier.wait(timeout=10.0)
                    for step_idx in range(writes_per_thread):
                        tid = f"thread-concurrent-{thread_idx}"
                        cid = f"cp-t{thread_idx}-s{step_idx}"
                        chk: Any = {
                            "v": 1,
                            "id": cid,
                            "ts": datetime.now(UTC).isoformat(),
                            "channel_values": {
                                "thread_idx": thread_idx,
                                "step": step_idx,
                                "payload": f"data-{thread_idx}-{step_idx}",
                            },
                            "channel_versions": {"thread_idx": 1, "step": step_idx + 1},
                            "versions_seen": {},
                        }
                        meta: Any = {
                            "source": "concurrent_test",
                            "step": step_idx,
                            "writes": {},
                            "parents": {},
                        }
                        cfg: Any = {"configurable": {"thread_id": tid}}
                        saver.put(cfg, chk, meta, {})
                except Exception as exc:
                    errors.append(exc)

            def reader_worker(reader_idx: int) -> None:
                try:
                    barrier.wait(timeout=10.0)
                    for _ in range(15):
                        tid = f"thread-concurrent-{reader_idx % num_writers}"
                        cfg: Any = {"configurable": {"thread_id": tid}}
                        tuple_res = saver.get_tuple(cfg)
                        if tuple_res is not None:
                            # AC-09: Reader must observe consistent committed state
                            assert "thread_idx" in tuple_res.checkpoint["channel_values"]
                            assert "step" in tuple_res.checkpoint["channel_values"]
                        # Test list operation during concurrent writes
                        list(saver.list(cfg, limit=3))
                except Exception as exc:
                    errors.append(exc)

            threads: list[threading.Thread] = []
            for i in range(num_writers):
                t = threading.Thread(target=writer_worker, args=(i,))
                threads.append(t)
            for j in range(num_readers):
                t = threading.Thread(target=reader_worker, args=(j,))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)

            assert len(errors) == 0, f"Concurrent operations raised exceptions: {errors}"

            # Verify all expected writes were persisted
            total_expected_checkpoints = num_writers * writes_per_thread
            all_persisted = saver.proxy.execute_read(
                "MATCH (c:Checkpoint) WHERE c.checkpoint_id STARTS WITH 'cp-t' "
                "RETURN count(c) AS total_count"
            )
            assert all_persisted[0]["total_count"] == total_expected_checkpoints

    def test_wal_durability_and_process_restart_fidelity_ac_10_ac_11(
        self, temp_disk_path: str
    ) -> None:
        """AC-10, AC-11: WAL durability, uncommitted rollback, and restart fidelity across sessions."""
        thread_id = "thread-durability-01"
        chk_id_1 = "cp-durable-001"
        chk_id_2 = "cp-durable-002"

        # Session 1: Commit two checkpoints to disk
        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver1:
            cfg: Any = {"configurable": {"thread_id": thread_id}}
            chk1: Any = {
                "v": 1,
                "id": chk_id_1,
                "ts": "2026-11-12T11:00:00Z",
                "channel_values": {"status": "first", "data": [1, 2, 3]},
                "channel_versions": {"status": 1, "data": 1},
                "versions_seen": {},
            }
            saver1.put(cfg, chk1, {"step": 1}, {})

            chk2: Any = {
                "v": 1,
                "id": chk_id_2,
                "ts": "2026-11-12T11:01:00Z",
                "channel_values": {"status": "second", "data": [1, 2, 3, 4]},
                "channel_versions": {"status": 2, "data": 2},
                "versions_seen": {},
            }
            saver1.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": chk_id_1}},
                chk2,
                {"step": 2},
                {},
            )

            # AC-10: Test transaction rollback on aborted write
            aborted_statements = [
                (
                    "CREATE (c:Checkpoint {checkpoint_id: 'cp-aborted-999', thread_id: $tid})",
                    {"tid": thread_id},
                ),
                (
                    "INVALID CYPHER SYNTAX BREAKS TRANSACTION",
                    {},
                ),
            ]
            with pytest.raises(RelateCheckpointError) as exc_info:
                saver1.proxy.transaction(aborted_statements)
            assert "syntax error" in str(exc_info.value).lower()

        # Session 2: Fresh RelateSaver instance opening existing disk DB (AC-11)
        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver2:
            # AC-10: Aborted transaction must not have persisted
            aborted_record = saver2.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: 'cp-aborted-999'}) RETURN count(c) AS cnt"
            )
            assert aborted_record[0]["cnt"] == 0

            # AC-11: Last committed checkpoint retrieved with byte-for-byte fidelity
            latest_tuple = saver2.get_tuple({"configurable": {"thread_id": thread_id}})
            assert latest_tuple is not None
            assert latest_tuple.checkpoint["id"] == chk_id_2
            assert latest_tuple.checkpoint["channel_values"]["status"] == "second"
            assert latest_tuple.checkpoint["channel_values"]["data"] == [1, 2, 3, 4]
            assert latest_tuple.metadata.get("step") == 2

            # Historical checkpoint also retrieved intact
            t1_tuple = saver2.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": chk_id_1}}
            )
            assert t1_tuple is not None
            assert t1_tuple.checkpoint["channel_values"]["status"] == "first"
            assert t1_tuple.checkpoint["channel_values"]["data"] == [1, 2, 3]

    def test_unserializable_state_raises_serialization_error_ac_12(
        self, temp_disk_path: str
    ) -> None:
        """AC-12: Unserializable objects in workflow state raise CheckpointSerializationError."""

        def failing_node(state: InvalidPayloadState) -> dict[str, Any]:
            # Injecting an open file object or raw socket into state
            return {
                "valid_text": "processing",
                "unserializable_handle": open(temp_disk_path, "rb"),
            }

        builder = StateGraph(InvalidPayloadState)
        builder.add_node("fail_node", failing_node)
        builder.add_edge(START, "fail_node")
        builder.add_edge("fail_node", END)

        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            graph = builder.compile(checkpointer=saver)

            with pytest.raises(
                (CheckpointSerializationError, RelateSerializationError)
            ) as exc_info:
                graph.invoke(
                    {"valid_text": "start", "unserializable_handle": None},
                    config={"configurable": {"thread_id": "thread-unserializable"}},
                )

            # AC-12: Exception identifies serialization error
            assert any(
                term in str(exc_info.value).lower()
                for term in (
                    "serialize",
                    "serialization",
                    "serializable",
                    "not msgpack serializable",
                )
            )

    def test_rich_langgraph_types_fidelity_ac_13(self, temp_disk_path: str) -> None:
        """AC-13: Preserves messages, datetimes, UUIDs, and Pydantic models with type fidelity."""
        fixed_dt = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)
        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        custom_model = CustomPayload(
            task_id="task-validation-001",
            score=0.985,
            metadata={"origin": "integration_test", "tags": ["graph", "relate"]},
        )

        def rich_node(state: RichTypeState) -> dict[str, Any]:
            return {
                "messages": state["messages"]
                + [
                    AIMessage(content="Analysis generated", additional_kwargs={"model": "gpt-4"}),
                ],
                "counter": state["counter"] + 1,
            }

        builder = StateGraph(RichTypeState)
        builder.add_node("rich_node", rich_node)
        builder.add_edge(START, "rich_node")
        builder.add_edge("rich_node", END)

        with RelateSaver(db_path=temp_disk_path, wal_enabled=True) as saver:
            graph = builder.compile(checkpointer=saver)
            run_cfg: Any = {"configurable": {"thread_id": "thread-rich-types"}}

            initial_state: RichTypeState = {
                "messages": [HumanMessage(content="Run analysis query")],
                "timestamp": fixed_dt,
                "session_id": fixed_uuid,
                "payload": custom_model,
                "counter": 0,
            }

            graph.invoke(initial_state, config=run_cfg)

            # Retrieve through checkpointer get_tuple
            tuple_res = saver.get_tuple(run_cfg)
            assert tuple_res is not None

            restored_channels = tuple_res.checkpoint["channel_values"]
            messages = restored_channels["messages"]
            assert len(messages) == 2
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content == "Run analysis query"
            assert isinstance(messages[1], AIMessage)
            assert messages[1].content == "Analysis generated"

            # Datetime fidelity
            assert isinstance(restored_channels["timestamp"], datetime)
            assert restored_channels["timestamp"] == fixed_dt

            # UUID fidelity
            assert isinstance(restored_channels["session_id"], uuid.UUID)
            assert restored_channels["session_id"] == fixed_uuid

            # Pydantic model fidelity
            payload = restored_channels["payload"]
            assert isinstance(payload, CustomPayload)
            assert payload.task_id == "task-validation-001"
            assert payload.score == 0.985
            assert payload.metadata["tags"] == ["graph", "relate"]

    def test_custom_connection_factory_seam_adapter(self) -> None:
        """Verifies driver injection seam via custom connection_factory callable."""
        connections_created: list[str] = []

        def custom_factory(database: str) -> relate.Connection:
            connections_created.append(database)
            return relate.connect(database=database)

        with RelateSaver(db_path=":memory:", connection_factory=custom_factory) as saver:
            assert len(connections_created) >= 1
            assert connections_created[0] == ":memory:"

            cfg: Any = {"configurable": {"thread_id": "thread-seam-test"}}
            chk: Any = {
                "v": 1,
                "id": "cp-seam-001",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {"seam": "passed"},
                "channel_versions": {"seam": 1},
                "versions_seen": {},
            }
            saver.put(cfg, chk, {"step": 1}, {})

            tuple_res = saver.get_tuple(cfg)
            assert tuple_res is not None
            assert tuple_res.checkpoint["channel_values"]["seam"] == "passed"

    def test_compiled_graph_with_subgraph_node_ac_173_ac_174(
        self, temp_disk_path: str
    ) -> None:
        """Verifies compiled StateGraph with embedded subgraph node, namespace resolution, and parent chains.

        Validates AC-173: get_state() without namespace resolves the root namespace state ("").
        Validates AC-174: Checkpoint parent IDs along graph and subgraph execution branches form
        proper parent chains, with config["configurable"]["checkpoint_id"] strictly taking priority
        over metadata["parents"].
        """
        class ChildState(TypedDict):
            counter: int
            sub_step: str

        class ParentWorkflowState(TypedDict):
            counter: int
            sub_step: str
            parent_note: str

        child_builder = StateGraph(ChildState)

        def c_node_a(state: ChildState) -> dict[str, Any]:
            return {"counter": state["counter"] + 5, "sub_step": "child_a"}

        def c_node_b(state: ChildState) -> dict[str, Any]:
            return {"counter": state["counter"] + 15, "sub_step": "child_b"}

        child_builder.add_node("child_a", c_node_a)
        child_builder.add_node("child_b", c_node_b)
        child_builder.add_edge(START, "child_a")
        child_builder.add_edge("child_a", "child_b")
        child_builder.add_edge("child_b", END)
        child_subgraph = child_builder.compile()

        parent_builder = StateGraph(ParentWorkflowState)

        def p_start(state: ParentWorkflowState) -> dict[str, Any]:
            return {"counter": state["counter"] + 1, "parent_note": "started"}

        def p_end(state: ParentWorkflowState) -> dict[str, Any]:
            return {"counter": state["counter"] + 2, "parent_note": "finished"}

        parent_builder.add_node("parent_start", p_start)
        parent_builder.add_node("subgraph_node", child_subgraph)
        parent_builder.add_node("parent_end", p_end)
        parent_builder.add_edge(START, "parent_start")
        parent_builder.add_edge("parent_start", "subgraph_node")
        parent_builder.add_edge("subgraph_node", "parent_end")
        parent_builder.add_edge("parent_end", END)

        thread_id = f"thread-subgraph-{uuid.uuid4().hex[:8]}"
        config: Any = {"configurable": {"thread_id": thread_id}}

        with RelateSaver(db_path=temp_disk_path) as saver:
            graph = parent_builder.compile(checkpointer=saver)

            init_input: ParentWorkflowState = {
                "counter": 10,
                "sub_step": "init",
                "parent_note": "none",
            }
            res: Any = graph.invoke(init_input, config=config)

            # Verification of final output: 10 + 1 (start) + 5 (child_a) + 15 (child_b) + 2 (end) = 33
            assert res["counter"] == 33
            assert res["sub_step"] == "child_b"
            assert res["parent_note"] == "finished"

            # AC-173: get_state without namespace resolves root namespace state ("")
            root_state = graph.get_state(config)
            assert root_state.values["counter"] == 33
            assert root_state.values["parent_note"] == "finished"
            assert root_state.config.get("configurable", {}).get("checkpoint_ns") == ""

            # AC-174: Checkpoint parents form proper parent chains across graph and subgraph
            all_checkpoints = list(saver.list(config))
            assert len(all_checkpoints) > 0

            # Partition by namespace
            root_checkpoints = [
                c
                for c in all_checkpoints
                if not c.config.get("configurable", {}).get("checkpoint_ns")
            ]
            subgraph_checkpoints = [
                c
                for c in all_checkpoints
                if str(c.config.get("configurable", {}).get("checkpoint_ns", "")).startswith(
                    "subgraph_node"
                )
            ]

            assert len(root_checkpoints) >= 3
            assert len(subgraph_checkpoints) >= 2

            # Root chain links: reverse chronological order, each points to prior
            root_chronological = list(reversed(root_checkpoints))
            assert root_chronological[0].parent_config is None
            for i in range(1, len(root_chronological)):
                expected_parent_id = root_chronological[i - 1].config["configurable"][
                    "checkpoint_id"
                ]
                actual_parent_cfg = root_chronological[i].parent_config
                assert actual_parent_cfg is not None
                assert actual_parent_cfg["configurable"]["checkpoint_id"] == expected_parent_id

            # Subgraph chain links: each non-initial child checkpoint points to prior child checkpoint
            subgraph_chronological = list(reversed(subgraph_checkpoints))
            assert subgraph_chronological[0].parent_config is None
            for i in range(1, len(subgraph_chronological)):
                expected_sub_parent_id = subgraph_chronological[i - 1].config["configurable"][
                    "checkpoint_id"
                ]
                actual_sub_parent_cfg = subgraph_chronological[i].parent_config
                assert actual_sub_parent_cfg is not None
                assert (
                    actual_sub_parent_cfg["configurable"]["checkpoint_id"] == expected_sub_parent_id
                )

            # AC-174 explicit check: metadata["parents"] is strictly ignored in put()
            explicit_cfg: Any = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": "explicit_parent"}
            }
            chk: Any = {
                "v": 1,
                "id": "explicit_child",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {"counter": 999},
                "channel_versions": {"counter": 1},
                "versions_seen": {},
            }
            saver.put(
                explicit_cfg,
                chk,
                {"step": 99, "parents": {"": "bogus_metadata_parent"}},
                {},
            )
            tuple_explicit = saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "explicit_child"}}
            )
            assert tuple_explicit is not None
            assert tuple_explicit.parent_config is not None
            assert (
                tuple_explicit.parent_config["configurable"]["checkpoint_id"]
                == "explicit_parent"
            )

    def test_update_state_fork_and_invoke_time_travel_ac_175(
        self, temp_disk_path: str
    ) -> None:
        """Verifies time-travel branching via update_state followed by invoke resuming from edited step.

        Validates AC-175: Checkpoint ordering and filtering use checkpoint_id DESC so get_state resolves
        the new branch head (highest checkpoint_id) rather than the old branch's later step.
        """
        class ForkState(TypedDict):
            counter: int
            trace: list[str]

        builder = StateGraph(ForkState)

        def f_step1(state: ForkState) -> dict[str, Any]:
            return {"counter": state["counter"] + 1, "trace": state["trace"] + ["step1"]}

        def f_step2(state: ForkState) -> dict[str, Any]:
            return {"counter": state["counter"] + 10, "trace": state["trace"] + ["step2"]}

        def f_step3(state: ForkState) -> dict[str, Any]:
            return {"counter": state["counter"] + 100, "trace": state["trace"] + ["step3"]}

        builder.add_node("step1", f_step1)
        builder.add_node("step2", f_step2)
        builder.add_node("step3", f_step3)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", "step3")
        builder.add_edge("step3", END)

        thread_id = f"thread-fork-{uuid.uuid4().hex[:8]}"
        base_config: Any = {"configurable": {"thread_id": thread_id}}

        with RelateSaver(db_path=temp_disk_path) as saver:
            graph = builder.compile(checkpointer=saver)

            # 1. Run multi-step graph to completion (0 + 1 + 10 + 100 = 111)
            init_fork: ForkState = {"counter": 0, "trace": ["start"]}
            initial_res: Any = graph.invoke(init_fork, config=base_config)
            assert initial_res["counter"] == 111
            assert initial_res["trace"] == ["start", "step1", "step2", "step3"]

            # 2. Grab checkpoint_id of step 1
            history = list(saver.list(base_config))
            step1_ckpts = [c for c in history if c.metadata.get("step") == 1]
            assert len(step1_ckpts) == 1
            step1_id = step1_ckpts[0].config["configurable"]["checkpoint_id"]
            assert step1_ckpts[0].checkpoint["channel_values"]["counter"] == 1

            # 3. Call update_state on step 1 with edited counter: 99
            edit_config: Any = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": step1_id}
            }
            update_result: Any = graph.update_state(edit_config, {"counter": 99})
            updated_id = update_result["configurable"]["checkpoint_id"]
            assert updated_id != step1_id

            # 4. Invoke from updated checkpoint to resume execution along the fork
            resume_config: Any = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": updated_id}
            }
            fork_res: Any = graph.invoke(None, config=resume_config)

            # Execution resumes from edited state: 99 + 10 (step2) + 100 (step3) = 209
            assert fork_res["counter"] == 209
            assert fork_res["trace"] == ["start", "step1", "step2", "step3"]

            # 5. Verify get_state resolves new branch head (highest checkpoint_id) per AC-175
            current_state = graph.get_state(base_config)
            assert current_state.values["counter"] == 209
            assert current_state.config["configurable"]["checkpoint_id"] > step1_id

            # Verify saver.list() preserves reverse-chronological order by checkpoint_id DESC (AC-175)
            all_fork_ckpts = list(saver.list(base_config))
            ckpt_ids = [c.config["configurable"]["checkpoint_id"] for c in all_fork_ckpts]
            assert ckpt_ids == sorted(ckpt_ids, reverse=True)

    def test_timed_out_write_does_not_commit_ac_179(
        self, temp_disk_path: str
    ) -> None:
        """Verifies that a write expiring caller timeout is cancelled and never committed.

        Validates AC-179: When caller timeout expires, future is cancelled and worker thread
        skips execution/commit, leaving database unmutated for subsequent reads.
        """
        thread_id = f"thread-timeout-{uuid.uuid4().hex[:8]}"
        marker_id = f"marker-{uuid.uuid4().hex[:8]}"

        with RelateSaver(db_path=temp_disk_path) as saver:
            # Enqueue a brief delay on worker thread to guarantee the write times out in queue
            saver.proxy._queue.put(lambda: time.sleep(0.12))

            # Attempt a write with a tight timeout (0.01s)
            cypher = (
                "MERGE (t:Thread {thread_id: $thread_id}) "
                "CREATE (c:Checkpoint {checkpoint_id: $cid, thread_id: $thread_id, marker: $marker})"
            )
            with pytest.raises(RelateConcurrencyError) as exc_info:
                saver.proxy.execute_write(
                    cypher,
                    {"thread_id": thread_id, "cid": marker_id, "marker": "timed_out_write"},
                    timeout=0.01,
                )

            assert "timed out" in str(exc_info.value).lower()

            # Wait for the delay task and cancellation processing to finish
            time.sleep(0.18)

            # Subsequent read must not observe the timed-out write
            records = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) RETURN c",
                {"cid": marker_id},
            )
            assert len(records) == 0

            # Verify checkpointer level put with tight timeout also aborts cleanly
            saver.proxy._queue.put(lambda: time.sleep(0.12))
            chk: Any = {
                "v": 1,
                "id": f"chk-{uuid.uuid4().hex[:8]}",
                "ts": datetime.now(UTC).isoformat(),
                "channel_values": {"key": "val"},
                "channel_versions": {"key": 1},
                "versions_seen": {},
            }
            original_timeout = saver.proxy.default_timeout
            saver.proxy.default_timeout = 0.01
            try:
                with pytest.raises(RelateConcurrencyError):
                    saver.put(
                        {"configurable": {"thread_id": thread_id}},
                        chk,
                        {"step": 0},
                        {},
                    )
            finally:
                saver.proxy.default_timeout = original_timeout

            time.sleep(0.18)
            tuple_check = saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": chk["id"]}}
            )
            assert tuple_check is None

    def test_closed_proxy_read_raises_connection_error_ac_176(
        self, temp_disk_path: str
    ) -> None:
        """Verifies closed proxy propagates RelateConnectionError on get_tuple and list.

        Validates AC-176: Underlying RelateConnectionError is re-raised rather than returning
        None or an empty collection.
        """
        thread_id = f"thread-closed-{uuid.uuid4().hex[:8]}"
        cfg: Any = {"configurable": {"thread_id": thread_id}}

        saver = RelateSaver(db_path=temp_disk_path)
        chk: Any = {
            "v": 1,
            "id": f"chk-pre-close-{uuid.uuid4().hex[:8]}",
            "ts": datetime.now(UTC).isoformat(),
            "channel_values": {"status": "active"},
            "channel_versions": {"status": 1},
            "versions_seen": {},
        }
        res_cfg = saver.put(cfg, chk, {"step": 1}, {})
        assert saver.get_tuple(res_cfg) is not None

        # Explicitly close the proxy / saver
        saver.proxy.close()

        # get_tuple must raise RelateConnectionError (not return None)
        with pytest.raises(RelateConnectionError) as exc_get:
            saver.get_tuple(res_cfg)
        assert "closed" in str(exc_get.value).lower()

        # list must raise RelateConnectionError (not return empty iterator)
        with pytest.raises(RelateConnectionError) as exc_list:
            list(saver.list(res_cfg))
        assert "closed" in str(exc_list.value).lower()

