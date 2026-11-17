"""Unit tests for Cypher schema DDL, DTO models, and in-memory FakeRelateDriver."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Ensure project root is available on sys.path when running pytest with pythonpath=["src"]
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import relate
from relate.exceptions import IntegrityError
from tests.fakes.fake_relate_driver import (
    FakeCursor,
    FakeRelateDriver,
    InMemoryRelateDriver,
)

from langgraph_checkpoint_relate.errors import RelateConnectionError
from langgraph_checkpoint_relate.models import (
    CheckpointNode,
    LoopReport,
    RootCauseReport,
    ThreadNode,
    ToolCallNode,
    WriteNode,
)
from langgraph_checkpoint_relate.schema import (
    ALL_SCHEMA_STATEMENTS,
    CONSTRAINT_CHECKPOINT_UNIQUE,
    CONSTRAINT_THREAD_UNIQUE,
    CONSTRAINT_TOOL_CALL_UNIQUE,
    CONSTRAINT_WRITE_UNIQUE,
    INDEX_CHECKPOINT_HASH,
    INDEX_CHECKPOINT_NS,
    INDEX_CHECKPOINT_THREAD,
    SCHEMA_CONSTRAINTS,
    SCHEMA_INDEXES,
    initialize_schema,
)


class TestThreadNode:
    """Test suite for ThreadNode DTO creation, validation, and serialization."""

    def test_valid_thread_node_creation(self) -> None:
        """Verify standard ThreadNode creation with explicit and default attributes."""
        node = ThreadNode(thread_id="th-100", created_at="2026-11-12T10:00:00Z")
        assert node.thread_id == "th-100"
        assert node.created_at == "2026-11-12T10:00:00Z"

        # Default created_at
        auto_node = ThreadNode(thread_id="th-200")
        assert auto_node.thread_id == "th-200"
        assert len(auto_node.created_at) > 10

    def test_thread_node_empty_id_validation(self) -> None:
        """Verify invalid empty or whitespace thread_id raises ValueError."""
        with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
            ThreadNode(thread_id="")

        with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
            ThreadNode(thread_id="   ")

        with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
            ThreadNode(thread_id=None)  # type: ignore[arg-type]

    def test_thread_node_empty_created_at_validation(self) -> None:
        """Verify invalid empty created_at raises ValueError."""
        with pytest.raises(ValueError, match="created_at must be a non-empty ISO-8601 string"):
            ThreadNode(thread_id="th-100", created_at="")

    def test_thread_node_immutability(self) -> None:
        """Verify ThreadNode is frozen and rejects mutation."""
        node = ThreadNode(thread_id="th-100")
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.thread_id = "th-mutated"  # type: ignore[misc]

    def test_thread_node_to_dict_and_from_dict(self) -> None:
        """Verify round-trip dictionary serialization and deserialization."""
        node = ThreadNode(thread_id="th-100", created_at="2026-11-12T10:00:00Z")
        data = node.to_dict()
        assert data == {"thread_id": "th-100", "created_at": "2026-11-12T10:00:00Z"}

        restored = ThreadNode.from_dict(data)
        assert restored == node


class TestCheckpointNode:
    """Test suite for CheckpointNode DTO creation, validation, and normalization."""

    def test_valid_checkpoint_node_creation(self) -> None:
        """Verify standard CheckpointNode creation with required and optional fields."""
        node = CheckpointNode(
            checkpoint_id="cp-001",
            thread_id="th-001",
            step=1,
            checkpoint_blob=b"channel_state_bytes",
            metadata_blob=b"metadata_bytes",
            state_hash="a1b2c3d4",
            checkpoint_ns="subgraph_a",
            parent_checkpoint_id="cp-000",
            created_at="2026-11-12T10:05:00Z",
        )
        assert node.checkpoint_id == "cp-001"
        assert node.thread_id == "th-001"
        assert node.step == 1
        assert node.checkpoint_blob == b"channel_state_bytes"
        assert node.metadata_blob == b"metadata_bytes"
        assert node.state_hash == "a1b2c3d4"
        assert node.checkpoint_ns == "subgraph_a"
        assert node.parent_checkpoint_id == "cp-000"
        assert node.created_at == "2026-11-12T10:05:00Z"

    def test_checkpoint_node_byte_normalization(self) -> None:
        """Verify list-of-ints and bytearray are normalized to bytes."""
        byte_list = [104, 101, 108, 108, 111]
        node = CheckpointNode(
            checkpoint_id="cp-002",
            thread_id="th-001",
            step=0,
            checkpoint_blob=byte_list,  # type: ignore[arg-type]
            metadata_blob=bytearray(b"meta"),  # type: ignore[arg-type]
            state_hash="hash123",
        )
        assert isinstance(node.checkpoint_blob, bytes)
        assert node.checkpoint_blob == b"hello"
        assert isinstance(node.metadata_blob, bytes)
        assert node.metadata_blob == b"meta"

    def test_checkpoint_node_validation_errors(self) -> None:
        """Verify boundary validations for empty ids, negative step, and malformed types."""
        # Empty checkpoint_id
        with pytest.raises(ValueError, match="checkpoint_id must be a non-empty string"):
            CheckpointNode(
                checkpoint_id="",
                thread_id="th-1",
                step=0,
                checkpoint_blob=b"",
                metadata_blob=b"",
                state_hash="hash",
            )

        # Empty thread_id
        with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
            CheckpointNode(
                checkpoint_id="cp-1",
                thread_id="",
                step=0,
                checkpoint_blob=b"",
                metadata_blob=b"",
                state_hash="hash",
            )

        # Negative step
        with pytest.raises(ValueError, match="step must be a non-negative integer"):
            CheckpointNode(
                checkpoint_id="cp-1",
                thread_id="th-1",
                step=-1,
                checkpoint_blob=b"",
                metadata_blob=b"",
                state_hash="hash",
            )

        # Non-string checkpoint_ns
        with pytest.raises(TypeError, match="checkpoint_ns must be a string"):
            CheckpointNode(
                checkpoint_id="cp-1",
                thread_id="th-1",
                step=0,
                checkpoint_blob=b"",
                metadata_blob=b"",
                state_hash="hash",
                checkpoint_ns=123,  # type: ignore[arg-type]
            )

        # Invalid blob type
        with pytest.raises(TypeError, match="checkpoint_blob must be bytes"):
            CheckpointNode(
                checkpoint_id="cp-1",
                thread_id="th-1",
                step=0,
                checkpoint_blob="not_bytes",  # type: ignore[arg-type]
                metadata_blob=b"",
                state_hash="hash",
            )

    def test_checkpoint_node_immutability(self) -> None:
        """Verify CheckpointNode cannot be mutated after creation."""
        node = CheckpointNode(
            checkpoint_id="cp-1",
            thread_id="th-1",
            step=0,
            checkpoint_blob=b"blob",
            metadata_blob=b"meta",
            state_hash="hash",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.step = 5  # type: ignore[misc]

    def test_checkpoint_node_roundtrip_dict(self) -> None:
        """Verify to_dict and from_dict fidelity for CheckpointNode."""
        node = CheckpointNode(
            checkpoint_id="cp-roundtrip",
            thread_id="th-1",
            step=3,
            checkpoint_blob=b"\x00\x01\x02",
            metadata_blob=b"\x03\x04\x05",
            state_hash="hash-xyz",
            checkpoint_ns="agent_worker",
            parent_checkpoint_id="cp-prev",
            created_at="2026-11-12T10:10:00Z",
        )
        d = node.to_dict()
        assert d["checkpoint_id"] == "cp-roundtrip"
        assert d["checkpoint_blob"] == b"\x00\x01\x02"

        restored = CheckpointNode.from_dict(d)
        assert restored == node

        # Test from_dict with list-of-ints representation from DB
        db_record = dict(d)
        db_record["checkpoint_blob"] = [0, 1, 2]
        db_record["metadata_blob"] = [3, 4, 5]
        from_db_node = CheckpointNode.from_dict(db_record)
        assert from_db_node == node


class TestWriteNode:
    """Test suite for WriteNode DTO creation, composite ID generation, and validation."""

    def test_write_node_build_write_id(self) -> None:
        """Verify deterministic composite write_id generation format."""
        cid = WriteNode.build_write_id(
            thread_id="th-1",
            checkpoint_ns="main",
            checkpoint_id="cp-99",
            task_id="task-apply",
            idx=2,
        )
        assert cid == "th-1:main:cp-99:task-apply:2"

    def test_write_node_create_helper(self) -> None:
        """Verify WriteNode.create derives write_id automatically."""
        node = WriteNode.create(
            thread_id="th-1",
            checkpoint_id="cp-1",
            task_id="task-01",
            channel="messages",
            write_blob=b"msg_write",
            checkpoint_ns="tools",
            idx=0,
        )
        assert node.write_id == "th-1:tools:cp-1:task-01:0"
        assert node.channel == "messages"
        assert node.write_blob == b"msg_write"

    def test_write_node_validation_errors(self) -> None:
        """Verify validation errors on invalid write attributes."""
        with pytest.raises(ValueError, match="write_id must be a non-empty string"):
            WriteNode(
                write_id="",
                thread_id="th-1",
                checkpoint_id="cp-1",
                task_id="t-1",
                channel="ch",
                write_blob=b"b",
            )

        with pytest.raises(ValueError, match="idx must be a non-negative integer"):
            WriteNode(
                write_id="wid",
                thread_id="th-1",
                checkpoint_id="cp-1",
                task_id="t-1",
                channel="ch",
                write_blob=b"b",
                idx=-2,
            )

    def test_write_node_roundtrip_dict(self) -> None:
        """Verify to_dict and from_dict for WriteNode."""
        node = WriteNode.create(
            thread_id="th-1",
            checkpoint_id="cp-1",
            task_id="t-1",
            channel="output",
            write_blob=b"result",
        )
        data = node.to_dict()
        restored = WriteNode.from_dict(data)
        assert restored == node


class TestToolCallNode:
    """Test suite for ToolCallNode DTO creation and validation."""

    def test_valid_tool_call_node(self) -> None:
        """Verify valid ToolCallNode creation and serialization."""
        node = ToolCallNode(
            call_id="call-123",
            tool_name="calculator",
            args_blob=b'{"x": 2, "y": 3}',
            result_blob=b'{"result": 5}',
            error=None,
            created_at="2026-11-12T10:15:00Z",
        )
        assert node.call_id == "call-123"
        assert node.tool_name == "calculator"
        assert node.result_blob == b'{"result": 5}'

        data = node.to_dict()
        restored = ToolCallNode.from_dict(data)
        assert restored == node

    def test_tool_call_node_with_error(self) -> None:
        """Verify ToolCallNode capturing failure message."""
        node = ToolCallNode(
            call_id="call-err-1",
            tool_name="web_search",
            args_blob=b'{"query": "neo4j"}',
            error="Connection timeout to search provider",
        )
        assert node.error == "Connection timeout to search provider"
        assert node.result_blob is None

    def test_tool_call_validation_errors(self) -> None:
        """Verify validation errors on empty tool name or call_id."""
        with pytest.raises(ValueError, match="call_id must be a non-empty string"):
            ToolCallNode(call_id="", tool_name="tool", args_blob=b"{}")

        with pytest.raises(ValueError, match="tool_name must be a non-empty string"):
            ToolCallNode(call_id="c-1", tool_name="", args_blob=b"{}")


class TestForensicReports:
    """Test suite for LoopReport and RootCauseReport diagnostic models."""

    def test_loop_report_creation_and_immutability(self) -> None:
        """Verify LoopReport creation, validation, and immutability."""
        report = LoopReport(
            thread_id="th-loop",
            cycle_nodes=["agent", "tools", "agent"],
            occurrence_count=3,
            checkpoint_id_range=("cp-1", "cp-7"),
        )
        assert report.thread_id == "th-loop"
        assert report.occurrence_count == 3
        assert report.checkpoint_id_range == ("cp-1", "cp-7")

        with pytest.raises(dataclasses.FrozenInstanceError):
            report.occurrence_count = 4  # type: ignore[misc]

    def test_loop_report_validation_errors(self) -> None:
        """Verify LoopReport boundary validation."""
        with pytest.raises(ValueError, match="occurrence_count must be non-negative"):
            LoopReport(
                thread_id="th-1",
                cycle_nodes=[],
                occurrence_count=-1,
                checkpoint_id_range=("cp-1", "cp-2"),
            )

        with pytest.raises(ValueError, match="checkpoint_id_range must be a 2-tuple"):
            LoopReport(
                thread_id="th-1",
                cycle_nodes=[],
                occurrence_count=1,
                checkpoint_id_range=("cp-1",),  # type: ignore[arg-type]
            )

    def test_root_cause_report_creation_and_validation(self) -> None:
        """Verify RootCauseReport fields and validation."""
        report = RootCauseReport(
            thread_id="th-fail",
            error_checkpoint_id="cp-9",
            error_message="Tool call failed with exit code 1",
            root_tool_call_id="call-5",
            root_channel_write="th-fail::cp-5:task-1:0",
            causal_path_length=4,
            diagnostic_summary="Tool execution error cascaded through 4 state steps",
        )
        assert report.causal_path_length == 4
        assert report.root_tool_call_id == "call-5"

        with pytest.raises(ValueError, match="error_checkpoint_id must be a non-empty string"):
            RootCauseReport(thread_id="th-1", error_checkpoint_id="", error_message="msg")


class TestSchemaDDLAndConstraints:
    """Test suite for Cypher schema DDL statements and initialize_schema function."""

    def test_schema_statements_definitions(self) -> None:
        """Verify schema constraint and index definitions match architecture specifications."""
        assert len(SCHEMA_CONSTRAINTS) == 4
        assert len(SCHEMA_INDEXES) == 3
        assert len(ALL_SCHEMA_STATEMENTS) == 7

        # Check required uniqueness constraints
        assert "thread_id_unique" in CONSTRAINT_THREAD_UNIQUE
        assert "IS UNIQUE" in CONSTRAINT_THREAD_UNIQUE
        assert "checkpoint_id_unique" in CONSTRAINT_CHECKPOINT_UNIQUE
        assert "IS UNIQUE" in CONSTRAINT_CHECKPOINT_UNIQUE
        assert "write_id_unique" in CONSTRAINT_WRITE_UNIQUE
        assert "IS UNIQUE" in CONSTRAINT_WRITE_UNIQUE
        assert "tool_call_id_unique" in CONSTRAINT_TOOL_CALL_UNIQUE
        assert "IS UNIQUE" in CONSTRAINT_TOOL_CALL_UNIQUE

        # Check required indexes
        assert "checkpoint_thread_idx" in INDEX_CHECKPOINT_THREAD
        assert "checkpoint_hash_idx" in INDEX_CHECKPOINT_HASH
        assert "checkpoint_ns_idx" in INDEX_CHECKPOINT_NS

    def test_initialize_schema_invalid_connection(self) -> None:
        """Verify initialize_schema raises RelateConnectionError when connection is invalid."""
        with pytest.raises(RelateConnectionError, match="Valid database connection required"):
            initialize_schema(None)

        with pytest.raises(RelateConnectionError, match="Valid database connection required"):
            initialize_schema("not_a_connection")

    # AC-01: Schema constraints initialized on non-existent or fresh database
    def test_initialize_schema_fake_driver_idempotence(self) -> None:
        """Verify initialize_schema runs idempotently against FakeRelateDriver connection (AC-01)."""
        driver = FakeRelateDriver()
        conn = driver.connect()

        # Run first time
        initialize_schema(conn)
        assert len(driver.constraints) == 4
        assert len(driver.indexes) == 3
        assert "thread_id_unique" in driver.constraints
        assert "checkpoint_id_unique" in driver.constraints
        assert "write_id_unique" in driver.constraints
        assert "tool_call_id_unique" in driver.constraints

        # Run second time for idempotence verification
        initialize_schema(conn)
        assert len(driver.constraints) == 4
        assert len(driver.indexes) == 3

    # AC-01: Schema constraints initialized on real in-memory Relate database
    def test_initialize_schema_real_relate_memory_idempotence(self) -> None:
        """Verify initialize_schema runs idempotently against real in-memory Relate database (AC-01)."""
        conn = relate.connect(":memory:")
        try:
            # First execution
            initialize_schema(conn)

            # Second execution: verify IF NOT EXISTS prevents error
            initialize_schema(conn)
        finally:
            conn.close()

    def test_constraint_uniqueness_enforcement_fake_driver(self) -> None:
        """Verify FakeRelateDriver enforces uniqueness constraints as defined in schema."""
        driver = FakeRelateDriver()
        conn = driver.connect()
        initialize_schema(conn)

        # 1. Thread constraint
        conn.execute(
            "CREATE (:Thread {thread_id: $thread_id, created_at: $created_at})",
            {"thread_id": "t1", "created_at": "2026-11-12T00:00:00Z"},
        )
        with pytest.raises(IntegrityError, match="uniqueness constraint violated"):
            conn.execute(
                "CREATE (:Thread {thread_id: $thread_id, created_at: $created_at})",
                {"thread_id": "t1", "created_at": "2026-11-12T01:00:00Z"},
            )

        # 2. Checkpoint constraint
        conn.execute(
            "CREATE (:Checkpoint {checkpoint_id: $checkpoint_id, thread_id: $thread_id})",
            {"checkpoint_id": "cp1", "thread_id": "t1"},
        )
        with pytest.raises(IntegrityError, match="uniqueness constraint violated"):
            conn.execute(
                "CREATE (:Checkpoint {checkpoint_id: $checkpoint_id, thread_id: $thread_id})",
                {"checkpoint_id": "cp1", "thread_id": "t1"},
            )

        # 3. Write constraint
        conn.execute(
            "CREATE (:Write {write_id: $write_id, channel: $channel})",
            {"write_id": "w1", "channel": "c"},
        )
        with pytest.raises(IntegrityError, match="uniqueness constraint violated"):
            conn.execute(
                "CREATE (:Write {write_id: $write_id, channel: $channel})",
                {"write_id": "w1", "channel": "c"},
            )

        # 4. ToolCall constraint
        conn.execute(
            "CREATE (:ToolCall {call_id: $call_id, tool_name: $tool_name})",
            {"call_id": "tc1", "tool_name": "calc"},
        )
        with pytest.raises(IntegrityError, match="uniqueness constraint violated"):
            conn.execute(
                "CREATE (:ToolCall {call_id: $call_id, tool_name: $tool_name})",
                {"call_id": "tc1", "tool_name": "calc"},
            )

    def test_constraint_uniqueness_enforcement_real_relate(self) -> None:
        """Verify real Relate database enforces uniqueness constraints matching schema DDL."""
        conn = relate.connect(":memory:")
        try:
            initialize_schema(conn)

            # 1. Thread uniqueness
            conn.execute(
                "CREATE (:Thread {thread_id: $thread_id, created_at: $created_at})",
                {"thread_id": "t_real_1", "created_at": "2026-11-12T00:00:00Z"},
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    "CREATE (:Thread {thread_id: $thread_id, created_at: $created_at})",
                    {"thread_id": "t_real_1", "created_at": "2026-11-12T01:00:00Z"},
                )

            # 2. Checkpoint uniqueness
            conn.execute(
                "CREATE (:Checkpoint {checkpoint_id: $checkpoint_id, thread_id: $thread_id})",
                {"checkpoint_id": "cp_real_1", "thread_id": "t_real_1"},
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    "CREATE (:Checkpoint {checkpoint_id: $checkpoint_id, thread_id: $thread_id})",
                    {"checkpoint_id": "cp_real_1", "thread_id": "t_real_1"},
                )
        finally:
            conn.close()


class TestFakeRelateDriverFeatures:
    """Test suite for FakeRelateDriver, FakeConnection, FakeCursor, and FakeRecord."""

    def test_fake_cursor_fetch_methods(self) -> None:
        """Verify fetchone, fetchall, peek, single, and len on FakeCursor."""
        records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        cur = FakeCursor(records)

        assert len(cur) == 2
        peeked = cur.peek()
        assert peeked is not None
        assert peeked["name"] == "Alice"

        one = cur.fetchone()
        assert one is not None
        assert one.name == "Alice"
        assert one["id"] == 1

        all_remaining = cur.fetchall()
        assert len(all_remaining) == 1
        assert all_remaining[0].name == "Bob"

        assert cur.fetchone() is None

    def test_fake_cursor_single_behavior(self) -> None:
        """Verify cursor single() returns exactly 1 item or raises ResultNotSingleError."""
        from relate.exceptions import ResultNotSingleError

        empty_cur = FakeCursor([])
        with pytest.raises(ResultNotSingleError):
            empty_cur.single()

        multi_cur = FakeCursor([{"x": 1}, {"x": 2}])
        with pytest.raises(ResultNotSingleError):
            multi_cur.single()

        single_cur = FakeCursor([{"x": 42}])
        rec = single_cur.single()
        assert rec.x == 42
        assert rec.data() == {"x": 42}

    def test_fake_connection_lifecycle(self) -> None:
        """Verify connection and cursor close lifecycle methods."""
        from relate.exceptions import InterfaceError

        driver = FakeRelateDriver()
        conn = driver.connect()

        conn.commit()
        conn.rollback()

        cur = conn.cursor()
        cur.close()
        assert cur.is_closed
        with pytest.raises(InterfaceError):
            cur.fetchone()

        conn.close()
        assert conn.is_closed
        with pytest.raises(InterfaceError):
            conn.execute("MATCH (n) RETURN n")

    def test_fake_driver_reset_and_alias(self) -> None:
        """Verify driver reset and InMemoryRelateDriver specification alias."""
        assert InMemoryRelateDriver is FakeRelateDriver

        driver = FakeRelateDriver()
        driver.connect().execute(
            "CREATE CONSTRAINT c1 IF NOT EXISTS FOR (t:Thread) REQUIRE t.thread_id IS UNIQUE"
        )
        assert len(driver.constraints) == 1

        driver.reset()
        assert len(driver.constraints) == 0
        assert len(driver.nodes) == 0
