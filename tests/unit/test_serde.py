"""Unit tests for Relate checkpoint state serialization engine (AC-12, AC-13, ADR-0003)."""

from __future__ import annotations

import socket
import tempfile
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from langgraph_checkpoint_relate.errors import CheckpointSerializationError
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.serde import (
    RelateStateSerializer,
    default_serializer,
)


class SampleAgentState(BaseModel):
    """Pydantic model representing structured agent channel state."""

    agent_id: str
    iteration: int
    flags: list[str] = []


@pytest.fixture
def serializer() -> RelateStateSerializer:
    """Provide a fresh RelateStateSerializer instance."""
    return RelateStateSerializer()


def test_serializer_protocol_conformance_ac_13(serializer: RelateStateSerializer) -> None:
    """AC-13: Verify RelateStateSerializer conforms to LangGraph SerializerProtocol."""
    assert isinstance(serializer, SerializerProtocol)
    assert hasattr(serializer, "dumps_typed")
    assert hasattr(serializer, "loads_typed")
    assert hasattr(serializer, "dumps")
    assert hasattr(serializer, "loads")
    assert hasattr(serializer, "dumps_base64")
    assert hasattr(serializer, "loads_base64")
    assert hasattr(serializer, "compute_state_hash")


def test_roundtrip_primitives_and_collections_ac_13(serializer: RelateStateSerializer) -> None:
    """AC-13: Verify round-trip serialization for primitives, nested dicts, and lists."""
    test_payload: dict[str, Any] = {
        "int_val": 42,
        "float_val": 3.14159,
        "str_val": "Relate Checkpointer",
        "bool_true": True,
        "bool_false": False,
        "none_val": None,
        "list_val": [1, "two", 3.0, [4, 5]],
        "nested_dict": {"level1": {"level2": {"leaf": "deep_value"}}},
    }

    # Test typed serialization
    typed_blob = serializer.dumps_typed(test_payload)
    restored_typed = serializer.loads_typed(typed_blob)
    assert restored_typed == test_payload

    # Test byte serialization
    binary_blob = serializer.dumps(test_payload)
    assert isinstance(binary_blob, bytes)
    restored_binary = serializer.loads(binary_blob)
    assert restored_binary == test_payload

    # Test base64 serialization
    b64_str = serializer.dumps_base64(test_payload)
    assert isinstance(b64_str, str)
    restored_b64 = serializer.loads_base64(b64_str)
    assert restored_b64 == test_payload


def test_roundtrip_standard_langgraph_types_ac_13(serializer: RelateStateSerializer) -> None:
    """AC-13: Verify preservation of LangChain messages, datetimes, UUIDs, and Pydantic models."""
    fixed_uuid = uuid.UUID("1ef82a3c-b034-6c2e-8000-000000000001")
    fixed_dt = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)

    human_msg = HumanMessage(content="Hello assistant", id="msg-001")
    ai_msg = AIMessage(
        content="Hello user, processing request",
        id="msg-002",
        tool_calls=[{"name": "fetch_data", "args": {"query": "weather"}, "id": "call_123"}],
    )
    sys_msg = SystemMessage(content="You are a helpful graph reasoning agent.")
    tool_msg = ToolMessage(content="Sunny 22C", tool_call_id="call_123")

    agent_model = SampleAgentState(
        agent_id="agent-neo",
        iteration=3,
        flags=["fast_path", "verified"],
    )

    state_payload: dict[str, Any] = {
        "uuid_field": fixed_uuid,
        "datetime_field": fixed_dt,
        "messages": [human_msg, ai_msg, sys_msg, tool_msg],
        "agent_state": agent_model,
    }

    serialized = serializer.dumps(state_payload)
    restored = serializer.loads(serialized)

    # Validate UUID fidelity
    assert restored["uuid_field"] == fixed_uuid
    assert isinstance(restored["uuid_field"], uuid.UUID)

    # Validate datetime fidelity
    assert restored["datetime_field"] == fixed_dt
    assert isinstance(restored["datetime_field"], datetime)

    # Validate LangChain message fidelity
    messages = restored["messages"]
    assert len(messages) == 4

    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == human_msg.content
    assert messages[0].id == human_msg.id

    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == ai_msg.content
    assert messages[1].tool_calls == ai_msg.tool_calls

    assert isinstance(messages[2], SystemMessage)
    assert messages[2].content == sys_msg.content

    assert isinstance(messages[3], ToolMessage)
    assert messages[3].content == tool_msg.content
    assert messages[3].tool_call_id == tool_msg.tool_call_id

    # Validate Pydantic model fidelity
    restored_model = restored["agent_state"]
    assert isinstance(restored_model, SampleAgentState)
    assert restored_model.agent_id == "agent-neo"
    assert restored_model.iteration == 3
    assert restored_model.flags == ["fast_path", "verified"]


def test_deterministic_state_hash_key_ordering_ac_13(serializer: RelateStateSerializer) -> None:
    """AC-13, ADR-0003: Verify state hash is deterministic regardless of dict key insertion order."""
    state_a = {
        "channel_alpha": "data",
        "channel_beta": 100,
        "channel_gamma": {"z": 1, "a": 2, "m": 3},
    }

    state_b = {
        "channel_gamma": {"m": 3, "z": 1, "a": 2},
        "channel_alpha": "data",
        "channel_beta": 100,
    }

    hash_a = serializer.compute_state_hash(state_a)
    hash_b = serializer.compute_state_hash(state_b)

    assert isinstance(hash_a, str)
    assert len(hash_a) == 64
    assert hash_a == hash_b


def test_deterministic_state_hash_checkpoint_extraction_ac_13(
    serializer: RelateStateSerializer,
) -> None:
    """AC-13, ADR-0003: Checkpoint dict hashing isolates channel_values for cycle detection."""
    channel_values = {
        "current_step": "evaluate",
        "retry_count": 2,
    }

    checkpoint_step_1: Checkpoint = {
        "v": 1,
        "id": "1ef82a3c-b034-6c2e-8000-000000000001",
        "ts": "2026-11-12T10:00:00Z",
        "channel_values": channel_values,
        "channel_versions": {"current_step": 1, "retry_count": 1},
        "versions_seen": {"node_a": {"current_step": 1}},
        "updated_channels": ["current_step"],
    }

    checkpoint_step_2: Checkpoint = {
        "v": 1,
        "id": "1ef82a3c-b034-6c2e-8000-000000000002",
        "ts": "2026-11-12T10:05:00Z",
        "channel_values": {
            "retry_count": 2,
            "current_step": "evaluate",
        },
        "channel_versions": {"current_step": 2, "retry_count": 2},
        "versions_seen": {"node_a": {"current_step": 2}},
        "updated_channels": ["retry_count"],
    }

    hash_direct = serializer.compute_state_hash(channel_values)
    hash_chk_1 = serializer.compute_state_hash(checkpoint_step_1)
    hash_chk_2 = serializer.compute_state_hash(checkpoint_step_2)

    assert hash_chk_1 == hash_direct
    assert hash_chk_2 == hash_direct
    assert hash_chk_1 == hash_chk_2


def test_state_hash_differentiates_distinct_states_ac_13(
    serializer: RelateStateSerializer,
) -> None:
    """AC-13: Verify differing state contents produce different SHA-256 hashes."""
    state_1 = {"counter": 1, "tag": "active"}
    state_2 = {"counter": 2, "tag": "active"}
    state_3 = {"counter": 1, "tag": "paused"}

    hash_1 = serializer.compute_state_hash(state_1)
    hash_2 = serializer.compute_state_hash(state_2)
    hash_3 = serializer.compute_state_hash(state_3)

    assert hash_1 != hash_2
    assert hash_2 != hash_3
    assert hash_1 != hash_3


def test_state_hash_raw_bytes_ac_13(serializer: RelateStateSerializer) -> None:
    """AC-13: Verify state_hash computes SHA-256 directly on raw bytes input."""
    data = b"raw_state_bytes_for_hashing"
    computed_hash = serializer.compute_state_hash(data)
    assert len(computed_hash) == 64
    assert computed_hash == serializer.state_hash(data)


def test_unserializable_socket_raises_error_naming_key_ac_12(
    serializer: RelateStateSerializer,
) -> None:
    """AC-12: An open socket in state payload raises CheckpointSerializationError identifying the offending key."""
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        payload = {
            "valid_key": "some_value",
            "active_socket": test_socket,
        }

        with pytest.raises(CheckpointSerializationError) as exc_info:
            serializer.dumps(payload)

        error_msg = str(exc_info.value)
        assert "active_socket" in error_msg
        assert "Failed to serialize checkpoint" in error_msg
    finally:
        test_socket.close()


def test_unserializable_file_handle_raises_error_naming_key_ac_12(
    serializer: RelateStateSerializer,
) -> None:
    """AC-12: An open raw file handle raises CheckpointSerializationError naming the offending key."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp_file:
        tmp_file.write("sample disk content")
        tmp_file.flush()

        raw_handle = open(tmp_file.name, "rb")
        try:
            checkpoint_payload: dict[str, Any] = {
                "channel_values": {
                    "persisted_data": [1, 2, 3],
                    "open_file_handle": raw_handle,
                }
            }

            with pytest.raises(CheckpointSerializationError) as exc_info:
                serializer.dumps(checkpoint_payload)

            error_msg = str(exc_info.value)
            assert "open_file_handle" in error_msg
            assert "channel_values" in error_msg
        finally:
            raw_handle.close()


def test_unserializable_nested_structure_pinpoints_path_ac_12(
    serializer: RelateStateSerializer,
) -> None:
    """AC-12: Un-serializable object deep inside nested structures identifies the offending path."""
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        payload = {
            "workflow": {
                "tasks": [
                    {"task_id": "t1", "result": "ok"},
                    {"task_id": "t2", "bad_conn": test_socket},
                ]
            }
        }

        with pytest.raises(CheckpointSerializationError) as exc_info:
            serializer.dumps(payload)

        error_msg = str(exc_info.value)
        assert "bad_conn" in error_msg
    finally:
        test_socket.close()


def test_unserializable_pydantic_attribute_pinpoints_key_ac_12(
    serializer: RelateStateSerializer,
) -> None:
    """AC-12: Model attribute containing un-serializable object identifies offending key."""

    class CustomContainer:
        def __init__(self, conn: Any) -> None:
            self.network_conn = conn

    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        payload = {"container": CustomContainer(conn=test_socket)}

        with pytest.raises(CheckpointSerializationError) as exc_info:
            serializer.dumps(payload)

        error_msg = str(exc_info.value)
        assert "network_conn" in error_msg or "container" in error_msg
    finally:
        test_socket.close()


def test_unserializable_root_object_ac_12(serializer: RelateStateSerializer) -> None:
    """AC-12: Standalone un-serializable object not in a dict raises typed CheckpointSerializationError."""
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(CheckpointSerializationError) as exc_info:
            serializer.dumps(test_socket)

        error_msg = str(exc_info.value)
        assert "socket" in error_msg
        assert "Failed to serialize checkpoint" in error_msg
    finally:
        test_socket.close()


def test_malformed_base64_raises_error_ac_12(serializer: RelateStateSerializer) -> None:
    """AC-12: Corrupted base64 payload raises CheckpointSerializationError."""
    malformed_b64 = "This is definitely NOT valid base64!!!@#$%"

    with pytest.raises(CheckpointSerializationError) as exc_info:
        serializer.loads_base64(malformed_b64)

    assert "Failed to decode base64 payload" in str(exc_info.value)


def test_corrupted_binary_payload_raises_error_ac_12(
    serializer: RelateStateSerializer,
) -> None:
    """AC-12: Corrupted binary bytes raise CheckpointSerializationError."""
    garbage_bytes = b"\xff\xfe\x00\x01\x02\x03corrupt_stream"

    with pytest.raises(CheckpointSerializationError) as exc_info:
        serializer.loads(garbage_bytes)

    assert "Failed to deserialize" in str(exc_info.value)


def test_malformed_typed_tuple_raises_error_ac_12(
    serializer: RelateStateSerializer,
) -> None:
    """AC-12: Invalid typed payload tuple raises CheckpointSerializationError."""
    with pytest.raises(CheckpointSerializationError) as exc_info:
        serializer.loads_typed("not_a_tuple")  # type: ignore[arg-type]

    assert "Malformed typed payload" in str(exc_info.value)


def test_convenience_aliases_and_default_instance_ac_13() -> None:
    """AC-13: Verify alias methods and module default_serializer work interchangeably."""
    data = {"sample_key": "sample_value"}

    b64_a = default_serializer.dumps_base64(data)
    b64_b = default_serializer.dumps_b64(data)
    assert b64_a == b64_b

    restored_a = default_serializer.loads_base64(b64_a)
    restored_b = default_serializer.loads_b64(b64_b)
    assert restored_a == restored_b == data

    hash_a = default_serializer.compute_state_hash(data)
    hash_b = default_serializer.state_hash(data)
    assert hash_a == hash_b


def test_relate_state_serializer_accepts_custom_serializer_protocol_ac_177() -> None:
    """AC-177: RelateStateSerializer accepts any conforming SerializerProtocol."""
    inner = JsonPlusSerializer()
    serializer = RelateStateSerializer(inner_serializer=inner)
    assert serializer.inner_serializer is inner

    payload = {"count": 42, "items": ["a", "b"]}
    t, b = serializer.dumps_typed(payload)
    restored = serializer.loads_typed((t, b))
    assert restored == payload


def test_relate_saver_auto_wraps_serializer_protocol_ac_177() -> None:
    """AC-177: RelateSaver automatically wraps custom SerializerProtocol in RelateStateSerializer."""
    inner = JsonPlusSerializer()
    saver = RelateSaver(db_path=":memory:", serde=inner)
    try:
        assert isinstance(saver.serde, RelateStateSerializer)
        assert saver.serde.inner_serializer is inner
    finally:
        saver.close()


def test_relate_saver_keeps_relate_state_serializer_without_double_wrapping_ac_177() -> None:
    """AC-177: Initializing RelateSaver with RelateStateSerializer does not double wrap."""
    existing = RelateStateSerializer()
    saver = RelateSaver(db_path=":memory:", serde=existing)
    try:
        assert saver.serde is existing
    finally:
        saver.close()

