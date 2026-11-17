"""Data Transfer Object (DTO) models for Relate graph nodes and forensic reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _normalize_bytes(val: Any, field_name: str) -> bytes:
    """Ensure value is converted to bytes, handling list-of-ints or bytearrays from database drivers."""
    if isinstance(val, bytes):
        return val
    if isinstance(val, (bytearray, memoryview)):
        return bytes(val)
    if isinstance(val, list):
        try:
            return bytes(val)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"{field_name} must contain byte integers (0-255)") from exc
    raise TypeError(f"{field_name} must be bytes, got {type(val).__name__}")


@dataclass(frozen=True)
class ThreadNode:
    """Represents an isolated LangGraph execution thread node (:Thread)."""

    thread_id: str
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.thread_id, str) or not self.thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty ISO-8601 string")

    def to_dict(self) -> dict[str, Any]:
        """Convert node properties to a Cypher-compatible parameter dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThreadNode:
        """Create a ThreadNode from a mapping or database record."""
        return cls(
            thread_id=str(data["thread_id"]),
            created_at=str(data.get("created_at", _utc_now_iso())),
        )


@dataclass(frozen=True)
class CheckpointNode:
    """Represents an immutable snapshot of workflow execution state (:Checkpoint)."""

    checkpoint_id: str
    thread_id: str
    step: int
    checkpoint_blob: bytes
    metadata_blob: bytes
    state_hash: str
    checkpoint_ns: str = ""
    parent_checkpoint_id: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string")
        if not isinstance(self.thread_id, str) or not self.thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"step must be a non-negative integer, got {self.step!r}")
        if not isinstance(self.checkpoint_ns, str):
            raise TypeError(
                f"checkpoint_ns must be a string, got {type(self.checkpoint_ns).__name__}"
            )
        if self.parent_checkpoint_id is not None and not isinstance(self.parent_checkpoint_id, str):
            raise TypeError("parent_checkpoint_id must be a string or None")
        if not isinstance(self.state_hash, str) or not self.state_hash.strip():
            raise ValueError("state_hash must be a non-empty string")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty ISO-8601 string")

        # Normalize bytes if needed
        norm_checkpoint_blob = _normalize_bytes(self.checkpoint_blob, "checkpoint_blob")
        norm_metadata_blob = _normalize_bytes(self.metadata_blob, "metadata_blob")
        object.__setattr__(self, "checkpoint_blob", norm_checkpoint_blob)
        object.__setattr__(self, "metadata_blob", norm_metadata_blob)

    def to_dict(self) -> dict[str, Any]:
        """Convert node properties to a Cypher-compatible parameter dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointNode:
        """Create a CheckpointNode from a mapping or database record."""
        raw_cp_blob = data["checkpoint_blob"]
        raw_md_blob = data["metadata_blob"]

        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            thread_id=str(data["thread_id"]),
            step=int(data["step"]),
            checkpoint_blob=_normalize_bytes(raw_cp_blob, "checkpoint_blob"),
            metadata_blob=_normalize_bytes(raw_md_blob, "metadata_blob"),
            state_hash=str(data["state_hash"]),
            checkpoint_ns=str(data.get("checkpoint_ns", "")),
            parent_checkpoint_id=str(data["parent_checkpoint_id"])
            if data.get("parent_checkpoint_id")
            else None,
            created_at=str(data.get("created_at", _utc_now_iso())),
        )


@dataclass(frozen=True)
class WriteNode:
    """Represents a pending channel update or intermediate write (:Write)."""

    write_id: str
    thread_id: str
    checkpoint_id: str
    task_id: str
    channel: str
    write_blob: bytes
    checkpoint_ns: str = ""
    idx: int = 0
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.write_id, str) or not self.write_id.strip():
            raise ValueError("write_id must be a non-empty string")
        if not isinstance(self.thread_id, str) or not self.thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("channel must be a non-empty string")
        if not isinstance(self.idx, int) or self.idx < 0:
            raise ValueError(f"idx must be a non-negative integer, got {self.idx!r}")
        if not isinstance(self.checkpoint_ns, str):
            raise TypeError(
                f"checkpoint_ns must be a string, got {type(self.checkpoint_ns).__name__}"
            )
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty ISO-8601 string")

        norm_write_blob = _normalize_bytes(self.write_blob, "write_blob")
        object.__setattr__(self, "write_blob", norm_write_blob)

    @staticmethod
    def build_write_id(
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: int,
    ) -> str:
        """Generate canonical composite write_id: {thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}."""
        return f"{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}"

    @classmethod
    def create(
        cls,
        thread_id: str,
        checkpoint_id: str,
        task_id: str,
        channel: str,
        write_blob: bytes,
        checkpoint_ns: str = "",
        idx: int = 0,
        write_id: str | None = None,
        created_at: str | None = None,
    ) -> WriteNode:
        """Convenience constructor that derives composite write_id if omitted."""
        actual_id = write_id or cls.build_write_id(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            idx=idx,
        )
        return cls(
            write_id=actual_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            channel=channel,
            write_blob=write_blob,
            idx=idx,
            created_at=created_at or _utc_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert node properties to a Cypher-compatible parameter dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WriteNode:
        """Create a WriteNode from a mapping or database record."""
        raw_blob = data["write_blob"]
        return cls(
            write_id=str(data["write_id"]),
            thread_id=str(data["thread_id"]),
            checkpoint_ns=str(data.get("checkpoint_ns", "")),
            checkpoint_id=str(data["checkpoint_id"]),
            task_id=str(data["task_id"]),
            channel=str(data["channel"]),
            write_blob=_normalize_bytes(raw_blob, "write_blob"),
            idx=int(data.get("idx", 0)),
            created_at=str(data.get("created_at", _utc_now_iso())),
        )


@dataclass(frozen=True)
class ToolCallNode:
    """Represents an external tool call invocation captured during execution (:ToolCall)."""

    call_id: str
    tool_name: str
    args_blob: bytes
    result_blob: bytes | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty ISO-8601 string")

        norm_args = _normalize_bytes(self.args_blob, "args_blob")
        object.__setattr__(self, "args_blob", norm_args)

        if self.result_blob is not None:
            norm_res = _normalize_bytes(self.result_blob, "result_blob")
            object.__setattr__(self, "result_blob", norm_res)

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        """Convert node properties to a Cypher-compatible parameter dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCallNode:
        """Create a ToolCallNode from a mapping or database record."""
        raw_args = data["args_blob"]
        raw_res = data.get("result_blob")
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            args_blob=_normalize_bytes(raw_args, "args_blob"),
            result_blob=_normalize_bytes(raw_res, "result_blob") if raw_res is not None else None,
            error=str(data["error"]) if data.get("error") is not None else None,
            created_at=str(data.get("created_at", _utc_now_iso())),
        )


@dataclass(frozen=True)
class LoopReport:
    """Diagnostic report identifying cyclical state oscillations in an execution thread."""

    thread_id: str
    cycle_nodes: list[str]
    occurrence_count: int
    checkpoint_id_range: tuple[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.thread_id, str) or not self.thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if not isinstance(self.occurrence_count, int) or self.occurrence_count < 0:
            raise ValueError(
                f"occurrence_count must be non-negative, got {self.occurrence_count!r}"
            )
        if not isinstance(self.checkpoint_id_range, tuple) or len(self.checkpoint_id_range) != 2:
            raise ValueError("checkpoint_id_range must be a 2-tuple of (start_id, end_id)")


@dataclass(frozen=True)
class RootCauseReport:
    """Forensic causal analysis report tracing failure origins backwards across DAG edges."""

    thread_id: str
    error_checkpoint_id: str
    error_message: str
    root_tool_call_id: str | None = None
    root_channel_write: str | None = None
    causal_path_length: int = 0
    diagnostic_summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.thread_id, str) or not self.thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if not isinstance(self.error_checkpoint_id, str):
            raise ValueError("error_checkpoint_id must be a non-empty string")
        if not self.error_checkpoint_id.strip() and self.error_message != "No error detected":
            raise ValueError("error_checkpoint_id must be a non-empty string")
        if not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string")
        if not isinstance(self.causal_path_length, int) or self.causal_path_length < 0:
            raise ValueError(
                f"causal_path_length must be non-negative, got {self.causal_path_length!r}"
            )


__all__ = [
    "CheckpointNode",
    "LoopReport",
    "RootCauseReport",
    "ThreadNode",
    "ToolCallNode",
    "WriteNode",
]
