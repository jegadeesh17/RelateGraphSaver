"""Core Relate graph checkpointer implementation for LangGraph.

Provides graph-native persistence, single-file WAL durability, and causal DAG tracking
conforming to LangGraph's BaseCheckpointSaver interface.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph_checkpoint_relate.errors import RelateCheckpointError
from langgraph_checkpoint_relate.proxy import RelateThreadProxy
from langgraph_checkpoint_relate.serde import RelateStateSerializer, default_serializer

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 string format."""
    return datetime.now(UTC).isoformat()


def _extract_bytes(raw: Any) -> bytes:
    """Convert raw database blob into bytes, handling list of integers or bytearray."""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, (bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, list):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("utf-8")
    raise TypeError(f"Expected binary payload from database, got {type(raw).__name__}")


class RelateSaver(BaseCheckpointSaver[str]):
    """Graph-native checkpointer for LangGraph workflows backed by Relate DB.

    Persists execution history as an immutable directed property graph with
    (:Thread) and (:Checkpoint) nodes connected via [:HAS_CHECKPOINT] and [:NEXT]
    edges, preserving execution topology and channel state with full serialization fidelity.
    """

    def __init__(
        self,
        db_path: str | RelateThreadProxy = ":memory:",
        *,
        wal_enabled: bool = True,
        proxy: RelateThreadProxy | None = None,
        serde: RelateStateSerializer | SerializerProtocol | None = None,
        connection_factory: Any | None = None,
    ) -> None:
        """Initialize RelateSaver checkpointer.

        Args:
            db_path: Path to the embedded Relate database file (default: ":memory:").
                Can also be an existing RelateThreadProxy instance.
            wal_enabled: Whether to enable Write-Ahead Logging for file-based databases.
            proxy: Optional pre-configured RelateThreadProxy actor instance.
            serde: Optional custom serializer protocol (defaults to RelateStateSerializer).
            connection_factory: Optional custom connection factory for driver injection.
        """
        self.serde: RelateStateSerializer | Any = serde or default_serializer
        super().__init__(serde=self.serde)

        if isinstance(db_path, RelateThreadProxy):
            self.proxy = db_path
            self.db_path = db_path.db_path
            self._owns_proxy = False
        elif proxy is not None:
            self.proxy = proxy
            self.db_path = proxy.db_path
            self._owns_proxy = False
        else:
            self.db_path = str(db_path)
            self.proxy = RelateThreadProxy(
                db_path=self.db_path,
                wal_enabled=wal_enabled,
                connection_factory=connection_factory,
                auto_init_schema=True,
            )
            self._owns_proxy = True

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        """Generate a monotonically increasing version identifier for channel writes.

        Args:
            current: Current version string or None if unversioned.
            channel: Unused, maintained for backwards compatibility.

        Returns:
            Monotonically increasing version string.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            try:
                current_v = int(str(current).split(".")[0])
            except (ValueError, TypeError):
                current_v = 0

        next_v = current_v + 1
        entropy = random.random()
        return f"{next_v:032}.{entropy:016}"

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a discrete workflow checkpoint node and connect causal NEXT edge.

        Implements AC-02 by storing the checkpoint and metadata as nodes and relationships
        in the Relate graph, computing deterministic state_hash, and returning an updated
        RunnableConfig containing thread_id, checkpoint_ns, and checkpoint_id.

        Args:
            config: Configuration dictionary specifying thread_id and optional namespace.
            checkpoint: Checkpoint dictionary containing channel_values and channel_versions.
            metadata: Step execution metadata (step index, writes, source).
            new_versions: Map of updated channel version identifiers.

        Returns:
            Updated RunnableConfig containing thread_id, checkpoint_ns, and checkpoint_id.

        Raises:
            ValueError: If thread_id is missing, invalid, or empty.
            CheckpointSerializationError: If checkpoint or metadata cannot be serialized.
            RelateCheckpointError: If database persistence fails.
        """
        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id")
        if not thread_id or not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string in config['configurable']")

        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))

        checkpoint_id = checkpoint.get("id") or configurable.get("checkpoint_id")
        if not checkpoint_id or not isinstance(checkpoint_id, str):
            checkpoint_id = str(uuid.uuid4())
            checkpoint["id"] = checkpoint_id

        # Determine parent checkpoint id for causal NEXT edge
        parent_candidate = configurable.get("checkpoint_id")
        if parent_candidate == checkpoint_id:
            parent_candidate = None

        parent_checkpoint_id: str | None = None
        raw_parent = dict(checkpoint).get("parent_checkpoint_id")
        if raw_parent:
            parent_checkpoint_id = str(raw_parent)
        elif parent_candidate:
            parent_checkpoint_id = str(parent_candidate)
        elif (
            isinstance(metadata, dict)
            and "parents" in metadata
            and isinstance(metadata["parents"], dict)
            and checkpoint_ns in metadata["parents"]
        ):
            parent_checkpoint_id = str(metadata["parents"][checkpoint_ns])

        if parent_checkpoint_id == checkpoint_id:
            parent_checkpoint_id = None

        step_val = metadata.get("step") if isinstance(metadata, dict) else None
        try:
            step = int(step_val) if step_val is not None else 0
        except (ValueError, TypeError):
            step = 0

        # Serialization
        checkpoint_blob = self.serde.dumps(checkpoint)
        enriched_metadata = (
            get_checkpoint_metadata(config, metadata) if metadata is not None else {}
        )
        metadata_blob = self.serde.dumps(enriched_metadata)
        state_hash = self.serde.compute_state_hash(checkpoint)
        created_at = _utc_now_iso()

        # Atomic Cypher execution: MERGE Thread, CREATE Checkpoint, link Thread->Checkpoint, link Parent->Checkpoint
        cypher = (
            "MERGE (t:Thread {thread_id: $thread_id}) "
            "ON CREATE SET t.created_at = $created_at "
            "CREATE (c:Checkpoint {"
            "  checkpoint_id: $checkpoint_id, "
            "  thread_id: $thread_id, "
            "  checkpoint_ns: $checkpoint_ns, "
            "  parent_checkpoint_id: $parent_checkpoint_id, "
            "  step: $step, "
            "  checkpoint_blob: $checkpoint_blob, "
            "  metadata_blob: $metadata_blob, "
            "  state_hash: $state_hash, "
            "  created_at: $created_at "
            "}) "
            "CREATE (t)-[:HAS_CHECKPOINT]->(c) "
            "WITH c "
            "OPTIONAL MATCH (p:Checkpoint {checkpoint_id: $parent_checkpoint_id}) "
            "FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END | CREATE (p)-[:NEXT]->(c))"
        )
        params = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
            "parent_checkpoint_id": parent_checkpoint_id,
            "step": step,
            "checkpoint_blob": checkpoint_blob,
            "metadata_blob": metadata_blob,
            "state_hash": state_hash,
            "created_at": created_at,
        }

        self.proxy.execute_write(cypher, params)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending channel writes and link them via HAS_WRITE relationships.

        Args:
            config: Configuration dictionary specifying thread_id and checkpoint_id.
            writes: Sequence of (channel, value) write pairs.
            task_id: Identifier of the task that produced the writes.
            task_path: Optional execution path within the subgraph.
        """
        if not writes:
            return

        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id")
        if not thread_id or not isinstance(thread_id, str):
            raise ValueError("thread_id must be a non-empty string in config['configurable']")

        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = configurable.get("checkpoint_id")
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required in config['configurable'] to store writes")

        created_at = _utc_now_iso()
        statements: list[tuple[str, dict[str, Any]]] = []

        for idx, (channel, value) in enumerate(writes):
            write_blob = self.serde.dumps(value)
            write_id = f"{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}"
            stmt = (
                "MERGE (w:Write {write_id: $write_id}) "
                "SET w.thread_id = $thread_id, "
                "    w.checkpoint_ns = $checkpoint_ns, "
                "    w.checkpoint_id = $checkpoint_id, "
                "    w.task_id = $task_id, "
                "    w.channel = $channel, "
                "    w.write_blob = $write_blob, "
                "    w.idx = $idx, "
                "    w.created_at = $created_at "
                "WITH w "
                "OPTIONAL MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id}) "
                "FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END | MERGE (c)-[:HAS_WRITE]->(w))"
            )
            params = {
                "write_id": write_id,
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": str(checkpoint_id),
                "task_id": str(task_id),
                "channel": str(channel),
                "write_blob": write_blob,
                "idx": idx,
                "created_at": created_at,
            }
            statements.append((stmt, params))

        self.proxy.execute_batch(statements)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve the latest or targeted checkpoint tuple with pending writes.

        Implements AC-03 (latest checkpoint for thread), AC-04 (targeted checkpoint by id),
        and AC-05 (returns None for non-existent thread or checkpoint).

        Args:
            config: Configuration dictionary specifying thread_id and optional checkpoint_id.

        Returns:
            Matching CheckpointTuple or None if not found.
        """
        if not config or "configurable" not in config:
            return None

        configurable = config["configurable"]
        thread_id = configurable.get("thread_id")
        if not thread_id or not isinstance(thread_id, str):
            return None

        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = configurable.get("checkpoint_ns")

        if checkpoint_id:
            query = (
                "MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_id: $checkpoint_id}) "
                "WHERE ($checkpoint_ns IS NULL OR c.checkpoint_ns = $checkpoint_ns) "
                "RETURN c.checkpoint_id AS checkpoint_id, "
                "       c.thread_id AS thread_id, "
                "       c.checkpoint_ns AS checkpoint_ns, "
                "       c.parent_checkpoint_id AS parent_checkpoint_id, "
                "       c.step AS step, "
                "       c.checkpoint_blob AS checkpoint_blob, "
                "       c.metadata_blob AS metadata_blob, "
                "       c.state_hash AS state_hash, "
                "       c.created_at AS created_at "
                "LIMIT 1"
            )
            params: dict[str, Any] = {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
            }
        else:
            query = (
                "MATCH (c:Checkpoint {thread_id: $thread_id}) "
                "WHERE ($checkpoint_ns IS NULL OR c.checkpoint_ns = $checkpoint_ns) "
                "RETURN c.checkpoint_id AS checkpoint_id, "
                "       c.thread_id AS thread_id, "
                "       c.checkpoint_ns AS checkpoint_ns, "
                "       c.parent_checkpoint_id AS parent_checkpoint_id, "
                "       c.step AS step, "
                "       c.checkpoint_blob AS checkpoint_blob, "
                "       c.metadata_blob AS metadata_blob, "
                "       c.state_hash AS state_hash, "
                "       c.created_at AS created_at "
                "ORDER BY c.step DESC, c.created_at DESC "
                "LIMIT 1"
            )
            params = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }

        try:
            records = self.proxy.execute_read(query, params)
        except RelateCheckpointError:
            return None

        if not records:
            return None

        row = records[0]
        return self._format_checkpoint_tuple(row)

    def _format_checkpoint_tuple(self, row: dict[str, Any]) -> CheckpointTuple:
        """Construct a CheckpointTuple from a database row mapping."""
        thread_id = str(row["thread_id"])
        checkpoint_id = str(row["checkpoint_id"])
        checkpoint_ns = str(row.get("checkpoint_ns", ""))
        parent_checkpoint_id = row.get("parent_checkpoint_id")
        if parent_checkpoint_id is not None:
            parent_checkpoint_id = str(parent_checkpoint_id)

        checkpoint_bytes = _extract_bytes(row["checkpoint_blob"])
        checkpoint = self.serde.loads(checkpoint_bytes)

        metadata_bytes = _extract_bytes(row["metadata_blob"])
        metadata = self.serde.loads(metadata_bytes)

        # Retrieve any pending writes associated with this checkpoint
        writes_query = (
            "MATCH (w:Write {thread_id: $thread_id, checkpoint_id: $checkpoint_id}) "
            "WHERE w.checkpoint_ns = $checkpoint_ns "
            "RETURN w.task_id AS task_id, "
            "       w.channel AS channel, "
            "       w.write_blob AS write_blob, "
            "       w.idx AS idx "
            "ORDER BY w.idx ASC"
        )
        writes_params = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
        }
        writes_records = self.proxy.execute_read(writes_query, writes_params)
        pending_writes = []
        for w_row in writes_records:
            wb_bytes = _extract_bytes(w_row["write_blob"])
            val = self.serde.loads(wb_bytes)
            pending_writes.append((str(w_row["task_id"]), str(w_row["channel"]), val))

        tuple_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
            if parent_checkpoint_id
            else None
        )

        return CheckpointTuple(
            config=tuple_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | str | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoint tuples in reverse-chronological order (by step DESC / created_at DESC).

        Implements AC-06 (reverse-chronological ordering) and AC-07 (before and limit filtering).

        Args:
            config: Optional base configuration for filtering by thread_id and checkpoint_ns.
            filter: Optional key-value dictionary for metadata property filtering.
            before: Optional checkpoint identifier (as string or RunnableConfig) defining
                an upper bound; only checkpoints created strictly before this are yielded.
            limit: Maximum number of checkpoint tuples to yield.

        Yields:
            Matching CheckpointTuple instances in reverse-chronological order.
        """
        if limit is not None and limit <= 0:
            return

        configurable = config.get("configurable", {}) if config else {}
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        checkpoint_id = get_checkpoint_id(config) if config else None

        query = (
            "MATCH (c:Checkpoint) "
            "WHERE ($thread_id IS NULL OR c.thread_id = $thread_id) "
            "  AND ($checkpoint_ns IS NULL OR c.checkpoint_ns = $checkpoint_ns) "
            "  AND ($checkpoint_id IS NULL OR c.checkpoint_id = $checkpoint_id) "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.thread_id AS thread_id, "
            "       c.checkpoint_ns AS checkpoint_ns, "
            "       c.parent_checkpoint_id AS parent_checkpoint_id, "
            "       c.step AS step, "
            "       c.checkpoint_blob AS checkpoint_blob, "
            "       c.metadata_blob AS metadata_blob, "
            "       c.state_hash AS state_hash, "
            "       c.created_at AS created_at "
            "ORDER BY c.step DESC, c.created_at DESC"
        )
        params: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }

        try:
            rows = self.proxy.execute_read(query, params)
        except RelateCheckpointError:
            return

        # Extract target before checkpoint id if specified
        before_id: str | None = None
        if isinstance(before, dict):
            before_id = get_checkpoint_id(before)
        elif isinstance(before, str) and before.strip():
            before_id = before.strip()

        before_step: int | None = None
        before_created_at: str | None = None
        if before_id:
            for r in rows:
                if r["checkpoint_id"] == before_id:
                    before_step = int(r["step"])
                    before_created_at = str(r["created_at"])
                    break
            if before_step is None:
                try:
                    b_query = (
                        "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                        "RETURN c.step AS step, c.created_at AS created_at "
                        "LIMIT 1"
                    )
                    b_rows = self.proxy.execute_read(b_query, {"cid": before_id})
                    if b_rows:
                        before_step = int(b_rows[0]["step"])
                        before_created_at = str(b_rows[0]["created_at"])
                except RelateCheckpointError:
                    pass

        yielded_count = 0
        for r in rows:
            cid = str(r["checkpoint_id"])
            step = int(r["step"])
            created_at = str(r["created_at"])

            # Filter by before checkpoint boundary
            if before_id:
                if cid == before_id:
                    continue
                if before_step is not None:
                    if step > before_step:
                        continue
                    if step == before_step and before_created_at is not None and created_at >= before_created_at:
                        continue
                else:
                    if cid >= before_id:
                        continue

            metadata_bytes = _extract_bytes(r["metadata_blob"])
            metadata = self.serde.loads(metadata_bytes)

            # Filter by metadata key-values
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue

            # Construct and yield tuple
            tup = self._format_checkpoint_tuple(r)
            yield tup

            yielded_count += 1
            if limit is not None and yielded_count >= limit:
                break

    def flush(self, timeout: float | None = None) -> None:
        """Explicitly flush pending changes to disk WAL.

        Args:
            timeout: Optional maximum duration in seconds to await flush completion.
        """
        if hasattr(self.proxy, "flush"):
            self.proxy.flush(timeout=timeout)

    def abort(self) -> None:
        """Abruptly terminate checkpointer and proxy without graceful flushing."""
        if hasattr(self.proxy, "abort"):
            self.proxy.abort()
        elif self._owns_proxy:
            self.proxy.close()

    def close(self) -> None:
        """Close the checkpointer and terminate the actor worker thread if owned."""
        if self._owns_proxy:
            self.proxy.close()

    def __enter__(self) -> RelateSaver:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = ["RelateSaver"]
