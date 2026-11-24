"""In-memory sandboxed time-travel replay engine for LangGraph execution graphs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langgraph_checkpoint_relate.errors import (
    CheckpointNotFoundError,
    RelateCheckpointError,
)
from langgraph_checkpoint_relate.saver import _extract_bytes, _utc_now_iso
from langgraph_checkpoint_relate.tool_calls import extract_tool_events

if TYPE_CHECKING:
    from langgraph_checkpoint_relate.models import ReplayResult
    from langgraph_checkpoint_relate.replay import GraphFactory
    from langgraph_checkpoint_relate.saver import RelateSaver

logger = logging.getLogger(__name__)


class TimeTravelEngine:
    """In-memory time-travel and replay engine for LangGraph execution graphs."""

    def __init__(self, source_saver: RelateSaver | None = None) -> None:
        """Initialize engine with an optional default source saver.

        Args:
            source_saver: Optional default RelateSaver checkpointer instance.
        """
        self.source_saver = source_saver

    @staticmethod
    def fork_in_memory(
        source_saver: RelateSaver,
        thread_id: str,
        checkpoint_id: str,
        branch_thread_id: str | None = None,
    ) -> RelateSaver:
        """Clone history up to checkpoint_id into an isolated :memory: database.

        Implements AC-19 by extracting the ancestor lineage of the target checkpoint
        and populating a fresh in-memory database with cloned Thread, Checkpoint, Write and
        ToolCall nodes and relationships (ADR-0010). A copied ToolCall keeps its outcome only
        when the tool's result message is already in the fork-point checkpoint's channel values.

        Args:
            source_saver: Active RelateSaver holding the durable execution graph.
            thread_id: Identifier of the source execution thread.
            checkpoint_id: Target checkpoint identifier to replay from.
            branch_thread_id: Optional new thread identifier for the replay branch.

        Returns:
            An isolated in-memory RelateSaver instance.

        Raises:
            CheckpointNotFoundError: If checkpoint_id does not exist for thread_id.
            RelateCheckpointError: If any source read fails, or the fork-point checkpoint
                cannot be deserialized.
        """
        # Validate target checkpoint existence and fetch candidates ordered by step ASC
        candidates_query = (
            "MATCH (c:Checkpoint {thread_id: $thread_id}) "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.thread_id AS thread_id, "
            "       c.checkpoint_ns AS checkpoint_ns, "
            "       c.parent_checkpoint_id AS parent_checkpoint_id, "
            "       c.step AS step, "
            "       c.checkpoint_blob AS checkpoint_blob, "
            "       c.metadata_blob AS metadata_blob, "
            "       c.state_hash AS state_hash, "
            "       c.created_at AS created_at "
            "ORDER BY c.step ASC, c.created_at ASC"
        )
        records = source_saver.proxy.execute_read(candidates_query, {"thread_id": thread_id})

        candidate_ids = [str(r["checkpoint_id"]) for r in records]
        if checkpoint_id not in candidate_ids:
            raise CheckpointNotFoundError(
                f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'. "
                f"Available checkpoints: {candidate_ids}"
            )

        checkpoints_by_id = {str(r["checkpoint_id"]): r for r in records}

        # Query ancestor lineage up to checkpoint_id along causal path
        lineage_map: dict[str, dict[str, Any]] = {}
        curr_id: str | None = checkpoint_id

        while curr_id and curr_id not in lineage_map:
            record = checkpoints_by_id.get(curr_id)
            if record is None:
                parent_query = (
                    "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
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
                p_records = source_saver.proxy.execute_read(parent_query, {"cid": curr_id})
                if p_records:
                    record = p_records[0]
                    checkpoints_by_id[curr_id] = record

            if record is None:
                break

            lineage_map[curr_id] = record
            parent_id = record.get("parent_checkpoint_id")
            curr_id = str(parent_id) if parent_id is not None else None

        ancestor_checkpoints = sorted(
            lineage_map.values(),
            key=lambda x: (int(x.get("step", 0)), str(x.get("created_at", ""))),
        )
        ancestor_ids = [str(c["checkpoint_id"]) for c in ancestor_checkpoints]

        # Query associated Write nodes connected via [:HAS_WRITE]
        writes_query = (
            "MATCH (c:Checkpoint)-[:HAS_WRITE]->(w:Write) "
            "WHERE c.checkpoint_id IN $ids "
            "RETURN w.write_id AS write_id, "
            "       w.thread_id AS thread_id, "
            "       w.checkpoint_ns AS checkpoint_ns, "
            "       w.checkpoint_id AS checkpoint_id, "
            "       w.task_id AS task_id, "
            "       w.task_path AS task_path, "
            "       w.channel AS channel, "
            "       w.write_blob AS write_blob, "
            "       w.idx AS idx, "
            "       w.created_at AS created_at "
            "ORDER BY w.idx ASC"
        )
        writes_records = source_saver.proxy.execute_read(writes_query, {"ids": ancestor_ids})

        # Query tool calls invoked from lineage checkpoints. A failure propagates: a fork
        # without its tool calls would make forensics on it differ silently from the source.
        tool_calls_query = (
            "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) WHERE c.checkpoint_id IN $ids "
            "RETURN c.checkpoint_id AS checkpoint_id, tc.call_id AS call_id, "
            "       tc.tool_name AS tool_name, tc.args_blob AS args_blob, "
            "       tc.result_blob AS result_blob, tc.error AS error, "
            "       tc.created_at AS created_at"
        )
        tool_call_records = source_saver.proxy.execute_read(tool_calls_query, {"ids": ancestor_ids})

        # Outcomes count only if the tool's result message is already in the fork point's
        # channel values (ADR-0010).
        fork_point = lineage_map[checkpoint_id]
        try:
            fork_checkpoint = source_saver.serde.loads(
                _extract_bytes(fork_point["checkpoint_blob"])
            )
        except Exception as exc:
            raise RelateCheckpointError(
                f"Cannot deserialize fork-point checkpoint '{checkpoint_id}' "
                f"of thread '{thread_id}': {exc}"
            ) from exc
        channel_values = (
            fork_checkpoint.get("channel_values") if isinstance(fork_checkpoint, dict) else None
        )
        fork_point_outcome_ids: set[str] = (
            {o.call_id for o in extract_tool_events(channel_values.values()).outcomes}
            if isinstance(channel_values, dict)
            else set()
        )

        # Query existing relationships between ancestor checkpoints
        edges_query = (
            "MATCH (p:Checkpoint)-[r:NEXT|FORKED_TO]->(c:Checkpoint) "
            "WHERE p.checkpoint_id IN $ids AND c.checkpoint_id IN $ids "
            "RETURN p.checkpoint_id AS source_id, type(r) AS rel_type, c.checkpoint_id AS target_id"
        )
        edges_records = source_saver.proxy.execute_read(edges_query, {"ids": ancestor_ids})

        if not edges_records:
            edges_records = []
            for chk in ancestor_checkpoints:
                pid = chk.get("parent_checkpoint_id")
                if pid and str(pid) in lineage_map:
                    edges_records.append(
                        {
                            "source_id": str(pid),
                            "rel_type": "NEXT",
                            "target_id": str(chk["checkpoint_id"]),
                        }
                    )

        # Query source thread metadata
        thread_query = (
            "MATCH (t:Thread {thread_id: $thread_id}) "
            "RETURN t.thread_id AS thread_id, t.created_at AS created_at "
            "LIMIT 1"
        )
        thread_records = source_saver.proxy.execute_read(thread_query, {"thread_id": thread_id})

        thread_created_at = (
            thread_records[0]["created_at"]
            if thread_records and thread_records[0].get("created_at")
            else _utc_now_iso()
        )

        # Create isolated in-memory saver
        from langgraph_checkpoint_relate.saver import RelateSaver

        conn_factory = getattr(source_saver, "_connection_factory", None)
        in_memory_saver = RelateSaver(
            db_path=":memory:",
            connection_factory=conn_factory,
            serde=source_saver.serde,
        )

        statements: list[tuple[str, dict[str, Any]]] = []

        # Populate Thread node
        statements.append(
            (
                "MERGE (t:Thread {thread_id: $thread_id}) ON CREATE SET t.created_at = $created_at",
                {"thread_id": thread_id, "created_at": thread_created_at},
            )
        )

        # Populate Checkpoint nodes and link (:Thread)-[:HAS_CHECKPOINT]->(:Checkpoint)
        for chk in ancestor_checkpoints:
            chk_params = {
                "thread_id": str(chk["thread_id"]),
                "checkpoint_id": str(chk["checkpoint_id"]),
                "checkpoint_ns": str(chk.get("checkpoint_ns", "")),
                "parent_checkpoint_id": (
                    str(chk["parent_checkpoint_id"])
                    if chk.get("parent_checkpoint_id") is not None
                    else None
                ),
                "step": int(chk.get("step", 0)),
                "checkpoint_blob": _extract_bytes(chk["checkpoint_blob"]),
                "metadata_blob": _extract_bytes(chk["metadata_blob"]),
                "state_hash": str(chk.get("state_hash", "")),
                "created_at": str(chk.get("created_at", "")),
            }
            chk_stmt = (
                "MERGE (t:Thread {thread_id: $thread_id}) "
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
                "CREATE (t)-[:HAS_CHECKPOINT]->(c)"
            )
            statements.append((chk_stmt, chk_params))

        # Recreate NEXT and FORKED_TO relationships
        for edge in edges_records:
            rel_type = edge.get("rel_type", "NEXT")
            if rel_type == "FORKED_TO":
                edge_stmt = (
                    "MATCH (p:Checkpoint {checkpoint_id: $source_id}), (c:Checkpoint {checkpoint_id: $target_id}) "
                    "CREATE (p)-[:FORKED_TO]->(c)"
                )
            else:
                edge_stmt = (
                    "MATCH (p:Checkpoint {checkpoint_id: $source_id}), (c:Checkpoint {checkpoint_id: $target_id}) "
                    "CREATE (p)-[:NEXT]->(c)"
                )
            statements.append(
                (edge_stmt, {"source_id": edge["source_id"], "target_id": edge["target_id"]})
            )

        # Recreate Write nodes and [:HAS_WRITE] relationships
        for w in writes_records:
            w_params = {
                "write_id": str(w["write_id"]),
                "thread_id": str(w["thread_id"]),
                "checkpoint_ns": str(w.get("checkpoint_ns", "")),
                "checkpoint_id": str(w["checkpoint_id"]),
                "task_id": str(w["task_id"]),
                "task_path": str(w.get("task_path", "")),
                "channel": str(w["channel"]),
                "write_blob": _extract_bytes(w["write_blob"]),
                "idx": int(w.get("idx", 0)),
                "created_at": str(w.get("created_at", "")),
            }
            w_stmt = (
                "MERGE (w:Write {write_id: $write_id}) "
                "SET w.thread_id = $thread_id, "
                "    w.checkpoint_ns = $checkpoint_ns, "
                "    w.checkpoint_id = $checkpoint_id, "
                "    w.task_id = $task_id, "
                "    w.task_path = $task_path, "
                "    w.channel = $channel, "
                "    w.write_blob = $write_blob, "
                "    w.idx = $idx, "
                "    w.created_at = $created_at "
                "WITH w "
                "MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id}) "
                "MERGE (c)-[:HAS_WRITE]->(w)"
            )
            statements.append((w_stmt, w_params))

        # Recreate ToolCall nodes (one per distinct call_id) and [:INVOKED] edges from
        # lineage checkpoints. Property names come from fixed lists, never from data.
        seen_call_ids: set[str] = set()
        for row in tool_call_records:
            call_id = str(row["call_id"])
            if call_id in seen_call_ids:
                continue
            seen_call_ids.add(call_id)
            props = ["tool_name", "args_blob", "created_at"]
            if call_id in fork_point_outcome_ids:
                props += ["result_blob", "error"]
            tc_params: dict[str, Any] = {"call_id": call_id}
            for prop in props:
                value = row.get(prop)
                if value is None:
                    continue
                tc_params[prop] = _extract_bytes(value) if prop.endswith("_blob") else value
            assignments = [f"tc.{prop} = ${prop}" for prop in tc_params if prop != "call_id"]
            set_clause = f" SET {', '.join(assignments)}" if assignments else ""
            statements.append(("MERGE (tc:ToolCall {call_id: $call_id})" + set_clause, tc_params))

        for row in tool_call_records:
            statements.append(
                (
                    "MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id}), "
                    "(tc:ToolCall {call_id: $call_id}) MERGE (c)-[:INVOKED]->(tc)",
                    {"checkpoint_id": str(row["checkpoint_id"]), "call_id": str(row["call_id"])},
                )
            )

        # Link target checkpoint to branch_thread_id via [:FORKED_TO] if specified
        if branch_thread_id:
            branch_params = {
                "branch_thread_id": str(branch_thread_id),
                "checkpoint_id": str(checkpoint_id),
                "created_at": _utc_now_iso(),
            }
            branch_stmt = (
                "MERGE (bt:Thread {thread_id: $branch_thread_id}) "
                "ON CREATE SET bt.created_at = $created_at "
                "WITH bt "
                "MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id}) "
                "CREATE (c)-[:FORKED_TO]->(bt)"
            )
            statements.append((branch_stmt, branch_params))

        # Execute atomic batch population
        try:
            in_memory_saver.proxy.execute_batch(statements)
        except Exception:
            in_memory_saver.close()
            raise

        return in_memory_saver

    @staticmethod
    def replay(
        source_saver: RelateSaver,
        thread_id: str,
        checkpoint_id: str,
        factory: GraphFactory,
        *,
        update: dict[str, Any] | None = None,
        as_node: str | None = None,
        set_messages: Mapping[int, str] | None = None,
    ) -> ReplayResult:
        """Replay the thread from a checkpoint in a sandbox and diff it (AC-108).

        Delegates to ``replay_from_checkpoint`` with the same arguments.
        """
        from langgraph_checkpoint_relate.replay import replay_from_checkpoint

        return replay_from_checkpoint(
            source_saver,
            thread_id,
            checkpoint_id,
            factory,
            update=update,
            as_node=as_node,
            set_messages=set_messages,
        )


def fork_in_memory(
    source_saver: RelateSaver,
    thread_id: str,
    checkpoint_id: str,
    branch_thread_id: str | None = None,
) -> RelateSaver:
    """Convenience function delegating to TimeTravelEngine.fork_in_memory."""
    return TimeTravelEngine.fork_in_memory(
        source_saver=source_saver,
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        branch_thread_id=branch_thread_id,
    )


__all__ = ["TimeTravelEngine", "fork_in_memory"]
