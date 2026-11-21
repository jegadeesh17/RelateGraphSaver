"""Cypher-native graph forensic engine for loop detection and root-cause analysis.

Provides automated cycle identification across execution states and backward causal
path traversals from failure nodes to offending antecedent writes or tool calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.models import (
    ArgumentOrigin,
    LoopReport,
    RootCauseReport,
    ToolFailureSummary,
)
from langgraph_checkpoint_relate.saver import RelateSaver, _extract_bytes

logger = logging.getLogger(__name__)

# Suffix LangGraph's ToolNode appends to tool error messages; hidden from reports (AC-65).
_LANGGRAPH_ERROR_SUFFIX = " Please fix your mistakes."


def _is_primitive(pattern: Sequence[Any]) -> bool:
    """Return True if sequence pattern cannot be divided into smaller repeating sub-units."""
    n = len(pattern)
    if n <= 1:
        return True
    for d in range(1, n // 2 + 1):
        if n % d == 0:
            sub = pattern[:d]
            if list(sub) * (n // d) == list(pattern):
                return False
    return True


def _detect_sequence_cycles(
    items: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
    threshold: int,
) -> list[tuple[int, int, int, list[str]]]:
    """Find maximal non-overlapping consecutive repeating cycles in items.

    Args:
        items: List of checkpoint info dicts.
        key_fn: Function extracting the cycle evaluation token.
        threshold: Minimum required repetitions.

    Returns:
        List of tuples: (start_idx, end_idx, occurrence_count, cycle_nodes)
    """
    tokens = [key_fn(it) for it in items]
    n = len(tokens)
    found_cycles: list[tuple[int, int, int, list[str]]] = []

    i = 0
    while i < n:
        best_match: tuple[int, int, list[str]] | None = None
        max_l = (n - i) // threshold
        for length in range(1, max_l + 1):
            pattern = tokens[i : i + length]
            if not _is_primitive(pattern):
                continue

            k = 1
            while (
                i + (k + 1) * length <= n
                and tokens[i + k * length : i + (k + 1) * length] == pattern
            ):
                k += 1

            if k >= threshold:
                cycle_nodes = [str(items[idx]["node"]) for idx in range(i, i + length)]
                if best_match is None or (k * length > best_match[0] * len(best_match[2])):
                    best_match = (k, length, cycle_nodes)

        if best_match:
            k, length, cycle_nodes = best_match
            end_idx = i + k * length - 1
            found_cycles.append((i, end_idx, k, cycle_nodes))
            i = end_idx + 1
        else:
            i += 1

    return found_cycles


_MIN_ARGUMENT_LENGTH = 3
_MAX_SEARCH_DEPTH = 6
_UNKNOWN_NODE = "<unknown>"


def _message_contains(message: BaseMessage, needle: str, depth: int) -> bool:
    """Return whether a message's text, or an AI message's tool-call arguments, hold needle."""
    content = message.content
    if isinstance(content, str):
        if needle in content:
            return True
    else:
        for part in content:
            if isinstance(part, str):
                if needle in part:
                    return True
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and needle in text:
                    return True
    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            if _value_contains(tool_call.get("args"), needle, depth + 1):
                return True
    return False


def _value_contains(value: Any, needle: str, depth: int = 0) -> bool:
    """Case-sensitive containment search through strings, messages and nested containers."""
    if depth > _MAX_SEARCH_DEPTH:
        return False
    if isinstance(value, str):
        return needle in value
    if isinstance(value, BaseMessage):
        return _message_contains(value, needle, depth)
    if isinstance(value, dict):
        return any(_value_contains(item, needle, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_value_contains(item, needle, depth + 1) for item in value)
    return False


def _holds_tool_message(value: Any, tool_call_id: str) -> bool:
    """Return whether a write value is, or is a sequence containing, the ToolMessage for an id."""
    if isinstance(value, ToolMessage):
        return value.tool_call_id == tool_call_id
    if isinstance(value, (list, tuple)):
        return any(isinstance(m, ToolMessage) and m.tool_call_id == tool_call_id for m in value)
    return False


def _expand_causal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand rows with direct and/or write-linked tool calls into separate causal rows.

    Prevents coalescing direct and linked tool calls from distinct invocations into a single
    inconsistent row (AC-171).
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        base = {
            k: v
            for k, v in row.items()
            if not k.startswith("direct_") and not k.startswith("linked_")
        }
        has_direct = row.get("direct_call_id") is not None
        has_linked = row.get("linked_call_id") is not None

        if has_direct:
            d_row = dict(base)
            d_row["tool_call_id"] = row["direct_call_id"]
            d_row["tool_error"] = row.get("direct_error")
            d_row["tool_name"] = row.get("direct_tool_name")
            out.append(d_row)

        if has_linked:
            l_row = dict(base)
            l_row["tool_call_id"] = row["linked_call_id"]
            l_row["tool_error"] = row.get("linked_error")
            l_row["tool_name"] = row.get("linked_tool_name")
            out.append(l_row)

        if not has_direct and not has_linked:
            n_row = dict(base)
            n_row["tool_call_id"] = None
            n_row["tool_error"] = None
            n_row["tool_name"] = None
            out.append(n_row)

    return out


class RelateForensics:
    """Forensic diagnostic engine for analyzing LangGraph execution graphs in Relate."""

    def __init__(
        self,
        saver_or_db_path: RelateSaver | str | None = None,
        *,
        db_path: str | None = None,
        connection_factory: Callable[..., Any] | None = None,
        wal_enabled: bool = True,
    ) -> None:
        """Initialize RelateForensics.

        Args:
            saver_or_db_path: An existing RelateSaver instance or a string path to database.
            db_path: Optional explicit database path if saver_or_db_path is omitted.
            connection_factory: Optional custom connection factory callable.
            wal_enabled: Whether WAL is enabled for newly initialized saver.
        """
        if isinstance(saver_or_db_path, RelateSaver):
            self._saver = saver_or_db_path
            self._owns_saver = False
        elif isinstance(saver_or_db_path, str):
            self._saver = RelateSaver(
                db_path=saver_or_db_path,
                connection_factory=connection_factory,
                wal_enabled=wal_enabled,
            )
            self._owns_saver = True
        elif saver_or_db_path is None:
            resolved_path = db_path if db_path is not None else "checkpoints.rl"
            self._saver = RelateSaver(
                db_path=resolved_path,
                connection_factory=connection_factory,
                wal_enabled=wal_enabled,
            )
            self._owns_saver = True
        else:
            raise TypeError(
                f"Expected RelateSaver or str db_path, got {type(saver_or_db_path).__name__}"
            )

    @property
    def saver(self) -> RelateSaver:
        """Return the underlying RelateSaver instance."""
        return self._saver

    def close(self) -> None:
        """Close database resources if owned by this forensics engine."""
        if self._owns_saver:
            self._saver.close()

    def __enter__(self) -> RelateForensics:
        """Enter runtime context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit runtime context and close resources."""
        self.close()

    def detect_loops(
        self,
        thread_id: str,
        threshold: int = 2,
    ) -> list[LoopReport]:
        """Identify state oscillation or repeating cycle paths in the Cypher DAG.

        Implements AC-25 and AC-26 by evaluating state hashes and node execution transitions
        along the ordered checkpoint sequence for thread_id.

        Args:
            thread_id: Execution thread identifier.
            threshold: Minimum consecutive cycle occurrence threshold (default: 2).

        Returns:
            List of detected LoopReport objects detailing cycle nodes, occurrence counts,
            and checkpoint ID intervals, or empty list if no cycles detected.

        Raises:
            CheckpointNotFoundError: If thread_id does not exist or has no checkpoints.
            ValueError: If threshold is less than 1.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise CheckpointNotFoundError("thread_id must be a non-empty string.")
        if not isinstance(threshold, int) or threshold < 1:
            raise ValueError(f"threshold must be an integer >= 1, got {threshold!r}")

        # Query ordered checkpoints for thread_id
        cypher = (
            "MATCH (c:Checkpoint {thread_id: $thread_id}) "
            "OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:Write) "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.step AS step, "
            "       c.state_hash AS state_hash, "
            "       c.metadata_blob AS metadata_blob, "
            "       c.created_at AS created_at, "
            "       collect(w.task_path) AS task_paths, "
            "       collect(w.channel) AS channels "
            "ORDER BY c.step ASC, c.created_at ASC"
        )
        records = self._saver.proxy.execute_read(cypher, {"thread_id": thread_id})
        if not records:
            raise CheckpointNotFoundError(
                f"Thread '{thread_id}' not found or contains no checkpoints."
            )

        # Parse records and extract node names and state hashes
        items: list[dict[str, Any]] = []
        for r in records:
            cid = str(r["checkpoint_id"])
            step = int(r.get("step", 0))
            state_hash = str(r.get("state_hash", ""))
            created_at = str(r.get("created_at", ""))

            node_name = ""
            raw_md = r.get("metadata_blob")
            if raw_md:
                try:
                    md_bytes = _extract_bytes(raw_md)
                    meta = self._saver.serde.loads(md_bytes)
                    if isinstance(meta, dict):
                        candidate = meta.get("langgraph_node") or meta.get("node")
                        if candidate:
                            node_name = str(candidate)
                        elif meta.get("source") and meta.get("source") != "loop":
                            node_name = str(meta.get("source"))
                except Exception:
                    logger.debug("Failed to deserialize metadata for checkpoint %s", cid)

            # If node_name is not yet resolved, inspect task_paths from writes
            if not node_name:
                task_paths = r.get("task_paths") or []
                for tp in task_paths:
                    if tp:
                        parts = [p.strip() for p in str(tp).split(",") if p.strip()]
                        if parts:
                            cand = parts[-1]
                            if cand and cand != "loop":
                                node_name = cand
                                break

            # If still not resolved, inspect channels from writes
            if not node_name:
                channels = r.get("channels") or []
                for ch in channels:
                    if ch and str(ch).startswith("branch:to:"):
                        node_name = str(ch).split("branch:to:")[-1]
                        break

            # Fallbacks: step -1 is input/start
            if not node_name:
                if step == -1:
                    node_name = "__start__"
                else:
                    node_name = f"step_{step}"

            items.append(
                {
                    "checkpoint_id": cid,
                    "step": step,
                    "state_hash": state_hash,
                    "node": node_name,
                    "created_at": created_at,
                }
            )

        # 1. Detect cycles in node execution sequence
        node_cycles = _detect_sequence_cycles(
            items,
            key_fn=lambda it: it["node"],
            threshold=threshold,
        )

        covered_ranges: set[tuple[int, int]] = set()
        reports: list[LoopReport] = []

        for start_idx, end_idx, k, cycle_nodes in node_cycles:
            covered_ranges.add((start_idx, end_idx))
            reports.append(
                LoopReport(
                    thread_id=thread_id,
                    cycle_nodes=cycle_nodes,
                    occurrence_count=k,
                    checkpoint_id_range=(
                        items[start_idx]["checkpoint_id"],
                        items[end_idx]["checkpoint_id"],
                    ),
                )
            )

        # 2. Detect cycles in state_hash sequence (for state oscillation where node names may be static)
        hash_cycles = _detect_sequence_cycles(
            items,
            key_fn=lambda it: it["state_hash"],
            threshold=threshold,
        )

        for start_idx, end_idx, k, cycle_nodes in hash_cycles:
            # Avoid duplicate reporting if already covered by node cycles
            if any(s <= start_idx and end_idx <= e for s, e in covered_ranges):
                continue
            reports.append(
                LoopReport(
                    thread_id=thread_id,
                    cycle_nodes=cycle_nodes,
                    occurrence_count=k,
                    checkpoint_id_range=(
                        items[start_idx]["checkpoint_id"],
                        items[end_idx]["checkpoint_id"],
                    ),
                )
            )

        return reports

    def _loads_blob(self, raw: Any, what: str) -> Any:
        """Deserialize a stored blob, returning None (and logging) when it cannot be read."""
        try:
            return self._saver.serde.loads(_extract_bytes(raw))
        except Exception as exc:
            logger.debug("Skipping undeserializable %s: %s", what, exc)
            return None

    def _writing_nodes(self, checkpoint_id: str, thread_id: str) -> tuple[str, ...]:
        """Name the node(s) whose versions_seen entry changes after a checkpoint (ADR-0018).

        Returns an empty tuple when the writing node(s) cannot be determined.
        """
        rows = self._saver.proxy.execute_read(
            "MATCH (a:Checkpoint {checkpoint_id: $id})-[:NEXT]->"
            "(b:Checkpoint {thread_id: $thread_id}) "
            "RETURN a.checkpoint_blob AS before_blob, b.checkpoint_blob AS after_blob "
            "ORDER BY b.step ASC LIMIT 1",
            {"id": checkpoint_id, "thread_id": thread_id},
        )
        if not rows:
            return ()
        before = self._loads_blob(rows[0].get("before_blob"), "checkpoint")
        after = self._loads_blob(rows[0].get("after_blob"), "checkpoint")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return ()
        seen_before = before.get("versions_seen")
        seen_after = after.get("versions_seen")
        if not isinstance(seen_before, dict) or not isinstance(seen_after, dict):
            return ()
        changed = sorted(
            str(node) for node in seen_after if seen_after[node] != seen_before.get(node)
        )
        return tuple(changed)

    def _argument_origin(
        self, thread_id: str, error_checkpoint_id: str, args: Mapping[str, Any]
    ) -> tuple[ArgumentOrigin, tuple[str, ...]] | None:
        """Find the earliest write holding a tool call's argument, returning origin and origin nodes."""
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise CheckpointNotFoundError("thread_id must be a non-empty string.")
        if not isinstance(error_checkpoint_id, str) or not error_checkpoint_id.strip():
            raise CheckpointNotFoundError("error_checkpoint_id must be a non-empty string.")
        if not isinstance(args, Mapping):
            return None
        candidates = [
            (str(name), value)
            for name, value in args.items()
            if isinstance(value, str) and len(value) >= _MIN_ARGUMENT_LENGTH
        ]
        if not candidates:
            return None

        rows = self._saver.proxy.execute_read(
            "MATCH path = (err:Checkpoint {checkpoint_id: $id})<-[:NEXT*0..]-"
            "(a:Checkpoint {thread_id: $thread_id})-[:HAS_WRITE]->(w:Write) "
            "RETURN a.checkpoint_id AS checkpoint_id, a.step AS step, length(path) AS hops, "
            "       w.channel AS channel, w.idx AS idx, w.write_blob AS write_blob",
            {"id": error_checkpoint_id, "thread_id": thread_id},
        )
        rows = sorted(rows, key=lambda r: (int(r.get("step") or 0), int(r.get("idx") or 0)))
        for row in rows:
            channel = str(row.get("channel") or "")
            if channel.startswith(("__", "branch:")) or not row.get("write_blob"):
                continue
            value = self._loads_blob(row["write_blob"], f"write on channel '{channel}'")
            if value is None:
                continue
            for name, needle in candidates:
                if _value_contains(value, needle):
                    checkpoint_id = str(row["checkpoint_id"])
                    nodes = self._writing_nodes(checkpoint_id, thread_id)
                    node = ", ".join(nodes) or _UNKNOWN_NODE
                    origin = ArgumentOrigin(
                        argument=name,
                        value=needle,
                        node=node,
                        step=int(row.get("step") or 0),
                        # length(path) also counts the closing [:HAS_WRITE] edge.
                        hops=max(int(row.get("hops") or 1) - 1, 0),
                        checkpoint_id=checkpoint_id,
                    )
                    return origin, nodes
        return None

    def trace_argument_origin(
        self, thread_id: str, error_checkpoint_id: str, args: Mapping[str, Any]
    ) -> ArgumentOrigin | None:
        """Find the earliest write that held a string argument of a failing tool call.

        Only top-level string arguments of at least 3 characters are traced, by
        case-sensitive containment; the earliest matching write on the ``[:NEXT]`` path
        ending at ``error_checkpoint_id`` wins (ADR-0018). This is a heuristic, not
        data-flow tracking.

        Args:
            thread_id: Execution thread identifier.
            error_checkpoint_id: Checkpoint where the failure was found; hops count back from it.
            args: The failing tool call's arguments.

        Returns:
            The origin, or None when no argument is traceable or no write holds one.

        Raises:
            CheckpointNotFoundError: If thread_id or error_checkpoint_id is not a non-empty string.
        """
        res = self._argument_origin(thread_id, error_checkpoint_id, args)
        return res[0] if res is not None else None

    def _failure_checkpoint_id(
        self, thread_id: str, error_checkpoint_id: str, tool_call_id: str
    ) -> str:
        """Return the first checkpoint at or after the error whose writes hold the tool's result."""
        rows = self._saver.proxy.execute_read(
            "MATCH path = (err:Checkpoint {checkpoint_id: $id})-[:NEXT*0..]->"
            "(c:Checkpoint {thread_id: $thread_id})-[:HAS_WRITE]->(w:Write) "
            "RETURN c.checkpoint_id AS checkpoint_id, length(path) AS distance, "
            "       w.channel AS channel, w.write_blob AS write_blob",
            {"id": error_checkpoint_id, "thread_id": thread_id},
        )
        for row in sorted(rows, key=lambda r: int(r.get("distance") or 0)):
            channel = str(row.get("channel") or "")
            if channel.startswith("__") or not row.get("write_blob"):
                continue
            value = self._loads_blob(row["write_blob"], f"write on channel '{channel}'")
            if _holds_tool_message(value, tool_call_id):
                return str(row["checkpoint_id"])
        return error_checkpoint_id

    def _trace_failing_call_origin(
        self,
        thread_id: str,
        error_checkpoint_id: str,
        tool_call_id: str,
        *,
        anchored: bool = False,
    ) -> ArgumentOrigin | None:
        """Trace the arguments of a recorded tool call back to the write that first held one.

        ``anchored`` means ``error_checkpoint_id`` is already the failure checkpoint.
        """
        rows = self._saver.proxy.execute_read(
            "MATCH (tc:ToolCall {call_id: $call_id}) RETURN tc.args_blob AS args_blob",
            {"call_id": tool_call_id},
        )
        if not rows:
            return None
        res = self._trace_call_args_origin(
            thread_id,
            error_checkpoint_id,
            tool_call_id,
            rows[0].get("args_blob"),
            anchored=anchored,
        )
        return res[0] if res is not None else None

    def _trace_call_args_origin(
        self,
        thread_id: str,
        error_checkpoint_id: str,
        tool_call_id: str,
        args_blob: Any,
        *,
        anchored: bool = False,
    ) -> tuple[ArgumentOrigin, tuple[str, ...]] | None:
        """Trace stored arguments from the checkpoint where the tool's result was written.

        ``anchored`` means ``error_checkpoint_id`` is already that checkpoint, so it is not
        resolved again.
        """
        if not args_blob:
            return None
        args = self._loads_blob(args_blob, f"arguments of call '{tool_call_id}'")
        if not isinstance(args, dict):
            return None
        start = (
            error_checkpoint_id
            if anchored
            else self._failure_checkpoint_id(thread_id, error_checkpoint_id, tool_call_id)
        )
        return self._argument_origin(thread_id, start, args)

    def failure_summary(self) -> list[ToolFailureSummary]:
        """Summarize failing tool calls across every thread in the store (AC-123).

        Counts distinct ``(thread_id, call_id)`` pairs per tool and traces each failed call's
        argument origin as ``trace_root_cause`` does. Runs one trace per failed call, so it
        suits demo-sized stores. Tools that never failed are omitted; the result is ordered
        by failures descending, then tool name.
        """
        # A call is linked either from its checkpoint or through one of that checkpoint's
        # writes (the reserved edge trace_root_cause also honours). Two reads with the same
        # columns, concatenated, feed the one dedupe below.
        return_columns = (
            "RETURN c.thread_id AS thread_id, c.checkpoint_id AS checkpoint_id, "
            "       tc.call_id AS call_id, tc.tool_name AS tool_name, "
            "       tc.error AS error, tc.args_blob AS args_blob"
        )
        rows = [
            *self._saver.proxy.execute_read(
                "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) " + return_columns
            ),
            *self._saver.proxy.execute_read(
                "MATCH (c:Checkpoint)-[:HAS_WRITE]->(:Write)-[:INVOKED]->(tc:ToolCall) "
                + return_columns
            ),
        ]
        # One entry per (thread, call); the earliest invoking checkpoint is the error checkpoint.
        calls: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sorted(rows, key=lambda r: str(r.get("checkpoint_id") or "")):
            thread_id = str(row.get("thread_id") or "")
            call_id = str(row.get("call_id") or "")
            calls.setdefault((thread_id, call_id), row)

        totals: dict[str, int] = {}
        failed_threads: dict[str, set[str]] = {}
        failures: dict[str, int] = {}
        origin_nodes: dict[str, set[str]] = {}
        for (thread_id, call_id), row in sorted(calls.items()):
            tool_name = str(row.get("tool_name") or "<unknown>")
            totals[tool_name] = totals.get(tool_name, 0) + 1
            if not row.get("error"):
                continue
            failures[tool_name] = failures.get(tool_name, 0) + 1
            failed_threads.setdefault(tool_name, set()).add(thread_id)
            res = self._trace_call_args_origin(
                thread_id, str(row["checkpoint_id"]), call_id, row.get("args_blob")
            )
            if res is not None:
                origin, nodes = res
                if nodes:
                    for n in nodes:
                        origin_nodes.setdefault(tool_name, set()).add(n)
                else:
                    origin_nodes.setdefault(tool_name, set()).add(_UNKNOWN_NODE)

        summaries = [
            ToolFailureSummary(
                tool_name=name,
                failures=count,
                total_calls=totals[name],
                threads=tuple(sorted(failed_threads[name])),
                origin_nodes=tuple(sorted(origin_nodes.get(name, ()))),
            )
            for name, count in failures.items()
        ]
        return sorted(summaries, key=lambda s: (-s.failures, s.tool_name))

    def trace_root_cause(self, thread_id: str) -> RootCauseReport:
        """Trace causal antecedent edges from an error checkpoint to offending write or tool call.

        Implements AC-27 and AC-28 by locating failure checkpoints and traversing backwards
        across causal [:NEXT], [:HAS_WRITE], and [:INVOKED] edges.

        Args:
            thread_id: Execution thread identifier.

        Returns:
            RootCauseReport detailing failure origins, or indicating no error detected.

        Raises:
            CheckpointNotFoundError: If thread_id does not exist or has no checkpoints.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise CheckpointNotFoundError("thread_id must be a non-empty string.")

        # Verify thread exists
        check_query = (
            "MATCH (c:Checkpoint {thread_id: $thread_id}) "
            "RETURN c.checkpoint_id AS checkpoint_id "
            "LIMIT 1"
        )
        check_records = self._saver.proxy.execute_read(check_query, {"thread_id": thread_id})
        if not check_records:
            raise CheckpointNotFoundError(
                f"Thread '{thread_id}' not found or contains no checkpoints."
            )

        # 1. Look for error checkpoint via __error__ write
        error_write_query = (
            "MATCH (c:Checkpoint {thread_id: $thread_id})-[:HAS_WRITE]->(w:Write) "
            "WHERE w.channel = '__error__' "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.step AS step, "
            "       w.write_id AS write_id, "
            "       w.write_blob AS write_blob, "
            "       c.metadata_blob AS metadata_blob "
            "ORDER BY c.step DESC, c.created_at DESC "
            "LIMIT 1"
        )
        error_records = self._saver.proxy.execute_read(error_write_query, {"thread_id": thread_id})

        # 2. Look for error checkpoint via tool call error
        if not error_records:
            return_columns = (
                "RETURN c.checkpoint_id AS checkpoint_id, "
                "       c.step AS step, "
                "       c.created_at AS created_at, "
                "       tc.call_id AS tool_call_id, "
                "       tc.error AS tool_error, "
                "       c.metadata_blob AS metadata_blob"
            )
            tool_rows = [
                *self._saver.proxy.execute_read(
                    "MATCH (c:Checkpoint {thread_id: $thread_id})-[:INVOKED]->(tc:ToolCall) "
                    "WHERE tc.error IS NOT NULL AND tc.error <> '' " + return_columns,
                    {"thread_id": thread_id},
                ),
                *self._saver.proxy.execute_read(
                    "MATCH (c:Checkpoint {thread_id: $thread_id})-[:HAS_WRITE]->(:Write)-[:INVOKED]->(tc:ToolCall) "
                    "WHERE tc.error IS NOT NULL AND tc.error <> '' " + return_columns,
                    {"thread_id": thread_id},
                ),
            ]
            if tool_rows:
                best = max(
                    tool_rows,
                    key=lambda r: (int(r.get("step") or 0), str(r.get("created_at") or "")),
                )
                error_records = [best]

        # 3. Look for error checkpoint via metadata tags/status
        if not error_records:
            all_cp_query = (
                "MATCH (c:Checkpoint {thread_id: $thread_id}) "
                "RETURN c.checkpoint_id AS checkpoint_id, "
                "       c.step AS step, "
                "       c.metadata_blob AS metadata_blob "
                "ORDER BY c.step DESC, c.created_at DESC"
            )
            all_cps = self._saver.proxy.execute_read(all_cp_query, {"thread_id": thread_id})
            for cp in all_cps:
                raw_md = cp.get("metadata_blob")
                if raw_md:
                    try:
                        md = self._saver.serde.loads(_extract_bytes(raw_md))
                        if isinstance(md, dict):
                            if (
                                md.get("error")
                                or md.get("exception")
                                or md.get("status") in ("error", "failed")
                                or "error" in md.get("tags", [])
                            ):
                                error_records = [cp]
                                break
                    except Exception:
                        pass

        # AC-28: No error detected in execution history
        if not error_records:
            return RootCauseReport(
                thread_id=thread_id,
                error_checkpoint_id="",
                error_message="No error detected",
                root_tool_call_id=None,
                root_channel_write=None,
                causal_path_length=0,
                diagnostic_summary="Workflow completed without errors.",
            )

        # Failure identified; extract error details
        err_rec = error_records[0]
        error_checkpoint_id = str(err_rec["checkpoint_id"])
        failed_call_id = err_rec.get("tool_call_id")
        if failed_call_id:
            # The saver attaches [:INVOKED] to whichever of the requesting step's writes or the
            # next checkpoint it saves first, and LangGraph saves them on different threads, so
            # the edge can sit on either. Anchor on the checkpoint where the tool failed, which
            # is the same in both cases, so the report does not depend on that race.
            error_checkpoint_id = self._failure_checkpoint_id(
                thread_id, error_checkpoint_id, str(failed_call_id)
            )

        error_message = ""
        if "write_blob" in err_rec and err_rec["write_blob"]:
            try:
                err_val = self._saver.serde.loads(_extract_bytes(err_rec["write_blob"]))
                if isinstance(err_val, dict):
                    error_message = str(
                        err_val.get("message")
                        or err_val.get("error")
                        or err_val.get("exception")
                        or err_val
                    )
                else:
                    error_message = str(err_val)
            except Exception:
                error_message = "Error payload deserialization failure"
        elif "tool_error" in err_rec and err_rec["tool_error"]:
            error_message = str(err_rec["tool_error"])
        elif "metadata_blob" in err_rec and err_rec["metadata_blob"]:
            try:
                md = self._saver.serde.loads(_extract_bytes(err_rec["metadata_blob"]))
                if isinstance(md, dict):
                    error_message = str(
                        md.get("error")
                        or md.get("exception")
                        or md.get("message")
                        or "Workflow execution failed"
                    )
            except Exception:
                pass

        if error_message.endswith(_LANGGRAPH_ERROR_SUFFIX):
            error_message = error_message.removesuffix(_LANGGRAPH_ERROR_SUFFIX).rstrip()

        if not error_message:
            error_message = f"Execution error at checkpoint {error_checkpoint_id}"

        # 4. Backward causal path traversal across [:NEXT], [:HAS_WRITE], [:INVOKED].
        # The variable-length pattern starts from the uniquely identified error checkpoint
        # and walks NEXT edges backwards; the reverse direction (antecedent first) makes
        # the engine expand from every checkpoint in the database.
        causal_query = (
            "MATCH path = (err:Checkpoint {checkpoint_id: $error_checkpoint_id})<-[:NEXT*0..]-(antecedent:Checkpoint {thread_id: $thread_id}) "
            "OPTIONAL MATCH (antecedent)-[:HAS_WRITE]->(w:Write) "
            "OPTIONAL MATCH (antecedent)-[:INVOKED]->(tc1:ToolCall) "
            "OPTIONAL MATCH (w)-[:INVOKED]->(tc2:ToolCall) "
            "RETURN antecedent.checkpoint_id AS checkpoint_id, "
            "       antecedent.step AS step, "
            "       length(path) AS distance_to_error, "
            "       w.write_id AS write_id, "
            "       w.channel AS channel, "
            "       w.task_id AS task_id, "
            "       w.write_blob AS write_blob, "
            "       tc1.call_id AS direct_call_id, "
            "       tc1.error AS direct_error, "
            "       tc1.tool_name AS direct_tool_name, "
            "       tc2.call_id AS linked_call_id, "
            "       tc2.error AS linked_error, "
            "       tc2.tool_name AS linked_tool_name "
            "ORDER BY distance_to_error ASC"
        )
        causal_rows = self._saver.proxy.execute_read(
            causal_query,
            {
                "thread_id": thread_id,
                "error_checkpoint_id": error_checkpoint_id,
            },
        )
        causal_rows = _expand_causal_rows(causal_rows)

        root_tool_call_id: str | None = None
        root_tool_name: str | None = None
        root_channel_write: str | None = None
        causal_path_length: int = 0

        # Prioritized search for offending antecedent:
        # Priority 1: ToolCall with explicit error
        for r in causal_rows:
            if r.get("tool_call_id") and r.get("tool_error"):
                root_tool_call_id = str(r["tool_call_id"])
                root_tool_name = str(r["tool_name"]) if r.get("tool_name") else None
                root_channel_write = str(r["write_id"]) if r.get("write_id") else None
                causal_path_length = int(r.get("distance_to_error", 0))
                break

        # Priority 2: Write with poison or error indicator (non-__error__)
        if not root_channel_write and not root_tool_call_id:
            for r in causal_rows:
                channel = str(r.get("channel", ""))
                if channel and channel != "__error__":
                    # Check write_blob for poison indicators
                    is_poison = "poison" in channel.lower()
                    if not is_poison and r.get("write_blob"):
                        try:
                            val = self._saver.serde.loads(_extract_bytes(r["write_blob"]))
                            val_str = str(val).lower()
                            if "poison" in val_str or "fault" in val_str:
                                is_poison = True
                        except Exception:
                            pass
                    if is_poison:
                        root_channel_write = str(r["write_id"])
                        root_tool_call_id = (
                            str(r["tool_call_id"]) if r.get("tool_call_id") else None
                        )
                        root_tool_name = str(r["tool_name"]) if r.get("tool_name") else None
                        causal_path_length = int(r.get("distance_to_error", 0))
                        break

        # Priority 3: Upstream tool invocation preceding failure
        if not root_channel_write and not root_tool_call_id:
            for r in causal_rows:
                if r.get("tool_call_id") and int(r.get("distance_to_error", 0)) > 0:
                    root_tool_call_id = str(r["tool_call_id"])
                    root_tool_name = str(r["tool_name"]) if r.get("tool_name") else None
                    root_channel_write = str(r["write_id"]) if r.get("write_id") else None
                    causal_path_length = int(r.get("distance_to_error", 0))
                    break

        # Priority 4: Nearest antecedent channel write preceding failure
        if not root_channel_write and not root_tool_call_id:
            for r in causal_rows:
                channel = str(r.get("channel", ""))
                dist = int(r.get("distance_to_error", 0))
                if channel and channel != "__error__" and dist > 0:
                    root_channel_write = str(r["write_id"])
                    root_tool_call_id = str(r["tool_call_id"]) if r.get("tool_call_id") else None
                    root_tool_name = str(r["tool_name"]) if r.get("tool_name") else None
                    causal_path_length = dist
                    break

        # Priority 5: Any non-error write at the failure node itself
        if not root_channel_write and not root_tool_call_id:
            for r in causal_rows:
                channel = str(r.get("channel", ""))
                if channel and channel != "__error__":
                    root_channel_write = str(r["write_id"])
                    root_tool_call_id = str(r["tool_call_id"]) if r.get("tool_call_id") else None
                    root_tool_name = str(r["tool_name"]) if r.get("tool_name") else None
                    causal_path_length = int(r.get("distance_to_error", 0))
                    break

        # Default path length to max distance traversed if not set
        if causal_path_length == 0 and causal_rows:
            causal_path_length = max(int(r.get("distance_to_error", 0)) for r in causal_rows)

        # Construct diagnostic summary
        parts: list[str] = [
            f"Failure at checkpoint '{error_checkpoint_id}' (error: {error_message})."
        ]
        if root_tool_call_id:
            parts.append(f"Root tool invocation: '{root_tool_call_id}'.")
            if root_tool_name:
                parts.append(f"Failing tool: '{root_tool_name}'.")
        if root_channel_write:
            parts.append(f"Offending antecedent write: '{root_channel_write}'.")
        parts.append(f"Causal path distance: {causal_path_length} step(s).")
        diagnostic_summary = " ".join(parts)

        origin: ArgumentOrigin | None = None
        if root_tool_call_id is not None and any(
            r.get("tool_call_id") == root_tool_call_id and r.get("tool_error") for r in causal_rows
        ):
            # error_checkpoint_id was anchored above for failed_call_id; reuse it when the
            # origin trace is for that same call (ADR-0021).
            origin = self._trace_failing_call_origin(
                thread_id,
                error_checkpoint_id,
                root_tool_call_id,
                anchored=bool(failed_call_id) and root_tool_call_id == str(failed_call_id),
            )

        return RootCauseReport(
            thread_id=thread_id,
            error_checkpoint_id=error_checkpoint_id,
            error_message=error_message,
            root_tool_call_id=root_tool_call_id,
            root_channel_write=root_channel_write,
            causal_path_length=causal_path_length,
            diagnostic_summary=diagnostic_summary,
            root_tool_name=root_tool_name,
            origin_argument=origin.argument if origin else None,
            origin_node=origin.node if origin else None,
            origin_step=origin.step if origin else None,
            origin_hops=origin.hops if origin else None,
        )


__all__ = ["RelateForensics"]
