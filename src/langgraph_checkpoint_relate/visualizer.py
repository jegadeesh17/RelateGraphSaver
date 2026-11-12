"""Terminal-native execution tree visualizer for Relate graph checkpoints.

Provides Unicode DAG rendering of LangGraph execution lineages, branch points,
and pending writes using rich.tree.Tree.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.tree import Tree

from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.saver import RelateSaver, _extract_bytes

logger = logging.getLogger(__name__)


class DAGVisualizer:
    """Terminal execution tree visualizer for Relate checkpoint graphs."""

    def __init__(
        self,
        saver_or_db_path: RelateSaver | str | None = None,
        *,
        db_path: str | None = None,
        connection_factory: Callable[..., Any] | None = None,
        wal_enabled: bool = True,
    ) -> None:
        """Initialize DAGVisualizer.

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
            if db_path is not None and not isinstance(db_path, str):
                raise TypeError(f"db_path must be a str, got {type(db_path).__name__}")
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
        """Close database resources if owned by this visualizer."""
        if self._owns_saver:
            self._saver.close()

    def __enter__(self) -> DAGVisualizer:
        """Enter runtime context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit runtime context and close resources."""
        self.close()

    def render_tree(self, thread_id: str) -> Tree:
        """Query execution graph for thread_id and construct a rich.tree.Tree.

        Implements AC-22 and AC-23 by building a structured Unicode DAG detailing
        step sequences, node and source names, short checkpoint IDs, branch points,
        and pending writes.

        Args:
            thread_id: Execution thread identifier.

        Returns:
            A rich.tree.Tree visualizing the workflow DAG.

        Raises:
            CheckpointNotFoundError: If thread_id has no checkpoints or does not exist.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise CheckpointNotFoundError("thread_id must be a non-empty string.")

        # 1. Query all checkpoints belonging to thread_id
        checkpoints_query = (
            "MATCH (c:Checkpoint {thread_id: $thread_id}) "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.thread_id AS thread_id, "
            "       c.checkpoint_ns AS checkpoint_ns, "
            "       c.parent_checkpoint_id AS parent_checkpoint_id, "
            "       c.step AS step, "
            "       c.metadata_blob AS metadata_blob, "
            "       c.state_hash AS state_hash, "
            "       c.created_at AS created_at "
            "ORDER BY c.step ASC, c.created_at ASC"
        )
        records = self._saver.proxy.execute_read(checkpoints_query, {"thread_id": thread_id})
        if not records:
            raise CheckpointNotFoundError(
                f"Thread '{thread_id}' not found or contains no checkpoints."
            )

        # 2. Query any external checkpoints forked from this thread
        external_forks_query = (
            "MATCH (p:Checkpoint {thread_id: $thread_id})-[:FORKED_TO]->(c:Checkpoint) "
            "WHERE c.thread_id <> $thread_id "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       c.thread_id AS thread_id, "
            "       c.checkpoint_ns AS checkpoint_ns, "
            "       c.parent_checkpoint_id AS parent_checkpoint_id, "
            "       c.step AS step, "
            "       c.metadata_blob AS metadata_blob, "
            "       c.state_hash AS state_hash, "
            "       c.created_at AS created_at"
        )
        try:
            external_forks = self._saver.proxy.execute_read(
                external_forks_query, {"thread_id": thread_id}
            )
        except Exception:
            external_forks = []

        all_checkpoint_records = list(records) + list(external_forks)

        # 3. Parse checkpoint records and deserialize metadata
        parsed_checkpoints: dict[str, dict[str, Any]] = {}
        for r in all_checkpoint_records:
            cid = str(r["checkpoint_id"])
            parsed_checkpoints[cid] = self._parse_checkpoint_record(r)

        # 4. Query relationships (:NEXT and :FORKED_TO)
        edges_query = (
            "MATCH (p:Checkpoint)-[r:NEXT|FORKED_TO]->(c:Checkpoint) "
            "WHERE p.thread_id = $thread_id OR c.thread_id = $thread_id "
            "RETURN p.checkpoint_id AS parent_id, "
            "       c.checkpoint_id AS child_id, "
            "       type(r) AS rel_type, "
            "       p.thread_id AS parent_thread_id, "
            "       c.thread_id AS child_thread_id"
        )
        try:
            edge_records = self._saver.proxy.execute_read(edges_query, {"thread_id": thread_id})
        except Exception:
            edge_records = []

        # 5. Query pending writes linked via [:HAS_WRITE]
        writes_query = (
            "MATCH (c:Checkpoint)-[:HAS_WRITE]->(w:Write) "
            "WHERE c.thread_id = $thread_id "
            "RETURN c.checkpoint_id AS checkpoint_id, "
            "       w.write_id AS write_id, "
            "       w.task_id AS task_id, "
            "       w.channel AS channel, "
            "       w.idx AS idx, "
            "       w.created_at AS created_at "
            "ORDER BY w.idx ASC"
        )
        try:
            write_records = self._saver.proxy.execute_read(writes_query, {"thread_id": thread_id})
        except Exception:
            write_records = []

        writes_by_cid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for w in write_records:
            writes_by_cid[str(w["checkpoint_id"])].append(w)

        # 6. Build DAG topology
        children_map: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        incoming_edges: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)

        for edge in edge_records:
            pid = str(edge["parent_id"])
            cid = str(edge["child_id"])
            rel = str(edge["rel_type"])
            if pid in parsed_checkpoints and cid in parsed_checkpoints:
                children_map[pid].append((cid, rel))
                incoming_edges[cid].append((pid, rel))

        # Check for unrecorded parent links from parent_checkpoint_id
        for cid, c_info in parsed_checkpoints.items():
            raw_pid = c_info.get("parent_checkpoint_id")
            if raw_pid and str(raw_pid) in parsed_checkpoints:
                parent_cid = str(raw_pid)
                existing_children = [child for child, _ in children_map[parent_cid]]
                if cid not in existing_children:
                    rel = "FORKED_TO" if existing_children else "NEXT"
                    children_map[parent_cid].append((cid, rel))
                    incoming_edges[cid].append((parent_cid, rel))

        # 7. Identify root checkpoints for thread_id
        roots = [
            cid
            for cid, c in parsed_checkpoints.items()
            if c["thread_id"] == thread_id and len(incoming_edges[cid]) == 0
        ]
        roots.sort(
            key=lambda cid: (
                parsed_checkpoints[cid]["step"],
                parsed_checkpoints[cid]["created_at"],
            )
        )

        if not roots:
            # Fallback to minimum step checkpoint in case of cycles
            thread_cids = [
                cid for cid, c in parsed_checkpoints.items() if c["thread_id"] == thread_id
            ]
            roots = [
                min(
                    thread_cids,
                    key=lambda cid: (
                        parsed_checkpoints[cid]["step"],
                        parsed_checkpoints[cid]["created_at"],
                    ),
                )
            ]

        # 8. Build rich Tree
        tree = Tree(f"[bold cyan]Thread: {escape(thread_id)}[/bold cyan]")
        visited: set[str] = set()

        for root_id in roots:
            self._add_checkpoint_subtree(
                tree_node=tree,
                cid=root_id,
                parsed_checkpoints=parsed_checkpoints,
                writes_by_cid=writes_by_cid,
                children_map=children_map,
                current_thread_id=thread_id,
                visited=visited,
            )

        return tree

    def render_to_string(self, thread_id: str, console: Console | None = None) -> str:
        """Render the execution tree to a plain text or ANSI string.

        Args:
            thread_id: Execution thread identifier.
            console: Optional rich.console.Console instance to use for capture.

        Returns:
            The captured text representation of the DAG.
        """
        tree = self.render_tree(thread_id)
        target_console = console or Console(record=True, width=120)
        with target_console.capture() as capture:
            target_console.print(tree)
        return capture.get()

    def print_tree(self, thread_id: str, console: Console | None = None) -> None:
        """Print the rendered execution tree for thread_id to console.

        Args:
            thread_id: Execution thread identifier.
            console: Optional rich.console.Console instance to print to.
        """
        tree = self.render_tree(thread_id)
        target_console = console or Console()
        target_console.print(tree)

    def _parse_checkpoint_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Safely extract metadata, node, source, and step from checkpoint record."""
        cid = str(record["checkpoint_id"])
        tid = str(record.get("thread_id", ""))
        ns = str(record.get("checkpoint_ns", ""))
        parent_id = record.get("parent_checkpoint_id")
        if parent_id is not None:
            parent_id = str(parent_id)
        step = int(record.get("step", 0))
        created_at = str(record.get("created_at", ""))

        metadata: dict[str, Any] = {}
        raw_md = record.get("metadata_blob")
        if raw_md:
            try:
                md_bytes = _extract_bytes(raw_md)
                deserialized = self._saver.serde.loads(md_bytes)
                if isinstance(deserialized, dict):
                    metadata = deserialized
            except Exception:
                logger.debug("Failed to deserialize metadata for checkpoint %s", cid)
                metadata = {}

        node = metadata.get("node")
        source = metadata.get("source")
        if "step" in metadata and isinstance(metadata["step"], int):
            step = metadata["step"]

        return {
            "checkpoint_id": cid,
            "thread_id": tid,
            "checkpoint_ns": ns,
            "parent_checkpoint_id": parent_id,
            "step": step,
            "created_at": created_at,
            "metadata": metadata,
            "node": node,
            "source": source,
        }

    def _determine_status(
        self,
        info: dict[str, Any],
        writes: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Determine checkpoint status tag and rich markup.

        Returns:
            Tuple of (plain_tag, rich_markup_tag).
        """
        write_channels = {str(w.get("channel", "")) for w in writes}
        metadata = info.get("metadata", {})

        # 1. Error status
        if (
            "__error__" in write_channels
            or metadata.get("status") == "error"
            or "error" in metadata
            or any("error" in str(w.get("channel", "")).lower() for w in writes)
        ):
            return "[ERROR]", "[bold red][ERROR][/bold red]"

        # 2. Interrupt status
        if (
            "__interrupt__" in write_channels
            or metadata.get("status") == "interrupt"
            or "interrupt" in metadata
            or any("interrupt" in str(w.get("channel", "")).lower() for w in writes)
        ):
            return "[INTERRUPT]", "[bold magenta][INTERRUPT][/bold magenta]"

        # 3. Pending status (pending writes exist)
        if writes or metadata.get("status") == "pending" or metadata.get("pending") is True:
            return "[PENDING]", "[bold yellow][PENDING][/bold yellow]"

        # 4. OK status
        return "[OK]", "[bold green][OK][/bold green]"

    def _format_checkpoint_label(
        self,
        info: dict[str, Any],
        status_markup: str,
    ) -> str:
        """Format the primary checkpoint display line for the tree."""
        step = info["step"]
        cid = info["checkpoint_id"]
        short_id = cid if len(cid) <= 12 else cid[:8]
        created_at = escape(str(info["created_at"]))

        node = info.get("node")
        source = info.get("source")
        if node and source and node != source:
            node_str = f"node: {escape(str(node))} (source: {escape(str(source))})"
        elif node:
            node_str = f"node: {escape(str(node))}"
        elif source:
            node_str = f"source: {escape(str(source))}"
        else:
            node_str = "node: <unknown>"

        ns_str = f" [ns: {escape(str(info['checkpoint_ns']))}]" if info.get("checkpoint_ns") else ""

        return f"{status_markup} Step {step}{ns_str}: {node_str} (id: {escape(short_id)}) - {created_at}"

    def _add_checkpoint_subtree(
        self,
        tree_node: Tree,
        cid: str,
        parsed_checkpoints: dict[str, dict[str, Any]],
        writes_by_cid: dict[str, list[dict[str, Any]]],
        children_map: dict[str, list[tuple[str, str]]],
        current_thread_id: str,
        visited: set[str],
    ) -> None:
        """Recursively append checkpoint nodes, pending writes, and branch subtrees."""
        if cid in visited:
            tree_node.add(f"[dim red](cycle detected: {escape(cid[:8])})[/dim red]")
            return
        visited.add(cid)

        info = parsed_checkpoints[cid]
        writes = writes_by_cid.get(cid, [])
        _, status_markup = self._determine_status(info, writes)
        label = self._format_checkpoint_label(info, status_markup)

        current_node = tree_node.add(label)

        # Annotate pending writes
        for w in writes:
            channel = escape(str(w.get("channel", "")))
            task_id = escape(str(w.get("task_id", "")))
            if channel == "__error__":
                current_node.add(
                    f"[bold red]pending write:[/bold red] channel='{channel}', task_id='{task_id}'"
                )
            elif channel == "__interrupt__":
                current_node.add(
                    f"[bold magenta]pending write:[/bold magenta] channel='{channel}', task_id='{task_id}'"
                )
            else:
                current_node.add(
                    f"[yellow]pending write:[/yellow] channel='{channel}', task_id='{task_id}'"
                )

        # Recurse for children
        children = list(children_map.get(cid, []))
        children.sort(
            key=lambda item: (
                1 if item[1] == "FORKED_TO" else 0,
                parsed_checkpoints.get(item[0], {}).get("step", 0),
                parsed_checkpoints.get(item[0], {}).get("created_at", ""),
            )
        )

        for child_id, rel_type in children:
            if child_id not in parsed_checkpoints:
                continue

            if rel_type == "FORKED_TO":
                child_info = parsed_checkpoints[child_id]
                fork_header = "[bold cyan]🔀 Branch: [BRANCH][/bold cyan]"
                if child_info["thread_id"] != current_thread_id:
                    fork_header = f"[bold cyan]🔀 Branch: [BRANCH] (thread: {escape(child_info['thread_id'])})[/bold cyan]"
                branch_node = current_node.add(fork_header)
                self._add_checkpoint_subtree(
                    tree_node=branch_node,
                    cid=child_id,
                    parsed_checkpoints=parsed_checkpoints,
                    writes_by_cid=writes_by_cid,
                    children_map=children_map,
                    current_thread_id=current_thread_id,
                    visited=visited,
                )
            else:
                self._add_checkpoint_subtree(
                    tree_node=current_node,
                    cid=child_id,
                    parsed_checkpoints=parsed_checkpoints,
                    writes_by_cid=writes_by_cid,
                    children_map=children_map,
                    current_thread_id=current_thread_id,
                    visited=visited,
                )


__all__ = ["DAGVisualizer"]
