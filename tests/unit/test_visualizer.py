"""Unit tests for Unicode terminal execution tree visualizer (DAGVisualizer).

Covers AC-22, AC-23, status tag annotations, branch indicators, and error handling.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.tree import Tree

from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.visualizer import DAGVisualizer


def _create_checkpoint(
    saver: RelateSaver,
    thread_id: str,
    checkpoint_id: str,
    step: int,
    node: str | None = None,
    source: str | None = None,
    parent_checkpoint_id: str | None = None,
    checkpoint_ns: str = "",
) -> None:
    """Helper to persist a checkpoint with specified metadata and optional parent."""
    cfg: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }
    if parent_checkpoint_id is not None:
        cfg["configurable"]["checkpoint_id"] = parent_checkpoint_id

    checkpoint_data = {
        "v": 1,
        "id": checkpoint_id,
        "channel_values": {"step": step},
        "channel_versions": {"step": str(step)},
        "versions_seen": {},
        "updated_channels": ["step"],
    }
    metadata: dict[str, Any] = {"step": step}
    if node is not None:
        metadata["node"] = node
    if source is not None:
        metadata["source"] = source

    saver.put(cfg, checkpoint_data, metadata, {})  # type: ignore[arg-type]


class TestDAGVisualizer:
    """Unit test suite for DAGVisualizer execution graph tree rendering."""

    def test_render_tree_multi_step_thread_ac_22(self) -> None:
        """AC-22: Render tree for a multi-step thread detailing step sequences, node names, and IDs."""
        with RelateSaver(db_path=":memory:") as saver:
            thread_id = "thread-ac22-test"
            _create_checkpoint(
                saver, thread_id, "cp-root-01", step=1, node="__start__", source="input"
            )
            _create_checkpoint(
                saver,
                thread_id,
                "cp-agent-02",
                step=2,
                node="agent",
                source="loop",
                parent_checkpoint_id="cp-root-01",
            )
            _create_checkpoint(
                saver,
                thread_id,
                "cp-tool-03",
                step=3,
                node="tools",
                source="update",
                parent_checkpoint_id="cp-agent-02",
            )

            visualizer = DAGVisualizer(saver)
            tree = visualizer.render_tree(thread_id)
            assert isinstance(tree, Tree)

            output_str = visualizer.render_to_string(thread_id)

            # AC-22 assertions
            assert thread_id in output_str
            assert "Step 1" in output_str
            assert "Step 2" in output_str
            assert "Step 3" in output_str
            assert "node: __start__" in output_str
            assert "node: agent" in output_str
            assert "node: tools" in output_str
            assert "cp-root-01" in output_str
            assert "cp-agent-02" in output_str
            assert "cp-tool-03" in output_str
            assert "[OK]" in output_str

    def test_render_tree_branch_and_pending_writes_ac_23(self) -> None:
        """AC-23: Render tree with branches and pending writes visually distinguished with status annotations."""
        with RelateSaver(db_path=":memory:") as saver:
            thread_id = "thread-ac23-branch-test"
            _create_checkpoint(saver, thread_id, "cp-init-01", step=1, node="__start__")

            # Main continuation
            _create_checkpoint(
                saver,
                thread_id,
                "cp-agent-02",
                step=2,
                node="agent",
                parent_checkpoint_id="cp-init-01",
            )
            # Add a pending write on cp-agent-02
            saver.put_writes(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-agent-02"}},
                [("messages", {"content": "pending response"})],
                task_id="task-agent-99",
            )

            # Branch continuation from cp-init-01
            _create_checkpoint(
                saver,
                thread_id,
                "cp-branch-02",
                step=2,
                node="human_feedback",
                parent_checkpoint_id="cp-init-01",
            )

            visualizer = DAGVisualizer(saver)
            output_str = visualizer.render_to_string(thread_id)

            # AC-23 assertions: branch indicators and pending write tags
            assert "[BRANCH]" in output_str or "Branch" in output_str
            assert "[PENDING]" in output_str
            assert "messages" in output_str
            assert "task-agent-99" in output_str
            assert "human_feedback" in output_str

    def test_render_tree_interrupt_and_error_status_tags_ac_23(self) -> None:
        """AC-23: Interrupted and error checkpoints display [INTERRUPT] and [ERROR] status tags."""
        with RelateSaver(db_path=":memory:") as saver:
            # 1. Thread with interrupt
            thread_interrupt = "thread-interrupt-test"
            _create_checkpoint(saver, thread_interrupt, "cp-int-01", step=1, node="router")
            saver.put_writes(
                {"configurable": {"thread_id": thread_interrupt, "checkpoint_id": "cp-int-01"}},
                [("__interrupt__", {"reason": "awaiting user approval"})],
                task_id="task-int-01",
            )

            # 2. Thread with error
            thread_error = "thread-error-test"
            _create_checkpoint(saver, thread_error, "cp-err-01", step=1, node="executor")
            saver.put_writes(
                {"configurable": {"thread_id": thread_error, "checkpoint_id": "cp-err-01"}},
                [("__error__", {"message": "unhandled tool exception"})],
                task_id="task-err-01",
            )

            visualizer = DAGVisualizer(saver)

            int_output = visualizer.render_to_string(thread_interrupt)
            assert "[INTERRUPT]" in int_output
            assert "__interrupt__" in int_output
            assert "task-int-01" in int_output

            err_output = visualizer.render_to_string(thread_error)
            assert "[ERROR]" in err_output
            assert "__error__" in err_output
            assert "task-err-01" in err_output

    def test_render_tree_nonexistent_thread_raises_checkpoint_not_found(self) -> None:
        """Non-existent thread_id raises CheckpointNotFoundError."""
        with RelateSaver(db_path=":memory:") as saver:
            visualizer = DAGVisualizer(saver)
            with pytest.raises(CheckpointNotFoundError, match="contains no checkpoints"):
                visualizer.render_tree("non-existent-thread-id")

            with pytest.raises(CheckpointNotFoundError, match="contains no checkpoints"):
                visualizer.render_to_string("non-existent-thread-id")

    def test_render_tree_empty_or_invalid_thread_raises_checkpoint_not_found(self) -> None:
        """Empty or whitespace thread_id raises CheckpointNotFoundError."""
        with RelateSaver(db_path=":memory:") as saver:
            visualizer = DAGVisualizer(saver)
            with pytest.raises(CheckpointNotFoundError, match="non-empty string"):
                visualizer.render_tree("")

            with pytest.raises(CheckpointNotFoundError, match="non-empty string"):
                visualizer.render_tree("   ")

    def test_render_tree_formatting_robustness_with_varied_metadata(self) -> None:
        """Clean Unicode rendering without crashing on varied metadata or missing fields."""
        with RelateSaver(db_path=":memory:") as saver:
            thread_id = "thread-varied-metadata"

            # Checkpoint with unicode node name and special symbols
            _create_checkpoint(
                saver,
                thread_id,
                "cp-unicode-01",
                step=1,
                node="🤖 assistant / worker (v2) 🚀",
                source="special_source [main]",
            )

            # Checkpoint with no node or source
            _create_checkpoint(
                saver,
                thread_id,
                "cp-empty-meta-02",
                step=2,
                parent_checkpoint_id="cp-unicode-01",
            )

            # Checkpoint with UUID format
            uuid_cp = "1ef82a3c-b034-6c2e-8000-000000000001"
            _create_checkpoint(
                saver,
                thread_id,
                uuid_cp,
                step=3,
                node="finalizer",
                parent_checkpoint_id="cp-empty-meta-02",
            )

            visualizer = DAGVisualizer(saver)
            output_str = visualizer.render_to_string(thread_id)

            assert "🤖 assistant / worker (v2) 🚀" in output_str
            assert "special_source [main]" in output_str
            assert "Step 1" in output_str
            assert "Step 2" in output_str
            assert "Step 3" in output_str
            assert "node: <unknown>" in output_str
            # Long UUID should display at least the first 8 characters
            assert uuid_cp[:8] in output_str

    def test_visualizer_initialization_options(self) -> None:
        """DAGVisualizer accepts RelateSaver, db_path string, or explicit db_path kwarg."""
        with RelateSaver(db_path=":memory:") as saver:
            # 1. From RelateSaver
            vis1 = DAGVisualizer(saver)
            assert vis1.saver is saver
            assert vis1._owns_saver is False
            vis1.close()
            assert saver.proxy.is_alive()

            # 2. From positional string db_path
            vis2 = DAGVisualizer(":memory:")
            assert vis2._owns_saver is True
            assert vis2.saver.proxy.is_alive()
            vis2.close()
            assert not vis2.saver.proxy.is_alive()

            # 3. From keyword db_path
            vis3 = DAGVisualizer(db_path=":memory:")
            assert vis3._owns_saver is True
            assert vis3.saver.proxy.is_alive()
            vis3.close()
            assert not vis3.saver.proxy.is_alive()

            # 4. Context manager protocol
            with DAGVisualizer(":memory:") as vis4:
                assert vis4.saver.proxy.is_alive()
            assert not vis4.saver.proxy.is_alive()

            # 5. Type validation
            with pytest.raises(TypeError, match="Expected RelateSaver or str db_path"):
                DAGVisualizer(12345)  # type: ignore[arg-type]

    def test_print_tree_writes_to_console(self) -> None:
        """print_tree outputs rendered DAG to terminal Console."""
        with RelateSaver(db_path=":memory:") as saver:
            thread_id = "thread-print-test"
            _create_checkpoint(saver, thread_id, "cp-p-01", step=1, node="node_a")

            visualizer = DAGVisualizer(saver)
            console = Console(record=True, width=120)

            visualizer.print_tree(thread_id, console=console)
            captured = console.export_text()

            assert thread_id in captured
            assert "node_a" in captured
            assert "[OK]" in captured

    def test_render_tree_cross_thread_fork_in_memory(self) -> None:
        """Render tree correctly visualizes cross-thread branching created via fork."""
        with RelateSaver(db_path=":memory:") as saver:
            parent_thread = "thread-parent-flow"
            child_thread = "thread-child-replay"

            _create_checkpoint(saver, parent_thread, "cp-p-01", step=1, node="entry")
            _create_checkpoint(
                saver,
                parent_thread,
                "cp-p-02",
                step=2,
                node="agent",
                parent_checkpoint_id="cp-p-01",
            )

            # Fork to child thread from cp-p-01
            _create_checkpoint(
                saver,
                child_thread,
                "cp-fork-01",
                step=2,
                node="forked_agent",
                parent_checkpoint_id="cp-p-01",
            )

            visualizer = DAGVisualizer(saver)
            parent_output = visualizer.render_to_string(parent_thread)

            assert "entry" in parent_output
            assert "agent" in parent_output
            assert "thread-child-replay" in parent_output
            assert "forked_agent" in parent_output
            assert "[BRANCH]" in parent_output


class TestFullCheckpointIds:
    """Checkpoint ids are rendered in full so that same-second ids stay distinguishable."""

    def test_tree_shows_full_distinct_ids_for_same_second_checkpoints_ac_62_ac_63(self) -> None:
        """AC-62, AC-63: ids sharing a long prefix each appear unshortened and distinct."""
        ids = [f"1f1e314c-8a21-6b3e-8000-00000000000{n}" for n in range(1, 7)]
        with RelateSaver(db_path=":memory:") as saver:
            thread_id = "thread-same-second"
            parent: str | None = None
            for step, cid in enumerate(ids):
                _create_checkpoint(
                    saver,
                    thread_id,
                    cid,
                    step=step,
                    node=f"node_{step}",
                    parent_checkpoint_id=parent,
                )
                parent = cid

            output_str = DAGVisualizer(saver).render_to_string(thread_id)

        for cid in ids:
            assert f"(id: {cid})" in output_str
        assert output_str.count("1f1e314c") == len(ids)
        assert len({cid for cid in ids if cid in output_str}) == len(ids)

    def test_cycle_detected_line_shows_full_id_ac_62(self) -> None:
        """AC-62: the cycle-detected line carries the full checkpoint id."""
        cid = "1f1e314c-8a21-6b3e-8000-000000000001"
        with RelateSaver(db_path=":memory:") as saver:
            visualizer = DAGVisualizer(saver)
            root = Tree("root")
            visualizer._add_checkpoint_subtree(
                tree_node=root,
                cid=cid,
                parsed_checkpoints={},
                writes_by_cid={},
                children_map={},
                current_thread_id="t",
                visited={cid},
            )
            console = Console(record=True, width=120)
            with console.capture() as capture:
                console.print(root)

        assert f"cycle detected: {cid}" in capture.get()
