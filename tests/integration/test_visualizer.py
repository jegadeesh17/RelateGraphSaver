"""Integration tests for the flat tree layout of DAGVisualizer (AC-185, ADR-0027)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.tree import Tree

from langgraph_checkpoint_relate.errors import RelateCheckpointError
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.visualizer import DAGVisualizer

LINEAR_STEPS = 1500


def _put(
    saver: RelateSaver,
    thread_id: str,
    checkpoint_id: str,
    step: int,
    parent: str | None = None,
) -> None:
    """Persist one checkpoint, chained to parent when given."""
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if parent is not None:
        configurable["checkpoint_id"] = parent
    cfg: RunnableConfig = {"configurable": configurable}
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "channel_values": {"step": step},
        "channel_versions": {"step": str(step)},
        "versions_seen": {},
        "updated_channels": ["step"],
    }
    metadata: dict[str, Any] = {"step": step, "node": f"n{step}"}
    saver.put(cfg, checkpoint, metadata, {})  # type: ignore[arg-type]


def _indent_of(output: str, checkpoint_id: str) -> int:
    """Return the column at which the checkpoint's label text starts."""
    line = next(ln for ln in output.splitlines() if f"(id: {checkpoint_id})" in ln)
    match = re.search(r"\[(OK|PENDING|ERROR|INTERRUPT)\]", line)
    assert match is not None, line
    return match.start()


def _depth(tree: Tree) -> int:
    """Return the nesting depth of a rich Tree, counted iteratively."""
    deepest = 0
    stack = [(tree, 0)]
    while stack:
        node, level = stack.pop()
        deepest = max(deepest, level)
        stack.extend((child, level + 1) for child in node.children)
    return deepest


def test_long_linear_run_renders_flat_without_recursion_error_ac_185() -> None:
    """AC-185: a 1500-checkpoint linear run renders as same-depth siblings."""
    assert LINEAR_STEPS > sys.getrecursionlimit()  # deeper than recursion could reach
    thread = "thread-linear-flat"
    with RelateSaver(db_path=":memory:", wal_enabled=False) as saver:
        parent: str | None = None
        for step in range(LINEAR_STEPS):
            cid = f"cp-lin-{step:05d}"
            _put(saver, thread, cid, step, parent)
            parent = cid
        visualizer = DAGVisualizer(saver)
        tree = visualizer.render_tree(thread)
        output = visualizer.render_to_string(thread, Console(width=200, record=True))

    assert len(tree.children) == LINEAR_STEPS
    assert _depth(tree) == 1
    assert f"cp-lin-{LINEAR_STEPS - 1:05d}" in output
    assert _indent_of(output, "cp-lin-00000") == _indent_of(output, "cp-lin-00750")
    assert _indent_of(output, "cp-lin-00000") == _indent_of(
        output, f"cp-lin-{LINEAR_STEPS - 1:05d}"
    )


def test_next_children_are_siblings_and_fork_is_nested_ac_185() -> None:
    """AC-185: NEXT steps share one indent; a FORKED_TO branch is indented deeper."""
    thread = "thread-flat-fork"
    with RelateSaver(db_path=":memory:") as saver:
        _put(saver, thread, "cp-a-01", 1)
        _put(saver, thread, "cp-a-02", 2, "cp-a-01")
        _put(saver, thread, "cp-a-03", 3, "cp-a-02")
        # A second child of cp-a-01 is a fork, not a continuation.
        _put(saver, thread, "cp-fork-02", 2, "cp-a-01")
        output = DAGVisualizer(saver).render_to_string(thread, Console(width=200, record=True))

    main = {_indent_of(output, cid) for cid in ("cp-a-01", "cp-a-02", "cp-a-03")}
    assert len(main) == 1
    assert "Branch" in output
    assert _indent_of(output, "cp-fork-02") > next(iter(main))


def test_nested_forks_render_iteratively_ac_185() -> None:
    """AC-185: a fork of a fork nests one level per FORKED_TO edge."""
    thread = "thread-flat-nested-fork"
    with RelateSaver(db_path=":memory:") as saver:
        _put(saver, thread, "cp-n-01", 1)
        _put(saver, thread, "cp-n-02", 2, "cp-n-01")
        _put(saver, thread, "cp-n-fork-a", 2, "cp-n-01")
        _put(saver, thread, "cp-n-fork-a2", 3, "cp-n-fork-a")
        _put(saver, thread, "cp-n-fork-b", 3, "cp-n-fork-a")
        output = DAGVisualizer(saver).render_to_string(thread, Console(width=200, record=True))

    assert _indent_of(output, "cp-n-fork-a") > _indent_of(output, "cp-n-01")
    assert _indent_of(output, "cp-n-fork-b") > _indent_of(output, "cp-n-fork-a")
    assert _indent_of(output, "cp-n-fork-a2") == _indent_of(output, "cp-n-fork-a")


@pytest.mark.parametrize(
    "failing_fragment",
    ["-[:FORKED_TO]->(c:Checkpoint) WHERE c.thread_id <>", "type(r) AS rel_type", "HAS_WRITE"],
    ids=["external-forks", "edges", "writes"],
)
def test_database_error_in_supporting_query_propagates(
    monkeypatch: pytest.MonkeyPatch, failing_fragment: str
) -> None:
    """A failing fork, edge or write query raises instead of rendering a partial tree."""
    thread = "thread-db-error"
    with RelateSaver(db_path=":memory:") as saver:
        _put(saver, thread, "cp-e-01", 1)
        _put(saver, thread, "cp-e-02", 2, "cp-e-01")
        proxy = saver.proxy
        real_execute_read = proxy.execute_read

        def failing_execute_read(
            cypher: str, params: dict[str, Any] | None = None, timeout: float | None = None
        ) -> list[dict[str, Any]]:
            if failing_fragment in cypher:
                raise RelateCheckpointError("simulated database failure")
            return real_execute_read(cypher, params, timeout)

        monkeypatch.setattr(proxy, "execute_read", failing_execute_read)
        visualizer = DAGVisualizer(saver)
        with pytest.raises(RelateCheckpointError, match="simulated database failure"):
            visualizer.render_tree(thread)


class TestReadOnlyDefault:
    """AC-187 (ADR-0029): DAGVisualizer opens a file store read-only unless told otherwise."""

    THREAD = "thread-ro-default"

    def _seed(self, db_path: str) -> RunnableConfig:
        with RelateSaver(db_path=db_path, wal_enabled=True) as saver:
            _put(saver, self.THREAD, "cp-ro-1", 0)
            return {"configurable": {"thread_id": self.THREAD, "checkpoint_id": "cp-ro-1"}}

    def test_ac187_path_defaults_to_read_only_and_leaves_file_unchanged(
        self, tmp_path: Path
    ) -> None:
        """AC-187: a path with no read_only argument opens read-only and the file is untouched."""
        from langgraph_checkpoint_relate.errors import RelateReadOnlyError

        db = str(tmp_path / "ro.rl")
        cfg = self._seed(db)
        before = Path(db).read_bytes()
        with DAGVisualizer(db) as visualizer:
            assert visualizer.saver.read_only is True
            assert "cp-ro-1"[:8] in visualizer.render_to_string(self.THREAD)
            with pytest.raises(RelateReadOnlyError):
                visualizer.saver.put(cfg, _checkpoint_for_write(), {"step": 1}, {})
        assert Path(db).read_bytes() == before

    def test_ac187_db_path_keyword_defaults_to_read_only(self, tmp_path: Path) -> None:
        """AC-187: the db_path keyword form also defaults to read-only."""
        db = str(tmp_path / "ro-kw.rl")
        self._seed(db)
        with DAGVisualizer(db_path=db) as visualizer:
            assert visualizer.saver.read_only is True

    def test_ac187_explicit_read_only_false_opens_writable(self, tmp_path: Path) -> None:
        """AC-187: read_only=False still opens a writable store."""
        db = str(tmp_path / "rw.rl")
        cfg = self._seed(db)
        with DAGVisualizer(db, read_only=False) as visualizer:
            assert visualizer.saver.read_only is False
            visualizer.saver.put(cfg, _checkpoint_for_write(), {"step": 1}, {})

    def test_ac187_in_memory_path_stays_writable(self) -> None:
        """AC-187: ':memory:' has nothing to protect, so the default does not reject it."""
        with DAGVisualizer(":memory:") as visualizer:
            assert visualizer.saver.read_only is False

    def test_ac187_missing_file_is_not_created_by_default(self, tmp_path: Path) -> None:
        """AC-187: the read-only default does not create a missing store."""
        db = tmp_path / "missing.rl"
        with pytest.raises(RelateCheckpointError):
            DAGVisualizer(str(db))
        assert not db.exists()


def _checkpoint_for_write() -> Any:
    """Return a minimal checkpoint for write-attempt assertions."""
    return {
        "v": 1,
        "id": "cp-ro-2",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "updated_channels": [],
    }
