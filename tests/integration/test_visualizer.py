"""Integration tests for the flat tree layout of DAGVisualizer (AC-185, ADR-0027)."""

from __future__ import annotations

import re
import sys
from typing import Any

from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.tree import Tree

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
