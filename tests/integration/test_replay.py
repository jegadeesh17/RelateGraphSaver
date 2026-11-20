"""Integration tests for the replay engine on real file stores (ADR-0017).

The first test proves the fork, edit and resume mechanism directly: a failing tool-agent run is
stored in a file, forked into memory, its first human message is replaced, and the sandbox is
resumed with ``invoke(None, ...)``. The rest exercise ``load_graph_factory`` and
``replay_from_checkpoint`` (AC-94..AC-108, AC-151). The source file must stay byte-identical
(AC-103).
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict

import pytest
from examples.research_agent import create_research_graph, run_research_workflow
from examples.tool_agent import (
    OK_URL,
    TOOL_NAME,
    UNREACHABLE_URL,
    create_tool_agent_graph,
    run_tool_agent,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import langgraph_checkpoint_relate as package
from langgraph_checkpoint_relate import replay as replay_module
from langgraph_checkpoint_relate.errors import (
    GraphSpecError,
    RelateCheckpointError,
    RelateConnectionError,
    ReplayInputError,
)
from langgraph_checkpoint_relate.models import RunDiff, RunSnapshot
from langgraph_checkpoint_relate.replay import (
    load_graph_factory,
    replay_from_checkpoint,
    snapshot_run,
)
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.time_travel import TimeTravelEngine

THREAD_ID = "replay-spike-thread"
RESEARCH_THREAD_ID = "replay-research-thread"
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_update_state_in_fork_replaces_human_message_and_reaches_ok_tool(tmp_path: Path) -> None:
    """Editing the first human message in a sandbox fork replays to an ok tool call.

    The earliest checkpoint ('input', step -1) has no state values yet: the first human message
    is only a pending write there. The first checkpoint that holds it is step 0, so that is the
    fork point for editing it with ``as_node="__start__"``.
    """
    db_path = tmp_path / "source.rl"

    writer = RelateSaver(str(db_path))
    try:
        run_tool_agent(writer, THREAD_ID, fail=True)
    finally:
        writer.close()
    digest_before = _sha256(db_path)

    source = RelateSaver(str(db_path), read_only=True)
    try:
        list_config: RunnableConfig = {
            "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}
        }
        by_step = {
            int(c.metadata["step"]): str(c.config["configurable"]["checkpoint_id"])
            for c in source.list(list_config)
            if c.metadata is not None and "step" in c.metadata
        }
        assert {-1, 0} <= set(by_step)
        fork_checkpoint_id = by_step[0]

        sandbox = TimeTravelEngine.fork_in_memory(source, THREAD_ID, fork_checkpoint_id)
        try:
            graph = create_tool_agent_graph(checkpointer=sandbox)
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": THREAD_ID,
                    "checkpoint_ns": "",
                    "checkpoint_id": fork_checkpoint_id,
                }
            }
            messages = graph.get_state(cfg).values.get("messages", [])
            assert messages, "the fork checkpoint must already hold the first human message"
            edited = messages[0].model_copy(update={"content": OK_URL})
            new_cfg = graph.update_state(cfg, {"messages": [edited]}, as_node="__start__")
            final = graph.invoke(
                None,
                {
                    "configurable": {
                        "thread_id": THREAD_ID,
                        "checkpoint_id": new_cfg["configurable"]["checkpoint_id"],
                    }
                },
            )

            final_messages = final["messages"]
            assert str(final_messages[-1].content).startswith("Source fetched")
            human = [m for m in final_messages if isinstance(m, HumanMessage)]
            assert len(human) == 1
            assert human[0].content == OK_URL
            tool_messages = [m for m in final_messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert tool_messages[0].status != "error"

            rows: list[dict[str, Any]] = sandbox.proxy.execute_read(
                "MATCH (tc:ToolCall {tool_name: $tool_name}) "
                "RETURN tc.result_blob AS result_blob, tc.error AS error",
                {"tool_name": TOOL_NAME},
            )
            # The fork also carries the recorded call, still without an outcome because its
            # result was not in the fork checkpoint's channel values; the replay adds its own.
            resolved = [row for row in rows if row["result_blob"] is not None]
            assert len(resolved) == 1
            assert not resolved[0]["error"]
            assert not any(row["error"] for row in rows)
        finally:
            sandbox.close()
    finally:
        source.close()

    assert _sha256(db_path) == digest_before


# ---------------------------------------------------------------------------
# Shared stores
# ---------------------------------------------------------------------------


def _by_step(saver: RelateSaver, thread_id: str) -> dict[int, str]:
    list_config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    return {
        int(c.metadata["step"]): str(c.config["configurable"]["checkpoint_id"])
        for c in saver.list(list_config)
        if c.metadata is not None and "step" in c.metadata
    }


@pytest.fixture(scope="module")
def tool_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file store holding one failing tool-agent run."""
    db_path = tmp_path_factory.mktemp("tool_store") / "tool.rl"
    writer = RelateSaver(str(db_path))
    try:
        run_tool_agent(writer, THREAD_ID, fail=True)
    finally:
        writer.close()
    return db_path


@pytest.fixture(scope="module")
def tool_source(tool_store: Path) -> Generator[RelateSaver]:
    source = RelateSaver(str(tool_store), read_only=True)
    try:
        yield source
    finally:
        source.close()


@pytest.fixture(scope="module")
def research_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file store holding one completed research run (it paused once at human_review)."""
    db_path = tmp_path_factory.mktemp("research_store") / "research.rl"
    run_research_workflow(
        db_path=str(db_path),
        thread_id=RESEARCH_THREAD_ID,
        auto_approve=True,
        print_output=False,
    )
    return db_path


@pytest.fixture(scope="module")
def research_source(research_store: Path) -> Generator[RelateSaver]:
    source = RelateSaver(str(research_store), read_only=True)
    try:
        yield source
    finally:
        source.close()


def _checkpoint_before_human_review(source: RelateSaver) -> str:
    """The earliest checkpoint whose research notes exist but whose summary does not."""
    candidates: list[tuple[int, str]] = []
    for step, checkpoint_id in _by_step(source, RESEARCH_THREAD_ID).items():
        found = source.get_tuple(
            {
                "configurable": {
                    "thread_id": RESEARCH_THREAD_ID,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint_id,
                }
            }
        )
        assert found is not None
        values = found.checkpoint["channel_values"]
        if values.get("research_notes") and not values.get("final_summary"):
            candidates.append((step, checkpoint_id))
    assert candidates
    return min(candidates)[1]


@pytest.fixture
def captured_sandboxes(monkeypatch: pytest.MonkeyPatch) -> list[RelateSaver]:
    """Record every sandbox ``fork_in_memory`` hands out during a test."""
    sandboxes: list[RelateSaver] = []
    original = TimeTravelEngine.fork_in_memory

    def recording(*args: Any, **kwargs: Any) -> RelateSaver:
        sandbox = original(*args, **kwargs)
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(TimeTravelEngine, "fork_in_memory", staticmethod(recording))
    return sandboxes


@pytest.fixture
def captured_snapshots(monkeypatch: pytest.MonkeyPatch) -> list[tuple[RunSnapshot, RunSnapshot]]:
    """Record the (recorded, replayed) snapshots each replay compares."""
    pairs: list[tuple[RunSnapshot, RunSnapshot]] = []
    original = replay_module.diff_runs

    def spy(recorded: RunSnapshot, replayed: RunSnapshot) -> RunDiff:
        pairs.append((recorded, replayed))
        return original(recorded, replayed)

    monkeypatch.setattr(replay_module, "diff_runs", spy)
    return pairs


def _assert_closed(sandbox: RelateSaver) -> None:
    with pytest.raises(RelateConnectionError):
        sandbox.proxy.execute_read("RETURN 1 AS one", {})


def _listing(path: Path) -> list[str]:
    return sorted(entry.name for entry in path.parent.iterdir())


# ---------------------------------------------------------------------------
# Graph spec loading (AC-94..AC-97)
# ---------------------------------------------------------------------------

_AGENT_FILE = """
from langgraph.graph import START, MessagesState, StateGraph


def echo(state: MessagesState) -> dict[str, list]:
    return {"messages": []}


def make(checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    return builder.compile(checkpointer=checkpointer)
"""


@pytest.mark.slow
def test_ac94_file_spec_loads_from_cwd_without_cwd_on_sys_path(tmp_path: Path) -> None:
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    (graphs / "agent.py").write_text(_AGENT_FILE, encoding="utf-8")
    script = textwrap.dedent(
        """
        import sys
        before = list(sys.path)
        from langgraph_checkpoint_relate import load_graph_factory

        factory = load_graph_factory("graphs/agent.py:make")
        assert sys.path == before, "sys.path must not change"
        assert callable(factory)
        assert factory.__name__ == "make"
        print("loaded", len([n for n in sys.modules if n.startswith("_relate_replay_")]))
        """
    )
    env = {**os.environ, "PYTHONPATH": _SRC_DIR}
    result = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "loaded 1"


def test_ac94_file_spec_module_name_is_derived_from_the_resolved_path(tmp_path: Path) -> None:
    agent_file = tmp_path / "agent.py"
    agent_file.write_text(_AGENT_FILE, encoding="utf-8")
    sys_path_before = list(sys.path)

    factory = load_graph_factory(str(agent_file) + ":make")

    expected = hashlib.sha256(str(agent_file.resolve()).encode("utf-8")).hexdigest()[:12]
    assert callable(factory)
    assert factory.__module__ == f"_relate_replay_{expected}"
    assert sys.path == sys_path_before


def test_ac95_module_spec_loads_the_demo_factory() -> None:
    factory = load_graph_factory("examples.tool_agent:create_tool_agent_graph")

    assert factory is create_tool_agent_graph


_FACTORY_AND_BUILDER_FILE = """
from examples.tool_agent import agent, fetch_source, route_after_agent
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def _builder():
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([fetch_source], handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder


def make(checkpointer=None):
    return _builder().compile(checkpointer=checkpointer)


uncompiled = _builder()
"""


@pytest.mark.parametrize("attribute", ["make", "uncompiled"])
def test_ac96_factory_and_uncompiled_state_graph_replay_to_completion(
    tool_source: RelateSaver, tmp_path: Path, attribute: str
) -> None:
    graph_file = tmp_path / "tool_graph.py"
    graph_file.write_text(_FACTORY_AND_BUILDER_FILE, encoding="utf-8")
    factory = load_graph_factory(f"{graph_file}:{attribute}")
    first = _by_step(tool_source, THREAD_ID)[-1]

    result = replay_from_checkpoint(tool_source, THREAD_ID, first, factory)

    assert not result.interrupted
    assert result.diff.is_empty


def test_ac97_bad_specs_raise_graph_spec_error_containing_the_spec(tmp_path: Path) -> None:
    raising = tmp_path / "raising.py"
    raising.write_text("raise RuntimeError('import exploded')\n", encoding="utf-8")
    wrong_type = tmp_path / "wrong_type.py"
    wrong_type.write_text(_AGENT_FILE + "\nnot_a_graph = 42\ncompiled = make()\n", encoding="utf-8")
    cases = {
        "no colon": "just_a_name",
        "empty attribute": f"{wrong_type}:",
        "missing file": f"{tmp_path / 'missing.py'}:make",
        "missing module": "no_such_module_for_replay:make",
        "import raises": f"{raising}:make",
        "missing attribute": f"{wrong_type}:nope",
        "wrong type": f"{wrong_type}:not_a_graph",
        "compiled graph": f"{wrong_type}:compiled",
    }

    for label, spec in cases.items():
        with pytest.raises(GraphSpecError) as excinfo:
            load_graph_factory(spec)
        assert spec in str(excinfo.value), label

    with pytest.raises(GraphSpecError) as raised:
        load_graph_factory(f"{raising}:make")
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_ac97_factory_that_ignores_the_checkpointer_is_rejected_before_any_node_runs(
    tool_source: RelateSaver, captured_sandboxes: list[RelateSaver]
) -> None:
    def ignoring(checkpointer: Any = None) -> Any:
        return create_tool_agent_graph(checkpointer=None)

    first = _by_step(tool_source, THREAD_ID)[-1]

    with pytest.raises(GraphSpecError, match="checkpointer"):
        replay_from_checkpoint(tool_source, THREAD_ID, first, ignoring)

    _assert_closed(captured_sandboxes[0])


def test_ac155_factory_without_a_checkpointer_parameter_is_a_graph_spec_error(
    tool_source: RelateSaver, captured_sandboxes: list[RelateSaver]
) -> None:
    def build() -> Any:
        return create_tool_agent_graph(checkpointer=None)

    first = _by_step(tool_source, THREAD_ID)[-1]

    with pytest.raises(GraphSpecError, match="checkpointer"):
        replay_from_checkpoint(tool_source, THREAD_ID, first, build)

    _assert_closed(captured_sandboxes[0])


def test_ac155_replay_input_error_is_a_value_error_and_a_checkpoint_error(
    tool_source: RelateSaver,
) -> None:
    fork = _by_step(tool_source, THREAD_ID)[0]

    with pytest.raises(ReplayInputError, match="out of range") as raised:
        replay_from_checkpoint(
            tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={5: "x"}
        )

    assert isinstance(raised.value, ValueError)
    assert isinstance(raised.value, RelateCheckpointError)
    assert not isinstance(raised.value, GraphSpecError)


# ---------------------------------------------------------------------------
# Replay behavior (AC-98..AC-103, AC-151)
# ---------------------------------------------------------------------------


def test_ac98_unedited_replay_of_the_failing_thread_has_an_empty_diff(
    tool_source: RelateSaver,
) -> None:
    steps = _by_step(tool_source, THREAD_ID)

    result = replay_from_checkpoint(tool_source, THREAD_ID, steps[-1], create_tool_agent_graph)

    assert result.diff.is_empty
    assert result.fork_checkpoint_id == steps[-1]
    assert result.updated_checkpoint_id == steps[-1]
    assert result.final_checkpoint_id != steps[-1]
    assert result.recorded_checkpoint_id == steps[max(steps)]
    assert not result.interrupted


def test_ac99_set_message_turns_the_failing_fetch_into_a_working_one(
    tool_source: RelateSaver,
) -> None:
    fork = _by_step(tool_source, THREAD_ID)[0]

    result = replay_from_checkpoint(
        tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={0: OK_URL}
    )

    assert len(result.diff.tool_calls) == 1
    change = result.diff.tool_calls[0]
    assert (change.tool_name, change.ordinal, change.kind) == (TOOL_NAME, 1, "changed")
    assert change.before is not None and change.after is not None
    assert (change.before.status, change.after.status) == ("error", "ok")
    assert change.before.args == {"url": UNREACHABLE_URL}
    assert change.after.args == {"url": OK_URL}
    by_channel = {m.channel: m for m in result.diff.messages}
    assert by_channel["messages[0]"].after == f"human: {OK_URL}"
    final = by_channel["messages[3]"]
    assert final.before is not None and final.before.startswith("ai: The source could not")
    assert final.after is not None and final.after.startswith("ai: Source fetched")


def test_ac100_edited_replay_holds_exactly_one_human_message_with_the_new_text(
    tool_source: RelateSaver, captured_snapshots: list[tuple[RunSnapshot, RunSnapshot]]
) -> None:
    fork = _by_step(tool_source, THREAD_ID)[0]

    replay_from_checkpoint(
        tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={0: OK_URL}
    )

    _, replayed = captured_snapshots[0]
    humans = [m for m in replayed.messages if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    assert replayed.messages[0] is humans[0]
    assert humans[0].content == OK_URL


def test_ac101_update_with_as_node_creates_a_new_checkpoint_and_the_run_continues_from_it(
    research_source: RelateSaver,
) -> None:
    fork = _checkpoint_before_human_review(research_source)

    # Writing as "planner" makes "researcher" the next node, so the run has a node to execute.
    result = replay_from_checkpoint(
        research_source,
        RESEARCH_THREAD_ID,
        fork,
        create_research_graph,
        update={"plan": ["edited plan"]},
        as_node="planner",
    )

    assert result.fork_checkpoint_id == fork
    assert result.updated_checkpoint_id != fork
    # The run executed "researcher" after the update, so it saved a checkpoint beyond the
    # updated one, and then paused at "human_review".
    assert result.final_checkpoint_id not in {fork, result.updated_checkpoint_id}
    assert result.interrupted
    plan = [c for c in result.diff.channels if c.channel == "plan"]
    assert len(plan) == 1
    assert plan[0].after == repr(["edited plan"])


def test_ac102_research_replay_from_before_human_review_reports_an_interrupt(
    research_source: RelateSaver,
) -> None:
    fork = _checkpoint_before_human_review(research_source)

    result = replay_from_checkpoint(
        research_source, RESEARCH_THREAD_ID, fork, create_research_graph
    )

    assert result.interrupted
    assert not result.diff.is_empty
    assert "final_summary" in {c.channel for c in result.diff.channels}


def test_ac103_source_store_is_untouched_and_sandbox_closed_on_success(
    tool_store: Path, tool_source: RelateSaver, captured_sandboxes: list[RelateSaver]
) -> None:
    digest = _sha256(tool_store)
    listing = _listing(tool_store)
    fork = _by_step(tool_source, THREAD_ID)[0]

    replay_from_checkpoint(
        tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={0: OK_URL}
    )

    assert _sha256(tool_store) == digest
    assert _listing(tool_store) == listing
    assert len(captured_sandboxes) == 1
    _assert_closed(captured_sandboxes[0])


def test_ac103_source_store_is_untouched_and_sandbox_closed_when_the_factory_raises(
    tool_store: Path, tool_source: RelateSaver, captured_sandboxes: list[RelateSaver]
) -> None:
    digest = _sha256(tool_store)
    listing = _listing(tool_store)
    fork = _by_step(tool_source, THREAD_ID)[0]

    def exploding(checkpointer: Any = None) -> Any:
        raise RuntimeError("factory exploded")

    with pytest.raises(RuntimeError, match="factory exploded"):
        replay_from_checkpoint(tool_source, THREAD_ID, fork, exploding)

    assert _sha256(tool_store) == digest
    assert _listing(tool_store) == listing
    assert len(captured_sandboxes) == 1
    _assert_closed(captured_sandboxes[0])


def test_ac151_set_message_on_a_checkpoint_without_messages_suggests_the_next_checkpoint(
    tool_store: Path, tool_source: RelateSaver, captured_sandboxes: list[RelateSaver]
) -> None:
    digest = _sha256(tool_store)
    steps = _by_step(tool_source, THREAD_ID)

    with pytest.raises(ValueError, match=r"no messages yet.*next \(step 0\) checkpoint") as info:
        replay_from_checkpoint(
            tool_source, THREAD_ID, steps[-1], create_tool_agent_graph, set_messages={0: OK_URL}
        )

    assert steps[0] in str(info.value)
    assert _sha256(tool_store) == digest
    _assert_closed(captured_sandboxes[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"update": ["not", "a", "dict"]},
        {"set_messages": {-1: "x"}},
        {"set_messages": {"0": "x"}},
        {"set_messages": {0: 5}},
        {"set_messages": {0: "x"}, "update": {"messages": []}},
    ],
)
def test_invalid_inputs_raise_value_error_before_any_fork(
    tool_source: RelateSaver, captured_sandboxes: list[RelateSaver], kwargs: dict[str, Any]
) -> None:
    fork = _by_step(tool_source, THREAD_ID)[0]

    with pytest.raises(ValueError):
        replay_from_checkpoint(tool_source, THREAD_ID, fork, create_tool_agent_graph, **kwargs)

    assert captured_sandboxes == []


def test_set_message_index_out_of_range_raises_value_error(tool_source: RelateSaver) -> None:
    fork = _by_step(tool_source, THREAD_ID)[0]

    with pytest.raises(ValueError, match="out of range"):
        replay_from_checkpoint(
            tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={5: "x"}
        )


# ---------------------------------------------------------------------------
# Recorded leaf (AC-107)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("created_at", "branch_wins"),
    [("9999-12-31T00:00:00+00:00", True), ("0001-01-01T00:00:00+00:00", False)],
)
def test_ac107_recorded_run_is_the_latest_leaf_below_the_fork_point(
    tool_store: Path, tmp_path: Path, created_at: str, branch_wins: bool
) -> None:
    store_dir = tmp_path / "branching"
    shutil.copytree(tool_store.parent, store_dir)
    saver = RelateSaver(str(store_dir / tool_store.name))
    try:
        steps = _by_step(saver, THREAD_ID)
        fork, original_leaf = steps[0], steps[max(steps)]
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
            "RETURN c.checkpoint_blob AS checkpoint_blob, c.metadata_blob AS metadata_blob, "
            "c.state_hash AS state_hash, c.step AS step",
            {"cid": original_leaf},
        )
        row = rows[0]
        # LangGraph never writes a second [:NEXT] child, so the second branch is added by hand.
        saver.proxy.execute_write(
            "MATCH (f:Checkpoint {checkpoint_id: $fork}) "
            "CREATE (n:Checkpoint {checkpoint_id: 'branch-leaf', thread_id: $thread_id, "
            "checkpoint_ns: '', parent_checkpoint_id: $fork, step: $step, "
            "checkpoint_blob: $checkpoint_blob, metadata_blob: $metadata_blob, "
            "state_hash: $state_hash, created_at: $created_at}) "
            "CREATE (f)-[:NEXT]->(n)",
            {
                "fork": fork,
                "thread_id": THREAD_ID,
                "step": row["step"],
                "checkpoint_blob": row["checkpoint_blob"],
                "metadata_blob": row["metadata_blob"],
                "state_hash": row["state_hash"],
                "created_at": created_at,
            },
        )

        result = replay_from_checkpoint(
            saver, THREAD_ID, fork, create_tool_agent_graph, set_messages={0: OK_URL}
        )
    finally:
        saver.close()

    assert result.recorded_checkpoint_id == ("branch-leaf" if branch_wins else original_leaf)


# ---------------------------------------------------------------------------
# Package surface (AC-108, RC-09)
# ---------------------------------------------------------------------------


def test_ac108_time_travel_engine_replay_returns_the_same_diff(tool_source: RelateSaver) -> None:
    fork = _by_step(tool_source, THREAD_ID)[0]

    direct = package.replay_from_checkpoint(
        tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={0: OK_URL}
    )
    delegated = package.TimeTravelEngine.replay(
        tool_source, THREAD_ID, fork, create_tool_agent_graph, set_messages={0: OK_URL}
    )

    assert delegated.diff == direct.diff
    assert not delegated.diff.is_empty
    assert package.replay_from_checkpoint is replay_from_checkpoint
    assert package.snapshot_run is snapshot_run
    assert package.load_graph_factory is load_graph_factory


@pytest.mark.parametrize(
    "name",
    [
        "CheckpointNotFoundError",
        "CheckpointSerializationError",
        "DAGVisualizer",
        "LoopReport",
        "RelateCheckpointError",
        "RelateCheckpointNotFoundError",
        "RelateConcurrencyError",
        "RelateConnectionError",
        "RelateForensics",
        "RelateReadOnlyError",
        "RelateSaver",
        "RelateSerializationError",
        "ReplayInputError",
        "RootCauseReport",
        "TimeTravelEngine",
        "fork_in_memory",
        "GraphFactory",
        "GraphSpecError",
        "load_graph_factory",
        "replay_from_checkpoint",
        "snapshot_run",
        "diff_runs",
        "ToolCallRecord",
        "RunSnapshot",
        "ChannelChange",
        "ToolCallChange",
        "RunDiff",
        "ReplayResult",
    ],
)
def test_rc09_ac108_every_export_imports_from_the_package(name: str) -> None:
    assert name in package.__all__
    assert hasattr(package, name)


_FORK_STEP_RUNS: list[str] = []


class _CounterState(TypedDict):
    value: int


def _fork_step_graph(checkpointer: Any = None) -> Any:
    def first(state: _CounterState) -> dict[str, int]:
        return {"value": state["value"] + 1}

    def second(state: _CounterState) -> dict[str, int]:
        _FORK_STEP_RUNS.append("second")
        return {"value": state["value"] * 10}

    builder = StateGraph(_CounterState)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=checkpointer)


def test_ac154_replay_without_edits_re_executes_the_fork_step() -> None:
    """A fork point that already holds stored writes must still run its node again."""
    source = RelateSaver(":memory:")
    try:
        graph = _fork_step_graph(source)
        graph.invoke({"value": 1}, {"configurable": {"thread_id": "fork-step"}})
        assert _FORK_STEP_RUNS == ["second"]

        # The checkpoint before "second" runs: its task writes are stored on that checkpoint.
        fork = next(
            str(snap.config["configurable"]["checkpoint_id"])
            for snap in graph.get_state_history({"configurable": {"thread_id": "fork-step"}})
            if snap.next == ("second",)
        )
        _FORK_STEP_RUNS.clear()

        result = replay_from_checkpoint(source, "fork-step", fork, _fork_step_graph)

        assert _FORK_STEP_RUNS == ["second"]
        assert result.updated_checkpoint_id == fork
        assert result.fork_checkpoint_id == fork
        assert not result.interrupted
    finally:
        _FORK_STEP_RUNS.clear()
        source.close()


class _PlainMessagesState(TypedDict):
    messages: list[Any]


def _plain_messages_graph(checkpointer: Any = None) -> Any:
    def echo(state: _PlainMessagesState) -> dict[str, list[Any]]:
        return {"messages": state["messages"]}

    builder = StateGraph(_PlainMessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=checkpointer)


def test_ac156_set_messages_rejects_a_messages_channel_without_add_messages() -> None:
    """Without an id-merging reducer the edit would replace the list and drop the rest."""
    source = RelateSaver(":memory:")
    try:
        graph = _plain_messages_graph(source)
        config: RunnableConfig = {"configurable": {"thread_id": "plain-messages"}}
        graph.invoke(
            {"messages": [HumanMessage(content=f"message {n}", id=f"m{n}") for n in range(3)]},
            config,
        )
        fork = next(
            str(snap.config["configurable"]["checkpoint_id"])
            for snap in reversed(list(graph.get_state_history(config)))
            if snap.values.get("messages")
        )
        before = len(list(source.list(config)))

        with pytest.raises(ReplayInputError, match="add_messages"):
            replay_from_checkpoint(
                source, "plain-messages", fork, _plain_messages_graph, set_messages={0: "new"}
            )

        assert len(list(source.list(config))) == before
    finally:
        source.close()
