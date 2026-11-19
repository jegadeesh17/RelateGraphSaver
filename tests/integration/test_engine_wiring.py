"""Cross-module checks for replay, forensics, the tree visualizer and the CLI.

Each test drives a journey through the public entry points (the ``relate-dag`` app, the
replay API, ``RelateForensics`` and ``DAGVisualizer``) against real file or in-memory stores.
The per-module suites cover each feature alone; these tests cover how the features share a
store and a sandbox, which is where the replay, forensics and visualizer features meet.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

import pytest
from examples.tool_agent import create_tool_agent_graph
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typer.testing import CliRunner

from langgraph_checkpoint_relate import replay as replay_module
from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.forensics import RelateForensics
from langgraph_checkpoint_relate.replay import replay_from_checkpoint
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.visualizer import DAGVisualizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_STORE = REPO_ROOT / "examples" / "data" / "demo.rl"
DEMO_THREAD = "demo-tool-fail"
GRAPH_SPEC = "examples/tool_agent.py:create_tool_agent_graph"
OK_URL = "https://example.org/report"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listing(path: Path) -> list[str]:
    return sorted(entry.name for entry in path.parent.iterdir())


def _step_ids(saver: RelateSaver, thread_id: str) -> dict[int, str]:
    """Map each step number of a thread to its checkpoint id."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    return {
        int(c.metadata["step"]): str(c.config["configurable"]["checkpoint_id"])
        for c in saver.list(config)
        if c.metadata is not None and "step" in c.metadata
    }


@pytest.fixture
def demo_copy(tmp_path: Path) -> str:
    """A copy of the committed demo store, so the committed file is never written."""
    db_file = tmp_path / "demo_copy.rl"
    shutil.copy(DEMO_STORE, db_file)
    return str(db_file)


# ---------------------------------------------------------------------------
# Replay and forensics share one source store (AC-110, AC-117, AC-151, RC-04, RC-05)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("checkpoint_step", "replay_exit"), [(0, 0), (-1, 1)])
def test_replay_leaves_failing_thread_forensics_byte_identical(
    demo_copy: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_step: int,
    replay_exit: int,
) -> None:
    """An edited replay (exit 0 from step 0) or a rejected one (exit 1 from step -1, AC-151) leaves
    the store file, its directory and the trace-error, failures and tree output byte-identical.
    """
    monkeypatch.chdir(REPO_ROOT)
    runner = CliRunner()
    db_path = Path(demo_copy)
    views = [
        ["forensics", "trace-error", DEMO_THREAD, "--db", demo_copy],
        ["forensics", "failures", "--db", demo_copy],
        ["tree", DEMO_THREAD, "--db", demo_copy],
    ]
    store_hash = _sha256(db_path)
    listing = _listing(db_path)
    before = [runner.invoke(app, view) for view in views]
    assert [r.exit_code for r in before] == [0, 0, 0]
    with RelateSaver(demo_copy, read_only=True) as source:
        checkpoint_id = _step_ids(source, DEMO_THREAD)[checkpoint_step]

    replay = runner.invoke(
        app,
        [
            "replay",
            DEMO_THREAD,
            checkpoint_id,
            "--graph",
            GRAPH_SPEC,
            "--set-message",
            f"0={OK_URL}",
            "--db",
            demo_copy,
        ],
    )

    assert replay.exit_code == replay_exit, replay.output
    if replay_exit == 0:
        assert "error -> ok" in replay.output
    else:
        assert "Error:" in replay.output
    after = [runner.invoke(app, view) for view in views]
    assert [r.exit_code for r in after] == [0, 0, 0]
    assert [r.output for r in after] == [r.output for r in before]
    assert _sha256(db_path) == store_hash
    assert _listing(db_path) == listing


# ---------------------------------------------------------------------------
# Sandbox tree labels after an edited replay (AC-88, AC-90, AC-99)
# ---------------------------------------------------------------------------


def test_sandbox_tree_after_edited_replay_has_no_error_or_interrupt_tags(
    demo_copy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox that an edited replay runs in renders a tree with OK labels and no ERROR or
    INTERRUPT tags, while the source thread is still rendered from the recorded run.

    ``replay_from_checkpoint`` closes its sandbox and does not return it, so the tree is rendered
    from inside the replay's own snapshot call, while the sandbox is open. ``DAGVisualizer`` does
    not own a saver it is given, so this does not close the sandbox.
    """
    rendered: list[str] = []
    with RelateSaver(demo_copy, read_only=True) as source:
        original = replay_module.snapshot_run

        def spy(saver: RelateSaver, thread_id: str, checkpoint_id: str) -> Any:
            if saver is not source:
                rendered.append(DAGVisualizer(saver).render_to_string(thread_id))
            return original(saver, thread_id, checkpoint_id)

        monkeypatch.setattr(replay_module, "snapshot_run", spy)
        steps = _step_ids(source, DEMO_THREAD)
        source_tree = DAGVisualizer(source).render_to_string(DEMO_THREAD)

        result = replay_from_checkpoint(
            source,
            DEMO_THREAD,
            steps[0],
            create_tool_agent_graph,
            set_messages={0: OK_URL},
        )
        assert DAGVisualizer(source).render_to_string(DEMO_THREAD) == source_tree

    assert [change.kind for change in result.diff.tool_calls] == ["changed"]
    assert len(rendered) == 1
    sandbox_tree = rendered[0]
    assert "[ERROR]" not in sandbox_tree
    assert "[INTERRUPT]" not in sandbox_tree
    assert "[OK] Step 0" in sandbox_tree
    assert sandbox_tree.count("[OK] Step") > source_tree.count("[OK] Step")


# ---------------------------------------------------------------------------
# Argument provenance: root cause, argument origin and failure summary agree (AC-119, AC-121, AC-123)
# ---------------------------------------------------------------------------


class _ChainState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool("probe_package")
def _probe_package(url: str) -> str:
    """Probe a package. Succeeds and does not echo the url."""
    return "probe ok"


@tool("fetch_package")
def _fetch_package(url: str) -> str:
    """Fetch a package. Fails unless the url ends with 'ok'."""
    if not url.endswith("ok"):
        raise RuntimeError("package host unreachable")
    return "fetched"


def _run_probe_then_fetch(saver: RelateSaver, thread_id: str, url: str) -> None:
    """Probe the url in one model step, then fetch it in a later one.

    The probe's tool-call arguments are the first write that holds the url, so they are the origin
    of the failing fetch's argument.
    """

    def probe_agent(state: _ChainState) -> dict[str, Any]:
        call = {"id": f"probe-{thread_id}", "name": "probe_package", "args": {"url": url}}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    def fetch_agent(state: _ChainState) -> dict[str, Any]:
        call = {"id": f"fetch-{thread_id}", "name": "fetch_package", "args": {"url": url}}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(_ChainState)
    builder.add_node("probe_agent", probe_agent)
    builder.add_node("probe_tools", ToolNode([_probe_package]))
    builder.add_node("fetch_agent", fetch_agent)
    builder.add_node("fetch_tools", ToolNode([_fetch_package], handle_tool_errors=True))
    builder.add_edge(START, "probe_agent")
    builder.add_edge("probe_agent", "probe_tools")
    builder.add_edge("probe_tools", "fetch_agent")
    builder.add_edge("fetch_agent", "fetch_tools")
    builder.add_edge("fetch_tools", END)
    builder.compile(checkpointer=saver).invoke(
        {"messages": [HumanMessage(content="check the package")]},
        {"configurable": {"thread_id": thread_id}},
    )


def test_root_cause_origin_matches_argument_origin_and_failure_summary(
    tmp_path: Path,
) -> None:
    """For the failing fetch, RootCauseReport's origin fields equal trace_argument_origin on the
    same error checkpoint and arguments, and failure_summary names the same origin node. The
    origin is the probe's tool-call argument (AC-119, tool-call path), not a state value.
    A thread whose fetch succeeds has no origin and does not appear in the summary.
    """
    failing_url = "pkg-alpha-9000"
    with RelateSaver(db_path=str(tmp_path / "chain.rl")) as saver:
        _run_probe_then_fetch(saver, "t-fail", failing_url)
        _run_probe_then_fetch(saver, "t-ok", "pkg-ok")

        with RelateForensics(saver) as forensics:
            report = forensics.trace_root_cause("t-fail")
            origin = forensics.trace_argument_origin(
                "t-fail", report.error_checkpoint_id, {"url": failing_url}
            )
            summary = forensics.failure_summary()
            ok_report = forensics.trace_root_cause("t-ok")

    assert report.root_tool_call_id == "fetch-t-fail"
    assert origin is not None
    assert origin.node == "probe_agent"
    assert (report.origin_argument, report.origin_node) == (origin.argument, origin.node)
    assert (report.origin_step, report.origin_hops) == (origin.step, origin.hops)
    assert [s.tool_name for s in summary] == ["fetch_package"]
    assert summary[0].origin_nodes == (report.origin_node,)
    assert summary[0].threads == ("t-fail",)
    assert summary[0].failures == 1
    assert summary[0].total_calls == 2
    assert ok_report.origin_node is None
    assert ok_report.origin_argument is None


def test_argument_written_in_the_prompt_traces_to_the_input_write(tmp_path: Path) -> None:
    """AC-119, message-content path: a url that first appears in the human message is traced to
    the input write, whose node is the graph entry ``__start__``.
    """
    url = "pkg-prompt-9000"

    def fetch_agent(state: _ChainState) -> dict[str, Any]:
        call = {"id": "fetch-prompt", "name": "fetch_package", "args": {"url": url}}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(_ChainState)
    builder.add_node("fetch_agent", fetch_agent)
    builder.add_node("fetch_tools", ToolNode([_fetch_package], handle_tool_errors=True))
    builder.add_edge(START, "fetch_agent")
    builder.add_edge("fetch_agent", "fetch_tools")
    builder.add_edge("fetch_tools", END)
    with RelateSaver(db_path=str(tmp_path / "prompt.rl")) as saver:
        builder.compile(checkpointer=saver).invoke(
            {"messages": [HumanMessage(content=f"Please fetch {url}")]},
            {"configurable": {"thread_id": "t-prompt"}},
        )

        with RelateForensics(saver) as forensics:
            report = forensics.trace_root_cause("t-prompt")

    assert report.origin_argument == "url"
    assert report.origin_node == "__start__"


@pytest.mark.parametrize("invoking_step", [2, 3], ids=["edge-on-request", "edge-on-result"])
def test_root_cause_origin_does_not_depend_on_where_the_invoked_edge_sits(
    tmp_path: Path, invoking_step: int
) -> None:
    """AC-121, AC-129: the saver may attach the failing call's [:INVOKED] edge to the checkpoint
    that requested the call (step 2) or to the next one (step 3), depending on which save lands
    first. Both placements give the same error checkpoint, and the report's origin hops equal
    trace_argument_origin from that checkpoint.
    """
    failing_url = "pkg-alpha-9000"
    call_id = "fetch-t-fail"
    with RelateSaver(db_path=str(tmp_path / "edge.rl")) as saver:
        _run_probe_then_fetch(saver, "t-fail", failing_url)
        saver.proxy.execute_write(
            "MATCH (:Checkpoint {thread_id: $thread_id})-[r:INVOKED]->(:ToolCall {call_id: $call}) "
            "DELETE r",
            {"thread_id": "t-fail", "call": call_id},
        )
        saver.proxy.execute_write(
            "MATCH (c:Checkpoint {thread_id: $thread_id, step: $step}), "
            "(tc:ToolCall {call_id: $call}) MERGE (c)-[:INVOKED]->(tc)",
            {"thread_id": "t-fail", "step": invoking_step, "call": call_id},
        )

        with RelateForensics(saver) as forensics:
            report = forensics.trace_root_cause("t-fail")
            origin = forensics.trace_argument_origin(
                "t-fail", report.error_checkpoint_id, {"url": failing_url}
            )
        failed_step = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {checkpoint_id: $id}) RETURN c.step AS step",
            {"id": report.error_checkpoint_id},
        )[0]["step"]

    assert report.root_tool_call_id == call_id
    assert failed_step == 3
    assert origin is not None
    assert (report.origin_step, report.origin_hops) == (origin.step, origin.hops)
    assert report.origin_hops == 3


# ---------------------------------------------------------------------------
# Tree branch label and unknown thread (RC-01)
# ---------------------------------------------------------------------------


def _put_checkpoint(
    saver: RelateSaver,
    thread_id: str,
    checkpoint_id: str,
    parent: str | None,
    step: int,
    node: str,
) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    if parent is not None:
        config["configurable"]["checkpoint_id"] = parent
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "channel_values": {"query": node},
        "channel_versions": {"query": "1"},
        "versions_seen": {},
        "updated_channels": ["query"],
    }
    metadata = {"step": step, "node": node, "source": "loop"}
    saver.put(config, cast(Any, checkpoint), cast(Any, metadata), {})


def test_tree_marks_branch_points_and_rejects_unknown_thread_rc01(tmp_path: Path) -> None:
    """RC-01: a branch point renders as exactly 'Branch: [BRANCH]', and an unknown thread exits 1
    with a not-found error.
    """
    db_file = str(tmp_path / "branch.rl")
    with RelateSaver(db_file) as saver:
        _put_checkpoint(saver, "t-branch", "cp-root", None, 1, "__start__")
        _put_checkpoint(saver, "t-branch", "cp-main", "cp-root", 2, "agent")
        _put_checkpoint(saver, "t-branch", "cp-side", "cp-root", 2, "human_feedback")

    runner = CliRunner()
    branch = runner.invoke(app, ["tree", "t-branch", "--db", db_file])
    unknown = runner.invoke(app, ["tree", "no-such-thread", "--db", db_file])

    assert branch.exit_code == 0, branch.output
    assert "Branch: [BRANCH]" in branch.output
    assert unknown.exit_code == 1
    assert "not found" in unknown.output.lower()
