"""Integration tests for relate-dag command-line interface.

Validates terminal visualization, in-memory replay branching, forensics diagnostics,
and boundary error handling (AC-24, AC-32).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

import pytest
import relate
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.errors import MarkupError
from rich.table import Table
from typer.testing import CliRunner

from langgraph_checkpoint_relate import cli
from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_STORE = REPO_ROOT / "examples" / "data" / "demo.rl"
DEMO_THREAD = "demo-tool-fail"
GRAPH_SPEC = "examples/tool_agent.py:create_tool_agent_graph"
OK_URL = "https://example.org/report"
UNREACHABLE_URL = "https://unreachable.invalid/report"


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing an isolated Typer CliRunner instance."""
    return CliRunner()


@pytest.fixture
def seeded_db() -> Generator[tuple[str, str, list[str]]]:
    """Fixture providing a temporary database seeded with a 3-step execution lineage."""
    temp_dir = tempfile.mkdtemp(prefix="relate_cli_test_")
    db_file = os.path.join(temp_dir, "cli_test.rl")
    thread_id = "test-session-42"
    checkpoint_ids = [
        "1ef82a3c-b034-6c2e-8000-000000000001",
        "1ef82a3c-b034-6c2e-8000-000000000002",
        "1ef82a3c-b034-6c2e-8000-000000000003",
    ]

    # Seed checkpoints into the database
    with RelateSaver(db_path=db_file) as saver:
        # Step 1: Input / Root node
        cfg_1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        chk_1 = {
            "v": 1,
            "id": checkpoint_ids[0],
            "channel_values": {"query": "Summarize quarterly report"},
            "channel_versions": {"query": "1"},
            "versions_seen": {},
            "updated_channels": ["query"],
        }
        md_1: dict[str, Any] = {"step": 1, "node": "router_agent", "source": "input"}
        saver.put(cfg_1, cast(Any, chk_1), cast(Any, md_1), {})

        # Step 2: Processing step
        cfg_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_ids[0],
                "checkpoint_ns": "",
            }
        }
        chk_2 = {
            "v": 1,
            "id": checkpoint_ids[1],
            "channel_values": {"query": "Summarize quarterly report", "draft": "Financial data"},
            "channel_versions": {"query": "1", "draft": "2"},
            "versions_seen": {},
            "updated_channels": ["draft"],
        }
        md_2: dict[str, Any] = {"step": 2, "node": "research_agent", "source": "loop"}
        saver.put(cfg_2, cast(Any, chk_2), cast(Any, md_2), {})

        # Step 3: Leaf / Final node
        cfg_3: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_ids[1],
                "checkpoint_ns": "",
            }
        }
        chk_3 = {
            "v": 1,
            "id": checkpoint_ids[2],
            "channel_values": {
                "query": "Summarize quarterly report",
                "draft": "Financial data",
                "summary": "Complete",
            },
            "channel_versions": {"query": "1", "draft": "2", "summary": "3"},
            "versions_seen": {},
            "updated_channels": ["summary"],
        }
        md_3: dict[str, Any] = {"step": 3, "node": "output_node", "source": "loop"}
        saver.put(cfg_3, cast(Any, chk_3), cast(Any, md_3), {})

    yield db_file, thread_id, checkpoint_ids

    # Cleanup temporary test files
    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.fixture
def seeded_loop_db() -> Generator[tuple[str, str]]:
    """Fixture providing a temporary database with an oscillating execution loop."""
    temp_dir = tempfile.mkdtemp(prefix="relate_cli_loop_")
    db_file = os.path.join(temp_dir, "cli_loop.rl")
    thread_id = "loop-thread-session"

    with RelateSaver(db_path=db_file) as saver:
        for i in range(4):
            node = "ping_agent" if i % 2 == 0 else "pong_agent"
            cid = f"chk-loop-step-{i}"
            parent = f"chk-loop-step-{i - 1}" if i > 0 else None
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": parent,
                    "checkpoint_ns": "",
                }
            }
            chk = {
                "v": 1,
                "id": cid,
                "channel_values": {"counter": i},
                "channel_versions": {"counter": str(i)},
                "versions_seen": {},
                "updated_channels": ["counter"],
            }
            md: dict[str, Any] = {"step": i, "node": node, "source": "loop"}
            saver.put(cfg, cast(Any, chk), cast(Any, md), {})

    yield db_file, thread_id

    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.fixture
def seeded_tool_error_db(tmp_path: Path) -> tuple[str, str]:
    """Fixture providing a database whose thread holds one failed tool call."""
    db_file = str(tmp_path / "cli_tool_error.rl")
    thread_id = "tool-fail-thread"
    with RelateSaver(db_path=db_file) as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        cp: Any = dict(empty_checkpoint())
        cp["id"] = "1f1e314c-8a21-6b3e-8000-000000000001"
        saved_cfg = saver.put(cfg, cp, {"step": 1, "source": "loop", "parents": {}}, {})
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "call-fetch-1", "name": "fetch_source", "args": {}}],
        )
        saver.put_writes(saved_cfg, [("messages", [ai])], task_id="task-ai")
        tool_msg = ToolMessage(
            content="Error: ConnectionError('down')\n Please fix your mistakes.",
            tool_call_id="call-fetch-1",
            status="error",
        )
        saver.put_writes(saved_cfg, [("messages", [tool_msg])], task_id="task-tool")
    return db_file, thread_id


@pytest.fixture
def seeded_error_db() -> Generator[tuple[str, str, str]]:
    """Fixture providing a temporary database with an antecedent channel write and error checkpoint."""
    temp_dir = tempfile.mkdtemp(prefix="relate_cli_error_")
    db_file = os.path.join(temp_dir, "cli_error.rl")
    thread_id = "error-thread-session"
    error_cid = "chk-err-step-2"

    with RelateSaver(db_path=db_file) as saver:
        # Step 1: Upstream write
        cfg_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        chk_1 = {
            "v": 1,
            "id": "chk-err-step-1",
            "channel_values": {"payload": "POISON_VALUE"},
            "channel_versions": {"payload": "1"},
            "versions_seen": {},
            "updated_channels": ["payload"],
        }
        md_1: dict[str, Any] = {"step": 1, "node": "data_loader", "source": "input"}
        saver.put(cfg_1, cast(Any, chk_1), cast(Any, md_1), {})
        saver.put_writes(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": "chk-err-step-1",
                    "checkpoint_ns": "",
                }
            },
            [("poison_payload", "malformed data packet")],
            "task_data_loader",
        )

        # Step 2: Failure checkpoint with error write
        cfg_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "chk-err-step-1",
                "checkpoint_ns": "",
            }
        }
        chk_2 = {
            "v": 1,
            "id": error_cid,
            "channel_values": {"payload": "POISON_VALUE"},
            "channel_versions": {"payload": "1"},
            "versions_seen": {},
            "updated_channels": [],
        }
        md_2: dict[str, Any] = {"step": 2, "node": "data_processor", "source": "loop"}
        saver.put(cfg_2, cast(Any, chk_2), cast(Any, md_2), {})
        saver.put_writes(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": error_cid,
                    "checkpoint_ns": "",
                }
            },
            [("__error__", {"message": "InvalidPayloadException: Malformed packet in channel"})],
            "task_data_processor",
        )

    yield db_file, thread_id, error_cid

    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


class TestCliRootAndHelp:
    """Tests verifying CLI metadata, help output, and subcommands discovery (AC-32)."""

    def test_cli_help_displays_subcommands_ac_32(self, runner: CliRunner) -> None:
        """AC-32: Verify relate-dag --help displays commands and exits with code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = result.output.lower()
        assert "tree" in output
        assert "replay" in output
        assert "forensics" in output

    def test_cli_tree_subcommand_help(self, runner: CliRunner) -> None:
        """Verify relate-dag tree --help displays options and exits with code 0."""
        result = runner.invoke(app, ["tree", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "thread_id" in result.output.lower()

    def test_cli_replay_subcommand_help(self, runner: CliRunner) -> None:
        """Verify relate-dag replay --help displays options and exits with code 0."""
        result = runner.invoke(app, ["replay", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "--branch" in result.output
        assert "checkpoint_id" in result.output.lower()

    def test_cli_forensics_subcommand_help(self, runner: CliRunner) -> None:
        """Verify relate-dag forensics --help displays subcommands and exits with code 0 (AC-32)."""
        result = runner.invoke(app, ["forensics", "--help"])
        assert result.exit_code == 0
        output = result.output.lower()
        assert "loops" in output
        assert "trace-error" in output

    def test_cli_forensics_help_documents_reserved_names(self, runner: CliRunner) -> None:
        """RC-04: group help lists the reserved subcommand names and the leading '-' rule."""
        result = runner.invoke(app, ["forensics", "--help"], env={"COLUMNS": "300"})
        assert result.exit_code == 0
        text = " ".join(result.output.split())
        assert "Reserved subcommand names" in text
        reserved = text.split("Reserved subcommand names", 1)[1]
        for name in ("loops", "trace-error", "trace", "failures", "diagnose"):
            assert name in reserved
        assert "cannot be used as a positional thread ID" in text
        assert "Thread IDs beginning with '-' are not accepted as positional thread IDs" in text

    def test_cli_forensics_loops_subcommand_help(self, runner: CliRunner) -> None:
        """Verify relate-dag forensics loops --help displays options and exits with code 0."""
        result = runner.invoke(app, ["forensics", "loops", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output or "-t" in result.output
        assert "--db" in result.output
        assert "thread_id" in result.output.lower()

    def test_cli_forensics_trace_error_subcommand_help(self, runner: CliRunner) -> None:
        """Verify relate-dag forensics trace-error --help displays options and exits with code 0."""
        result = runner.invoke(app, ["forensics", "trace-error", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "thread_id" in result.output.lower()


class TestCliTreeCommand:
    """Tests verifying DAG tree visualization command (AC-24)."""

    def test_tree_renders_execution_dag(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify relate-dag tree renders Unicode DAG for valid thread with exit code 0."""
        db_file, thread_id, _ = seeded_db
        result = runner.invoke(app, ["tree", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert thread_id in result.output
        assert "router_agent" in result.output
        assert "research_agent" in result.output
        assert "output_node" in result.output

    def test_tree_renders_with_file_alias(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify relate-dag tree accepts --file alias."""
        db_file, thread_id, _ = seeded_db
        result = runner.invoke(app, ["tree", thread_id, "--file", db_file])
        assert result.exit_code == 0
        assert thread_id in result.output

    def test_tree_non_existent_thread_ac_24(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """AC-24: Verify non-existent thread_id exits with code 1 and prints informative error."""
        db_file, _, _ = seeded_db
        result = runner.invoke(app, ["tree", "non-existent-thread-id", "--db", db_file])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "non-existent-thread-id" in result.output
        assert "not found" in result.output.lower()

    def test_tree_non_existent_database_file(self, runner: CliRunner) -> None:
        """Verify non-existent database path exits with code 1 and prints missing file error."""
        missing_db = "non_existent_database_path_12345.rl"
        result = runner.invoke(app, ["tree", "test-thread", "--db", missing_db])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert missing_db in result.output

    def test_tree_empty_thread_id(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify empty thread_id string exits with code 1."""
        db_file, _, _ = seeded_db
        result = runner.invoke(app, ["tree", "   ", "--db", db_file])
        assert result.exit_code == 1
        assert "thread_id must be a non-empty string" in result.output


class TestCliReplayCommand:
    """Tests verifying in-memory time-travel replay branching command."""

    def test_replay_forks_target_checkpoint_into_memory(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify replay from checkpoint clones lineage into memory with exit code 0."""
        db_file, thread_id, checkpoint_ids = seeded_db
        target_cid = checkpoint_ids[1]  # Step 2
        result = runner.invoke(app, ["replay", thread_id, target_cid, "--db", db_file])
        assert result.exit_code == 0
        assert "Sandboxed replay initialized successfully" in result.output
        assert thread_id in result.output
        assert target_cid in result.output
        assert "Step 2" in result.output or "2" in result.output

    def test_replay_with_branch_option(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify replay with explicit --branch option creates designated branch thread."""
        db_file, thread_id, checkpoint_ids = seeded_db
        target_cid = checkpoint_ids[0]  # Step 1
        branch_id = "what-if-experiment-branch"
        result = runner.invoke(
            app,
            ["replay", thread_id, target_cid, "--db", db_file, "--branch", branch_id],
        )
        assert result.exit_code == 0
        assert "Sandboxed replay initialized successfully" in result.output
        assert branch_id in result.output

    def test_replay_invalid_checkpoint_displays_candidates(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify replay with invalid checkpoint exits with code 1 and lists candidates."""
        db_file, thread_id, checkpoint_ids = seeded_db
        invalid_cid = "non-existent-chk-9999"
        result = runner.invoke(app, ["replay", thread_id, invalid_cid, "--db", db_file])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert invalid_cid in result.output
        assert "Available checkpoints:" in result.output
        # Check that existing candidate IDs are displayed
        for cid in checkpoint_ids:
            assert cid in result.output

    def test_replay_non_existent_database_file(self, runner: CliRunner) -> None:
        """Verify replay with non-existent database file exits with code 1."""
        missing_db = "non_existent_replay_db.rl"
        result = runner.invoke(app, ["replay", "th-1", "cid-1", "--db", missing_db])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert missing_db in result.output

    def test_replay_empty_arguments(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify replay with whitespace arguments exits with code 1."""
        db_file, thread_id, checkpoint_ids = seeded_db
        result = runner.invoke(app, ["replay", thread_id, "  ", "--db", db_file])
        assert result.exit_code == 1
        assert "checkpoint_id must be a non-empty string" in result.output

    def test_replay_with_checkpoint_option_flag(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify replay with --checkpoint option flag exits with code 0."""
        db_file, thread_id, checkpoint_ids = seeded_db
        target_cid = checkpoint_ids[1]
        result = runner.invoke(
            app, ["replay", thread_id, "--checkpoint", target_cid, "--db", db_file]
        )
        assert result.exit_code == 0
        assert "Sandboxed replay initialized successfully" in result.output
        assert target_cid in result.output


class TestCliForensicsCommand:
    """Tests verifying forensics diagnostics initial structure."""

    def test_forensics_valid_thread(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify forensics on existing thread outputs diagnostics with exit code 0."""
        db_file, thread_id, _ = seeded_db
        result = runner.invoke(app, ["forensics", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "Thread Forensics Diagnostic" in result.output
        assert thread_id in result.output
        assert "Total Checkpoints: 3" in result.output
        assert "OK" in result.output

    def test_forensics_non_existent_thread(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify forensics on non-existent thread exits with code 1."""
        db_file, _, _ = seeded_db
        result = runner.invoke(app, ["forensics", "unknown-thread-id", "--db", db_file])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "unknown-thread-id" in result.output

    def test_forensics_non_existent_database_file(self, runner: CliRunner) -> None:
        """Verify forensics with missing database exits with code 1."""
        missing_db = "non_existent_forensics_db.rl"
        result = runner.invoke(app, ["forensics", "th-1", "--db", missing_db])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert missing_db in result.output

    def test_forensics_loops_with_loop_thread(
        self, runner: CliRunner, seeded_loop_db: tuple[str, str]
    ) -> None:
        """AC-25: Verify relate-dag forensics loops renders table of detected repeating cycles."""
        db_file, thread_id = seeded_loop_db
        result = runner.invoke(app, ["forensics", "loops", thread_id, "--db", db_file])
        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert sum("Detected Loops" in line for line in lines) == 1
        assert re.search(
            r"│ Cycle Nodes\s+│ Occurrences │ First Checkpoint │ Last Checkpoint │", result.output
        )
        # Four alternating checkpoints hold exactly one ping/pong cycle seen twice, so the table
        # has exactly one data row carrying the node sequence, count and checkpoint interval.
        data_rows = [line for line in lines if "ping_agent" in line or "pong_agent" in line]
        assert len(data_rows) == 1
        assert re.fullmatch(
            r"│ ping_agent -> pong_agent │\s+2 │ chk-loop-step-0\s+│ chk-loop-step-3\s+│",
            data_rows[0],
        )

    def test_forensics_loops_with_linear_thread(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """AC-26: Verify relate-dag forensics loops shows clean confirmation for linear thread."""
        db_file, thread_id, _ = seeded_db
        result = runner.invoke(app, ["forensics", "loops", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "No loops" in result.output or "No repeating cycles" in result.output
        assert thread_id in result.output

    def test_forensics_loops_non_existent_thread(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify relate-dag forensics loops exits with code 1 on non-existent thread."""
        db_file, _, _ = seeded_db
        result = runner.invoke(app, ["forensics", "loops", "non-existent-thread", "--db", db_file])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "non-existent-thread" in result.output

    def test_forensics_loops_non_existent_database_file(self, runner: CliRunner) -> None:
        """Verify relate-dag forensics loops exits with code 1 on missing DB file."""
        missing_db = "missing_loops_db.rl"
        result = runner.invoke(app, ["forensics", "loops", "th-1", "--db", missing_db])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert missing_db in result.output

    def test_forensics_trace_error_with_error_thread(
        self, runner: CliRunner, seeded_error_db: tuple[str, str, str]
    ) -> None:
        """AC-27: Verify relate-dag forensics trace-error renders root cause trace for failed thread."""
        db_file, thread_id, error_cid = seeded_error_db
        result = runner.invoke(app, ["forensics", "trace-error", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "Root Cause Analysis" in result.output
        assert error_cid in result.output
        assert "InvalidPayloadException" in result.output
        assert "Diagnostic Summary" in result.output

    def test_forensics_trace_error_shows_root_tool_name_ac_45(
        self, runner: CliRunner, seeded_error_db: tuple[str, str, str]
    ) -> None:
        """AC-45: trace-error output includes a Root Tool Name line, None when unknown."""
        db_file, thread_id, _ = seeded_error_db
        result = runner.invoke(app, ["forensics", "trace-error", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "Root Tool Name:" in result.output

    def test_forensics_trace_alias(
        self, runner: CliRunner, seeded_error_db: tuple[str, str, str]
    ) -> None:
        """Verify relate-dag forensics trace alias functions identically to trace-error."""
        db_file, thread_id, error_cid = seeded_error_db
        result = runner.invoke(app, ["forensics", "trace", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "Root Cause Analysis" in result.output
        assert error_cid in result.output

    def test_forensics_trace_error_with_healthy_thread(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """AC-28: Verify relate-dag forensics trace-error confirms no error detected on healthy thread."""
        db_file, thread_id, _ = seeded_db
        result = runner.invoke(app, ["forensics", "trace-error", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "No error detected" in result.output
        assert thread_id in result.output

    def test_forensics_trace_error_non_existent_thread(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify relate-dag forensics trace-error exits with code 1 on non-existent thread."""
        db_file, _, _ = seeded_db
        result = runner.invoke(
            app, ["forensics", "trace-error", "unknown-error-thread", "--db", db_file]
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "unknown-error-thread" in result.output

    def test_forensics_trace_error_non_existent_database_file(self, runner: CliRunner) -> None:
        """Verify relate-dag forensics trace-error exits with code 1 on missing DB file."""
        missing_db = "missing_trace_db.rl"
        result = runner.invoke(app, ["forensics", "trace-error", "th-1", "--db", missing_db])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert missing_db in result.output

    def test_forensics_empty_thread_id(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """Verify empty thread_id string exits with code 1 across subcommands."""
        db_file, _, _ = seeded_db
        result_loops = runner.invoke(app, ["forensics", "loops", "   ", "--db", db_file])
        assert result_loops.exit_code == 1
        assert "thread_id must be a non-empty string" in result_loops.output

        result_trace = runner.invoke(app, ["forensics", "trace-error", "   ", "--db", db_file])
        assert result_trace.exit_code == 1
        assert "thread_id must be a non-empty string" in result_trace.output

        result_diag = runner.invoke(app, ["forensics", "   ", "--db", db_file])
        assert result_diag.exit_code == 1
        assert "thread_id must be a non-empty string" in result_diag.output


class TestCliForensicsToolName:
    """The combined forensics summary names the failing tool (AC-67, AC-68)."""

    def test_forensics_summary_shows_root_tool_name_ac_67(
        self, runner: CliRunner, seeded_tool_error_db: tuple[str, str]
    ) -> None:
        """AC-67: the combined summary shows tool_name alongside call_id."""
        db_file, thread_id = seeded_tool_error_db
        result = runner.invoke(app, ["forensics", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "Root Tool Call ID:" in result.output
        assert "call-fetch-1" in result.output
        assert "Root Tool Name:" in result.output
        assert "fetch_source" in result.output

    def test_forensics_summary_without_failure_has_no_root_tool_ac_68(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """AC-68: a thread with no failing tool call shows no root tool name or call id."""
        db_file, thread_id, _ = seeded_db
        result = runner.invoke(app, ["forensics", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert "Root Tool" not in result.output
        assert "Status:            OK" in result.output


def _file_state(db_file: str) -> tuple[str, list[str]]:
    """Return the store's sha256 and the sorted listing of its directory."""
    path = Path(db_file)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, sorted(p.name for p in path.parent.iterdir())


def _inspection_args(command: str, thread_id: str, checkpoint_id: str, db_file: str) -> list[str]:
    commands = {
        "tree": ["tree", thread_id, "--db", db_file],
        "forensics": ["forensics", thread_id, "--db", db_file],
        "loops": ["forensics", "loops", thread_id, "--db", db_file],
        "trace-error": ["forensics", "trace-error", thread_id, "--db", db_file],
        "replay": ["replay", thread_id, "--checkpoint", checkpoint_id, "--db", db_file],
    }
    return commands[command]


class TestCliReadOnlyInspection:
    """Inspection commands leave the store byte-identical (AC-75, AC-84, ADR-0015)."""

    @pytest.mark.parametrize("command", ["tree", "forensics", "loops", "trace-error", "replay"])
    def test_command_leaves_store_unchanged_ac_84(
        self,
        runner: CliRunner,
        seeded_db: tuple[str, str, list[str]],
        command: str,
    ) -> None:
        """AC-84: the command succeeds; sha256 and directory listing are unchanged."""
        db_file, thread_id, checkpoint_ids = seeded_db
        before = _file_state(db_file)

        result = runner.invoke(
            app, _inspection_args(command, thread_id, checkpoint_ids[1], db_file)
        )

        assert result.exit_code == 0
        assert _file_state(db_file) == before

    @pytest.mark.parametrize("command", ["tree", "forensics", "loops", "trace-error", "replay"])
    def test_exit_code_matches_a_writable_copy_ac_84(
        self,
        runner: CliRunner,
        seeded_db: tuple[str, str, list[str]],
        tmp_path: Path,
        command: str,
    ) -> None:
        """AC-84: the exit code equals the one from a copy that a writable saver has opened."""
        db_file, thread_id, checkpoint_ids = seeded_db
        copy_file = str(tmp_path / "writable_copy.rl")
        shutil.copy(db_file, copy_file)
        with RelateSaver(db_path=copy_file) as writable:
            writable.get_tuple({"configurable": {"thread_id": thread_id}})

        args = _inspection_args(command, thread_id, checkpoint_ids[1], db_file)
        copy_args = _inspection_args(command, thread_id, checkpoint_ids[1], copy_file)
        assert runner.invoke(app, args).exit_code == runner.invoke(app, copy_args).exit_code

    @pytest.mark.parametrize("command", ["forensics", "trace-error"])
    def test_tool_failure_commands_leave_store_unchanged_ac_84(
        self,
        runner: CliRunner,
        seeded_tool_error_db: tuple[str, str],
        command: str,
    ) -> None:
        """AC-84: a store with a failed tool call is inspected without modification."""
        db_file, thread_id = seeded_tool_error_db
        before = _file_state(db_file)
        result = runner.invoke(app, _inspection_args(command, thread_id, "", db_file))
        assert result.exit_code == 0
        assert "fetch_source" in result.output
        assert _file_state(db_file) == before

    def test_loops_command_leaves_loop_store_unchanged_ac_84(
        self, runner: CliRunner, seeded_loop_db: tuple[str, str]
    ) -> None:
        """AC-84: loop detection on a store with a loop leaves it unchanged."""
        db_file, thread_id = seeded_loop_db
        before = _file_state(db_file)
        result = runner.invoke(app, ["forensics", "loops", thread_id, "--db", db_file])
        assert result.exit_code == 0
        assert _file_state(db_file) == before

    @pytest.mark.parametrize("command", ["tree", "forensics", "loops", "trace-error", "replay"])
    def test_missing_file_is_reported_and_not_created_ac_85(
        self, runner: CliRunner, tmp_path: Path, command: str
    ) -> None:
        """AC-85: a missing store exits 1 with the not-found message and is not created."""
        missing = str(tmp_path / "missing.rl")
        result = runner.invoke(app, _inspection_args(command, "t", "c", missing))
        assert result.exit_code == 1
        assert "Database file not found" in result.output
        assert list(tmp_path.iterdir()) == []

    def test_memory_database_keeps_working_ac_84(self, runner: CliRunner) -> None:
        """An empty :memory: store keeps a writable saver, so it fails as before, not as read-only."""
        result = runner.invoke(app, ["tree", "t", "--db", ":memory:"])
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "read-only" not in result.output.lower()


def _console_script() -> str | None:
    """Return the relate-dag console script installed next to the running interpreter."""
    scripts_dir = str(Path(sys.executable).parent)
    return shutil.which("relate-dag", path=scripts_dir) or shutil.which("relate-dag")


def _run_cli_cp1252(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run relate-dag in a subprocess whose stdout and stderr are cp1252 pipes.

    Uses the installed console script, which calls the Typer app directly; falls back to
    ``python -m`` only when the script is not installed.
    """
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "COLUMNS": "200"}
    env.pop("PYTHONUTF8", None)
    script = _console_script()
    command = (
        [script, *args]
        if script
        else [sys.executable, "-m", "langgraph_checkpoint_relate.cli", *args]
    )
    return subprocess.run(command, capture_output=True, env=env, timeout=120, check=False, cwd=cwd)


class TestCliCp1252Output:
    """Every relate-dag command prints only text a cp1252 stdout can encode (AC-59..AC-61)."""

    def test_all_commands_exit_zero_on_cp1252_stdout_ac_59_ac_60(
        self,
        seeded_db: tuple[str, str, list[str]],
        seeded_loop_db: tuple[str, str],
        seeded_tool_error_db: tuple[str, str],
    ) -> None:
        """AC-59, AC-60: each command exits 0 with its expected text on a cp1252 pipe."""
        clean_db, clean_thread, clean_ids = seeded_db
        loop_db, loop_thread = seeded_loop_db
        tool_db, tool_thread = seeded_tool_error_db

        cases: list[tuple[list[str], str]] = [
            (["--help"], "RelateGraphSaver CLI"),
            (["tree", clean_thread, "--db", clean_db], clean_thread),
            (["tree", clean_thread, "--db", clean_db], clean_ids[0]),
            (["replay", clean_thread, clean_ids[1], "--db", clean_db], clean_ids[1]),
            (["forensics", clean_thread, "--db", clean_db], "Total Checkpoints: 3"),
            (["forensics", tool_thread, "--db", tool_db], "fetch_source"),
            (["forensics", "loops", loop_thread, "--db", loop_db], "Detected Loops"),
            (["forensics", "loops", clean_thread, "--db", clean_db], "No loops"),
            (["forensics", "trace-error", tool_thread, "--db", tool_db], "fetch_source"),
            (["forensics", "trace-error", clean_thread, "--db", clean_db], "No error detected"),
        ]
        for args, expected in cases:
            proc = _run_cli_cp1252(args)
            stdout = proc.stdout.decode("cp1252")
            stderr = proc.stderr.decode("cp1252")
            assert proc.returncode == 0, f"{args}: {stderr}"
            assert "UnicodeEncodeError" not in stderr, args
            assert expected in stdout, f"{args}: {stdout}"

    def test_no_finding_messages_are_ascii_ac_60(
        self,
        seeded_db: tuple[str, str, list[str]],
    ) -> None:
        """AC-60: the no-finding messages use ASCII only."""
        db_file, thread_id, _ = seeded_db
        for args in (
            ["forensics", "trace-error", thread_id, "--db", db_file],
            ["forensics", "loops", thread_id, "--db", db_file],
        ):
            proc = _run_cli_cp1252(args)
            assert proc.returncode == 0, proc.stderr.decode("cp1252")
            assert proc.stdout.decode("cp1252").isascii(), args

    def test_cp1252_suite_uses_installed_console_script_ac_59(self) -> None:
        """AC-59: the subprocess checks exercise the console script, not only ``python -m``."""
        script = _console_script()
        assert script is not None, "relate-dag console script is not installed in this venv"
        assert Path(script).resolve().parent == Path(sys.executable).resolve().parent

    def test_stored_non_cp1252_text_does_not_crash_tree_ac_59(self, tmp_path: Path) -> None:
        """AC-59: a stored node name outside cp1252 is replaced, not raised, on output."""
        db_file = str(tmp_path / "emoji.rl")
        thread_id = "emoji-thread"
        with RelateSaver(db_path=db_file) as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp: Any = dict(empty_checkpoint())
            cp["id"] = "1f1e314c-8a21-6b3e-8000-000000000001"
            saver.put(cfg, cp, {"step": 1, "source": "loop", "node": "\U0001f916 worker"}, {})  # type: ignore[typeddict-unknown-key]

        proc = _run_cli_cp1252(["tree", thread_id, "--db", db_file])
        stderr = proc.stderr.decode("cp1252")
        assert proc.returncode == 0, stderr
        assert "UnicodeEncodeError" not in stderr
        stdout = proc.stdout.decode("cp1252")
        assert thread_id in stdout
        assert "worker" in stdout

    def test_cli_source_is_ascii_ac_61(self) -> None:
        """AC-61: the CLI module contains only ASCII characters."""
        import langgraph_checkpoint_relate.cli as cli_module

        source = Path(cli_module.__file__).read_text(encoding="utf-8")
        assert source.isascii()

    def test_replay_accepts_full_id_copied_from_tree_ac_64(
        self,
        runner: CliRunner,
        seeded_db: tuple[str, str, list[str]],
    ) -> None:
        """AC-64: an id shown by tree is accepted by replay on the same store."""
        db_file, thread_id, checkpoint_ids = seeded_db
        tree_result = runner.invoke(app, ["tree", thread_id, "--db", db_file])
        assert tree_result.exit_code == 0
        for cid in checkpoint_ids:
            assert cid in tree_result.output
            replay_result = runner.invoke(app, ["replay", thread_id, cid, "--db", db_file])
            assert replay_result.exit_code == 0
            assert cid in replay_result.output


_STEP_ID = re.compile(r"Step (-?\d+): [^()]*\(id:\s*([0-9a-f-]{36})\)")


@pytest.fixture
def demo_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict[int, str]]:
    """A copy of the committed demo store, plus its step -> checkpoint id map from ``tree``.

    Runs from the repository root so the relative graph spec resolves. Ids come from the
    ``tree`` output, not from constants, because the committed store is rebuilt from time to time.
    """
    monkeypatch.chdir(REPO_ROOT)
    db_file = str(tmp_path / "demo_copy.rl")
    shutil.copy(DEMO_STORE, db_file)
    result = CliRunner().invoke(app, ["tree", DEMO_THREAD, "--db", db_file])
    assert result.exit_code == 0, result.output
    steps = {int(step): cid for step, cid in _STEP_ID.findall(result.output)}
    assert {-1, 0} <= steps.keys()
    return db_file, steps


def _replay_args(db_file: str, checkpoint_id: str, *extra: str) -> list[str]:
    return ["replay", DEMO_THREAD, checkpoint_id, *extra, "--db", db_file]


def _squash(text: str) -> str:
    """Collapse whitespace so assertions do not depend on terminal line wrapping."""
    return " ".join(text.split())


_EDIT_ARGS = ["--graph", GRAPH_SPEC, "--set-message", f"0={OK_URL}"]


class TestCliReplayGraph:
    """``replay --graph`` re-runs the user's graph in a sandbox and prints the diff."""

    def test_without_graph_keeps_summary_and_adds_hint_ac_109(
        self, runner: CliRunner, seeded_db: tuple[str, str, list[str]]
    ) -> None:
        """AC-109: the six summary lines stay and one hint line names --graph."""
        db_file, thread_id, checkpoint_ids = seeded_db
        result = runner.invoke(app, ["replay", thread_id, checkpoint_ids[1], "--db", db_file])
        assert result.exit_code == 0
        # Compare whitespace-squashed text: the Isolation line wraps on an 80-column terminal.
        text = _squash(result.output)
        expected_start = _squash(
            "Sandboxed replay initialized successfully. "
            f"Source Thread: {thread_id} Replay Thread: {thread_id} "
            f"Checkpoint ID: {checkpoint_ids[1]} Replay Step: 2 "
            "Isolation: Isolated :memory: sandbox (source database unchanged) "
        )
        assert text.startswith(expected_start)
        hint = text[len(expected_start) :].strip()
        assert hint.startswith("Hint:")
        assert "--graph" in hint
        assert text.count("Hint:") == 1

    def test_edited_replay_prints_result_and_tables_ac_110(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-110: the edited replay shows the result, the fixed tool call and the state diff."""
        db_file, steps = demo_copy
        result = runner.invoke(app, _replay_args(db_file, steps[0], *_EDIT_ARGS))
        assert result.exit_code == 0, result.output
        output = result.output
        assert "Graph replay completed" in output
        assert "Replay result" in output
        assert steps[0] in output
        assert "New final:" in output
        assert "Recorded final:" in output
        assert "Stopped at interrupt: no" in output
        assert "Tool calls" in output
        assert "fetch_source #1" in output
        assert "error -> ok" in output
        assert f"url: {UNREACHABLE_URL} -> {OK_URL}" in output
        assert "Final state changes" in output
        assert "messages[0]" in output
        assert "No tool calls or state changed." not in output
        assert output.isascii()

    def test_committed_demo_store_command_leaves_it_untouched_ac_110(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-110: the documented command works on the committed store without changing it."""
        monkeypatch.chdir(REPO_ROOT)
        db_arg = "examples/data/demo.rl"
        before = _file_state(str(DEMO_STORE))
        tree = runner.invoke(app, ["tree", DEMO_THREAD, "--db", db_arg])
        step0 = {int(step): cid for step, cid in _STEP_ID.findall(tree.output)}[0]

        result = runner.invoke(
            app,
            ["replay", DEMO_THREAD, step0, "--graph", GRAPH_SPEC]
            + ["--set-message", f"0={OK_URL}", "--db", db_arg],
        )

        assert result.exit_code == 0, result.output
        assert "fetch_source #1" in result.output
        assert "error -> ok" in result.output
        assert _file_state(str(DEMO_STORE)) == before

    def test_set_message_repeats_and_short_graph_flag_works_ac_110(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-110, AC-114: --set-message may repeat with distinct indexes; -g precedes --db."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app,
            _replay_args(
                db_file,
                steps[1],
                "-g",
                GRAPH_SPEC,
                "--set-message",
                f"0={OK_URL}",
                "--set-message",
                "1=edited reply",
            ),
        )
        assert result.exit_code == 0, result.output
        assert "Replay result" in result.output
        assert "messages[0]" in result.output
        assert "messages[1]" in result.output

    def test_duplicate_set_message_index_is_rejected_ac_153(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-153: a repeated --set-message INDEX exits 1, names the index and runs no graph."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app,
            _replay_args(
                db_file,
                steps[0],
                "--graph",
                GRAPH_SPEC,
                "--set-message",
                f"0={OK_URL}",
                "--set-message",
                "0=another text",
            ),
        )
        assert result.exit_code == 1
        output = _squash(result.output)
        assert "Error:" in output
        assert "--set-message index 0" in output
        assert "Replay result" not in output
        assert "Traceback" not in output

    def test_unedited_replay_reports_no_changes_ac_111(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-111: an unchanged graph and no edits give an empty diff."""
        db_file, steps = demo_copy
        result = runner.invoke(app, _replay_args(db_file, steps[0], "--graph", GRAPH_SPEC))
        assert result.exit_code == 0, result.output
        assert "Replay result" in result.output
        assert "No tool calls or state changed." in result.output
        assert "Tool calls" not in result.output

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--update", '{"a": 1}'), ("--set-message", "0=x"), ("--as-node", "agent")],
    )
    def test_edit_flag_without_graph_is_rejected_ac_112(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], flag: str, value: str
    ) -> None:
        """AC-112: an edit flag without --graph exits 1 with an Error line naming --graph."""
        db_file, steps = demo_copy
        result = runner.invoke(app, _replay_args(db_file, steps[0], flag, value))
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "--graph" in result.output
        assert flag in result.output
        assert "Sandboxed replay initialized" not in result.output

    @pytest.mark.parametrize("update_args", [[], ["--update", "{}"]])
    def test_as_node_without_an_edit_is_rejected_ac_152(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], update_args: list[str]
    ) -> None:
        """AC-152: --as-node with no effective edit exits 1 and runs no graph."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app,
            _replay_args(
                db_file, steps[0], "--graph", GRAPH_SPEC, "--as-node", "agent", *update_args
            ),
        )
        assert result.exit_code == 1
        output = _squash(result.output)
        assert "Error:" in output
        assert "--as-node needs an edit" in output
        assert "Replay result" not in output
        assert "Traceback" not in output

    @pytest.mark.parametrize("bad_update", ["{not json", "[1, 2]", '"text"', "42", ""])
    def test_update_must_be_a_json_object_ac_113(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], bad_update: str
    ) -> None:
        """AC-113: invalid JSON or a non-object value exits 1 with an Error line."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app,
            _replay_args(db_file, steps[0], "--graph", GRAPH_SPEC, "--update", bad_update),
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "--update" in result.output
        assert "Traceback" not in result.output

    @pytest.mark.parametrize(
        "bad_value",
        ["no-equals-sign", "first=x", "-1=x", "1.5=x", "=x", "99=out of range"],
    )
    def test_set_message_forms_and_index_are_validated_ac_114(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], bad_value: str
    ) -> None:
        """AC-114: a malformed or out-of-range --set-message exits 1 with an Error line."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app,
            _replay_args(db_file, steps[0], "--graph", GRAPH_SPEC, "--set-message", bad_value),
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Traceback" not in result.output

    def test_set_message_splits_on_first_equals_ac_114(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-114: only the first '=' separates the index, so the text may contain '='."""
        db_file, steps = demo_copy
        text = "https://example.org/report?a=b"
        result = runner.invoke(
            app,
            _replay_args(db_file, steps[0], "--graph", GRAPH_SPEC, "--set-message", f"0={text}"),
        )
        assert result.exit_code == 0, result.output
        assert text in _squash(result.output)

    @pytest.mark.parametrize(
        "spec", ["nope.py:missing", "no-colon-spec", "examples/tool_agent.py:not_there"]
    )
    def test_bad_graph_spec_names_the_spec_without_traceback_ac_115(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], spec: str
    ) -> None:
        """AC-115: a spec that fails to load exits 1, repeats the spec and prints no traceback."""
        db_file, steps = demo_copy
        result = runner.invoke(app, _replay_args(db_file, steps[0], "--graph", spec))
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert spec in _squash(result.output)
        assert "Traceback" not in result.output

    def test_as_node_rejected_by_the_graph_prints_hint_ac_116(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-116: an --as-node the graph rejects exits 1 with the --as-node hint."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app,
            _replay_args(db_file, steps[0], *_EDIT_ARGS, "--as-node", "no_such_node"),
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Hint: pass --as-node with the node that should be treated as the writer." in (
            _squash(result.output)
        )
        assert "Traceback" not in result.output

    def test_graph_code_failure_is_reported_without_traceback(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], tmp_path: Path
    ) -> None:
        """Any other exception from user graph code is reported as 'replay failed: <type>: <msg>'."""
        db_file, steps = demo_copy
        graph_file = tmp_path / "broken_graph.py"
        graph_file.write_text(
            "def make(checkpointer=None):\n    raise RuntimeError('factory exploded')\n",
            encoding="utf-8",
        )
        spec = f"{graph_file}:make"
        result = runner.invoke(app, _replay_args(db_file, steps[0], "--graph", spec))
        assert result.exit_code == 1
        assert "Error: replay failed: RuntimeError: factory exploded" in _squash(result.output)
        assert "Traceback" not in result.output

    def test_ac155_value_error_from_a_graph_node_keeps_its_context(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], tmp_path: Path
    ) -> None:
        """AC-155: a ValueError raised by a node is not mistaken for a replay input error."""
        db_file, steps = demo_copy
        graph_file = tmp_path / "boom_graph.py"
        source_lines = [
            "from langgraph.graph import END, START, MessagesState, StateGraph",
            "",
            "def explode(state: MessagesState) -> dict:",
            "    raise ValueError('boom')",
            "",
            "def make(checkpointer=None):",
            "    builder = StateGraph(MessagesState)",
            "    builder.add_node('agent', explode)",
            "    builder.add_node('tools', explode)",
            "    builder.add_edge(START, 'agent')",
            "    builder.add_edge('agent', 'tools')",
            "    builder.add_edge('tools', END)",
            "    return builder.compile(checkpointer=checkpointer)",
        ]
        graph_file.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
        spec = f"{graph_file}:make"
        result = runner.invoke(app, _replay_args(db_file, steps[0], "--graph", spec))
        assert result.exit_code == 1, result.output
        assert "replay failed: ValueError: boom" in _squash(result.output)
        assert "Traceback" not in result.output

    def test_ac166_invalid_update_error_during_graph_execution_does_not_print_as_node_hint(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], tmp_path: Path
    ) -> None:
        """AC-166: InvalidUpdateError during graph execution prints failure without --as-node hint."""
        db_file, steps = demo_copy
        graph_file = tmp_path / "boom_graph.py"
        source_lines = [
            "from langgraph.errors import InvalidUpdateError",
            "from langgraph.graph import END, START, MessagesState, StateGraph",
            "",
            "def explode(state: MessagesState) -> dict:",
            "    raise InvalidUpdateError('concurrent update rejected')",
            "",
            "def make(checkpointer=None):",
            "    builder = StateGraph(MessagesState)",
            "    builder.add_node('agent', explode)",
            "    builder.add_node('tools', explode)",
            "    builder.add_edge(START, 'agent')",
            "    builder.add_edge('agent', 'tools')",
            "    builder.add_edge('tools', END)",
            "    return builder.compile(checkpointer=checkpointer)",
        ]
        graph_file.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
        spec = f"{graph_file}:make"
        result = runner.invoke(app, _replay_args(db_file, steps[0], "--graph", spec))
        assert result.exit_code == 1, result.output
        assert "Error: replay failed: InvalidUpdateError: concurrent update rejected" in _squash(
            result.output
        )
        assert "--as-node" not in result.output
        assert "Hint:" not in result.output
        assert "Traceback" not in result.output

    @pytest.mark.parametrize("spec", [r"C:\work\graph.py", "path/to/graph.py"])
    def test_ac165_file_spec_without_attribute_rejected_by_cli(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]], spec: str
    ) -> None:
        """AC-165: file path spec without attribute is rejected by CLI with code 1 and spec format error."""
        db_file, steps = demo_copy
        result = runner.invoke(app, _replay_args(db_file, steps[0], "--graph", spec))
        assert result.exit_code == 1
        assert "must look like 'module:attribute' or 'path/to/file.py:attribute'" in _squash(
            result.output
        )
        assert "Traceback" not in result.output

    def test_branch_with_graph_is_rejected(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """--branch cannot be combined with --graph."""
        db_file, steps = demo_copy
        result = runner.invoke(
            app, _replay_args(db_file, steps[0], "--graph", GRAPH_SPEC, "--branch", "what-if")
        )
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "--branch" in result.output

    def test_unknown_checkpoint_lists_candidates_with_graph(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """An unknown checkpoint keeps the existing error text when --graph is given."""
        db_file, _ = demo_copy
        result = runner.invoke(app, _replay_args(db_file, "no-such-checkpoint", *_EDIT_ARGS))
        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Available checkpoints:" in result.output

    def test_step_without_messages_suggests_next_checkpoint_ac_151(
        self, runner: CliRunner, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-151: editing a message at the input checkpoint points at the step 0 checkpoint."""
        db_file, steps = demo_copy
        before = _file_state(db_file)
        result = runner.invoke(app, _replay_args(db_file, steps[-1], *_EDIT_ARGS))
        assert result.exit_code == 1
        squashed = _squash(result.output)
        assert "Error:" in squashed
        assert "no messages yet" in squashed
        assert "next (step 0) checkpoint" in squashed
        assert steps[0] in squashed
        assert _file_state(db_file) == before

    @pytest.mark.parametrize(
        ("variant", "expected_exit"),
        [
            ("summary", 0),
            ("graph_edit", 0),
            ("graph_unedited", 0),
            ("bad_spec", 1),
            ("bad_as_node", 1),
            ("bad_update", 1),
            ("step_without_messages", 1),
            ("unknown_checkpoint", 1),
        ],
    )
    def test_store_and_directory_are_unchanged_ac_117(
        self,
        runner: CliRunner,
        demo_copy: tuple[str, dict[int, str]],
        variant: str,
        expected_exit: int,
    ) -> None:
        """AC-117: sha256 and directory listing are unchanged, succeeding or failing."""
        db_file, steps = demo_copy
        variants: dict[str, list[str]] = {
            "summary": _replay_args(db_file, steps[0]),
            "graph_edit": _replay_args(db_file, steps[0], *_EDIT_ARGS),
            "graph_unedited": _replay_args(db_file, steps[0], "--graph", GRAPH_SPEC),
            "bad_spec": _replay_args(db_file, steps[0], "--graph", "nope.py:missing"),
            "bad_as_node": _replay_args(
                db_file, steps[0], *_EDIT_ARGS, "--as-node", "no_such_node"
            ),
            "bad_update": _replay_args(db_file, steps[0], "-g", GRAPH_SPEC, "--update", "[1]"),
            "step_without_messages": _replay_args(db_file, steps[-1], *_EDIT_ARGS),
            "unknown_checkpoint": _replay_args(db_file, "no-such-checkpoint", *_EDIT_ARGS),
        }
        before = _file_state(db_file)
        result = runner.invoke(app, variants[variant])
        assert result.exit_code == expected_exit, result.output
        assert _file_state(db_file) == before

    def test_edited_replay_on_cp1252_stdout_ac_118(
        self, demo_copy: tuple[str, dict[int, str]]
    ) -> None:
        """AC-118: the AC-110 command exits 0 on a cp1252 pipe and prints 'error -> ok'."""
        db_file, steps = demo_copy
        proc = _run_cli_cp1252(_replay_args(db_file, steps[0], *_EDIT_ARGS), cwd=REPO_ROOT)
        stderr = proc.stderr.decode("cp1252")
        assert proc.returncode == 0, stderr
        assert "UnicodeEncodeError" not in stderr
        stdout = proc.stdout.decode("cp1252")
        assert "error -> ok" in stdout
        assert "fetch_source #1" in stdout


class _OriginState(TypedDict):
    """State for a planner -> agent -> tools graph whose tool consumes the planner's value."""

    target: str
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def _fetch_package(url: str) -> str:
    """Fetch a package; fails for every url unless it ends with 'ok'."""
    if not url.endswith("ok"):
        raise RuntimeError(f"no such package: {url}")
    return f"fetched {url}"


_fetch_package.name = "fetch_package"


def _run_origin_graph(db_file: str, thread_id: str, target: str) -> None:
    """Persist one run in which the planner writes ``target`` and the tool call uses it."""

    def planner(state: _OriginState) -> dict[str, Any]:
        return {"target": target}

    def agent(state: _OriginState) -> dict[str, Any]:
        call = {"id": "call-origin-1", "name": "fetch_package", "args": {"url": state["target"]}}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(_OriginState)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([_fetch_package], handle_tool_errors=True))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    with RelateSaver(db_path=db_file) as saver:
        builder.compile(checkpointer=saver).invoke(
            {"messages": [HumanMessage(content="look the package up")], "target": ""},
            {"configurable": {"thread_id": thread_id}},
        )


def _flatten(output: str) -> str:
    """Collapse panel borders and wrapping so a wrapped line can be matched as one string."""
    return re.sub(r"[|│\s]+", " ", output)


def test_trace_error_prints_bad_argument_origin_ac_122(runner: CliRunner, tmp_path: Path) -> None:
    """AC-122: trace-error names the argument, node, step and hops of the bad argument."""
    db_file = str(tmp_path / "origin.rl")
    _run_origin_graph(db_file, "origin-fail", "pkg-alpha-9000")

    result = runner.invoke(app, ["forensics", "trace-error", "origin-fail", "--db", db_file])

    assert result.exit_code == 0
    assert (
        "Bad argument origin: url came from node planner (step 0, 2 hops before failure)"
        in _flatten(result.output)
    )


def test_trace_error_explains_both_distance_figures_ac_122(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-122: each distance figure says what it counts and keeps its value."""
    db_file = str(tmp_path / "origin.rl")
    _run_origin_graph(db_file, "origin-fail", "pkg-alpha-9000")

    result = runner.invoke(app, ["forensics", "trace-error", "origin-fail", "--db", db_file])

    flat = _flatten(result.output)
    assert result.exit_code == 0
    assert "2 hops before failure)" in flat
    assert (
        "hops: checkpoint steps from the write where the argument first appeared "
        "to the error checkpoint" in flat
    )
    assert re.search(
        r"Causal Path Length: \d+ \(checkpoint steps from the offending write "
        r"to the error checkpoint\)",
        flat,
    )


def test_trace_error_omits_origin_line_without_origin_ac_122(
    runner: CliRunner, seeded_error_db: tuple[str, str, str]
) -> None:
    """AC-122: a failure with no traceable tool argument prints no origin line."""
    db_file, thread_id, _ = seeded_error_db

    result = runner.invoke(app, ["forensics", "trace-error", thread_id, "--db", db_file])

    assert result.exit_code == 0
    assert "Root Tool Name:" in result.output
    assert "Bad argument origin" not in result.output


def test_combined_forensics_prints_bad_argument_origin_ac_122(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-122: the combined forensics summary shows the same origin line as trace-error."""
    db_file = str(tmp_path / "origin.rl")
    _run_origin_graph(db_file, "origin-fail", "pkg-alpha-9000")

    result = runner.invoke(app, ["forensics", "origin-fail", "--db", db_file])

    flat = _flatten(result.output)
    assert result.exit_code == 0
    assert "Bad argument origin: url came from node planner (step 0, 2 hops before failure)" in flat
    assert "hops: checkpoint steps from the write where the argument first appeared" in flat
    assert "Status: ERROR DETECTED" in flat


def test_combined_forensics_omits_origin_line_without_origin_ac_122(
    runner: CliRunner, seeded_error_db: tuple[str, str, str]
) -> None:
    """AC-122: with no traceable argument the combined summary has no origin line or blank gap."""
    db_file, thread_id, _ = seeded_error_db

    result = runner.invoke(app, ["forensics", thread_id, "--db", db_file])

    assert result.exit_code == 0
    assert "Root Tool Name:" in result.output
    assert "Bad argument origin" not in result.output
    assert not re.search(r"Root Tool Name:.*\n[│|\s]*\n", result.output)


def test_trace_error_demo_tool_fail_still_names_the_tool_rc_05(runner: CliRunner) -> None:
    """RC-05: the committed demo store still reports fetch_source and its call id."""
    result = runner.invoke(app, ["forensics", "trace-error", DEMO_THREAD, "--db", str(DEMO_STORE)])

    assert result.exit_code == 0
    assert "fetch_source" in result.output
    assert re.search(r"call_[0-9a-f]{32}", result.output)


def test_failures_prints_table_for_failing_tools_ac_123(runner: CliRunner, tmp_path: Path) -> None:
    """AC-123: the table lists each failing tool with its counts, threads and origin nodes."""
    db_file = str(tmp_path / "failures.rl")
    _run_origin_graph(db_file, "origin-fail", "pkg-alpha-9000")

    result = runner.invoke(app, ["forensics", "failures", "--db", db_file])

    assert result.exit_code == 0
    for column in ("Tool", "Failures", "Calls", "Threads", "Origin nodes"):
        assert column in result.output
    row = next(line for line in result.output.splitlines() if "fetch_package" in line)
    assert [cell.strip() for cell in row.strip("|").split("|")] == [
        "fetch_package",
        "1",
        "1",
        "origin-fail",
        "planner",
    ]


def test_failures_on_demo_store_lists_fetch_source_ac_124(runner: CliRunner) -> None:
    """AC-124: the committed demo store shows a fetch_source row."""
    result = runner.invoke(app, ["forensics", "failures", "--db", str(DEMO_STORE)])

    assert result.exit_code == 0
    assert "fetch_source" in result.output


def test_failures_leaves_demo_store_unchanged_ac_124(runner: CliRunner) -> None:
    """ADR-0015: inspecting the committed store with failures does not modify it."""
    before = DEMO_STORE.read_bytes()

    runner.invoke(app, ["forensics", "failures", "--db", str(DEMO_STORE)])

    assert DEMO_STORE.read_bytes() == before


def test_failures_without_failing_calls_prints_message_ac_125(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-125: a store with no failing tool call prints the message and exits 0."""
    db_file = str(tmp_path / "no_failures.rl")
    _run_origin_graph(db_file, "origin-ok", "pkg-ok")

    result = runner.invoke(app, ["forensics", "failures", "--db", db_file])

    assert result.exit_code == 0
    assert "No failing tool calls found." in result.output


def test_failures_missing_database_fails_without_creating_it_ac_125(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The missing-file check runs before opening, so the path is not created."""
    missing = tmp_path / "missing.rl"

    result = runner.invoke(app, ["forensics", "failures", "--db", str(missing)])

    assert result.exit_code == 1
    assert "Database file not found" in result.output
    assert not missing.exists()


def test_failures_is_not_read_as_a_thread_id_rc_04(runner: CliRunner, tmp_path: Path) -> None:
    """RC-04: 'forensics failures' routes to its own command, not the thread diagnostics."""
    db_file = str(tmp_path / "no_failures.rl")
    _run_origin_graph(db_file, "origin-ok", "pkg-ok")

    result = runner.invoke(app, ["forensics", "failures", "--db", db_file])

    assert "Thread 'failures'" not in result.output
    assert "not found" not in result.output


_MARKUP_ERROR_TEXT = "bad tag [/b] and [red]x[/red]"


def _seed_markup_error_store(db_file: str, thread_id: str) -> None:
    """Persist one failed tool call whose error text looks like Rich markup."""
    with RelateSaver(db_path=db_file) as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        cp: Any = dict(empty_checkpoint())
        cp["id"] = "1f1e314c-8a21-6b3e-8000-000000000001"
        saved_cfg = saver.put(cfg, cp, {"step": 1, "source": "loop", "parents": {}}, {})
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "call-markup-1", "name": "fetch_source", "args": {}}],
        )
        saver.put_writes(saved_cfg, [("messages", [ai])], task_id="task-ai")
        tool_msg = ToolMessage(
            content=f"Error: {_MARKUP_ERROR_TEXT}", tool_call_id="call-markup-1", status="error"
        )
        saver.put_writes(saved_cfg, [("messages", [tool_msg])], task_id="task-tool")


@pytest.mark.parametrize("command", [["forensics", "trace-error"], ["forensics", "diagnose"]])
def test_ac160_root_cause_panel_prints_stored_markup_literally(
    runner: CliRunner, tmp_path: Path, command: list[str]
) -> None:
    """AC-160: tool error text that looks like Rich markup prints as written, never styled."""
    db_file = str(tmp_path / "markup.rl")
    _seed_markup_error_store(db_file, "markup-thread")

    result = runner.invoke(app, [*command, "markup-thread", "--db", db_file])

    assert result.exit_code == 0, result.output
    assert "Unexpected error" not in result.output
    assert "[/b]" in result.output
    assert "[red]x[/red]" in result.output


@pytest.mark.parametrize("command", ["trace-error", "diagnose"])
def test_ac160_markup_like_thread_id_prints_literally(
    runner: CliRunner, tmp_path: Path, command: str
) -> None:
    """AC-160: a thread id that looks like markup is escaped in headings and panel titles."""
    db_file = str(tmp_path / "markup_thread.rl")
    thread_id = "t-[red]x[/red]"
    _seed_markup_error_store(db_file, thread_id)

    result = runner.invoke(app, ["forensics", command, thread_id, "--db", db_file])

    assert result.exit_code == 0, result.output
    assert "t-[red]x[/red]" in result.output
    if command == "trace-error":
        assert "Root Cause Analysis: t-[red]x[/red]" in result.output
    else:
        assert "Thread Forensics Diagnostic: t-[red]x[/red]" in result.output


@pytest.mark.parametrize("command", ["trace-error", "diagnose"])
def test_ac160_error_paths_print_markup_like_text_literally(
    runner: CliRunner, tmp_path: Path, command: str
) -> None:
    """AC-160: not-found and missing-file errors echo markup-like input without crashing."""
    db_file = str(tmp_path / "markup_errors.rl")
    _seed_markup_error_store(db_file, "markup-thread")
    thread_id = "no-such-[/b]-[red]x[/red]"

    unknown = runner.invoke(app, ["forensics", command, thread_id, "--db", db_file])
    missing_db = str(tmp_path / "gone-[red]x[bold].rl")
    missing = runner.invoke(app, ["forensics", command, "markup-thread", "--db", missing_db])

    assert unknown.exit_code == 1
    assert "[/b]-[red]x[/red]" in _flatten(unknown.output)
    assert unknown.exception is None or isinstance(unknown.exception, SystemExit)
    assert missing.exit_code == 1
    assert "Database file not found" in missing.output
    assert "gone-[red]x[bold].rl" in _flatten(missing.output)


@pytest.mark.parametrize(
    "command_args",
    [
        ["tree", "run[/x]"],
        ["forensics", "loops", "run[/x]"],
        ["replay", "run[/x]", "chk-1"],
    ],
)
def test_ac161_markup_thread_id_exits_one_without_markup_error(
    runner: CliRunner, tmp_path: Path, command_args: list[str]
) -> None:
    """AC-161: tree, loops and replay exit 1 with literal text and result.exception not MarkupError."""
    db_file = str(tmp_path / "markup_store.rl")
    _seed_markup_error_store(db_file, "markup-thread")

    result = runner.invoke(app, [*command_args, "--db", db_file])

    assert result.exit_code == 1
    assert "run[/x]" in _flatten(result.output)
    assert not isinstance(result.exception, MarkupError)
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize(
    "command_args",
    [
        ["tree", "any-thread"],
        ["forensics", "loops", "any-thread"],
        ["replay", "any-thread", "chk-1"],
    ],
)
def test_ac161_missing_db_with_markup_name_prints_literally(
    runner: CliRunner, tmp_path: Path, command_args: list[str]
) -> None:
    """AC-161: a missing --db named [red]missing.rl prints literally for tree, loops, and replay."""
    missing_db = str(tmp_path / "[red]missing.rl")

    result = runner.invoke(app, [*command_args, "--db", missing_db])

    assert result.exit_code == 1
    assert "Database file not found" in result.output
    assert "[red]missing.rl" in _flatten(result.output)
    assert not isinstance(result.exception, MarkupError)
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_ac161_loops_prints_markup_thread_id_literally(runner: CliRunner, tmp_path: Path) -> None:
    """AC-161: loops on a tmp store containing a thread named [bold]x prints literally without styling."""
    db_file = str(tmp_path / "bold_thread.rl")
    thread_id = "[bold]x"
    _seed_markup_error_store(db_file, thread_id)

    result = runner.invoke(app, ["forensics", "loops", thread_id, "--db", db_file])

    assert result.exit_code == 0, result.output
    assert "[bold]x" in result.output
    assert not isinstance(result.exception, MarkupError)


def test_ac161_tree_and_replay_print_markup_thread_id_literally(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-161: tree and replay on an existing thread named [bold]x print literally without styling."""
    db_file = str(tmp_path / "bold_thread_all.rl")
    thread_id = "[bold]x"
    _seed_markup_error_store(db_file, thread_id)

    tree_result = runner.invoke(app, ["tree", thread_id, "--db", db_file])
    assert tree_result.exit_code == 0, tree_result.output
    assert "[bold]x" in tree_result.output
    assert not isinstance(tree_result.exception, MarkupError)

    replay_result = runner.invoke(
        app,
        ["replay", thread_id, "1f1e314c-8a21-6b3e-8000-000000000001", "--db", db_file],
    )
    assert replay_result.exit_code == 0, replay_result.output
    assert "[bold]x" in replay_result.output
    assert "1f1e314c-8a21-6b3e-8000-000000000001" in replay_result.output
    assert not isinstance(replay_result.exception, MarkupError)


def test_ac163_cli_trace_error_finds_write_linked_failed_tool_call(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-163: CLI trace-error identifies a Write-linked tool failure as root cause."""
    db_file = str(tmp_path / "write_linked.rl")
    thread_id = "t-write-fail"
    _seed_markup_error_store(db_file, thread_id)
    with RelateSaver(db_path=db_file) as saver:
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) "
            "RETURN c.checkpoint_id AS checkpoint_id",
            {},
        )
        assert rows
        checkpoint_id = str(rows[0]["checkpoint_id"])
        saver.proxy.execute_write(
            "MATCH (:Checkpoint)-[r:INVOKED]->(:ToolCall) DELETE r",
            {},
        )
        saver.proxy.execute_write(
            "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
            "CREATE (w:Write {write_id: 'w-task', thread_id: $thread_id, "
            "  checkpoint_ns: '', checkpoint_id: $cid, task_id: 'task-w', "
            "  channel: 'results', idx: 0}) "
            "CREATE (c)-[:HAS_WRITE]->(w)",
            {"cid": checkpoint_id, "thread_id": thread_id},
        )
        saver.proxy.execute_write(
            "MATCH (w:Write {write_id: 'w-task'}), (tc:ToolCall) CREATE (w)-[:INVOKED]->(tc)",
            {},
        )

    result = runner.invoke(app, ["forensics", "trace-error", thread_id, "--db", db_file])
    assert result.exit_code == 0, result.output
    assert f"Root Cause Analysis: {thread_id}" in result.output
    assert "fetch_source" in result.output
    assert "call-markup-1" in result.output


@tool
def _failing_fanout_tool(url: str) -> str:
    """Failing tool for testing origin node rendering in CLI."""
    raise RuntimeError(f"cannot fetch {url}")


def test_ac164_cli_forensics_failures_displays_fanout_origin_nodes(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-164: CLI forensics failures displays distinct fan-out origin nodes in the table."""

    class FanoutState(TypedDict):
        target: str
        extra: str
        messages: list[Any]

    def fetcher(state: FanoutState) -> dict[str, Any]:
        return {"target": "manifest-url-1"}

    def parser(state: FanoutState) -> dict[str, Any]:
        return {"extra": "parsed"}

    def agent(state: FanoutState) -> dict[str, Any]:
        call = {
            "id": "call-fanout-cli",
            "name": "_failing_fanout_tool",
            "args": {"url": state["target"]},
        }
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(FanoutState)
    builder.add_node("fetcher", fetcher)
    builder.add_node("parser", parser)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([_failing_fanout_tool], handle_tool_errors=True))
    builder.add_edge(START, "fetcher")
    builder.add_edge(START, "parser")
    builder.add_edge("fetcher", "agent")
    builder.add_edge("parser", "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)

    db_file = str(tmp_path / "fanout_cli.rl")
    with RelateSaver(db_path=db_file) as saver:
        builder.compile(checkpointer=saver).invoke(
            {"messages": [HumanMessage(content="go")], "target": "", "extra": ""},
            {"configurable": {"thread_id": "t-fanout-cli"}},
        )

    result = runner.invoke(app, ["forensics", "failures", "--db", db_file])
    assert result.exit_code == 0, result.output
    assert "_failing_fanout_tool" in result.output
    assert "fetcher" in result.output
    assert "fetcher, parser" not in result.output


@pytest.mark.parametrize("initial_width", [None, 80, 120])
def test_ac167_print_wide_preserves_console_width(
    monkeypatch: pytest.MonkeyPatch, initial_width: int | None
) -> None:
    """AC-167: _print_wide preserves and restores console._width to pre-call value (None or fixed width)."""
    fake_stream = io.StringIO()
    test_console = Console(file=fake_stream, width=initial_width)
    monkeypatch.setattr(cli, "console", test_console)

    assert test_console._width == initial_width

    table = Table(title="Wide Table")
    table.add_column("Column 1")
    table.add_row("Value 1")

    cli._print_wide(table)

    assert test_console._width == initial_width
    assert "Wide Table" in fake_stream.getvalue()


def test_print_wide_keeps_cell_wider_than_console_on_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_print_wide prints cell text wider than the console unwrapped and untruncated."""
    fake_stream = io.StringIO()
    test_console = Console(file=fake_stream, width=40)
    monkeypatch.setattr(cli, "console", test_console)
    wide_cell = "error -> " + "x" * 100

    table = Table(title="Wide Table")
    table.add_column("Change")
    table.add_row(wide_cell)

    cli._print_wide(table)

    lines = fake_stream.getvalue().splitlines()
    assert sum(wide_cell in line for line in lines) == 1
    assert len([line for line in lines if "x" in line]) == 1
    assert "…" not in fake_stream.getvalue()
    assert max(len(line) for line in lines) > 40
    assert test_console.width == 40


class _FaultyDriverConnection:
    """Forward to a real relate-db connection, failing any statement that contains a fragment.

    Stands in for a driver fault on one query; every other statement reaches the real driver.
    """

    def __init__(self, inner: Any, failing_fragment: str) -> None:
        self._inner = inner
        self._failing_fragment = failing_fragment

    def execute(self, query: str, params: Any = None) -> Any:
        if self._failing_fragment in query:
            raise RuntimeError("simulated driver fault: page read failed")
        if params is None:
            return self._inner.execute(query)
        return self._inner.execute(query, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_tree_reports_database_error_and_exits_one(
    runner: CliRunner, seeded_db: tuple[str, str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database error raised by the tree query reaches the CLI as 'Database error', exit 1."""
    db_file, thread_id, _ = seeded_db
    real_connect = relate.connect
    tree_query = (
        "MATCH (c:Checkpoint {thread_id: $thread_id}) RETURN c.checkpoint_id AS checkpoint_id"
    )

    def faulty_connect(*args: Any, **kwargs: Any) -> _FaultyDriverConnection:
        return _FaultyDriverConnection(real_connect(*args, **kwargs), tree_query)

    monkeypatch.setattr(relate, "connect", faulty_connect)
    result = runner.invoke(app, ["tree", thread_id, "--db", db_file])

    assert result.exit_code == 1
    assert "Database error" in result.output
    assert "simulated driver fault" in result.output
    assert "Unexpected error" not in result.output


def _flatten_usage_error(text: str) -> str:
    """Join a boxed, wrapped usage error into one line so its wording can be matched."""
    return " ".join(text.replace("|", " ").replace("│", " ").split())


@pytest.mark.parametrize("command", ["loops", "diagnose"])
@pytest.mark.parametrize("threshold", ["1", "0"])
def test_threshold_below_two_is_a_usage_error_before_the_store_opens(
    runner: CliRunner, seeded_loop_db: tuple[str, str], command: str, threshold: str
) -> None:
    """AC-188: --threshold below 2 is a usage error (exit 2) naming the option and the minimum,
    not an unexpected error, and the store file is left byte-identical."""
    db_file, thread_id = seeded_loop_db
    before = _file_state(db_file)

    result = runner.invoke(
        app, ["forensics", command, thread_id, "--threshold", threshold, "--db", db_file]
    )

    assert result.exit_code == 2
    assert "--threshold" in result.stderr
    assert "greater than or equal to 2" in _flatten_usage_error(result.stderr)
    assert "Unexpected error" not in result.output
    assert "Unexpected error" not in result.stderr
    assert "Detected Loops" not in result.output
    assert _file_state(db_file) == before


def test_threshold_below_two_fails_before_a_missing_store_is_checked(
    runner: CliRunner, tmp_path: Path
) -> None:
    """AC-188: validation runs before the store is looked up, so a bad threshold wins over a
    missing database file."""
    missing = str(tmp_path / "absent.rl")

    result = runner.invoke(app, ["forensics", "loops", "t", "--threshold", "1", "--db", missing])

    assert result.exit_code == 2
    assert "--threshold" in result.stderr
    assert "not found" not in result.output


@pytest.mark.parametrize("command", ["loops", "diagnose"])
def test_threshold_two_is_accepted(
    runner: CliRunner, seeded_loop_db: tuple[str, str], command: str
) -> None:
    """AC-188: --threshold 2, the minimum, still runs the command."""
    db_file, thread_id = seeded_loop_db

    result = runner.invoke(
        app, ["forensics", command, thread_id, "--threshold", "2", "--db", db_file]
    )

    assert result.exit_code == 0
    assert "Unexpected error" not in result.output


@pytest.fixture(scope="module")
def origin_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Module-scoped store with one run whose failing tool call carries a traceable argument."""
    db_file = str(tmp_path_factory.mktemp("origin_store") / "origin.rl")
    _run_origin_graph(db_file, "origin-fail", "pkg-alpha-9000")
    return db_file


def test_combined_forensics_by_path_shows_origin_and_leaves_store_unchanged(
    runner: CliRunner, origin_db: str
) -> None:
    """AC-122, AC-84: the combined summary of a store opened by path shows the origin line
    and leaves the store file and its directory unchanged."""
    before = _file_state(origin_db)

    result = runner.invoke(app, ["forensics", "origin-fail", "--db", origin_db])

    assert result.exit_code == 0, result.output
    assert (
        "Bad argument origin: url came from node planner (step 0, 2 hops before failure)"
        in _flatten(result.output)
    )
    assert _file_state(origin_db) == before


def test_combined_forensics_unknown_thread_leaves_store_unchanged(
    runner: CliRunner, origin_db: str
) -> None:
    """AC-84: a thread with no checkpoints exits 1 and does not modify the store."""
    before = _file_state(origin_db)

    result = runner.invoke(app, ["forensics", "no-such-thread", "--db", origin_db])

    assert result.exit_code == 1
    assert "no-such-thread" in result.output
    assert _file_state(origin_db) == before


@pytest.mark.parametrize("command", [["tree"], ["forensics"]], ids=["tree", "forensics"])
def test_committed_demo_store_tree_and_forensics_leave_it_untouched(
    runner: CliRunner, command: list[str]
) -> None:
    """AC-84, ADR-0015: tree and combined forensics leave the committed demo store unchanged."""
    before = _file_state(str(DEMO_STORE))

    result = runner.invoke(app, [*command, DEMO_THREAD, "--db", str(DEMO_STORE)])

    assert result.exit_code == 0, result.output
    assert _file_state(str(DEMO_STORE)) == before
