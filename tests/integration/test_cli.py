"""Integration tests for relate-dag command-line interface.

Validates terminal visualization, in-memory replay branching, forensics diagnostics,
and boundary error handling (AC-24, AC-32).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.saver import RelateSaver


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
        assert "Detected Loops" in result.output
        assert "ping_agent" in result.output
        assert "pong_agent" in result.output
        assert "Cycle Nodes" in result.output
        assert "Occurrences" in result.output

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
