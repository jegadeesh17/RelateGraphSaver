"""Command-line interface for Relate graph checkpointer (relate-dag).

Provides terminal commands for execution DAG visualization, time-travel replay branching,
and forensic diagnostics.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperGroup

from langgraph_checkpoint_relate.errors import (
    CheckpointNotFoundError,
    RelateCheckpointError,
)
from langgraph_checkpoint_relate.forensics import RelateForensics
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.time_travel import TimeTravelEngine
from langgraph_checkpoint_relate.visualizer import DAGVisualizer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="relate-dag",
    help="RelateGraphSaver CLI - Graph-native LangGraph persistence & forensics",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def _make_output_streams_lossy() -> None:
    """Replace unencodable characters instead of raising on stdout and stderr (AC-59).

    Stored data (node names, messages) can hold characters the terminal encoding lacks;
    the stream keeps its encoding and only its error handler changes.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            logger.debug("Could not reconfigure %r to replace unencodable characters", stream)


@app.callback()
def _configure_output() -> None:
    # No docstring: Typer would use it as the root help text instead of ``help=`` above.
    # Runs before every command, including when the installed script calls ``app`` directly.
    _make_output_streams_lossy()


def _open_inspection_saver(db: str) -> RelateSaver:
    """Open the store for an inspection command without modifying the file (ADR-0015).

    An in-memory store starts empty, so there is nothing to protect and it stays writable.
    """
    return RelateSaver(db_path=db, read_only=db != ":memory:")


@app.command(name="tree", help="Visualize execution DAG for a thread.")
def tree(
    thread_id: str = typer.Argument(
        ...,
        help="Target thread identifier to visualize.",
    ),
    db: str = typer.Option(
        "checkpoints.rl",
        "--db",
        "-d",
        "--file",
        "-f",
        help="Path to the Relate graph database file.",
    ),
) -> None:
    """Visualize the execution DAG for a thread in the terminal (AC-24)."""
    if not thread_id.strip():
        err_console.print("[bold red]Error:[/bold red] thread_id must be a non-empty string.")
        raise typer.Exit(code=1)

    if db != ":memory:" and not os.path.exists(db):
        err_console.print(f"[bold red]Error:[/bold red] Database file not found: {db}")
        raise typer.Exit(code=1)

    try:
        with _open_inspection_saver(db) as saver, DAGVisualizer(saver) as visualizer:
            tree_obj = visualizer.render_tree(thread_id)
            console.print(tree_obj)
    except typer.Exit:
        raise
    except CheckpointNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except RelateCheckpointError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Database error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command(
    name="replay",
    help="Fork and replay execution from a historical checkpoint in memory.",
)
def replay(
    thread_id: str = typer.Argument(
        ...,
        help="Target thread identifier containing the historical lineage.",
    ),
    checkpoint_id: str | None = typer.Argument(
        None,
        help="Target checkpoint identifier to replay from.",
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Target checkpoint identifier to replay from (flag syntax).",
    ),
    db: str = typer.Option(
        "checkpoints.rl",
        "--db",
        "-d",
        "--file",
        "-f",
        help="Path to the Relate graph database file.",
    ),
    branch: str | None = typer.Option(
        None,
        "--branch",
        "-b",
        help="Optional new branch thread identifier for the replay lineage.",
    ),
) -> None:
    """Fork an execution thread at a target checkpoint into an isolated in-memory database."""
    if not thread_id.strip():
        err_console.print("[bold red]Error:[/bold red] thread_id must be a non-empty string.")
        raise typer.Exit(code=1)

    effective_checkpoint_id = checkpoint or checkpoint_id
    if not effective_checkpoint_id or not effective_checkpoint_id.strip():
        err_console.print(
            "[bold red]Error:[/bold red] checkpoint_id must be a non-empty string (passed as an argument or via --checkpoint/-c)."
        )
        raise typer.Exit(code=1)

    if db != ":memory:" and not os.path.exists(db):
        err_console.print(f"[bold red]Error:[/bold red] Database file not found: {db}")
        raise typer.Exit(code=1)

    source_saver: RelateSaver | None = None
    sandboxed_saver: RelateSaver | None = None
    try:
        source_saver = _open_inspection_saver(db)
        sandboxed_saver = TimeTravelEngine.fork_in_memory(
            source_saver=source_saver,
            thread_id=thread_id,
            checkpoint_id=effective_checkpoint_id,
            branch_thread_id=branch,
        )

        step_records = sandboxed_saver.proxy.execute_read(
            "MATCH (c:Checkpoint {checkpoint_id: $cid}) RETURN c.step AS step",
            {"cid": effective_checkpoint_id},
        )
        step = step_records[0]["step"] if step_records else 0

        target_thread = branch if branch else thread_id
        console.print("[bold green]Sandboxed replay initialized successfully.[/bold green]")
        console.print(f"  Source Thread:    [cyan]{thread_id}[/cyan]")
        console.print(f"  Replay Thread:    [cyan]{target_thread}[/cyan]")
        console.print(f"  Checkpoint ID:    [cyan]{effective_checkpoint_id}[/cyan]")
        console.print(f"  Replay Step:      [bold yellow]{step}[/bold yellow]")
        console.print("  Isolation:        Isolated :memory: sandbox (source database unchanged)")
    except typer.Exit:
        raise
    except CheckpointNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except RelateCheckpointError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Database error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if sandboxed_saver is not None:
            sandboxed_saver.close()
        if source_saver is not None:
            source_saver.close()


class ForensicsGroup(TyperGroup):
    """Custom TyperGroup routing 'relate-dag forensics <thread_id>' to default diagnose command."""

    # ``ctx`` is typed Any because the Context class differs between Typer releases
    # (real click or Typer's vendored copy).
    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if args:
            first_non_option = next((a for a in args if not a.startswith("-")), None)
            if first_non_option and first_non_option not in self.commands:
                args = ["diagnose"] + list(args)
            elif not first_non_option and not any(a in ("--help", "-h") for a in args):
                args = ["diagnose"] + list(args)
        return super().parse_args(ctx, args)


forensics_app = typer.Typer(
    cls=ForensicsGroup,
    name="forensics",
    help="Detect cycles, infinite loops, and trace root causes of failures.",
    no_args_is_help=True,
)
app.add_typer(forensics_app)


@forensics_app.command(
    name="loops",
    help="Detect cyclical state oscillations and repeating node sequences.",
)
def forensics_loops(
    thread_id: str = typer.Argument(
        ...,
        help="Target thread identifier to analyze.",
    ),
    threshold: int = typer.Option(
        2,
        "--threshold",
        "-t",
        help="Minimum consecutive cycle occurrence threshold.",
    ),
    db: str = typer.Option(
        "checkpoints.rl",
        "--db",
        "-d",
        "--file",
        "-f",
        help="Path to the Relate graph database file.",
    ),
) -> None:
    """Identify repeating node cycles or state oscillation in execution graph (AC-25, AC-26)."""
    if not thread_id.strip():
        err_console.print("[bold red]Error:[/bold red] thread_id must be a non-empty string.")
        raise typer.Exit(code=1)

    if db != ":memory:" and not os.path.exists(db):
        err_console.print(f"[bold red]Error:[/bold red] Database file not found: {db}")
        raise typer.Exit(code=1)

    try:
        with _open_inspection_saver(db) as saver, RelateForensics(saver) as forensics_engine:
            reports = forensics_engine.detect_loops(thread_id, threshold=threshold)
            if reports:
                table = Table(title=f"Detected Loops: {thread_id}")
                table.add_column("Cycle Nodes", style="cyan")
                table.add_column("Occurrences", justify="right", style="magenta")
                table.add_column("First Checkpoint", style="green")
                table.add_column("Last Checkpoint", style="yellow")
                for rep in reports:
                    table.add_row(
                        " -> ".join(rep.cycle_nodes),
                        str(rep.occurrence_count),
                        rep.checkpoint_id_range[0],
                        rep.checkpoint_id_range[1],
                    )
                console.print(table)
            else:
                console.print(
                    f"[green]OK[/green] No loops or repeating cycles detected in thread '{thread_id}' (threshold: {threshold})."
                )
    except typer.Exit:
        raise
    except CheckpointNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except RelateCheckpointError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Database error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc


def forensics_trace_error(
    thread_id: str = typer.Argument(
        ...,
        help="Target thread identifier to trace root cause for.",
    ),
    db: str = typer.Option(
        "checkpoints.rl",
        "--db",
        "-d",
        "--file",
        "-f",
        help="Path to the Relate graph database file.",
    ),
) -> None:
    """Trace backward causal path to pinpoint root cause of failure (AC-27, AC-28)."""
    if not thread_id.strip():
        err_console.print("[bold red]Error:[/bold red] thread_id must be a non-empty string.")
        raise typer.Exit(code=1)

    if db != ":memory:" and not os.path.exists(db):
        err_console.print(f"[bold red]Error:[/bold red] Database file not found: {db}")
        raise typer.Exit(code=1)

    try:
        with _open_inspection_saver(db) as saver, RelateForensics(saver) as forensics_engine:
            report = forensics_engine.trace_root_cause(thread_id)
            if report.error_message == "No error detected" and not report.error_checkpoint_id:
                console.print(
                    f"[green]OK[/green] No error detected in execution thread '{thread_id}'. Workflow completed cleanly."
                )
            else:
                panel_content = (
                    f"[bold cyan]Error Checkpoint ID:[/bold cyan] {report.error_checkpoint_id}\n"
                    f"[bold red]Error Message:[/bold red]       {report.error_message}\n"
                    f"[bold yellow]Root Tool Call ID:[/bold yellow]   {report.root_tool_call_id or 'None'}\n"
                    f"[bold yellow]Root Tool Name:[/bold yellow]      {report.root_tool_name or 'None'}\n"
                    f"[bold yellow]Root Channel Write:[/bold yellow]  {report.root_channel_write or 'None'}\n"
                    f"[bold green]Causal Path Length:[/bold green]  {report.causal_path_length}\n"
                    f"[bold white]Diagnostic Summary:[/bold white]  {report.diagnostic_summary}"
                )
                console.print(
                    Panel(panel_content, title=f"Root Cause Analysis: {thread_id}", expand=False)
                )
    except typer.Exit:
        raise
    except CheckpointNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except RelateCheckpointError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Database error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc


forensics_app.command(
    name="trace-error",
    help="Traverse backward causal paths to pinpoint root causes of failures.",
)(forensics_trace_error)

forensics_app.command(
    name="trace",
    help="Alias for trace-error.",
)(forensics_trace_error)


@forensics_app.command(
    name="diagnose",
    hidden=True,
    help="Default forensic diagnostics combining loop detection and root-cause analysis.",
)
def forensics_diagnose(
    thread_id: str = typer.Argument(
        ...,
        help="Target thread identifier to diagnose.",
    ),
    threshold: int = typer.Option(
        2,
        "--threshold",
        "-t",
        help="Minimum consecutive cycle occurrence threshold.",
    ),
    db: str = typer.Option(
        "checkpoints.rl",
        "--db",
        "-d",
        "--file",
        "-f",
        help="Path to the Relate graph database file.",
    ),
) -> None:
    """Run comprehensive forensics diagnostics and cycle checks on execution thread."""
    if not thread_id.strip():
        err_console.print("[bold red]Error:[/bold red] thread_id must be a non-empty string.")
        raise typer.Exit(code=1)

    if db != ":memory:" and not os.path.exists(db):
        err_console.print(f"[bold red]Error:[/bold red] Database file not found: {db}")
        raise typer.Exit(code=1)

    try:
        with _open_inspection_saver(db) as saver, RelateForensics(saver) as forensics_engine:
            records = forensics_engine.saver.proxy.execute_read(
                "MATCH (c:Checkpoint {thread_id: $thread_id}) RETURN count(c) AS count",
                {"thread_id": thread_id},
            )
            count = records[0]["count"] if records else 0
            if count == 0:
                err_console.print(
                    f"[bold red]Error:[/bold red] Thread '{thread_id}' not found or contains no checkpoints."
                )
                raise typer.Exit(code=1)

            loops = forensics_engine.detect_loops(thread_id, threshold=threshold)
            root_cause = forensics_engine.trace_root_cause(thread_id)

            console.print(f"[bold cyan]Thread Forensics Diagnostic:[/bold cyan] {thread_id}")
            console.print(f"  Total Checkpoints: {count}")

            if loops:
                table = Table(title=f"Detected Loops: {thread_id}")
                table.add_column("Cycle Nodes", style="cyan")
                table.add_column("Occurrences", justify="right", style="magenta")
                table.add_column("First Checkpoint", style="green")
                table.add_column("Last Checkpoint", style="yellow")
                for rep in loops:
                    table.add_row(
                        " -> ".join(rep.cycle_nodes),
                        str(rep.occurrence_count),
                        rep.checkpoint_id_range[0],
                        rep.checkpoint_id_range[1],
                    )
                console.print(table)
            else:
                console.print("  Loop Detection:    No infinite execution loops detected")

            if root_cause.error_message != "No error detected" and root_cause.error_checkpoint_id:
                panel_content = (
                    f"[bold cyan]Error Checkpoint ID:[/bold cyan] {root_cause.error_checkpoint_id}\n"
                    f"[bold red]Error Message:[/bold red]       {root_cause.error_message}\n"
                    f"[bold yellow]Root Tool Call ID:[/bold yellow]   {root_cause.root_tool_call_id or 'None'}\n"
                    f"[bold yellow]Root Tool Name:[/bold yellow]      {root_cause.root_tool_name or 'None'}\n"
                    f"[bold yellow]Root Channel Write:[/bold yellow]  {root_cause.root_channel_write or 'None'}\n"
                    f"[bold green]Causal Path Length:[/bold green]  {root_cause.causal_path_length}\n"
                    f"[bold white]Diagnostic Summary:[/bold white]  {root_cause.diagnostic_summary}"
                )
                console.print(Panel(panel_content, title="Root Cause Analysis", expand=False))
                console.print("  Status:            ERROR DETECTED")
            else:
                console.print("  Causal Lineage:    Valid DAG structure verified")
                console.print("  Status:            OK")
    except typer.Exit:
        raise
    except CheckpointNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except RelateCheckpointError as exc:
        err_console.print(f"[bold red]Error:[/bold red] Database error: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    """CLI application entry point."""
    app()


if __name__ == "__main__":
    main()
