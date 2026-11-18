"""Command-line interface for Relate graph checkpointer (relate-dag).

Provides terminal commands for execution DAG visualization, time-travel replay branching,
and forensic diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Annotated, Any

import typer
from langgraph.errors import InvalidUpdateError
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperGroup

from langgraph_checkpoint_relate.errors import (
    CheckpointNotFoundError,
    GraphSpecError,
    RelateCheckpointError,
)
from langgraph_checkpoint_relate.forensics import RelateForensics
from langgraph_checkpoint_relate.models import ChannelChange, RunDiff, ToolCallChange
from langgraph_checkpoint_relate.replay import load_graph_factory, replay_from_checkpoint
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


def _fail(message: str) -> typer.Exit:
    """Print an error line and return the exit to raise."""
    err_console.print(f"[bold red]Error:[/bold red] {escape(message)}")
    return typer.Exit(code=1)


def _parse_update(raw: str) -> dict[str, Any]:
    """Parse the ``--update`` value, which must be a JSON object (AC-113)."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(f"--update must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail('--update must be a JSON object, for example \'{"key": "value"}\'.')
    return value


def _parse_set_messages(raw_items: list[str]) -> dict[int, str]:
    """Parse repeated ``--set-message INDEX=TEXT`` values (AC-114)."""
    parsed: dict[int, str] = {}
    for item in raw_items:
        index_text, separator, text = item.partition("=")
        if not separator:
            raise _fail(f"--set-message must look like INDEX=TEXT, got: {item!r}")
        try:
            index = int(index_text.strip())
        except ValueError as exc:
            raise _fail(f"--set-message index must be an integer, got: {index_text!r}") from exc
        if index < 0:
            raise _fail(f"--set-message index must not be negative, got: {index}")
        parsed[index] = text
    return parsed


def _print_wide(table: Table) -> None:
    """Print a table at its natural width so cell text such as ``error -> ok`` never wraps."""
    natural_width = console.measure(table, options=console.options.update_width(10_000)).maximum
    available_width = console.width
    console.width = max(natural_width, available_width)
    try:
        console.print(table)
    finally:
        console.width = available_width


def _argument_changes(change: ToolCallChange) -> str:
    """Describe the arguments of a tool call change, one ``name: before -> after`` per line."""
    before = change.before.args if change.before is not None else None
    after = change.after.args if change.after is not None else None
    if isinstance(before, dict) and isinstance(after, dict):
        keys = [*before, *(key for key in after if key not in before)]
        lines = [
            f"{key}: {before.get(key, '-')} -> {after.get(key, '-')}"
            for key in keys
            if (key in before) != (key in after) or before[key] != after[key]
        ]
        return "\n".join(lines) if lines else "-"
    if change.kind == "added":
        return f"{after}"
    if change.kind == "removed":
        return f"{before}"
    return f"{before} -> {after}"


def _tool_call_result(change: ToolCallChange) -> str:
    """Describe the status of a tool call change, as ``before -> after`` when it moved."""
    before = change.before.status if change.before is not None else None
    after = change.after.status if change.after is not None else None
    if change.kind == "added":
        return f"{after}"
    if change.kind == "removed":
        return f"{before}"
    return f"{before} -> {after}" if before != after else f"{after}"


def _print_diff(diff: RunDiff) -> None:
    """Print the tool-call and final-state tables of a replay diff (AC-110, AC-111)."""
    if diff.is_empty:
        console.print("No tool calls or state changed.")
        return

    if diff.tool_calls:
        calls = Table(title="Tool calls", box=box.ASCII)
        calls.add_column("Tool call", style="cyan", no_wrap=True)
        calls.add_column("Change", no_wrap=True)
        calls.add_column("Result", style="magenta", no_wrap=True)
        calls.add_column("Arguments", no_wrap=True)
        for call in diff.tool_calls:
            calls.add_row(
                escape(f"{call.tool_name} #{call.ordinal}"),
                call.kind,
                escape(_tool_call_result(call)),
                escape(_argument_changes(call)),
            )
        _print_wide(calls)

    state_changes: list[ChannelChange] = [*diff.messages, *diff.channels]
    if state_changes:
        states = Table(title="Final state changes", box=box.ASCII)
        states.add_column("Channel", style="cyan", no_wrap=True)
        states.add_column("Change", no_wrap=True)
        states.add_column("Before", overflow="fold")
        states.add_column("After", overflow="fold")
        for state in state_changes:
            states.add_row(
                escape(state.channel),
                state.kind,
                escape(state.before if state.before is not None else "-"),
                escape(state.after if state.after is not None else "-"),
            )
        console.print(states)


def _replay_with_graph(
    db: str,
    thread_id: str,
    checkpoint_id: str,
    spec: str,
    update: dict[str, Any] | None,
    set_messages: dict[int, str] | None,
    as_node: str | None,
) -> None:
    """Re-run the user's graph from a checkpoint in a sandbox and print the diff (AC-110)."""
    source_saver: RelateSaver | None = None
    try:
        source_saver = _open_inspection_saver(db)
        factory = load_graph_factory(spec)
        result = replay_from_checkpoint(
            source_saver,
            thread_id,
            checkpoint_id,
            factory,
            update=update,
            as_node=as_node,
            set_messages=set_messages,
        )
        console.print("[bold green]Graph replay completed in an isolated sandbox.[/bold green]")
        console.print("[bold cyan]Replay result[/bold cyan]")
        console.print(f"  Source Thread:        [cyan]{escape(thread_id)}[/cyan]")
        console.print(f"  Fork checkpoint:      [cyan]{result.fork_checkpoint_id}[/cyan]")
        console.print(f"  Updated checkpoint:   [cyan]{result.updated_checkpoint_id}[/cyan]")
        console.print(f"  New final:            [cyan]{result.final_checkpoint_id}[/cyan]")
        console.print(f"  Recorded final:       [cyan]{result.recorded_checkpoint_id}[/cyan]")
        stopped = "yes" if result.interrupted else "no"
        console.print(f"  Stopped at interrupt: [bold yellow]{stopped}[/bold yellow]")
        console.print(
            "  Isolation:            Isolated :memory: sandbox (source database unchanged)"
        )
        _print_diff(result.diff)
    except typer.Exit:
        raise
    except GraphSpecError as exc:
        raise _fail(str(exc)) from exc
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    except CheckpointNotFoundError as exc:
        raise _fail(str(exc)) from exc
    except RelateCheckpointError as exc:
        raise _fail(f"Database error: {exc}") from exc
    except InvalidUpdateError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        err_console.print(
            "Hint: pass --as-node with the node that should be treated as the writer."
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        raise _fail(f"replay failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if source_saver is not None:
            source_saver.close()


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
    graph: str | None = typer.Option(
        None,
        "--graph",
        "-g",
        help="Graph to re-run from the checkpoint, as path/to/file.py:factory or module:factory.",
    ),
    update: str | None = typer.Option(
        None,
        "--update",
        help="JSON object of channel values to write before re-running (needs --graph).",
    ),
    set_message: Annotated[
        list[str] | None,
        typer.Option(
            "--set-message",
            help="Replace message INDEX with TEXT, as INDEX=TEXT; repeatable (needs --graph).",
        ),
    ] = None,
    as_node: str | None = typer.Option(
        None,
        "--as-node",
        help="Node to treat as the writer of the update (needs --graph).",
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

    if graph is None:
        edit_flags = [
            flag
            for flag, given in (
                ("--update", update is not None),
                ("--set-message", bool(set_message)),
                ("--as-node", as_node is not None),
            )
            if given
        ]
        if edit_flags:
            raise _fail(f"{' and '.join(edit_flags)} need --graph to say which graph to re-run.")
    else:
        if branch:
            raise _fail("--branch cannot be combined with --graph.")
        update_value = _parse_update(update) if update is not None else None
        set_messages = _parse_set_messages(set_message) if set_message else None
        _replay_with_graph(
            db, thread_id, effective_checkpoint_id, graph, update_value, set_messages, as_node
        )
        return

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
        console.print(
            "Hint: add --graph path/to/file.py:factory to re-run the graph from this checkpoint."
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
