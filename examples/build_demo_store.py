"""Regenerate the committed demo store at ``examples/data/demo.rl``.

Runs the example agents into a single-file Relate store with exactly four threads, so an
evaluator can explore real checkpoints with ``relate-dag`` straight after cloning
(AC-69..AC-75, AC-133):

    demo-tool-fail   tool agent, ``fetch_source`` fails on purpose
    demo-tool-ok     tool agent, ``fetch_source`` succeeds
    demo-research    research agent, human review auto-approved
    demo-real-api    dependency agent, real PyPI lookups; the ``relategraphsaver`` lookup
                     returns HTTP 404

The first three threads need no API key and no network. The fourth calls https://pypi.org, so
rebuilding the store needs network access (ADR-0019); there is no offline option. The store is
built beside its destination and moved into place only after the saver is closed and the PyPI
lookups are checked, so an existing file is never left half-written or replaced by a store
recorded during an outage, and no sidecar files remain (AC-136).

Usage:
    python examples/build_demo_store.py [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Allow ``python examples/build_demo_store.py`` to import the sibling example modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples import dependency_agent  # noqa: E402
from examples.dependency_agent import FetchJson, run_dependency_agent  # noqa: E402
from examples.research_agent import run_research_workflow  # noqa: E402
from examples.tool_agent import run_tool_agent  # noqa: E402

from langgraph_checkpoint_relate.saver import RelateSaver  # noqa: E402

DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "data" / "demo.rl"
TOOL_FAIL_THREAD = "demo-tool-fail"
TOOL_OK_THREAD = "demo-tool-ok"
RESEARCH_THREAD = "demo-research"
REAL_API_THREAD = "demo-real-api"
DEMO_THREADS = (TOOL_FAIL_THREAD, TOOL_OK_THREAD, RESEARCH_THREAD, REAL_API_THREAD)
MISSING_PACKAGE = "relategraphsaver"
STORE_SUFFIX = ".rl"


def _remove_if_exists(path: Path) -> None:
    """Delete ``path`` when present; a missing file is not an error."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _cleanup_staging(staging: Path) -> None:
    """Remove the staging file after a build, without ever raising.

    A single-file Relate store (``.rl``) writes nothing but that one file: opening, writing,
    aborting and killing the process all leave no ``-wal``, ``-shm``, ``-journal`` or lock file
    beside it. So the staging path is the only path the build can create, and the only one removed;
    anything else in the directory is not ours, whatever its name.

    This runs in a ``finally`` block, so an exception here would replace the build error that is
    already propagating. A removal that fails is reported on stderr instead.
    """
    try:
        _remove_if_exists(staging)
    except OSError as exc:
        print(f"[WARN] Could not remove leftover {staging}: {exc}", file=sys.stderr)


class PyPIUnreachableError(RuntimeError):
    """The real-API thread did not record the expected PyPI lookups (AC-136)."""


def _check_real_api_lookups(saver: RelateSaver) -> None:
    """Fail unless the ``relategraphsaver`` lookup holds a 404 error and the rest a result.

    ``ToolNode`` turns a network error into a tool error, so a run during an outage completes
    normally; the recorded tool calls are the only place the outage shows (AC-135).

    Raises:
        PyPIUnreachableError: With the first unexpected tool error, or a short description when
            a lookup is missing or holds neither a result nor the expected 404.
    """
    rows: list[dict[str, Any]] = saver.proxy.execute_read(
        "MATCH (c:Checkpoint {thread_id: $tid})-[:INVOKED]->(tc:ToolCall) "
        "RETURN tc.args_blob AS args, tc.error AS error, tc.result_blob AS result",
        {"tid": REAL_API_THREAD},
    )
    lookups: list[tuple[str, str | None, bool]] = []
    for row in rows:
        args = saver.serde.loads(bytes(row["args"]))
        # ToolNode appends a retry hint on a second line; the first line names the failure.
        error_lines = (row["error"] or "").strip().splitlines()
        error = error_lines[0].strip() if error_lines else None
        lookups.append((str(args.get("name", "")), error, row["result"] is not None))
    names = [name for name, _, _ in lookups]
    for name, error, has_result in lookups:
        if name == MISSING_PACKAGE:
            if error is None or "404" not in error:
                raise PyPIUnreachableError(error or f"the {name} lookup did not fail with 404")
        elif error is not None:
            raise PyPIUnreachableError(error)
        elif not has_result:
            raise PyPIUnreachableError(f"the {name} lookup recorded no result")
    if MISSING_PACKAGE not in names or len(names) < 2:
        raise PyPIUnreachableError("the dependency lookups were not recorded")


def build_demo_store(
    out_path: Path | str = DEFAULT_OUT_PATH, *, fetch_json: FetchJson | None = None
) -> Path:
    """Build the four-thread demo store at ``out_path``, replacing any existing file.

    Args:
        out_path: Destination of the single-file store; it must end in ``.rl``. Parent
            directories are created.
        fetch_json: Fetcher the ``demo-real-api`` thread calls for each package name. ``None``
            means ``examples.dependency_agent.fetch_pypi_json``, looked up when called.

    Returns:
        The destination path, holding a closed store; the build leaves no other file beside it.

    Raises:
        ValueError: If ``out_path`` is empty or does not end in ``.rl``. Nothing is created.
        PyPIUnreachableError: If the lookups do not hold the expected 404 and results. The
            existing store, if any, is left unchanged and no staging file remains.
    """
    destination = Path(out_path)
    if not str(out_path).strip():
        raise ValueError("out_path must be a non-empty path")
    if destination.suffix != STORE_SUFFIX:
        raise ValueError(f'out_path "{out_path}" must end in "{STORE_SUFFIX}"')
    fetcher = fetch_json if fetch_json is not None else dependency_agent.fetch_pypi_json
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Relate picks single-file mode from the ``.rl`` suffix, so the staging name must keep it.
    staging = destination.with_name(f"{destination.stem}.building{destination.suffix}")
    _remove_if_exists(staging)
    try:
        saver = RelateSaver(db_path=str(staging))
        try:
            run_tool_agent(saver, TOOL_FAIL_THREAD, fail=True)
            run_tool_agent(saver, TOOL_OK_THREAD, fail=False)
            run_research_workflow(
                thread_id=RESEARCH_THREAD,
                auto_approve=True,
                saver=saver,
                print_output=False,
            )
            run_dependency_agent(saver, REAL_API_THREAD, fetch_json=fetcher)
            _check_real_api_lookups(saver)
        finally:
            saver.close()
        os.replace(staging, destination)
    finally:
        # After a failed build the staging file must not be left behind.
        _cleanup_staging(staging)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the demo store builder."""
    parser = argparse.ArgumentParser(
        description="Regenerate the committed Relate demo store (needs network access to pypi.org)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help=f"Where to write the store (default: {DEFAULT_OUT_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: build the store and report where it was written."""
    args = parse_args(argv)
    try:
        written = build_demo_store(args.out)
    except PyPIUnreachableError as exc:
        print(f"Error: could not reach pypi.org: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Could not build the demo store: {exc}", file=sys.stderr)
        return 1
    print(f"[*] Wrote {written} with threads: {', '.join(DEMO_THREADS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
