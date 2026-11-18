"""Regenerate the committed demo store at ``examples/data/demo.rl``.

Runs both example agents offline (no API key, no network) into a single-file Relate store with
exactly three threads, so an evaluator can explore real checkpoints with ``relate-dag`` straight
after cloning (AC-69..AC-75):

    demo-tool-fail   tool agent, ``fetch_source`` fails on purpose
    demo-tool-ok     tool agent, ``fetch_source`` succeeds
    demo-research    research agent, human review auto-approved

The store is built beside its destination and moved into place only after the saver is closed,
so an existing file is never left half-written and no sidecar files remain.

Usage:
    python examples/build_demo_store.py [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Allow ``python examples/build_demo_store.py`` to import the sibling example modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.research_agent import run_research_workflow  # noqa: E402
from examples.tool_agent import run_tool_agent  # noqa: E402

from langgraph_checkpoint_relate.saver import RelateSaver  # noqa: E402

DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "data" / "demo.rl"
TOOL_FAIL_THREAD = "demo-tool-fail"
TOOL_OK_THREAD = "demo-tool-ok"
RESEARCH_THREAD = "demo-research"
DEMO_THREADS = (TOOL_FAIL_THREAD, TOOL_OK_THREAD, RESEARCH_THREAD)
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


def build_demo_store(out_path: Path | str = DEFAULT_OUT_PATH) -> Path:
    """Build the three-thread demo store at ``out_path``, replacing any existing file.

    Args:
        out_path: Destination of the single-file store; it must end in ``.rl``. Parent
            directories are created.

    Returns:
        The destination path, holding a closed store; the build leaves no other file beside it.

    Raises:
        ValueError: If ``out_path`` is empty or does not end in ``.rl``. Nothing is created.
    """
    destination = Path(out_path)
    if not str(out_path).strip():
        raise ValueError("out_path must be a non-empty path")
    if destination.suffix != STORE_SUFFIX:
        raise ValueError(f'out_path "{out_path}" must end in "{STORE_SUFFIX}"')
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
        description="Regenerate the committed Relate demo store without API keys or network",
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
    except Exception as exc:
        print(f"[ERROR] Could not build the demo store: {exc}", file=sys.stderr)
        return 1
    print(f"[*] Wrote {written} with threads: {', '.join(DEMO_THREADS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
