"""Compare RelateSaver with SqliteSaver on the same checkpoint workload (ADR-0008, AC-56).

Run ``python benchmarks/bench_checkpointers.py --checkpoints 1000`` from a development
install. Standard output carries only Markdown (an environment block and one table);
progress goes to standard error. Exit codes: 0 success, 1 when the two savers return
different ancestry counts, 2 when ``langgraph-checkpoint-sqlite`` is not installed.

This file is a standalone script, not a test, and it deliberately asserts nothing about
which saver is faster.
"""

from __future__ import annotations

import argparse
import math
import platform
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.base.id import uuid6

from langgraph_checkpoint_relate import RelateSaver

THREAD_ID = "bench-thread"
RELATE_FILE = "relate.rl"
RELATE_NO_FLUSH_FILE = "relate-noflush.rl"
SQLITE_FILE = "sqlite.db"
GET_SAMPLE_LIMIT = 200
ANCESTRY_REPEATS = 5

RELATE_ANCESTRY_CYPHER = (
    "MATCH (leaf:Checkpoint {checkpoint_id: $leaf})<-[:NEXT|FORKED_TO*0..]-(a:Checkpoint) "
    "RETURN count(DISTINCT a) AS n"
)
SQLITE_ANCESTRY_SQL = (
    "WITH RECURSIVE ancestry(checkpoint_id) AS ("
    " SELECT checkpoint_id FROM checkpoints"
    "  WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id = ?"
    " UNION ALL"
    " SELECT c.parent_checkpoint_id FROM checkpoints c"
    "  JOIN ancestry a ON c.checkpoint_id = a.checkpoint_id"
    "  WHERE c.thread_id = ? AND c.checkpoint_ns = '' AND c.parent_checkpoint_id IS NOT NULL"
    ") SELECT count(*) FROM ancestry"
)


def log(message: str) -> None:
    """Write a progress line to standard error."""
    print(message, file=sys.stderr, flush=True)


def make_checkpoint(index: int, payload: str) -> tuple[Checkpoint, CheckpointMetadata]:
    """Build the checkpoint and metadata for one step of the workload."""
    checkpoint: Checkpoint = {
        "v": 1,
        "id": str(uuid6()),
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"step": index, "payload": payload},
        "channel_versions": {},
        "versions_seen": {},
        "updated_channels": None,
    }
    metadata: CheckpointMetadata = {"source": "loop", "step": index, "parents": {}}
    return checkpoint, metadata


def config_for(checkpoint_id: str | None = None) -> RunnableConfig:
    """Build a config for the benchmark thread, optionally pointing at a checkpoint."""
    configurable: dict[str, Any] = {"thread_id": THREAD_ID, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of an ascending, non-empty sequence."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def run_workload(
    saver: Any,
    count: int,
    payload: str,
    get_ids: Callable[[list[str]], list[str]],
    ancestry: Callable[[str], int],
    label: str,
) -> dict[str, Any]:
    """Write the chain and branch, then time gets and the ancestry query.

    The same workload runs against every saver: ``count`` timed puts in one chain, an
    untimed branch of ``max(1, count // 10)`` checkpoints from index ``count // 2``,
    ``min(count, 200)`` timed ``get_tuple`` calls, and the ancestry query from the
    branch leaf repeated five times.
    """
    chain_ids: list[str] = []
    config = config_for()
    log(f"[{label}] writing {count} checkpoints")
    started = time.perf_counter()
    for index in range(count):
        checkpoint, metadata = make_checkpoint(index, payload)
        config = saver.put(config, checkpoint, metadata, {})
        chain_ids.append(config["configurable"]["checkpoint_id"])
    elapsed = time.perf_counter() - started

    branch_config = config_for(chain_ids[count // 2])
    branch_ids: list[str] = []
    log(f"[{label}] writing branch of {max(1, count // 10)} checkpoints")
    for index in range(max(1, count // 10)):
        checkpoint, metadata = make_checkpoint(count // 2 + 1 + index, payload)
        branch_config = saver.put(branch_config, checkpoint, metadata, {})
        branch_ids.append(branch_config["configurable"]["checkpoint_id"])

    log(f"[{label}] timing get_tuple")
    get_latencies: list[float] = []
    for checkpoint_id in get_ids(chain_ids + branch_ids):
        started_get = time.perf_counter()
        found = saver.get_tuple(config_for(checkpoint_id))
        get_latencies.append((time.perf_counter() - started_get) * 1000.0)
        if found is None:
            raise RuntimeError(f"{label} could not read back checkpoint {checkpoint_id}")
    get_latencies.sort()

    log(f"[{label}] timing ancestry query")
    leaf = branch_ids[-1]
    ancestry_times: list[float] = []
    ancestry_count = -1
    for _ in range(ANCESTRY_REPEATS):
        started_query = time.perf_counter()
        ancestry_count = ancestry(leaf)
        ancestry_times.append((time.perf_counter() - started_query) * 1000.0)

    return {
        "put_per_second": count / elapsed,
        "get_p50_ms": percentile(get_latencies, 0.50),
        "get_p95_ms": percentile(get_latencies, 0.95),
        "ancestry_ms": statistics.median(ancestry_times),
        "ancestry_count": ancestry_count,
    }


def disk_size_kib(directory: Path, prefix: str) -> float:
    """Return the total size in KiB of every file in ``directory`` whose name starts with prefix."""
    total = sum(p.stat().st_size for p in directory.iterdir() if p.name.startswith(prefix))
    return total / 1024.0


def bench_relate(
    workdir: Path,
    count: int,
    payload: str,
    get_ids: Callable[[list[str]], list[str]],
    *,
    wal_enabled: bool = True,
) -> dict[str, Any]:
    """Run the workload against RelateSaver on a file-backed database.

    ``wal_enabled`` selects the flush mode: True flushes after every write (the default),
    False skips the per-write flush.
    """
    filename = RELATE_FILE if wal_enabled else RELATE_NO_FLUSH_FILE
    label = "RelateSaver" if wal_enabled else "RelateSaver (wal_enabled=False)"
    saver = RelateSaver(str(workdir / filename), wal_enabled=wal_enabled)

    def ancestry(leaf: str) -> int:
        records = saver.proxy.execute_read(RELATE_ANCESTRY_CYPHER, {"leaf": leaf})
        return int(records[0]["n"])

    try:
        result = run_workload(saver, count, payload, get_ids, ancestry, label)
    finally:
        saver.close()
    result["size_kib"] = disk_size_kib(workdir, filename)
    return result


def bench_sqlite(
    workdir: Path, count: int, payload: str, get_ids: Callable[[list[str]], list[str]]
) -> dict[str, Any]:
    """Run the workload against SqliteSaver on a file-backed database."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(str(workdir / SQLITE_FILE), check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        saver.setup()

        def ancestry(leaf: str) -> int:
            row = connection.execute(SQLITE_ANCESTRY_SQL, (THREAD_ID, leaf, THREAD_ID)).fetchone()
            return int(row[0])

        result = run_workload(saver, count, payload, get_ids, ancestry, "SqliteSaver")
    finally:
        connection.close()
    result["size_kib"] = disk_size_kib(workdir, SQLITE_FILE)
    return result


def package_version(name: str) -> str:
    """Return the installed version of a distribution, or 'not installed'."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def render_markdown(
    count: int,
    payload_bytes: int,
    relate: dict[str, Any],
    relate_no_flush: dict[str, Any],
    sqlite: dict[str, Any],
) -> str:
    """Render the environment block and the results table as Markdown."""
    lines = [
        "## Checkpointer benchmark",
        "",
        f"- Checkpoints (N): {count}",
        f"- Payload bytes (B): {payload_bytes}",
        f"- OS: {platform.platform()}",
        f"- CPU: {platform.processor() or 'unknown'}",
        f"- Python: {platform.python_version()}",
        f"- relate-db: {package_version('relate-db')}",
        f"- langgraph-checkpoint: {package_version('langgraph-checkpoint')}",
        f"- langgraph-checkpoint-sqlite: {package_version('langgraph-checkpoint-sqlite')}",
        f"- SQLite: {sqlite3.sqlite_version}",
        "",
        "| Metric | RelateSaver (flush per write, default) | RelateSaver (wal_enabled=False) "
        "| SqliteSaver |",
        "| :--- | ---: | ---: | ---: |",
    ]
    rows = [
        ("Put throughput (checkpoints/s)", "put_per_second", "{:.1f}"),
        ("Get p50 (ms)", "get_p50_ms", "{:.3f}"),
        ("Get p95 (ms)", "get_p95_ms", "{:.3f}"),
        ("On-disk size (KiB)", "size_kib", "{:.1f}"),
        ("Ancestry query, median of 5 (ms)", "ancestry_ms", "{:.3f}"),
    ]
    for title, key, fmt in rows:
        cells = [fmt.format(result[key]) for result in (relate, relate_no_flush, sqlite)]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare RelateSaver and SqliteSaver on the same checkpoint workload."
    )
    parser.add_argument("--checkpoints", type=int, default=200, help="main-chain length N")
    parser.add_argument("--payload-bytes", type=int, default=256, help="payload size B")
    parser.add_argument("--seed", type=int, default=0, help="seed for the get-latency sample")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "directory for the database files (default: a temporary directory deleted at "
            f"exit); existing {RELATE_FILE}*, {RELATE_NO_FLUSH_FILE}* and {SQLITE_FILE}* files in "
            "it are replaced"
        ),
    )
    args = parser.parse_args(argv)
    if args.checkpoints < 1:
        parser.error("--checkpoints must be at least 1")
    if args.payload_bytes < 0:
        parser.error("--payload-bytes must not be negative")
    return args


def run(args: argparse.Namespace, workdir: Path) -> int:
    """Run both benchmarks in ``workdir`` and print the Markdown report."""
    count: int = args.checkpoints
    payload = "x" * args.payload_bytes
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in workdir.iterdir():
        if stale.name.startswith((RELATE_FILE, RELATE_NO_FLUSH_FILE, SQLITE_FILE)):
            stale.unlink()

    sample_size = min(count, GET_SAMPLE_LIMIT)

    def get_ids(written: list[str]) -> list[str]:
        # Seeded afresh on every call so both savers read back the same sample.
        indexes = random.Random(args.seed).sample(range(len(written)), sample_size)
        return [written[i] for i in indexes]

    relate = bench_relate(workdir, count, payload, get_ids)
    relate_no_flush = bench_relate(workdir, count, payload, get_ids, wal_enabled=False)
    sqlite = bench_sqlite(workdir, count, payload, get_ids)

    expected = count // 2 + 1 + max(1, count // 10)
    counts = (
        relate["ancestry_count"],
        relate_no_flush["ancestry_count"],
        sqlite["ancestry_count"],
    )
    if any(found != expected for found in counts):
        log(
            f"Ancestry count mismatch: expected {expected}, RelateSaver returned "
            f"{relate['ancestry_count']}, RelateSaver (wal_enabled=False) returned "
            f"{relate_no_flush['ancestry_count']}, SqliteSaver returned {sqlite['ancestry_count']}"
        )
        return 1

    print(render_markdown(count, args.payload_bytes, relate, relate_no_flush, sqlite))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        import langgraph.checkpoint.sqlite  # noqa: F401
    except ImportError:
        log(
            "langgraph-checkpoint-sqlite is not installed. "
            'Install the dev extra: pip install -e ".[dev]"'
        )
        return 2

    if args.workdir is not None:
        return run(args, args.workdir)
    scratch = Path(tempfile.mkdtemp(prefix="bench-checkpointers-"))
    try:
        return run(args, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
