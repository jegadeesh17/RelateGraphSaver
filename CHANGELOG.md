# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.4.html).

## [Unreleased]

### Added
- Key-free dependency agent (`examples/dependency_agent.py`) that looks up the project's runtime dependencies on the PyPI JSON API and records each `lookup_pypi_package` call; tests that reach pypi.org carry a `network` marker and are deselected by default.
- Optional recorded LLM run: `examples/openrouter_agent.py` behind an `llm` extra (`langchain-openai`), and `examples/data/demo-llm.rl` holding one run recorded with `mistralai/mistral-small-3.2-24b-instruct` through OpenRouter. The key is read only from the process environment.
- `replay_from_checkpoint()` and `relate-dag replay --graph`: re-run a user graph from a historical checkpoint in an isolated sandbox, with `--update`, `--set-message` and `--as-node` edits, and print the tool-call and final-state diff against the recorded run (ADR-0017).
- `trace_argument_origin()` and the `Bad argument origin` line in `relate-dag forensics trace-error`: trace a failing tool argument back to the step that first wrote it.
- `failure_summary()` and `relate-dag forensics failures`: summarize failing tool calls across every thread in the store.
- Tool-call capture: `RelateSaver.put()` and `put_writes()` record each tool call and its outcome as a `:ToolCall` node; `fork_in_memory()` copies tool calls only up to the fork point (ADR-0010).
- `root_tool_name` on `RootCauseReport`, `trace_root_cause()` and the `relate-dag forensics trace-error` panel, naming the tool behind a failure.
- Key-free tool-calling demo (`examples/tool_agent.py`) with an end-to-end forensics test.
- Async checkpointer methods (`aget_tuple`, `alist`, `aput`, `aput_writes`).
- Benchmark script comparing `RelateSaver` with LangGraph's `SqliteSaver` (`benchmarks/bench_checkpointers.py`, ADR-0008, ADR-0012).
- Read-only mode, `RelateSaver(read_only=True)`, used by the inspection commands so they leave a store byte-identical (ADR-0015).
- Committed demo store (`examples/data/demo.rl`) and `examples/build_demo_store.py` to rebuild it.
- README sections: use case, how we use Relate and why, where the data lives, exploring the demo store, measured benchmark results and the single-process limit.
- Cypher forensic diagnostic engine (`RelateForensics`, `detect_loops()`, `trace_root_cause()`) detecting state oscillations, repeating node cycles, and backward causal failure root-cause analysis.
- CLI forensic diagnostic subcommands (`relate-dag forensics loops` and `relate-dag forensics trace-error`) with rich tabular rendering.
- End-to-end multi-agent research demo application (`examples/research_agent.py`) orchestrating Planner, Researcher, Human Review interrupt, and Critic with live terminal DAG visualization.
- Standardized PyPI packaging specifications, Trove classifiers, license metadata, and wheel build validation with Hatchling.
- Human-in-the-loop (HITL) support and intermediate task channel writes via `put_writes()` and `:Write` nodes connected via `[:HAS_WRITE]` relationships.
- Hierarchical namespaced subgraph isolation using indexed `checkpoint_ns` properties and parent namespace routing.
- In-memory sandboxed time-travel replay engine (`TimeTravelEngine` and `RelateSaver.fork_in_memory()`) cloning execution subgraphs into zero-side-effect `:memory:` database instances.
- Terminal-native Unicode execution tree visualizer (`DAGVisualizer`) rendering step counters, node names, checkpoint IDs, branch indicators, and status tags (`[OK]`, `[PENDING]`, `[INTERRUPT]`, `[ERROR]`).
- Command-line interface (`relate-dag`) offering interactive `tree` inspection and `replay` sandboxed execution.
- Core `RelateSaver` checkpointer implementing `BaseCheckpointSaver` interface for LangGraph with thread-safe persistence.
- Embedded graph storage backed by Relate DB engine with Cypher DDL schema constraints and `:NEXT` relationship modeling.
- Dedicated `RelateThreadProxy` actor queue maintaining single-thread affinity for Relate DB connections.
- State serialization with LangGraph typed serialization and deterministic SHA-256 state hashing.
- Base exception taxonomy and error mapping for database, concurrency, and serialization failures.
- Write-ahead log (WAL) synchronization and crash recovery support for durable checkpointing.
- Integration test suite validating end-to-end checkpointing, pagination, history retrieval, and multi-threaded execution.

### Changed
- Project URLs point to the public repository, and the `relate-db` requirement is now `>=1.1.0` after the full suite and the benchmark passed on 1.1.0 (ADR-0020).
- README: a plain tagline, a "Why not X?" comparison with Neo4jSaver, SqliteSaver, built-in time travel and tracing tools, install from source, four explore commands with real output, and the benchmark rerun on relate-db 1.1.0.
- `examples/data/demo.rl` holds a fourth thread, `demo-real-api`, recorded against pypi.org, with one failed lookup to trace; the builder leaves the existing store in place when pypi.org cannot be reached (ADR-0019).
- The research agent's notes make no latency or speed claims.
- `relate-dag forensics trace-error` and `forensics` explain what `Causal Path Length` and the origin hop count measure.
- Replay message previews longer than 120 characters end with `...`; the marker is display-only and never affects the diff.
- Variable-length ancestry queries start from the target checkpoint instead of scanning the whole chain.
- Tool-call capture sends only new statements per saver (ADR-0011) and commits without its own flush; the store is flushed on close (ADR-0014).

### Fixed
- `relate-dag replay --graph` no longer hides an exception raised by the user's graph behind a bare `Error:` line: it prints `replay failed: <Type>: <message>`, a factory that does not accept `checkpointer=` raises `GraphSpecError`, and invalid replay inputs raise `ReplayInputError`, a `ValueError` subclass (AC-155, ADR-0022).
- `replay_from_checkpoint(..., set_messages=...)` rejects a graph whose `messages` channel is not reduced by `add_messages` instead of silently dropping the unedited messages (AC-156, ADR-0022).
- The replay diff compares full message content and AI tool-call names and arguments, so changes after the first 120 characters are reported; the shown text is still truncated (AC-106, AC-159, ADR-0023).
- `failure_summary()` counts failing tool calls linked through a `:Write`, as `trace_root_cause()` already did (AC-158).
- `trace_root_cause()` resolves the failure checkpoint once per trace instead of twice (ADR-0021).
- `relate-dag forensics trace-error` and `relate-dag forensics` print stored error text and thread ids containing Rich markup literally instead of failing (AC-160).
- `examples/build_demo_store.py` reports unreadable lookup arguments as an unreachable PyPI instead of an unrelated `TypeError` (AC-136).
- `trace_root_cause()` reports the same error checkpoint and origin hop count on every run for a failed tool call, whichever step the tool call is attached to (ADR-0021).
- `mypy src tests` passes on a `dev`-only install, without the optional `llm` extra.
- `examples/build_demo_store.py` refuses to overwrite the recorded LLM store, and `examples/openrouter_agent.py` refuses to add a second run to a thread that already holds one.
- `relate-dag tree` marks completed steps `[OK]` and reserves `PENDING` for leaves with writes.
- `relate-dag replay` rejects `--as-node` without an edit and a repeated `--set-message` index instead of ignoring them.
- `click` and `langchain-core` are declared as runtime dependencies, so a clean install runs the CLI.
- `ruff check`, `ruff format --check` and `mypy src tests` pass.
- `put()` latency no longer grows with the length of the checkpoint chain.
- A tool call could go missing under concurrent async runs.
- The clean-install test skips only when the package index is unreachable (ADR-0013).
- The benchmark uses an equivalent SQLite ancestry query and removes only its own files (ADR-0012).
- CLI output is readable on any stdout encoding; `tree` shows full checkpoint ids; the LangGraph error suffix is trimmed from forensics output.
- `examples/build_demo_store.py` rejects an output path without the `.rl` suffix, and on failure removes only its own staging file.
- Aligned research demo pipeline default database path to `checkpoints.rl` for zero-configuration CLI visualization.
- Propagated startup file-lock concurrency errors and added automatic transaction rollback on failure in `RelateThreadProxy`.
- Added flag syntax `--checkpoint / -c` support in `relate-dag replay` CLI and guarded metadata dictionary access.
