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
- `RelateForensics` and `DAGVisualizer` open a store file read-only unless `read_only=False` is passed; with no arguments they now raise `RelateConnectionError` when `checkpoints.rl` is missing instead of creating it (AC-187, ADR-0029).
- `detect_loops` and `relate-dag forensics loops --threshold` require a threshold of at least 2 and raise `ValueError` below it (AC-188).
- `relate-dag tree` renders a thread's checkpoints as a flat list of siblings, with forks nested under the checkpoint they branch from, so long runs no longer indent off screen (AC-185).
- `relate-dag forensics --help` lists the reserved subcommand names and states that thread IDs starting with `-` are not accepted as positional thread IDs.
- The combined `relate-dag forensics <thread>` panel shows the bad-argument origin line, as `trace-error` does.

### Fixed
- `get_tuple()` defaults omitted `checkpoint_ns` to `""` matching SqliteSaver behavior, while `list()` remains unfiltered when no namespace is specified (AC-173).
- `put()` resolves parent checkpoint IDs strictly from `config["configurable"]["checkpoint_id"]`, ignoring any misleading mappings in `metadata["parents"]` (AC-174).
- Latest checkpoint resolution, `list()` pagination, and `before` boundaries sort reverse-chronologically by `checkpoint_id DESC` (UUIDv6) rather than step order, matching LangGraph runtime and fork semantics (AC-175, ADR-0026, amending AC-06).
- `get_tuple()` and `list()` re-raise `RelateConnectionError` and `RelateConcurrencyError` rather than converting database or network failures to empty collections or `None` (AC-176).
- Passing a custom `SerializerProtocol` to `RelateSaver` automatically wraps it in `RelateStateSerializer(inner_serializer=serde)` (AC-177).
- Configuration parameters follow strict three-tier precedence: explicit constructor argument -> `RELATE_*` environment variables (`RELATE_DB_PATH`, `RELATE_WAL_ENABLED`, `RELATE_PROXY_TIMEOUT_SEC`, `RELATE_LOG_LEVEL`) -> default values (`:memory:`, `True`, `30.0`, `INFO`), raising `ValueError` on malformed inputs (AC-178, ADR-0028).
- Calling proxy commands cancels the future on caller timeout and the actor worker skips cancelled commands without executing or committing them to the database (AC-179).
- `put_writes()` implements first-write-wins (`ON CREATE SET`) for standard channels while preserving overwrite semantics for `WRITES_IDX_MAP` special channels, and orders writes deterministically by `w.task_id, w.idx` (AC-180).
- `time_travel.py` propagates database read errors directly rather than converting them to `CheckpointNotFoundError` or empty writes, and closes the in-memory saver actor if batch population fails.
- Actor proxy handles worker initialization timeouts by signalling shutdown and closing database handles, protecting lifecycle state transitions with a unified lock.
- Concurrency and connection error classification in `proxy.py` uses word-boundary regular expressions, eliminating false-positive matches on substrings like "unlocked" or "block".
- The replay diff compares non-message channel scalars by type and value (`0 != False`, `1 != 1.0`), preventing false matches across types, and normalizes `float('nan')` so identical NaN values produce an empty diff (AC-168, AC-169, ADR-0025).
- `GraphFactory` is exported as a runtime type union (`Callable[..., Any] | StateGraph`) without subscripting `StateGraph` at runtime, enabling `GraphFactory | None` and `typing.get_type_hints()` without runtime errors while preserving strict type checker verification (AC-170, ADR-0025).
- `update_state` exceptions are tagged with an internal attribute (`_relate_update_state_error`) instead of an exception note, eliminating framework clutter from user tracebacks while preserving accurate `--as-node` hint scoping (AC-172, ADR-0025).
- `trace_root_cause()` disentangles direct and write-linked tool calls into separate candidate rows, preventing a successful direct call and a failing write-linked call on the same checkpoint from coalescing into a single contradictory row (AC-171, ADR-0025).
- `failure_summary()` and `trace_argument_origin()` parse write task paths to isolate only the fan-out nodes that actually authored the failing tool argument, falling back to writing nodes if unparsable (AC-172, ADR-0025).
- `GraphFactory` uses a string type annotation so `langgraph_checkpoint_relate` imports successfully even in environments where `StateGraph` cannot be parametrized at runtime (AC-162).
- The replay diff compares non-message channel values recursively by key rather than by string representation (`repr`), ignoring dictionary key order and normalizing message lists (AC-106, ADR-0024).
- `trace_root_cause()` detects failing tool calls linked to checkpoints through intermediate `:Write` nodes, achieving parity with `failure_summary()` (AC-163).
- `relate-dag tree`, `loops`, and `replay` escape user and database text (thread IDs, database paths, target threads, and checkpoint IDs) so Rich markup is rendered literally without formatting crashes (AC-161).
- The `--as-node` CLI hint is displayed only when `graph.update_state()` rejects the edit; graph execution failures print `replay failed: InvalidUpdateError: <message>` without the misleading hint (AC-116, AC-166, ADR-0024).
- `failure_summary()` lists each distinct writing node of a concurrent fan-out step separately in `origin_nodes` rather than as a single joined string (AC-164).
- Graph specifications without an attribute delimiter (such as `C:\work\graph.py`) are rejected with `GraphSpecError` and a clear format message before execution (AC-165).
- `_print_wide()` saves and restores the console's width state (`console._width`), ensuring wide tables do not leave the console permanently clamped or pinned (AC-167).
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
- Forensics lineage walks stay inside one thread while following forks within it (AC-181), and loop detection runs per namespace and per fork branch, so subgraphs and sibling branches are never merged (AC-182).
- Loop detection no longer reports a single pass through the graph as a loop, resolves node names from task paths, and orders candidates deterministically (AC-25, AC-182).
- Root-cause tracing matches poison checks on channel names only instead of value text (AC-183), prefers a tool call on the `__error__` path only at distance <= 1, and orders ties deterministically (AC-184).
- Argument provenance narrows the origin to named nodes before falling back to `versions_seen` (ADR-0025, AC-172).
- Replay stops the recorded leaf before the next invoke's input checkpoint (AC-186).
- Database errors raised by tree queries reach the CLI error handler instead of being swallowed.
- `trace_root_cause` no longer toggles the process-wide garbage collector, and long threads trace in linear time: about 1.1 s at 4,000 checkpoints, down from 6.6 s (AC-27).
- Wide CLI tables print through a local Rich `Console` instead of mutating the shared console's private width (AC-167).
