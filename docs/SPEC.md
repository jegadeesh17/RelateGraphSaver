# Product Specification: RelateGraphSaver

A detailed product specification defining the user journeys, behavioral requirements, and acceptance criteria for RelateGraphSaver, an embedded graph-native checkpointing and time-travel persistence engine for LangGraph.

Reference documents:
- [Architecture](ARCHITECTURE.md)
- [Decisions](DECISIONS.md)

Date: 2026-11-12

---

## 1. Summary

### 1.1 Vision
RelateGraphSaver is an open-source, embedded, graph-native checkpoint and time-travel persistence engine for LangGraph powered by Relate by Neo4j. It brings single-file Write-Ahead Log (WAL) durability, native Cypher execution Directed Acyclic Graph (DAG) traversals, and zero-cleanup in-memory time-travel replays to autonomous agent workflows without requiring an external PostgreSQL server, Docker container, or daemon process.

### 1.2 Posture
- Posture: Production
- Standard: High reliability, crash safety, strict thread safety, zero data corruption on abnormal process termination, and deterministic time-ordered UUIDv6 checkpoint lineage.

### 1.3 Target Persona
AI Agent Engineers building, debugging, and observing autonomous workflows in LangGraph who require durable, crash-safe execution persistence and rapid time-travel debugging without database server infrastructure.

---

## 2. User Journeys

The product capabilities were delivered in these stages:
- **Core persistence:** Core `RelateSaver` engine, `RelateThreadProxy` actor, single-file WAL persistence, `put`/`get_tuple`/`list`.
- **Time travel and visualizer:** `put_writes` human-in-the-loop, `checkpoint_ns` subgraphs, `:memory:` sandboxed replay, `relate-dag tree` terminal visualizer.
- **Forensics and packaging:** Cypher loop detection, poison-tool root-cause analysis, multi-agent demo application, PyPI packaging.
- **Tool-call forensics and async:** clean-install CLI fix, lint and type-check baseline, README Relate rationale and limits, `:ToolCall` capture from real runs, async checkpointer methods, SQLite comparison benchmark.
- **Replay and provenance:** tree status fix, graph re-execution replay with a run diff, argument provenance, cross-thread tool failures, a key-free real-API demo thread, README and metadata corrections (updated 2026-11-18).
- **Replay and forensics hardening (AC-161..AC-172):** CLI markup escaping, import type-annotation robustness, indirect tool-call root-cause parity, fan-out origin nodes, path spec validation, update-state hint scoping, console width restoration, and typed replay channel comparison (updated 2026-11-21).
- **Write-path and read-side correctness (AC-173..AC-191):** write-path namespace normalization, parent-id extraction, UUIDv6 checkpoint ordering, proxy cancellation, custom serializer wrapping, first-write-wins channel semantics, thread-scoped write and tool-call linking, root-to-leaf loop detection, deterministic failure-checkpoint resolution, non-recursive tree rendering, and read-only diagnostics defaults (updated 2026-11-23).

### 2.1 Core persistence

#### Journey 1: Basic Workflow Persistence with RelateSaver
1. The developer installs `langgraph-checkpoint-relate` and instantiates `saver = RelateSaver("checkpoints.rl")`.
2. The developer compiles a LangGraph workflow with `checkpointer=saver`.
3. The developer runs the workflow with a configuration containing `{"configurable": {"thread_id": "thread-1"}}`.
4. As nodes execute, each step's state, channel values, metadata, and parent linkage are persisted as graph nodes and relationships in `checkpoints.rl`.
5. The developer calls `saver.get_tuple(config)` to inspect the latest checkpoint state.
6. The developer calls `saver.list(config)` to retrieve the full execution history for the thread in chronological order.

#### Journey 2: Thread-Safe Multi-Worker Execution
1. The developer configures a LangGraph workflow that executes branching parallel nodes or handles requests concurrently across a Python thread pool.
2. Multiple worker threads invoke `saver.put()` and `saver.get_tuple()` concurrently for shared and distinct `thread_id` values.
3. The internal `RelateThreadProxy` actor receives write and read requests via dedicated thread queues, executing transactions against the embedded database without database thread-affinity exceptions or lock collisions.
4. All parallel branch states are persisted with accurate parent checkpoint pointers in the execution DAG.

#### Journey 3: Crash Recovery and WAL Durability
1. An executing LangGraph workflow writes checkpoints to `checkpoints.rl` under active Write-Ahead Logging (WAL).
2. The host process abruptly terminates (e.g., power loss, SIGKILL, unhandled host crash) mid-execution.
3. On application restart, the developer re-instantiates `RelateSaver("checkpoints.rl")`.
4. The embedded Relate engine inspects the WAL log, reconciles incomplete transactions, and restores the database to the last atomically committed checkpoint.
5. The developer invokes `saver.get_tuple(config)` and retrieves intact state without corruption or partial graph dangling nodes.

#### Journey 4: Serialization Resilience with Complex State
1. A LangGraph node attempts to persist channel states containing standard LangGraph objects (messages, UUIDs, datetimes, Pydantic models).
2. `RelateSaver` serializes the payload using `JsonPlusSerializer` and stores the binary blob safely in the checkpoint node.
3. If a node accidentally includes an un-serializable payload (e.g., active network socket, file descriptor), `RelateSaver` catches the error and raises a clear `CheckpointSerializationError` indicating the offending state key.

---

### 2.2 Time travel and visualizer

#### Journey 5: Human-In-The-Loop (HITL) Interruption and Pending Writes
1. The developer configures a LangGraph graph with an interrupt condition (e.g., requiring human approval before executing a critical tool).
2. The graph pauses at the interrupt point; pending tool calls and state updates are recorded via `saver.put_writes()`.
3. The developer queries `saver.get_tuple(config)` and inspects `tuple.pending_writes` to evaluate the proposed action.
4. The human reviewer approves or updates the state and invokes `graph.invoke(None, config)`.
5. The workflow resumes seamlessly, writing the subsequent checkpoint with a direct causal edge to the interrupted node in the Cypher DAG.

#### Journey 6: Hierarchical Subgraphs and Namespaced Lineage
1. The developer defines a hierarchical multi-agent workflow featuring a parent orchestrator and nested subgraphs (e.g., `"researcher"` and `"writer"`).
2. During execution, checkpoints are emitted with qualified `checkpoint_ns` identifiers (e.g., `""`, `"researcher"`, `"writer.subgraph"`).
3. The developer lists checkpoints filtered by `checkpoint_ns` to inspect the isolated history of a single sub-agent.
4. Cypher queries traversing the graph maintain parent-child hierarchy edges (`CHILD_OF`, `NEXT`), preserving the nested execution structure.

#### Journey 7: Sandboxed In-Memory Time-Travel Replay
1. The developer inspects an erroneous run in `checkpoints.rl` and decides to test a prompt change at step 3 without altering the durable database.
2. The developer calls `saver.fork_in_memory(thread_id, checkpoint_id)` or CLI equivalent to clone the execution graph up to step 3 into a `:memory:` Relate database.
3. The developer modifies state channels and runs the workflow forward in the in-memory sandbox.
4. The developer evaluates the alternative outcome; `checkpoints.rl` remains completely unchanged.
5. If the developer specifies a non-existent `checkpoint_id`, `RelateSaver` raises `CheckpointNotFoundError` listing valid checkpoint IDs.

#### Journey 8: Terminal Execution Tree Visualization (`relate-dag tree`)
1. The developer opens a terminal and runs `relate-dag tree <thread_id> --file checkpoints.rl`.
2. The CLI connects to the Relate graph database, executes a Cypher DAG traversal, and constructs the execution tree.
3. The CLI prints a structured Unicode DAG to stdout displaying step numbers, node names, checkpoint IDs, timestamps, and branch points.
4. Checkpoints with pending writes or errors are visually styled with distinct status tags.

---

### 2.3 Forensics and packaging

#### Journey 9: Cypher Forensic Diagnostics (Loop Detection & Root Cause Analysis)
1. An autonomous agent enters an unintended cycle or fails with an exception during execution.
2. The developer runs `relate-dag forensics loops <thread_id>` or calls `saver.detect_loops(thread_id)`.
3. A native Cypher query evaluates state hashes and node transitions, returning any repeating cycles with node names and frequency counts.
4. The developer runs `relate-dag forensics trace-error <thread_id>` or calls `saver.trace_root_cause(thread_id)`.
5. A Cypher path traversal inspects the lineage backward from the failure node, identifying the specific upstream tool call or channel write that introduced the erroneous data.

#### Journey 10: Multi-Agent Research Application Demo
1. The developer executes the packaged demonstration application `python -m examples.research_agent`.
2. A multi-agent team (Planner, Researcher, Critic) runs cooperatively, storing all execution state in `checkpoints.rl`.
3. An intentional interrupt prompts the user for feedback in the console.
4. Upon feedback submission, execution completes, printing final output and summarizing Cypher graph statistics.

#### Journey 11: PyPI Installation and CLI Usage
1. The developer runs `pip install langgraph-checkpoint-relate`.
2. The developer imports `from langgraph_checkpoint_relate import RelateSaver, RelateThreadProxy, RelateForensics`.
3. The developer executes `relate-dag --help` in the terminal to view subcommands (`tree`, `replay`, `forensics`).

### 2.4 Tool-call forensics and async

This stage changes the finished forensics and packaging stage. Each journey states the existing behavior it starts from; synchronous behavior and the graph schema stay unchanged except for the new `:ToolCall` writes.

#### Journey 12: Clean Install Runs the CLI
Starts from Journey 11, where `relate-dag` failed in a clean environment because `click` was imported but not declared.
1. The evaluator creates a fresh virtual environment and runs `pip install .` (or installs the built wheel).
2. The evaluator runs `relate-dag --help`.
3. The CLI prints its subcommands and exits with code 0; no missing-module error appears.

#### Journey 13: Repository Passes Its Quality Gates
Starts from the existing codebase, where lint, format, and type checks did not pass.
1. The contributor runs `ruff check .`, `ruff format --check .`, and `mypy src tests` from the repository root.
2. Each command exits with code 0.
3. The contributor runs `pytest`; every test module imports and the suite passes with the same behavior as before.

#### Journey 14: Evaluator Learns How and Why Relate Is Used
Starts from the existing `README.md`.
1. The evaluator opens `README.md` and finds the section "How we use Relate, and why".
2. The section explains which Relate capabilities the package depends on and why each matters for checkpointing.
3. A limits note states that RelateSaver is single-process and points multi-worker or multi-pod users to PostgresSaver.

#### Journey 15: Tool Calls from Real Runs Feed Forensics
Starts from Journey 9, where root-cause analysis reads `:ToolCall` nodes that previously had to be inserted by hand.
1. The developer runs an agent graph that calls tools, persisting through RelateSaver.
2. When the model emits an AI message with tool calls, each call is stored as a `:ToolCall` node linked from the current checkpoint by `[:INVOKED]`.
3. When the tool's result message arrives, its outcome (result or error) is recorded on the `:ToolCall` node with the matching `call_id`.
4. A tool fails during the run. The developer runs `relate-dag forensics trace-error <thread_id>` or `saver.trace_root_cause(thread_id)`.
5. The report names the failing tool call, using only data the run itself produced.

#### Journey 16: Async Graph Execution
Starts from the existing synchronous RelateSaver used with `graph.invoke` / `graph.stream`.
1. The developer compiles a graph with RelateSaver and runs it with `await graph.ainvoke(...)` or `async for chunk in graph.astream(...)`.
2. The run completes and its checkpoints are stored exactly as a synchronous run would store them.
3. The developer launches several async runs concurrently on different threads with `asyncio.gather`.
4. All runs complete without errors; database work runs off the event loop and writes stay serialized through `RelateThreadProxy`.

#### Journey 17: Benchmark Against SqliteSaver
New journey; no prior behavior.
1. The developer installs the dev extras and runs the benchmark script with a checkpoint count N.
2. The script writes N checkpoints with RelateSaver and with SqliteSaver and measures put throughput, get latency, on-disk size, and a branch/ancestry query.
3. The script prints the four metrics for both savers and exits with code 0.
4. The README publishes the measured numbers as measured, including metrics where SQLite is faster.

### 2.5 Replay and provenance

These journeys change the finished package so that replay re-runs the graph, forensics traces a bad argument back to the step that produced it and across threads, the committed store holds a failure from a real public API, and the README is correct for a first-time evaluator. Synchronous checkpointer behavior and the graph schema stay unchanged.

#### Journey 18: Reading Step Status in the Tree
Starts from Journey 8, where every checkpoint with a stored write was tagged `[PENDING]`, including completed steps.
1. The developer runs `relate-dag tree <thread>` on a finished run.
2. Completed steps that have a next step show `[OK]`; their writes are labelled `write:`.
3. Only a last step (no next checkpoint) that still holds writes shows `[PENDING]`; errors and interrupts keep their tags.

#### Journey 19: Replaying a Run with an Edit from the CLI
Starts from Journey 7, where `relate-dag replay` only cloned history into `:memory:` and did not run anything.
1. The developer runs `relate-dag replay <thread> <checkpoint_id>` without `--graph` and sees the same summary as before plus a hint about `--graph`.
2. The developer adds `--graph path/to/file.py:factory` and an edit (`--set-message INDEX=TEXT`, or `--update JSON` with optional `--as-node NAME`).
3. The CLI re-runs the graph from that checkpoint in an in-memory sandbox and prints the replay result, a tool-call table (for example `fetch_source #1  error -> ok`) and the final state changes.
4. Bad input prints an `Error:` line and exits with code 1. The source store is never modified.

#### Journey 20: Replaying from Python
1. The developer calls `replay_from_checkpoint(saver, thread_id, checkpoint_id, factory, set_messages=...)`.
2. The result holds the fork, updated and final checkpoint ids, whether the run stopped at an interrupt, and a diff against the recorded run.
3. The developer calls `diff_runs(recorded, replayed)` on two run snapshots to compare any two runs.

#### Journey 21: Finding Where a Bad Argument Came From
Starts from Journey 15, where the root-cause report named the failing tool call only.
1. A tool fails on an argument that an earlier node produced.
2. The developer runs `relate-dag forensics trace-error <thread>`.
3. The report adds a line naming the node and step that first wrote the argument, and how many steps before the failure that was.

#### Journey 22: Tool Failures Across Threads
1. The developer runs `relate-dag forensics failures --db <store>`.
2. A table lists each failing tool with its failure count, total calls, the threads it failed in, and the nodes its bad arguments came from.

#### Journey 23: Exploring a Real-API Failure in the Committed Store
1. The evaluator clones the repository and, without API keys, runs the README's four explore commands against `examples/data/demo.rl`.
2. Thread `demo-real-api` shows a dependency audit whose planner added a package name that does not exist on PyPI; the lookup failed with a real 404.
3. `trace-error` traces the bad name back to the planner, `forensics failures` lists the failing lookup, and the replay command turns `fetch_source` from error to ok. The store stays byte-identical.

#### Journey 24: Evaluator Reads the README
Starts from Journey 14 and the committed-store README.
1. The evaluator reads a plain-language tagline, installs from source (no PyPI install line), and reads a "Why not X?" table comparing Neo4jSaver, SqliteSaver, LangGraph's built-in time travel and tracing tools.
2. The project URLs in the package metadata lead to the real repository.

#### Journey 25: Recorded LLM Run
1. The developer sets `OPENROUTER_API_KEY` in the environment, installs the `llm` extra, and runs `examples/openrouter_agent.py --model <model>`.
2. The run is stored once in `examples/data/demo-llm.rl`, separate from the key-free store.

### 2.6 Replay and Forensics Hardening

These journeys define the expected user and API behavior across code-review hardening fixes, ensuring reliable CLI markup rendering, import stability, forensic root-cause tracing, graph spec validation, update-state error scoping, and terminal console width management.

#### Journey 26: Robust Diagnostics, Spec Validation, and CLI Rendering
Starts from Journeys 8, 9, 19, and 22.
1. The developer runs `relate-dag tree`, `relate-dag forensics loops`, or `relate-dag replay` against threads, database paths, or errors containing Rich markup-like syntax (such as `[red]` or `run[/x]`), and the CLI renders all text literally without formatting artifacts or markup errors.
2. The developer imports `langgraph_checkpoint_relate` in Python environments where `StateGraph` cannot be parametrized at runtime, and the import succeeds without raising `TypeError`.
3. The developer runs root-cause analysis on a thread where tool failures are linked via writes or where fan-out nodes execute concurrently; `trace_root_cause` accurately identifies the failed tool call, and `failure_summary` reports distinct fan-out origin nodes separately.
4. The developer specifies a filesystem graph path without a function or attribute delimiter (`C:\work\graph.py`); `load_graph_factory` raises `GraphSpecError` explaining the required `module:attribute` or `path/to/file.py:attribute` format.
5. The developer replays a graph that triggers `InvalidUpdateError` during graph execution; the CLI prints the failure error without the misleading `--as-node` update hint.
6. The CLI displays wide tables via `_print_wide`; after printing, the terminal width setting is restored, preserving standard rendering for subsequent outputs.

### 2.7 Write-Path and Read-Side Correctness

These journeys define behavioral correctness on write operations, thread proxy execution, DAG traversal lineage, loop detection sequencing, and diagnostics defaults.

#### Journey 27: Strict Write-Path Correctness and Concurrency Handling
Starts from Journeys 1, 2, and 5.
1. The developer compiles and executes a workflow with or without explicit subgraphs. When `get_tuple()` is invoked without a `checkpoint_ns`, it defaults to `""` matching SqliteSaver semantics, while `list()` returns checkpoints across all namespaces when no namespace filter is provided.
2. During state checkpointing, `put()` resolves parent checkpoint linkage strictly from `config["configurable"]["checkpoint_id"]`, ignoring any caller-supplied `metadata["parents"]`.
3. Checkpoint ordering for latest tuple retrieval and pagination (`before`) orders strictly by time-ordered UUIDv6 `checkpoint_id` descending, independent of step numbering.
4. If connection loss or concurrency conflicts occur during reads, `get_tuple()` and `list()` re-raise `RelateConnectionError` and `RelateConcurrencyError` rather than returning empty results or swallowing failures.
5. The developer configures the saver with custom serialization or environment variables; any plain `SerializerProtocol` is automatically wrapped into `RelateStateSerializer`, and configuration values resolve through explicit argument -> `RELATE_*` environment variable -> default (`:memory:`), raising descriptive `ValueError` naming any invalid variable.
6. When client timeouts occur on thread proxy requests, pending futures are cancelled, and the worker thread drops cancelled commands without executing or committing them. For pending channel writes, regular channels apply first-write-wins (`ON CREATE SET`), while index map channels overwrite, maintaining deterministic ordering by `task_id` and `idx`.

#### Journey 28: Forensic Lineage, Loop Detection, and Tree Diagnostics
Starts from Journeys 8, 9, and 21.
1. The developer inspects execution lineage across thread branches; traversals follow `[:NEXT|FORKED_TO*0..]` strictly restricted to the specified `thread_id`.
2. The developer runs loop detection via `detect_loops()`; sequences are constructed by walking `NEXT` relationships from root to leaf per namespace rather than sorting across steps, and thresholds below 2 are rejected with `ValueError`.
3. The developer runs `trace_root_cause()`; poison analysis inspects channel names without false-positive matching against serialized string values, and antecedent tool-call errors on the `__error__` path are attributed as root cause only when within `distance_to_error <= 1`.
4. The developer inspects execution history via `relate-dag tree` or `DAGVisualizer`; linear `NEXT` sequences render as siblings at equal indentation iteratively without recursion, while branches (`FORKED_TO`) nest hierarchically.
5. Replay boundaries determine recorded leaf nodes by halting prior to subsequent input steps where metadata `source == "input"`.
6. `RelateForensics` and `DAGVisualizer` instances default to `read_only=True` on open, preserving file immutability during analysis.

---

## 3. Acceptance Criteria

### 3.1 Core persistence
AC-01: Given a non-existent database file path, When RelateSaver is initialized with that path, Then the file is created with initialized schema, constraints, and WAL mode enabled.
AC-02: Given an initialized RelateSaver, When put() is invoked with a valid config, Checkpoint, CheckpointMetadata, and ChannelVersions, Then the state is persisted as a checkpoint node in the Relate graph and the returned RunnableConfig contains the new checkpoint_id.
AC-03: Given an existing thread execution with multiple checkpoints, When get_tuple() is called with a thread_id, Then it returns the latest CheckpointTuple with matching checkpoint, metadata, and parent_config.
AC-04: Given an existing thread execution with multiple checkpoints, When get_tuple() is called with a specific checkpoint_id in config, Then it returns the exact CheckpointTuple matching that checkpoint_id.
AC-05: Given a non-existent thread_id or checkpoint_id, When get_tuple() is invoked, Then it returns None without raising an exception.
AC-06: Given an execution with multiple checkpoints, When list() or latest get_tuple() is called with a thread_id, Then checkpoints are ordered reverse-chronologically by checkpoint_id (uuid6), superseding step order.
AC-07: Given an execution with multiple checkpoints, When list() is called with before and limit parameters, Then it yields at most limit CheckpointTuple records created before the specified checkpoint_id.
AC-08: Given multiple OS threads concurrently invoking put() on the same RelateSaver instance, When transactions execute, Then RelateThreadProxy serializes all write operations without raising database thread-affinity or lock contention exceptions.
AC-09: Given concurrent reads and writes across worker threads, When get_tuple() executes while writes are queued, Then reads return consistent committed checkpoint states without dirty reads or partial writes.
AC-10: Given an uncommitted checkpoint transaction interrupted by an abrupt process termination, When RelateSaver reopens checkpoints.rl, Then WAL recovery rolls back the uncommitted transaction leaving the database consistent.
AC-11: Given a committed checkpoint in checkpoints.rl, When the host process terminates and restarts, Then get_tuple() retrieves the exact state of the last committed checkpoint with byte-for-byte fidelity.
AC-12: Given a checkpoint payload containing un-serializable objects (such as open sockets or raw file handles), When put() is invoked, Then a CheckpointSerializationError is raised identifying the offending state key.
AC-13: Given standard LangGraph types including messages, datetimes, UUIDs, and Pydantic models, When persisted and retrieved through RelateSaver, Then types and data values are preserved across serialization.

### 3.2 Time travel and visualizer
AC-14: Given an execution pausing at an interrupt, When put_writes() is called with a task_id and writes sequence, Then writes are persisted and linked to the corresponding checkpoint in the Relate graph.
AC-15: Given a checkpoint with pending writes stored via put_writes(), When get_tuple() is called, Then the returned CheckpointTuple contains the matching pending_writes list.
AC-16: Given an interrupted workflow with pending writes, When resumed with put(), Then the newly created checkpoint records the interrupted checkpoint as its parent.
AC-17: Given an execution containing nested subgraphs, When put() is invoked with varying checkpoint_ns values, Then checkpoints are stored with their namespace attributes intact.
AC-18: Given checkpoints across multiple namespaces within a thread, When list() is called with a specific checkpoint_ns filter, Then only checkpoints matching that namespace are returned.
AC-19: Given an existing durable checkpoint in checkpoints.rl, When fork_in_memory() is invoked with a thread_id and checkpoint_id, Then a new RelateSaver instance backed by an isolated :memory: database is returned containing the historical graph up to that checkpoint.
AC-20: Given an active in-memory sandbox session, When new checkpoints are written, Then all modifications occur exclusively in memory and checkpoints.rl remains bitwise identical.
AC-21: Given a non-existent checkpoint_id passed to fork_in_memory(), When invocation occurs, Then CheckpointNotFoundError is raised listing all available checkpoint IDs for that thread.
AC-22: Given a thread with committed checkpoints in checkpoints.rl, When relate-dag tree <thread_id> is executed in the CLI, Then a structured Unicode DAG is output to stdout detailing step sequences, node names, and checkpoint IDs.
AC-23: Given a thread with branches or interrupted pending writes, When relate-dag tree is run, Then branch points and pending writes are visually distinguished with status annotations.
AC-24: Given a non-existent thread_id passed to relate-dag tree, When the command runs, Then the CLI exits with a non-zero exit code and prints an error indicating the thread was not found.

### 3.3 Forensics and packaging
AC-25: Given an agent execution with state oscillation across nodes exceeding a threshold, When detect_loops() is invoked, Then a list of detected execution cycles is returned containing node sequences, occurrence counts, and checkpoint ID intervals.
AC-26: Given an execution with no repeating cycles, When detect_loops() is invoked, Then an empty cycle report is returned.
AC-27: Given an execution terminating in an error checkpoint, When trace_root_cause() is invoked, Then a causal path is returned tracing back to the earliest antecedent tool invocation or state write that preceded the error.
AC-28: Given a thread without errors, When trace_root_cause() is invoked, Then the report indicates no failure root cause was identified.
AC-29: Given the packaged examples/research_agent.py application, When executed from the command line, Then it completes the multi-agent planning and critique cycle, saving all checkpoints to checkpoints.rl.
AC-30: Given the demo application halting at a human review prompt, When the user supplies input, Then the graph resumes execution from the pending checkpoint and reflects the input in the final output.
AC-31: Given a clean Python 3.11+ virtual environment, When pip install . is run, Then the package installs successfully without dependency resolution errors.
AC-32: Given the installed package, When relate-dag --help is executed in terminal, Then command options and subcommands (tree, replay, forensics) are displayed with exit code 0.

### 3.4 Tool-call forensics and async

#### Clean install and CLI dependency
AC-33: Given a fresh Python 3.11+ virtual environment with no other packages installed, When `pip install .` is run followed by `relate-dag --help`, Then the command exits with code 0 and prints the subcommands without an `ImportError` or `ModuleNotFoundError`.
AC-34: Given `pyproject.toml`, When the `[project].dependencies` list is read, Then it declares `click` as a runtime dependency (not only as a dev or optional dependency).

#### Static quality gates
AC-35: Given the repository root, When `ruff check .` is run, Then it exits with code 0 and reports no errors.
AC-36: Given the repository root, When `ruff format --check .` is run, Then it exits with code 0 and reports no files that would be reformatted.
AC-37: Given the repository root, When `mypy src tests` is run, Then it exits with code 0 with no configuration errors (including duplicate-module errors) and no type errors.
AC-38: Given the lint and type fixes applied, When the full `pytest` suite is run in the project virtual environment, Then every test module imports successfully, all tests pass, and no previously existing test is removed, skipped, or marked expected-to-fail to achieve this.

#### README: how we use Relate
AC-39: Given `README.md`, When it is read, Then it contains a section headed "How we use Relate, and why" that names each Relate capability the package relies on (embedded single-file database, WAL durability, native Cypher queries, `:memory:` databases) and the product reason for each.
AC-40: Given `README.md`, When the "How we use Relate, and why" section is read, Then it states that RelateSaver is a single-process engine and recommends PostgresSaver for multi-worker or multi-pod deployments.

#### Tool call capture from real runs
AC-41: Given a RelateSaver and a checkpoint, When `put_writes()` is called with a write whose messages include an AI message carrying one or more tool calls, Then one `:ToolCall` node per tool call is created with its `call_id`, `tool_name`, and serialized arguments, linked as `(:Checkpoint)-[:INVOKED]->(:ToolCall)` from the checkpoint identified by the config.
AC-42: Given a RelateSaver, When `put()` is called with checkpoint channel values whose messages include an AI message carrying tool calls, Then a `:ToolCall` node per tool call is created and linked by `[:INVOKED]` from the stored checkpoint.
AC-43: Given an existing `:ToolCall` node, When a later write contains a tool result message whose `tool_call_id` matches that node's `call_id` and whose status is not an error, Then the result is recorded on that node's `result_blob` and `error` remains null.
AC-44: Given an existing `:ToolCall` node, When a later write contains a tool result message whose `tool_call_id` matches that node's `call_id` and whose status is error, Then the error text is recorded on that node's `error` property.
AC-45: Given the demo agent run end to end through `graph.invoke` with a tool that raises or returns an error result, and no data inserted into the database by any other means, When `trace_root_cause()` is invoked for that thread, Then the report identifies the failing tool by its `tool_name` and `call_id`.
AC-46: Given a write or checkpoint that contains no messages, When `put_writes()` or `put()` is called, Then no `:ToolCall` nodes and no `[:INVOKED]` relationships are created.
AC-47: Given a write or checkpoint containing a malformed message (for example, a tool call without an id or name, or a non-message object in a messages list), When `put_writes()` or `put()` is called, Then the malformed entry is skipped, the call returns normally without raising, and all other writes and well-formed tool calls are persisted.
AC-48: Given a tool call already persisted as a `:ToolCall` node, When the same tool call (same `call_id`) is written again through `put_writes()` or `put()`, Then exactly one `:ToolCall` node with that `call_id` exists afterward and the call does not raise.
AC-49: Given a tool result message whose `tool_call_id` matches no existing `:ToolCall` node, When it is written through `put_writes()` or `put()`, Then no `:ToolCall` node is created from it and the call returns normally.

#### Async checkpointer
AC-50: Given a RelateSaver with stored checkpoints and pending writes, When `aget_tuple()`, `alist()`, `aput()`, and `aput_writes()` are awaited, Then each returns results equal to its synchronous counterpart (`get_tuple`, `list`, `put`, `put_writes`) for the same inputs.
AC-51: Given a compiled LangGraph graph using RelateSaver, When `await graph.ainvoke(...)` is run, Then it completes and the resulting checkpoints are retrievable with `get_tuple()` for that thread.
AC-52: Given a compiled LangGraph graph using RelateSaver, When `graph.astream(...)` is iterated to completion, Then it yields every step's output and the final checkpoint is retrievable for that thread.
AC-53: Given a compiled LangGraph graph using RelateSaver, When at least five `graph.ainvoke(...)` runs on distinct thread_ids are executed concurrently with `asyncio.gather`, Then all runs complete without raising, and each thread's checkpoint history is complete and contains only that thread's state.
AC-54: Given a running event loop, When any of `aget_tuple()`, `alist()`, `aput()`, or `aput_writes()` is awaited, Then the underlying synchronous method executes on a thread other than the event loop's thread, and writes are still submitted through `RelateThreadProxy`.
AC-55: Given RelateSaver, When its public methods are inspected, Then `adelete_thread` is overridden if and only if a synchronous `delete_thread` is defined on RelateSaver, and when present it deletes the same data as `delete_thread`.

#### Benchmark
AC-56: Given a development install with `langgraph-checkpoint-sqlite` available, When the benchmark script is run with a checkpoint count N, Then it exits with code 0 and reports, for both RelateSaver and SqliteSaver, put throughput, get latency, on-disk size after N checkpoints, and the time of a branch/ancestry query.
AC-57: Given `pyproject.toml`, When dependencies are read, Then `langgraph-checkpoint-sqlite` appears only in a development dependency group and not in `[project].dependencies`.
AC-58: Given `README.md`, When the benchmark section is read, Then it contains the four metrics for both savers as measured values, together with the N and the environment used, regardless of which saver performed better on each metric.

### 3.5 CLI output, demo store and read-only inspection

These criteria cover what an evaluator hits when following the README on a stock Windows terminal, and the committed store the Devpost rules require. The committed demo store is `examples/data/demo.rl` with three fixed thread ids: `demo-tool-fail` (tool agent, `fetch_source` fails), `demo-tool-ok` (tool agent, no failure) and `demo-research` (research agent).

#### CLI output on any stdout encoding
AC-59: Given stdout encoded as cp1252 (for example `PYTHONIOENCODING=cp1252` with stdout redirected to a file or pipe), When each of `relate-dag --help`, `relate-dag tree`, `relate-dag replay`, `relate-dag forensics`, `relate-dag forensics loops` and `relate-dag forensics trace-error` is run against a store holding the relevant thread, Then each exits with code 0, no `UnicodeEncodeError` is raised, and the output contains the command's expected text (for example the thread id or the failing tool name).
AC-60: Given stdout encoded as cp1252, When `relate-dag forensics trace-error` is run on a thread with no tool failure and `relate-dag forensics loops` is run on a thread with no loop, Then each exits with code 0 and prints its no-finding message using only ASCII characters (for example `OK` or `[ok]` in place of a check-mark symbol).
AC-61: Given the `relate-dag` source, When every literal status message the CLI prints (not user or stored data) is inspected, Then it contains only ASCII characters.

#### Full checkpoint ids in the tree
AC-62: Given a thread whose checkpoints were all created within the same second, When `relate-dag tree <thread>` is run, Then each checkpoint's full `checkpoint_id` appears in the output, unshortened.
AC-63: Given a thread with N checkpoints, When `relate-dag tree <thread>` is run, Then the output shows N distinct checkpoint id strings, and no two checkpoint nodes render with the same id text.
AC-64: Given any checkpoint id copied from `relate-dag tree <thread>` output, When `relate-dag replay <thread> <that id>` is run on the same store, Then it exits with code 0 and replays from that checkpoint.

#### Trace-error message text
AC-65: Given a `:ToolCall` whose stored `error` ends with " Please fix your mistakes.", When `relate-dag forensics trace-error` reports it, Then the displayed error text is the stored text with that trailing suffix removed, and the stored `error` property is unchanged afterwards.
AC-66: Given a `:ToolCall` whose stored `error` does not end with " Please fix your mistakes." (including one that contains the phrase elsewhere), When `relate-dag forensics trace-error` reports it, Then the displayed error text equals the stored text exactly.

#### Root tool name in the combined summary
AC-67: Given a thread with a failing tool call, When `relate-dag forensics <thread>` is run, Then the summary shows the root tool's `tool_name` alongside its `call_id`.
AC-68: Given a thread with no failing tool call, When `relate-dag forensics <thread>` is run, Then it exits with code 0 and the summary shows no tool name or call id for a root cause.

#### Committed demo store
AC-69: Given a fresh clone of the repository, When the working tree is listed, Then `examples/data/demo.rl` is present and tracked by git, and no `demo.rl-wal`, `demo.rl-lock` or other sidecar file for it is tracked or required.
AC-70: Given a fresh clone with no other files created, When `relate-dag forensics trace-error demo-tool-fail --db examples/data/demo.rl` is run, Then it exits with code 0 and the report names `fetch_source` and its call id.
AC-71: Given a fresh clone, When `relate-dag forensics trace-error demo-tool-ok --db examples/data/demo.rl` is run, Then it exits with code 0 and reports no tool failure.
AC-72: Given a fresh clone, When `relate-dag tree demo-research --db examples/data/demo.rl` is run, Then it exits with code 0 and shows at least one checkpoint.
AC-73: Given an environment with no LLM API keys set and no network access, When `python examples/build_demo_store.py` is run, Then it exits with code 0 and writes a closed single-file store at `examples/data/demo.rl` containing exactly the threads `demo-tool-fail`, `demo-tool-ok` and `demo-research`, with no sidecar files left beside it. (Superseded by AC-134 and AC-136: the store now also holds `demo-real-api`, which needs network access to pypi.org.)
AC-74: Given two runs of `python examples/build_demo_store.py`, When the resulting stores are compared, Then they hold the same thread ids, the same number of checkpoints per thread, and the same `:ToolCall` tool names, call ids and error/no-error outcomes (checkpoint ids and timestamps may differ).
AC-75: Given `examples/data/demo.rl` is read by any `relate-dag` command, When the command finishes, Then the tracked file is unchanged (`git status` reports no modification to it) and no sidecar file remains beside it.

#### README answers the submission template
AC-76: Given `README.md`, When it is read, Then it contains a "Use case" section stating the problem solved and which relationships matter (checkpoint lineage, branches, tool calls).
AC-77: Given `README.md`, When the "How it uses Relate" section is read, Then it contains the line `Mode: single file`, a reason for choosing that mode, and a description of what the graph models (`:Checkpoint`, `:Write`, `:ToolCall` nodes and their relationships).
AC-78: Given `README.md`, When it is read, Then it contains a "Where the data lives" section naming `examples/data/demo.rl` and the thread ids `demo-tool-fail`, `demo-tool-ok` and `demo-research`.
AC-79: Given `README.md`, When its explore-the-committed-store commands are run in order from a fresh clone after installation, without running any agent first, Then `relate-dag tree`, `relate-dag forensics trace-error` and `relate-dag replay` each exit with code 0 against `examples/data/demo.rl`.
AC-80: Given the `relate-dag replay` command shown in `README.md`, When its checkpoint id is compared with the output of `relate-dag tree` for the same thread on the committed store, Then that id appears in the tree output.

#### Read-only inspection
These criteria cover opening a store read-only for inspection ([ADR-0015](DECISIONS.md#adr-0015-read-only-opening-for-inspection-commands)).
AC-81: Given a closed file store, When `RelateSaver(path, read_only=True)` is opened and closed with no queries, Then the file's sha256 is unchanged and no sidecar file exists beside it.
AC-82: Given a read-only saver on a file store, When `put`, `put_writes`, `aput` or `aput_writes` is called, Then `RelateReadOnlyError` is raised and the file's sha256 is unchanged after close.
AC-83: Given a file store with checkpoints and tool calls, When `get_tuple`, `list`, `detect_loops`, `trace_root_cause` and `fork_in_memory` run on a read-only saver, Then the results equal those from a writable saver on a copy of the same store.
AC-84: Given a closed file store, When `relate-dag tree`, `relate-dag forensics`, `relate-dag forensics loops`, `relate-dag forensics trace-error` or `relate-dag replay` runs against it, Then the exit code is the same as against a writable copy of the store, and the file's sha256 and the directory listing are unchanged.
AC-85: Given a path that does not exist, When `RelateSaver(path, read_only=True)` is constructed, Then `RelateConnectionError` is raised and no file or directory is created.
AC-86: Given `read_only=True` with `":memory:"` or with a passed writable proxy, When the saver is constructed, Then `ValueError` is raised.
AC-87: Given `RelateSaver(path)` without `read_only`, When a graph runs and the saver is closed and reopened, Then schema setup, saves and flush-on-close behave as before and the saved checkpoints are readable after reopening.

### 3.6 Replay and provenance

#### Tree step status
AC-88: Given a checkpoint with stored writes that has a child linked by `[:NEXT]` and is neither an error nor an interrupt, When `relate-dag tree <thread>` is run, Then that checkpoint is tagged `[OK]` and not `[PENDING]`.
AC-89: Given a checkpoint with stored writes and no `[:NEXT]` child (a leaf) that is neither an error nor an interrupt, When `relate-dag tree <thread>` is run, Then it is tagged `[PENDING]`.
AC-90: Given an error checkpoint, with or without writes or a `[:NEXT]` child, When the tree is rendered, Then it is tagged `[ERROR]` (error outranks every other status).
AC-91: Given an interrupted checkpoint that is not an error, with or without writes or a `[:NEXT]` child, When the tree is rendered, Then it is tagged `[INTERRUPT]`.
AC-92: Given a checkpoint whose only child is linked by `[:FORKED_TO]` (no `[:NEXT]` child) and that has stored writes, When the tree is rendered, Then it is tagged `[PENDING]`.
AC-93: Given writes on a checkpoint with a `[:NEXT]` child, When the tree is rendered, Then each write is labelled `write: channel=..., task_id=...`; writes on a leaf keep the label `pending write:`.

#### Replay engine (Python API)
AC-94: Given a graph spec `path/to/file.py:attr` relative to the working directory, When `load_graph_factory(spec)` is called from a process whose `sys.path` does not contain the working directory, Then it returns the attribute loaded from that file.
AC-95: Given a graph spec `package.module:attr` for an importable module, When `load_graph_factory(spec)` is called, Then it returns that attribute.
AC-96: Given an attribute that is either a factory accepting `checkpointer=` or an uncompiled `StateGraph`, When it is passed to `replay_from_checkpoint`, Then the replay compiles it against the sandbox saver and completes in both cases.
AC-97: Given a spec with no `:`, a missing file or module, a missing attribute, or an attribute that is neither callable nor a `StateGraph`, When `load_graph_factory(spec)` is called, Then a `ValueError` (or subclass) is raised whose message contains the spec.
AC-98: Given the recorded failing tool-agent thread, When `replay_from_checkpoint` runs from its first checkpoint with no update and no message edit, Then the returned `ReplayResult.diff` has no tool-call changes, no message changes and no channel changes.
AC-99: Given the recorded failing tool-agent thread, When `replay_from_checkpoint` runs from its step 0 checkpoint (the first checkpoint whose state holds the human message) with `set_messages={0: <working URL>}`, Then the diff reports `fetch_source` call 1 changing from error to ok with its `url` argument before and after, and the final answer message differs from the recorded one.
AC-100: Given a replay with `set_messages={0: <text>}`, When the replayed run finishes, Then its message list holds exactly one human message at index 0, with the new text, and no copy of the original human message.
AC-156: Given a graph whose `messages` channel is not reduced by `add_messages`, When `replay_from_checkpoint` is called with `set_messages`, Then `ReplayInputError` is raised mentioning `add_messages`, the graph is not run, and the source store is unchanged ([ADR-0022](DECISIONS.md#adr-0022-replay-input-errors-and-message-edit-safety)).
AC-101: Given `update=<dict>` and `as_node=<node>`, When `replay_from_checkpoint` runs, Then `updated_checkpoint_id` differs from `fork_checkpoint_id` and the replayed run continues from the updated checkpoint.
AC-154: Given a recorded run and a checkpoint whose next task already has stored writes, When `replay_from_checkpoint` is called with no edits, Then that task's node executes again on the sandbox, and `updated_checkpoint_id` equals `fork_checkpoint_id` ([ADR-0022](DECISIONS.md#adr-0022-replay-input-errors-and-message-edit-safety)).
AC-102: Given a research-agent thread and a checkpoint before its human-review interrupt, When `replay_from_checkpoint` runs, Then it returns normally with `interrupted=True` and a diff covering the steps that ran.
AC-103: Given a file store, When `replay_from_checkpoint` completes or raises, Then the file's sha256 and its directory listing are unchanged and the sandbox saver is closed.
AC-104: Given two run snapshots whose tool calls have different call ids but the same tool names in the same order and the same outcomes, When `diff_runs` is called, Then no tool-call change is reported.
AC-105: Given two run snapshots, When the second has an extra call to a tool, lacks a call, or a call whose arguments or outcome differ (matched by tool name and ordinal), Then `diff_runs` reports that call as added, removed or changed respectively.
AC-106: Given two run snapshots that differ only in checkpoint ids, timestamps, message ids, tool-call ids, or channels named `__*` or `branch:*`, When `diff_runs` is called, Then the diff is empty ([ADR-0023](DECISIONS.md#adr-0023-replay-diff-compares-full-message-content)).
AC-159: Given two run snapshots whose messages differ only after the first 120 characters, or only in an AI message's tool-call names or arguments, When `diff_runs` is called, Then that message is reported as `changed`, and the shown before/after text is truncated to 120 characters plus `...` ([ADR-0023](DECISIONS.md#adr-0023-replay-diff-compares-full-message-content)).
AC-107: Given a fork point with several descendant branches, When a replay builds the recorded run, Then it uses the latest `[:NEXT]` leaf that descends from the fork point.
AC-108: Given the package, When `replay_from_checkpoint`, `diff_runs`, `ReplayResult` and `RunDiff` are imported from `langgraph_checkpoint_relate` and `TimeTravelEngine.replay(...)` is called with the same inputs, Then the imports succeed and both calls return equal diffs.

#### Replay CLI
AC-109: Given a stored thread, When `relate-dag replay <thread> <checkpoint_id>` is run without `--graph`, Then it exits with code 0, prints the same six summary lines as before, and prints one further line that mentions `--graph`.
AC-110: Given the committed store, When `relate-dag replay demo-tool-fail <step 0 checkpoint id> --graph examples/tool_agent.py:create_tool_agent_graph --set-message 0=https://example.org/report --db examples/data/demo.rl` is run, Then it exits with code 0 and prints a success line, a "Replay result" section with the new and recorded final checkpoints, a tool-call table containing `fetch_source #1` and `error -> ok` with the argument before and after, and a final state changes table.
AC-111: Given `--graph` and no edit flags, When `relate-dag replay` is run on an unchanged graph, Then it exits with code 0 and states that no tool calls or state changed.
AC-112: Given any of `--update`, `--set-message` or `--as-node` without `--graph`, When `relate-dag replay` is run, Then it exits with code 1 and prints an `Error:` line naming `--graph`.
AC-113: Given `--update` whose value is not valid JSON or not a JSON object, When `relate-dag replay --graph ...` is run, Then it exits with code 1 and prints an `Error:` line.
AC-114: Given `--set-message` without `=`, with a non-integer index, or with an index outside the message list, When `relate-dag replay --graph ...` is run, Then it exits with code 1 and prints an `Error:` line.
AC-152: Given `--graph` and `--as-node` but no effective edit (neither `--update` nor `--set-message`, or `--update '{}'`), When `relate-dag replay` is run, Then it exits with code 1, prints an `Error:` line saying `--as-node` needs an edit, and does not run the graph (no "Replay result" section is printed).
AC-153: Given `--set-message` repeated with the same INDEX, When `relate-dag replay --graph ...` is run, Then it exits with code 1, prints an `Error:` line naming the duplicate index, and does not run the graph (no "Replay result" section is printed).
AC-115: Given a `--graph` spec that fails to load (AC-97), When `relate-dag replay` is run, Then it exits with code 1, prints an `Error:` line containing the spec, and prints no Python traceback.
AC-116: Given `--as-node` naming a node the graph rejects for the update, When `relate-dag replay --graph ...` is run, Then it exits with code 1 and prints an `Error:` line with a hint about `--as-node`.
AC-155: Given a `--graph` factory that does not accept `checkpointer=`, When replay runs, Then `GraphSpecError` is raised naming `checkpointer`; Given a graph node that raises an exception during replay, When `relate-dag replay --graph ...` is run, Then it exits with code 1 and prints `Error: replay failed: <ExceptionType>: <message>`; and invalid replay inputs raise `ReplayInputError`, a `ValueError` subclass, with their existing messages (AC-151) ([ADR-0022](DECISIONS.md#adr-0022-replay-input-errors-and-message-edit-safety)).
AC-117: Given a closed file store, When `relate-dag replay` runs with or without `--graph`, succeeding or failing, Then the store's sha256 and its directory listing are unchanged.
AC-118: Given stdout encoded as cp1252, When the AC-110 command is run, Then it exits with code 0 without `UnicodeEncodeError` and its output contains `error -> ok`.

#### Argument provenance
AC-119: Given a failing tool call whose string argument (3 or more characters) first appears in a write of an earlier checkpoint on the same `[:NEXT]` path (in message content or in tool-call arguments), When `trace_argument_origin(thread_id, error_checkpoint_id, args)` is called, Then it returns the node, step and hop count (steps between that checkpoint and the error checkpoint) of the earliest such write.
AC-120: Given arguments that are not strings, are shorter than 3 characters, or appear in no earlier write, When `trace_argument_origin` is called, Then it returns `None`.
AC-121: Given a thread whose root cause is a failing tool call with a traceable argument, When `trace_root_cause` runs, Then `origin_node`, `origin_step` and `origin_hops` are set; for a thread with no failure or no traceable argument they are `None`, and `RootCauseReport` can still be built without them.
AC-122: Given a root cause with an origin, When `relate-dag forensics trace-error <thread>` is run, Then it prints `Bad argument origin: <arg> came from node <node> (step <N>, <H> hops before failure)`, followed by a note saying what the hops count, and `Causal Path Length:` is followed by a note saying what it counts; without an origin no origin line is printed.

#### Cross-thread tool failures
AC-123: Given a store with failing tool calls in several threads, When `failure_summary()` is called, Then it returns one entry per tool with at least one failure, holding `tool_name`, `failures`, `total_calls`, the `threads` it failed in and the `origin_nodes` of its bad arguments, ordered by `failures` descending; tools that never failed are omitted.
AC-158: Given a failing tool call linked only through `(:Checkpoint)-[:HAS_WRITE]->(:Write)-[:INVOKED]->(:ToolCall)`, When `failure_summary()` is called, Then that call is counted.
AC-124: Given the committed store, When `relate-dag forensics failures --db examples/data/demo.rl` is run, Then it exits with code 0 and prints a table whose rows include `fetch_source` and `lookup_pypi_package`.
AC-125: Given a store with no failing tool call, When `relate-dag forensics failures --db <store>` is run, Then it exits with code 0 and prints an ASCII no-failures message.

#### Stored text in forensics output
AC-160: Given stored error text or a thread id containing Rich markup such as `[/b]` or `[red]`, When `relate-dag forensics trace-error <thread>` or `relate-dag forensics <thread>` (the `diagnose` command) is run, Then the text prints literally and the command does not crash.

#### Research agent notes
AC-126: Given `examples/research_agent.py`, When its generated notes text is read, Then it contains none of "sub-millisecond", "instantaneous" or "eliminate branch-tracking ambiguities", nor any other latency or speed claim.

#### Key-free dependency agent
AC-127: Given a fake fetcher, When `run_dependency_agent(saver, thread_id, fetch_json=fake)` runs the request "Audit the dependencies of RelateGraphSaver", Then the stored run visits `intake`, `planner`, `agent`, `tools`, `agent`, `report` in that order and the planner's package list includes `relategraphsaver`.
AC-128: Given a fake fetcher that raises a 404 for `relategraphsaver`, When the dependency agent runs, Then the `lookup_pypi_package` call for that name is stored with an error, the run still reaches `report`, and the report mentions the failed lookup.
AC-129: Given the AC-128 thread, When `trace_root_cause` runs, Then the root tool is `lookup_pypi_package`, `origin_node` is `planner` and `origin_hops` is at least 2.
AC-130: Given a stubbed `urlopen`, When `fetch_pypi_json(name)` is called, Then it requests the PyPI JSON URL for that name with a 10 second timeout and a generic User-Agent containing no personal data, and returns only `name`, `version` and `summary`.
AC-131: Given the default `pytest` run, When tests are collected, Then the test marked `network` is deselected and no unknown-marker warning appears; When `pytest -m network` is run with network access, Then the live test gets a real 404 for `relategraphsaver`.
AC-132: Given an environment with no LLM API key set, When the dependency agent runs, Then it completes without reading any key.

#### Committed demo store with a real-API thread
AC-133: Given `examples/data/demo.rl`, When the AC-70, AC-71 and AC-72 commands are run, Then they still pass and the store also holds thread `demo-real-api`.
AC-134: Given network access to pypi.org, When `python examples/build_demo_store.py` is run, Then it exits with code 0 and writes a closed single-file store containing exactly `demo-tool-fail`, `demo-tool-ok`, `demo-research` and `demo-real-api`, with no sidecar files.
AC-135: Given the committed store, When the `demo-real-api` tool calls are read, Then a `lookup_pypi_package` call for `relategraphsaver` holds an error recording a 404 and at least one other lookup holds a result.
AC-136: Given an existing store at the output path and a fetcher that cannot reach the network, When `build_demo_store` runs, Then it exits with code 1 with a message naming the network failure, the existing store's sha256 is unchanged, and no staging file is left.
AC-137: Given two builds, When the stores are compared, Then all four threads have the same checkpoint counts per thread and the same tool names and error/no-error outcomes; tool result text may differ.
AC-138: Given a fresh clone, When `relate-dag tree demo-real-api`, `relate-dag forensics trace-error demo-real-api` and `relate-dag forensics failures` are run with `--db examples/data/demo.rl`, Then each exits with code 0, trace-error prints a `Bad argument origin:` line at least 2 hops back, the tree shows completed steps as `[OK]`, and the store is byte-identical afterwards.

#### Recorded LLM run
AC-139: (optional) Given `pyproject.toml`, When dependencies are read, Then an `llm` extra declares `langchain-openai` and `[project].dependencies` does not.
AC-140: (optional) Given no `OPENROUTER_API_KEY` in the environment, When `examples/openrouter_agent.py` is run, Then it exits with code 1 with an `Error:` line and makes no network call; `.env.example` lists `OPENROUTER_API_KEY=`.
AC-141: (optional) Given `examples/data/`, When it is listed, Then it holds `demo.rl` and at most `demo-llm.rl`, and `build_demo_store.py` never writes or reads `demo-llm.rl`.

#### Package metadata
AC-142: Given `pyproject.toml`, When `[project.urls]` is read, Then every URL starts with `https://github.com/jegadeesh17/RelateGraphSaver` and the `Issues` URL ends with `/issues`.
AC-143: Given `pyproject.toml`, When the `relate-db` requirement is read, Then it is either `>=1.1.0` (and the full suite passes on 1.1.0 and the README benchmark numbers come from 1.1.0) or `>=1.0.2,<1.1`, and the README's statements about the relate-db version match it.

#### README
AC-144: Given `README.md`, When it is read, Then it contains no `pip install langgraph-checkpoint-relate` line and contains a source install line `pip install -e ".[dev]"`.
AC-145: Given `README.md`, When the tagline under the title is read, Then it names a failing agent step and finding its cause from one file offline, and contains none of "WAL", "DAG" or "graph-native".
AC-146: Given `README.md`, When the "Why not X?" section is read, Then its table has rows for Neo4jSaver, SqliteSaver, LangGraph built-in time travel and tracing tools, and the Neo4jSaver row states that it needs a Neo4j server.
AC-147: Given `README.md`, When its explore section is read, Then it holds exactly four `relate-dag` commands: `tree demo-real-api`, `forensics trace-error demo-real-api`, `forensics failures`, and `replay demo-tool-fail <step 0 checkpoint id> --graph examples/tool_agent.py:create_tool_agent_graph --set-message 0=https://example.org/report --db examples/data/demo.rl`, with every flag before `--db`.
AC-148: Given a fresh clone after a source install, When the four explore commands are run in order, Then each exits with code 0, the replay prints `fetch_source #1` and `error -> ok`, and `git status` shows no change.
AC-149: Given the README replay command, When its checkpoint id is compared with `relate-dag tree demo-tool-fail` on the committed store, Then it is the id of that thread's step 0 checkpoint (the second checkpoint in the tree, the first whose state holds the human message), not the step -1 input checkpoint.
AC-150: Given `README.md`, When "Where the data lives" is read, Then it names all four thread ids of the committed store.
AC-151: Given a checkpoint whose state holds no messages yet (such as the step -1 input checkpoint), When `relate-dag replay <thread> <checkpoint id> --graph SPEC --set-message 0=TEXT` is run, or `replay_from_checkpoint(..., set_messages={0: TEXT})` is called, Then the CLI exits with code 1 and prints an `Error:` line (the API raises `ValueError` with the same message) saying the checkpoint holds no messages yet and suggesting the next (step 0) checkpoint, and the source store is unchanged.

### 3.7 Replay and Forensics Hardening

AC-161: Given a thread ID, database path, checkpoint ID, exception message, or table element containing Rich markup-like syntax (such as `run[/x]` or missing `--db` `[red]missing.rl`), When `relate-dag tree`, `relate-dag forensics loops`, or `relate-dag replay` (without `--graph`) is run, Then all interpolated text is escaped and rendered literally without `MarkupError` or unintended formatting artifacts.
AC-162: Given a Python runtime environment where `StateGraph` cannot be subscripted or parametrized at runtime, When `langgraph_checkpoint_relate` or any of its submodules is imported, Then the import succeeds without raising `TypeError`.
AC-163: Given a thread where a failing tool call is linked to a checkpoint only through `(:Checkpoint)-[:HAS_WRITE]->(:Write)-[:INVOKED]->(:ToolCall)`, When `trace_root_cause(thread_id)` is invoked, Then it detects the failed tool call and identifies it as the root cause with its `tool_name` and `call_id` in parity with `failure_summary()`.
AC-164: Given a workflow step produced by multiple concurrent fan-out nodes whose writes contain a bad argument, When `failure_summary()` is called, Then `origin_nodes` contains each origin node as a distinct separate string element in the list rather than as a single comma-joined string.
AC-165: Given a file-path graph spec without an attribute delimiter and attribute name (such as `C:\work\graph.py` or `path/to/graph.py`), When `load_graph_factory(spec)` is called, Then `GraphSpecError` is raised with a message stating the spec must look like `module:attribute` or `path/to/file.py:attribute`.
AC-166: Given a replayed graph execution where `InvalidUpdateError` is raised by a node during graph execution, When `relate-dag replay --graph ...` is run, Then the CLI exits with code 1 and prints `Error: replay failed: InvalidUpdateError: <message>` without appending the `--as-node` hint; Given `update_state` itself rejects the update prior to execution, When `relate-dag replay --graph ...` is run, Then the CLI exits with code 1 and prints the `--as-node` hint.
AC-167: Given a Rich Console instance with an initial width state, When `_print_wide(console, ...)` is called and completes, Then `console._width` is preserved and restored to its exact pre-call value so subsequent table prints do not stay clamped or pinned.
AC-168: Given a recorded and a replayed run whose non-message channel holds values equal under Python == but of different scalar types (0 and False, 1 and True, 1 and 1.0), at top level or nested in lists or dicts, When diff_runs compares them, Then the channel is reported as changed.
AC-169: Given a recorded and a replayed run whose non-message channel holds float('nan') at the same position, When diff_runs compares them, Then no change is reported for that channel.
AC-170: Given langgraph_checkpoint_relate is imported, When a caller uses GraphFactory at runtime (GraphFactory | None, or typing.get_type_hints on a function annotated with it), Then it is a type, not a str, and no error is raised; AC-162 still holds.
AC-171: Given a checkpoint linked to a successful tool call by (:Checkpoint)-[:INVOKED]-> and to a failing tool call through (:Write)-[:INVOKED]->, When trace_root_cause runs, Then root_tool_call_id and root_tool_name both name the failing call.
AC-172: Given a fan-out step where only some nodes wrote the failing tool argument, When failure_summary() or trace_argument_origin() runs, Then origin nodes list only the nodes whose writes contain the argument; and Given update_state rejects a replay update, When the InvalidUpdateError propagates, Then it carries no added exception note and is_update_state_error still returns True.

### 3.8 Write-Path and Read-Side Correctness

AC-173: Given a config without checkpoint_ns, When get_tuple() is called, Then checkpoint_ns defaults to "" (matching SqliteSaver behavior); and When list() is called without checkpoint_ns, Then results remain unfiltered by namespace.
AC-174: Given a checkpoint configuration and metadata containing parents, When put() is called, Then the parent checkpoint ID is resolved strictly from config["configurable"]["checkpoint_id"] and any metadata["parents"] field is ignored.
AC-175: Given checkpoints stored in a thread, When get_tuple() resolves the latest checkpoint or list() queries with a before filter, Then ordering and filtering use checkpoint_id (ORDER BY c.checkpoint_id DESC and c.checkpoint_id < $before).
AC-176: Given an underlying RelateConnectionError or RelateConcurrencyError during execution, When get_tuple() or list() is invoked, Then the error is re-raised rather than returning None or an empty collection.
AC-177: Given a custom serializer implementing SerializerProtocol passed to RelateSaver, When initialized, Then it is automatically wrapped in RelateStateSerializer(inner_serializer=serde).
AC-178: Given RelateSaver initialization, When configuration parameters are resolved, Then resolution priority is explicit constructor argument, then RELATE_* environment variable (RELATE_DB_PATH, RELATE_WAL_ENABLED, RELATE_PROXY_TIMEOUT_SEC, RELATE_LOG_LEVEL), then defaults (db_path=":memory:", wal enabled=True, timeout=30.0, log_level=INFO); and When an environment variable contains an invalid value, Then ValueError is raised naming the offending variable.
AC-179: Given a RelateThreadProxy command awaiting execution, When a caller timeout expires, Then the future is cancelled and the worker thread skips the cancelled command without executing or committing it.
AC-180: Given concurrent or multiple writes to channel state, When put_writes() executes, Then non-special channels use first-write-wins semantics via ON CREATE SET, WRITES_IDX_MAP channels overwrite, and write records are retrieved ordered by w.task_id, w.idx.
AC-181: Given an execution DAG containing branches and checkpoint relationships, When forensics walks a checkpoint's ancestors, Then the walk follows [:NEXT|FORKED_TO] edges backward and visits only checkpoints within the same thread_id.
AC-182: Given an execution history with multiple checkpoints and namespaces, When detect_loops() is invoked, Then sequences are constructed by walking NEXT relationships from root to leaf per namespace rather than sorting across steps.
AC-183: Given channel state inspected for poison pills during forensics, When root-cause analysis runs, Then poison checks match exclusively against the channel name, and substring matches for 'poison' and 'fault' within serialized value text are not performed.
AC-184: Given an execution terminating in an error along an __error__ path, When trace_root_cause() analyzes antecedent tool calls, Then a tool-call error is accepted as root cause only when distance_to_error <= 1.
AC-185: Given an execution DAG with linear checkpoints and branches, When relate-dag tree or DAGVisualizer renders the graph, Then NEXT children render as siblings at the same depth built iteratively without recursion, and only FORKED_TO relationships nest child branches.
AC-186: Given recorded execution checkpoints spanning multiple invoke runs, When replay determines the recorded leaf via _recorded_leaf, Then traversal halts before the next checkpoint whose metadata source is "input".
AC-187: Given a store path opened for diagnostics, When RelateForensics or DAGVisualizer is instantiated without an explicit read_only argument, Then read_only defaults to True.
AC-188: Given loop detection configuration, When detect_loops() is invoked with threshold < 2, Then ValueError is raised.
AC-189: Given a failing tool call whose [:INVOKED] checkpoint has a [:NEXT] successor and a [:FORKED_TO] branch in the same thread that both hold a ToolMessage for that call id, When trace_root_cause() resolves the failure checkpoint, Then it walks only [:NEXT] edges forward from the [:INVOKED] checkpoint, reports the nearest checkpoint on that chain whose writes hold the ToolMessage as error_checkpoint_id, reports the [:INVOKED] checkpoint when no checkpoint on that chain holds it, and returns the same error_checkpoint_id and origin_hops on every run ([ADR-0030](DECISIONS.md#adr-0030-the-failure-checkpoint-walk-follows-next-edges-only)).
AC-190: Given a checkpoint stored in thread A, When put_writes() is called with a config whose thread_id is thread B or whose checkpoint_ns differs, and whose checkpoint_id is that checkpoint's id, Then the :Write nodes are stored with the config's thread_id and checkpoint_ns, no [:HAS_WRITE] edge and no [:INVOKED] edge from thread A's checkpoint is created for them, and get_tuple() for thread A's checkpoint returns none of them in pending_writes ([ADR-0031](DECISIONS.md#adr-0031-writes-and-tool-calls-attach-only-to-a-checkpoint-of-their-own-thread)).
AC-191: Given :Write nodes stored by put_writes() before their checkpoint exists, When put() stores that checkpoint, Then it links with [:HAS_WRITE] exactly the writes whose thread_id and checkpoint_ns equal its own, and no others ([ADR-0031](DECISIONS.md#adr-0031-writes-and-tool-calls-attach-only-to-a-checkpoint-of-their-own-thread)).

### 3.9 Regression Criteria

RC-01: Given a thread with branches, When `relate-dag tree` is run, Then branch points still show `Branch: [BRANCH]` and an unknown thread still exits non-zero with a not-found error (AC-23, AC-24).
RC-02: Given `relate-dag replay` without `--graph`, When the checkpoint id is empty or unknown, Then it still exits with code 1 and prints `checkpoint_id must be a non-empty string` or `Available checkpoints:` respectively.
RC-03: Given `TimeTravelEngine.fork_in_memory` and module `fork_in_memory`, When called with the same arguments as before, Then they return an in-memory saver with the cloned history and run no graph code.
RC-04: Given the committed store, When `relate-dag forensics <thread>`, `forensics loops <thread>`, `forensics trace-error <thread>` and `forensics trace <thread>` are run, Then they behave as before; `failures` is not taken as a thread id.
RC-05: Given `demo-tool-fail` and `demo-tool-ok`, When `trace-error` runs on each, Then the first still names `fetch_source` and its call id and the second still reports no failure (AC-70, AC-71).
RC-06: Given a rebuilt store, When `demo-tool-fail`, `demo-tool-ok` and `demo-research` are compared with the previous store, Then their checkpoint counts, tool names and outcomes are unchanged.
RC-07: Given `build_demo_store.py --out` with an empty value or a path without the `.rl` suffix, When run, Then it is still rejected.
RC-08: Given `examples/tool_agent.py` and `examples/research_agent.py`, When run with their existing options, Then they accept the same options and print the same kind of output.
RC-09: Given the package, When every previously public name in `langgraph_checkpoint_relate` is imported, Then each import succeeds.
RC-10: Given `README.md` and `pyproject.toml`, When the strings asserted before this change are checked (for example "How we use Relate, and why", `Mode: single file`, "Use case", `RelateSaver`, `fork_in_memory`, `pip install`, the three `relate-dag` commands, and the URL keys `Homepage`, `Repository`, `Issues`), Then all are still present.
RC-11: Given no network access, When `pytest -m "not slow"` is run, Then it passes.
RC-12: Given an existing execution with subgraphs, When get_tuple() is called with an explicit checkpoint_ns, Then it returns checkpoints matching that specific namespace (AC-18).
RC-13: Given an existing execution history, When put() is called with standard config, Then new checkpoints are created and linked to parent checkpoints as before (AC-02).
RC-14: Given relate-dag tree on normal DAG structures, When executed, Then it renders valid step numbers, checkpoint IDs, and node names (AC-22).
RC-15: Given standard serialization without a custom serializer, When put() and get_tuple() run, Then LangGraph state channels are saved and restored with type fidelity (AC-13).

---

## 4. Non-Goals

1. **Web UI or Browser-Based Dashboard:** RelateGraphSaver focuses strictly on an embedded Python engine and terminal-native CLI (`relate-dag`). Visual graphs can be rendered in terminals or exported to Graphviz/Mermaid formats.
2. **Distributed Multi-Node Database Server:** Relate is an embedded graph engine designed for single-process, local file, or in-memory persistence. Distributed clustering and remote client-server connections are out of scope.
3. **Automatic Migration Utilities for SqliteSaver Databases:** No automated schema or data migration tooling from existing SQLite-based checkpointers is provided in initial releases.
4. **Arbitrary Graph Query Language Extensions:** The engine supports standard Cypher queries supported natively by the embedded Relate engine without custom query language transpilation.
5. **Tool-call forensics and async: Web UI:** This stage adds no web or browser interface; the terminal CLI remains the only user interface.
6. **Tool-call forensics and async: Multi-Process Support:** RelateSaver remains a single-process engine. Multi-worker or multi-pod deployments are documented as a limit (use PostgresSaver), not solved.
7. **Tool-call forensics and async: Native Async Relate Driver:** The async checkpointer methods delegate to the synchronous methods off the event loop; no native async driver for Relate is built.
8. **Tool-call forensics and async: Sync Behavior and Schema Changes:** Existing synchronous behavior and the graph schema are not changed, except for adding `:ToolCall` node and `[:INVOKED]` relationship writes from real runs.
9. **Tool-call forensics and async: Benchmark Claims:** The benchmark reports measured numbers only; it makes no claim that Relate is faster than SQLite, and no tuning work is done to make it so.
10. **CLI output, demo store and read-only inspection: Stored Data Rewrites:** Shortening the trace-error message is display-only; no stored `error` property is rewritten or migrated.
11. **CLI output, demo store and read-only inspection: Byte-Identical Demo Store:** Regenerating `examples/data/demo.rl` reproduces the same threads and tool-call outcomes, not a byte-identical file; checkpoint ids are time-based.
12. **CLI output, demo store and read-only inspection: Fork and Capture Limits:** Forks that miss tool results after trimmed message history, unbounded growth of the applied-tool-call set in long-lived savers, a resent INVOKED edge landing one step late, and silent fork writes-read failures remain documented known limits, not fixes.
13. **CLI output, demo store and read-only inspection: Submission Assets:** Making the repository public, the demo video and the Devpost description are outside this specification.

Replay and provenance non-goals:

14. **Publishing to PyPI:** the package is installed from source; no release is published.
15. **Making the repository public, recording the video, submitting on Devpost:** the project owner does these.
16. **Web or WASM UI:** the terminal CLI stays the only interface.
17. **`[TOOL ERROR]` tree tag:** failing tool calls are not tagged in the tree; `forensics` reports them.
18. **Multi-process support:** RelateSaver stays single-process.
19. **Replay without user graph code:** the CLI cannot know a graph, so re-execution requires `--graph`.
20. **Offline build of `demo-real-api`:** no offline or staged-data flag; the real-API thread needs pypi.org.
21. **Exact provenance:** argument origin is a heuristic (string containment, earliest write), not data-flow tracking.
22. **Refactors not required by these criteria.**

Code-review fixes non-goals:

23. **`[:FORKED_TO]` failure-anchor diagnostic note:** modifying the failure-anchor diagnostic logic for forked threads is out of scope.
24. **LangGraph async message-id note:** documenting or altering LangGraph async message ID handling is out of scope.
25. **Store rebuild or README updates:** rebuilding `examples/data/demo.rl` or updating benchmark figures and README copy is out of scope.
26. **Dependency or `pyproject.toml` changes:** adding or modifying runtime or package dependencies is out of scope.

Write-path and read-side correctness non-goals:

27. **Substring error-text matching for poison channels:** matching arbitrary serialized value text for substrings "poison" or "fault" is removed in favor of explicit channel name matching.
28. **Multi-process proxy synchronization:** RelateThreadProxy continues to manage single-process concurrency.

---

## 5. Open Questions & Assumptions

1. **Relate Python Driver Surface:**
   - *Question:* What exact Python binding interfaces does the embedded Relate engine provide for executing local Cypher transactions and configuring WAL modes?
   - *Assumption:* Relate provides an embedded Python connection/session API (or DB-API 2.0 compliant interface) accepting file paths (`path/to/db.rl`) and `:memory:` URI targets, supporting direct `session.run(cypher_query, params)` execution.

2. **Graph Schema Model for LangGraph:**
   - *Question:* Should checkpoints be stored as individual nodes with `PARENT_OF` relationships, or should channels and writes have dedicated first-class nodes?
   - *Assumption:* A hybrid graph model: `:Checkpoint` nodes represent steps connected by `[:PREV_CHECKPOINT]`, with `:Write` nodes connected via `[:HAS_WRITE]` and `:Channel` nodes linked via `[:HAS_CHANNEL]`, allowing Cypher queries to traverse state mutations natively.

3. **CLI Terminal Formatting Library:**
   - *Question:* Which library should be used to render the terminal tree?
   - *Assumption:* Use `rich` for rendering the Unicode tree, color highlights, and table formatting in `relate-dag`, with `click` or `typer` for CLI argument parsing.

4. **Tool-call forensics and async: Orphan Tool Results:**
   - *Question:* What happens when a tool result message references a `tool_call_id` for which no `:ToolCall` node exists (for example, the AI message was never persisted)?
   - *Assumption:* The result is skipped silently: no `:ToolCall` node is created from a tool result alone, and the write succeeds (AC-49).

5. **Tool-call forensics and async: Async Thread Deletion:**
   - *Question:* Should `adelete_thread` be added?
   - *Assumption:* Only if a synchronous `delete_thread` exists on `RelateSaver`. As of 2026-11-17 it does not, so no `adelete_thread` override is added and the base-class behavior is unchanged (AC-55).

6. **Tool-call forensics and async: Benchmark Location and Size:**
   - *Question:* Where does the benchmark script live and what checkpoint count N is used?
   - *Assumption:* A standalone script under `benchmarks/` that accepts N as a command-line argument with a small default suitable for a laptop run; the README records the N and environment used for the published numbers.

7. **CLI output, demo store and read-only inspection: Committing a gitignored file type:**
   - *Question:* `.rl` files are gitignored; how is `examples/data/demo.rl` committed?
   - *Assumption:* A negation rule in `.gitignore` (for example `!examples/data/demo.rl`) tracks only that file; all other `.rl` files stay ignored (AC-69).

8. **CLI output, demo store and read-only inspection: Demo store without API keys:**
   - *Question:* The research agent normally calls an LLM; how does `build_demo_store.py` populate `demo-research` without keys?
   - *Assumption:* The script drives both agents with scripted or fake model responses and auto-approves any human-in-the-loop interrupt, so it runs offline (AC-73).

9. **CLI output, demo store and read-only inspection: Read-only use of the committed store:**
   - *Question:* Opening a single-file Relate store may create WAL or lock sidecars or touch the file; does that dirty the clone?
   - *Assumption:* Inspection commands open the store read-only ([ADR-0015](DECISIONS.md#adr-0015-read-only-opening-for-inspection-commands)); verified byte-identical with no sidecars (AC-75, AC-81, AC-84).

10. **CLI output, demo store and read-only inspection: Replay id in the README:**
    - *Question:* Checkpoint ids change when the store is regenerated; how does the README stay correct?
    - *Assumption:* The README quotes an id from the committed store and is updated whenever the store is regenerated; AC-80 detects drift.

11. **CLI output, demo store and read-only inspection: Tree width with full ids:**
    - *Question:* Full UUID checkpoint ids make tree lines longer; is wrapping acceptable?
    - *Assumption:* Yes, provided each id remains a single unbroken, copyable token in redirected output (AC-62, AC-64).

7. **Tool-call forensics and async: Which Writes Are Scanned for Tool Calls:**
   - *Question:* Are tool calls extracted from `put_writes` writes, from `put` checkpoint channel values, or both?
   - *Assumption:* Both: any message list written through `put_writes` or contained in the channel values passed to `put` is scanned. The `call_id` uniqueness constraint makes the overlap harmless (AC-48).

12. **Replay and provenance: Editing the first human message:**
    - *Question:* Can a sandbox fork replace the first human message through `update_state` with `as_node="__start__"`?
    - *Answer (verified):* Yes, at the step 0 checkpoint, when the new message reuses the original message id (messages are replaced by id; a new id would append a duplicate, AC-100). The step -1 checkpoint holds no messages yet, so the set-message path forks at step 0 (AC-99, AC-110, AC-147, AC-149) and step -1 is an input error (AC-151).

13. **Replay and provenance: Graph spec errors:**
    - *Question:* Which exception does `load_graph_factory` raise for a bad spec?
    - *Assumption:* `ValueError` or a subclass, with the spec in the message (AC-97); the CLI turns it into an `Error:` line (AC-115).

14. **Replay and provenance: Failure summary contents:**
    - *Question:* Does `failure_summary()` include tools that never failed?
    - *Assumption:* No; only tools with at least one failure, ordered by failure count (AC-123).

15. **Replay and provenance: Provenance false matches:**
    - *Question:* A short or common argument string can appear in an unrelated earlier write.
    - *Assumption:* Accepted as a documented heuristic limit; arguments under 3 characters are ignored and the earliest containing write wins (AC-119, AC-120).

16. **Replay and provenance: Rebuilding the store depends on pypi.org:**
    - *Question:* The committed store can now only be rebuilt with network access, and PyPI data drifts.
    - *Assumption:* Accepted: the store is rebuilt once, comparisons use outcomes not result text (AC-137), and a failed fetch leaves the existing store untouched (AC-136). The test for AC-73's offline build is replaced accordingly.

17. **Replay and provenance: relate-db version:**
    - *Question:* Should the package require relate-db 1.1.0?
    - *Assumption:* Decided by running the full suite and benchmark on 1.1.0 in a fresh environment; either outcome satisfies AC-143.

18. **Replay and provenance: Homepage URL form:**
    - *Question:* Should `Homepage` point at the repository root or its `#readme` anchor?
    - *Assumption:* `Homepage` uses `#readme`, `Repository` the root, `Issues` `/issues`; AC-142 accepts any URL under the repository.

19. **Code-review fixes: Non-message channel comparison:**
    - *Question:* Is a separate acceptance criterion required for channel value comparison?
    - *Assumption:* No dedicated acceptance criterion assigned; existing AC-106 and regression criteria assert channel diff invariance, while AC-161 through AC-167 define the specific behavior required for the code-review fixes.

20. **Code-review fixes: Graph spec error hierarchy:**
    - *Question:* Does `GraphSpecError` inherit from `ValueError` to preserve existing exception handling?
    - *Assumption:* Yes, `GraphSpecError` remains a `ValueError` subclass per ADR-0022 and AC-97/AC-155.

21. **Write-path and read-side correctness: detect_loops minimum threshold:**
    - *Question:* Should threshold values below 2 be clamped or rejected?
    - *Assumption:* Rejected with `ValueError` (AC-188), as cycle detection requires at least 2 occurrences to define a repetitive sequence.

22. **Write-path and read-side correctness: Checkpoint ordering in non-UUIDv6 tests:**
    - *Question:* Do legacy or synthetic tests using manual string checkpoint IDs work with UUIDv6 ordering?
    - *Assumption:* Synthetic test IDs must sort chronologically in ASCII string comparison to conform to `c.checkpoint_id DESC` ordering (AC-175).
