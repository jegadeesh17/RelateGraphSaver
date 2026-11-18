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
- Standard: High reliability, crash safety, strict thread safety, and zero data corruption on abnormal process termination.

### 1.3 Target Persona
AI Agent Engineers building, debugging, and observing autonomous workflows in LangGraph who require durable, crash-safe execution persistence and rapid time-travel debugging without database server infrastructure.

---

## 2. User Journeys

The product capabilities are structured across four delivery milestones:
- **Milestone 1:** MVP Vertical Slice (Core `RelateSaver` engine, `RelateThreadProxy` actor, single-file WAL persistence, `put`/`get_tuple`/`list`).
- **Milestone 2:** Core Flows & Time-Travel Debugger (`put_writes` human-in-the-loop, `checkpoint_ns` subgraphs, `:memory:` sandboxed replay, `relate-dag tree` terminal visualizer).
- **Milestone 3:** Forensic Diagnostics & Production Polish (Cypher loop detection, poison-tool root-cause analysis, multi-agent demo application, PyPI packaging).
- **Milestone 4:** Judge Readiness (clean-install CLI fix, lint and type-check baseline, README Relate rationale and limits, `:ToolCall` capture from real runs, async checkpointer methods, SQLite comparison benchmark).

### 2.1 Milestone 1: MVP Vertical Slice

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

### 2.2 Milestone 2: Core Flows & Time-Travel Debugger

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

### 2.3 Milestone 3: Forensics & Production Polish

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

### 2.4 Milestone 4: Judge Readiness

Milestone 4 changes the finished Milestone 3 package. Each journey states the existing behavior it starts from; synchronous behavior and the graph schema stay unchanged except for the new `:ToolCall` writes.

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

---

## 3. Acceptance Criteria

### 3.1 Milestone 1: MVP Vertical Slice
AC-01: Given a non-existent database file path, When RelateSaver is initialized with that path, Then the file is created with initialized schema, constraints, and WAL mode enabled.
AC-02: Given an initialized RelateSaver, When put() is invoked with a valid config, Checkpoint, CheckpointMetadata, and ChannelVersions, Then the state is persisted as a checkpoint node in the Relate graph and the returned RunnableConfig contains the new checkpoint_id.
AC-03: Given an existing thread execution with multiple checkpoints, When get_tuple() is called with a thread_id, Then it returns the latest CheckpointTuple with matching checkpoint, metadata, and parent_config.
AC-04: Given an existing thread execution with multiple checkpoints, When get_tuple() is called with a specific checkpoint_id in config, Then it returns the exact CheckpointTuple matching that checkpoint_id.
AC-05: Given a non-existent thread_id or checkpoint_id, When get_tuple() is invoked, Then it returns None without raising an exception.
AC-06: Given an execution with multiple checkpoints, When list() is called with a thread_id, Then it yields CheckpointTuple records ordered reverse-chronologically by checkpoint step.
AC-07: Given an execution with multiple checkpoints, When list() is called with before and limit parameters, Then it yields at most limit CheckpointTuple records created before the specified checkpoint_id.
AC-08: Given multiple OS threads concurrently invoking put() on the same RelateSaver instance, When transactions execute, Then RelateThreadProxy serializes all write operations without raising database thread-affinity or lock contention exceptions.
AC-09: Given concurrent reads and writes across worker threads, When get_tuple() executes while writes are queued, Then reads return consistent committed checkpoint states without dirty reads or partial writes.
AC-10: Given an uncommitted checkpoint transaction interrupted by an abrupt process termination, When RelateSaver reopens checkpoints.rl, Then WAL recovery rolls back the uncommitted transaction leaving the database consistent.
AC-11: Given a committed checkpoint in checkpoints.rl, When the host process terminates and restarts, Then get_tuple() retrieves the exact state of the last committed checkpoint with byte-for-byte fidelity.
AC-12: Given a checkpoint payload containing un-serializable objects (such as open sockets or raw file handles), When put() is invoked, Then a CheckpointSerializationError is raised identifying the offending state key.
AC-13: Given standard LangGraph types including messages, datetimes, UUIDs, and Pydantic models, When persisted and retrieved through RelateSaver, Then types and data values are preserved across serialization.

### 3.2 Milestone 2: Core Flows & Time-Travel Debugger
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

### 3.3 Milestone 3: Forensics & Production Polish
AC-25: Given an agent execution with state oscillation across nodes exceeding a threshold, When detect_loops() is invoked, Then a list of detected execution cycles is returned containing node sequences, occurrence counts, and checkpoint ID intervals.
AC-26: Given an execution with no repeating cycles, When detect_loops() is invoked, Then an empty cycle report is returned.
AC-27: Given an execution terminating in an error checkpoint, When trace_root_cause() is invoked, Then a causal path is returned tracing back to the earliest antecedent tool invocation or state write that preceded the error.
AC-28: Given a thread without errors, When trace_root_cause() is invoked, Then the report indicates no failure root cause was identified.
AC-29: Given the packaged examples/research_agent.py application, When executed from the command line, Then it completes the multi-agent planning and critique cycle, saving all checkpoints to checkpoints.rl.
AC-30: Given the demo application halting at a human review prompt, When the user supplies input, Then the graph resumes execution from the pending checkpoint and reflects the input in the final output.
AC-31: Given a clean Python 3.11+ virtual environment, When pip install . is run, Then the package installs successfully without dependency resolution errors.
AC-32: Given the installed package, When relate-dag --help is executed in terminal, Then command options and subcommands (tree, replay, forensics) are displayed with exit code 0.

### 3.4 Milestone 4: Judge Readiness

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

### 3.5 Judge-run fixes

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
AC-73: Given an environment with no LLM API keys set and no network access, When `python examples/build_demo_store.py` is run, Then it exits with code 0 and writes a closed single-file store at `examples/data/demo.rl` containing exactly the threads `demo-tool-fail`, `demo-tool-ok` and `demo-research`, with no sidecar files left beside it.
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

---

## 4. Non-Goals

1. **Web UI or Browser-Based Dashboard:** RelateGraphSaver focuses strictly on an embedded Python engine and terminal-native CLI (`relate-dag`). Visual graphs can be rendered in terminals or exported to Graphviz/Mermaid formats.
2. **Distributed Multi-Node Database Server:** Relate is an embedded graph engine designed for single-process, local file, or in-memory persistence. Distributed clustering and remote client-server connections are out of scope.
3. **Automatic Migration Utilities for SqliteSaver Databases:** No automated schema or data migration tooling from existing SQLite-based checkpointers is provided in initial releases.
4. **Arbitrary Graph Query Language Extensions:** The engine supports standard Cypher queries supported natively by the embedded Relate engine without custom query language transpilation.
5. **Milestone 4: Web UI:** Judge readiness adds no web or browser interface; the terminal CLI remains the only user interface.
6. **Milestone 4: Multi-Process Support:** RelateSaver remains a single-process engine. Multi-worker or multi-pod deployments are documented as a limit (use PostgresSaver), not solved.
7. **Milestone 4: Native Async Relate Driver:** The async checkpointer methods delegate to the synchronous methods off the event loop; no native async driver for Relate is built.
8. **Milestone 4: Sync Behavior and Schema Changes:** Existing synchronous behavior and the graph schema are not changed, except for adding `:ToolCall` node and `[:INVOKED]` relationship writes from real runs.
9. **Milestone 4: Benchmark Claims:** The benchmark reports measured numbers only; it makes no claim that Relate is faster than SQLite, and no tuning work is done to make it so.
10. **Judge-run fixes: Stored Data Rewrites:** Shortening the trace-error message is display-only; no stored `error` property is rewritten or migrated.
11. **Judge-run fixes: Byte-Identical Demo Store:** Regenerating `examples/data/demo.rl` reproduces the same threads and tool-call outcomes, not a byte-identical file; checkpoint ids are time-based.
12. **Judge-run fixes: Fork and Capture Limits:** Forks that miss tool results after trimmed message history, unbounded growth of the applied-tool-call set in long-lived savers, a resent INVOKED edge landing one step late, and silent fork writes-read failures remain documented known limits, not fixes.
13. **Judge-run fixes: Submission Assets:** Making the repository public, the demo video and the Devpost description are outside this specification.

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

4. **Milestone 4: Orphan Tool Results:**
   - *Question:* What happens when a tool result message references a `tool_call_id` for which no `:ToolCall` node exists (for example, the AI message was never persisted)?
   - *Assumption:* The result is skipped silently: no `:ToolCall` node is created from a tool result alone, and the write succeeds (AC-49).

5. **Milestone 4: Async Thread Deletion:**
   - *Question:* Should `adelete_thread` be added?
   - *Assumption:* Only if a synchronous `delete_thread` exists on `RelateSaver`. As of 2026-11-17 it does not, so no `adelete_thread` override is added and the base-class behavior is unchanged (AC-55).

6. **Milestone 4: Benchmark Location and Size:**
   - *Question:* Where does the benchmark script live and what checkpoint count N is used?
   - *Assumption:* A standalone script under `benchmarks/` that accepts N as a command-line argument with a small default suitable for a laptop run; the README records the N and environment used for the published numbers.

7. **Judge-run fixes: Committing a gitignored file type:**
   - *Question:* `.rl` files are gitignored; how is `examples/data/demo.rl` committed?
   - *Assumption:* A negation rule in `.gitignore` (for example `!examples/data/demo.rl`) tracks only that file; all other `.rl` files stay ignored (AC-69).

8. **Judge-run fixes: Demo store without API keys:**
   - *Question:* The research agent normally calls an LLM; how does `build_demo_store.py` populate `demo-research` without keys?
   - *Assumption:* The script drives both agents with scripted or fake model responses and auto-approves any human-in-the-loop interrupt, so it runs offline (AC-73).

9. **Judge-run fixes: Read-only use of the committed store:**
   - *Question:* Opening a single-file Relate store may create WAL or lock sidecars or touch the file; does that dirty the clone?
   - *Assumption:* Inspection commands open the store read-only ([ADR-0015](DECISIONS.md#adr-0015-read-only-opening-for-inspection-commands)); verified byte-identical with no sidecars (AC-75, AC-81, AC-84).

10. **Judge-run fixes: Replay id in the README:**
    - *Question:* Checkpoint ids change when the store is regenerated; how does the README stay correct?
    - *Assumption:* The README quotes an id from the committed store and is updated whenever the store is regenerated; AC-80 detects drift.

11. **Judge-run fixes: Tree width with full ids:**
    - *Question:* Full UUID checkpoint ids make tree lines longer; is wrapping acceptable?
    - *Assumption:* Yes, provided each id remains a single unbroken, copyable token in redirected output (AC-62, AC-64).

7. **Milestone 4: Which Writes Are Scanned for Tool Calls:**
   - *Question:* Are tool calls extracted from `put_writes` writes, from `put` checkpoint channel values, or both?
   - *Assumption:* Both: any message list written through `put_writes` or contained in the channel values passed to `put` is scanned. The `call_id` uniqueness constraint makes the overlap harmless (AC-48).
