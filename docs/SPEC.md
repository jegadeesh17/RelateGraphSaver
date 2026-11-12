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

The product capabilities are structured across three delivery milestones:
- **Milestone 1:** MVP Vertical Slice (Core `RelateSaver` engine, `RelateThreadProxy` actor, single-file WAL persistence, `put`/`get_tuple`/`list`).
- **Milestone 2:** Core Flows & Time-Travel Debugger (`put_writes` human-in-the-loop, `checkpoint_ns` subgraphs, `:memory:` sandboxed replay, `relate-dag tree` terminal visualizer).
- **Milestone 3:** Forensic Diagnostics & Production Polish (Cypher loop detection, poison-tool root-cause analysis, multi-agent demo application, PyPI packaging).

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
AC-31: Given a clean Python 3.10+ virtual environment, When pip install . is run, Then the package installs successfully without dependency resolution errors.
AC-32: Given the installed package, When relate-dag --help is executed in terminal, Then command options and subcommands (tree, replay, forensics) are displayed with exit code 0.

---

## 4. Non-Goals

1. **Web UI or Browser-Based Dashboard:** RelateGraphSaver focuses strictly on an embedded Python engine and terminal-native CLI (`relate-dag`). Visual graphs can be rendered in terminals or exported to Graphviz/Mermaid formats.
2. **Distributed Multi-Node Database Server:** Relate is an embedded graph engine designed for single-process, local file, or in-memory persistence. Distributed clustering and remote client-server connections are out of scope.
3. **Automatic Migration Utilities for SqliteSaver Databases:** No automated schema or data migration tooling from existing SQLite-based checkpointers is provided in initial releases.
4. **Arbitrary Graph Query Language Extensions:** The engine supports standard Cypher queries supported natively by the embedded Relate engine without custom query language transpilation.

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
