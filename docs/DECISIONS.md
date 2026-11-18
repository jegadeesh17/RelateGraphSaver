# Architecture Decision Records

Key architectural and technical decisions made for RelateGraphSaver, capturing context, rationale, and consequences.

Reference documents:
- [architecture](ARCHITECTURE.md)
- [specification](SPEC.md)

---

## Index

| ID | Title | Status | Date |
| :--- | :--- | :--- | :--- |
| [ADR-0001](#adr-0001-embedded-single-file-graph-storage-with-relate-by-neo4j) | Embedded Single-File Graph Storage with Relate by Neo4j | Accepted | 2026-11-12 |
| [ADR-0002](#adr-0002-dedicated-actor-proxy-for-thread-affinity-and-single-writer-concurrency) | Dedicated Actor Proxy for Thread-Affinity and Single-Writer Concurrency | Accepted | 2026-11-12 |
| [ADR-0003](#adr-0003-hybrid-graph-blob-schema-for-agent-execution-states) | Hybrid Graph-Blob Schema for Agent Execution States | Accepted | 2026-11-12 |
| [ADR-0004](#adr-0004-in-memory-graph-cloning-for-zero-side-effect-time-travel-sandboxing) | In-Memory Graph Cloning for Zero-Side-Effect Time-Travel Sandboxing | Accepted | 2026-11-12 |
| [ADR-0005](#adr-0005-capture-tool-calls-from-persisted-langchain-messages) | Capture Tool Calls from Persisted LangChain Messages | Accepted | 2026-11-17 |
| [ADR-0006](#adr-0006-async-checkpointer-methods-delegate-to-the-synchronous-methods-on-worker-threads) | Async Checkpointer Methods Delegate to the Synchronous Methods on Worker Threads | Accepted | 2026-11-17 |
| [ADR-0007](#adr-0007-static-quality-gates-pass-through-configuration-and-behavior-neutral-fixes) | Static Quality Gates Pass Through Configuration and Behavior-Neutral Fixes | Accepted | 2026-11-17 |
| [ADR-0008](#adr-0008-standalone-benchmark-script-against-sqlitesaver) | Standalone Benchmark Script Against SqliteSaver | Accepted | 2026-11-17 |
| [ADR-0009](#adr-0009-declare-every-directly-imported-runtime-dependency-and-prove-it-with-a-clean-install) | Declare Every Directly Imported Runtime Dependency and Prove It with a Clean Install | Accepted | 2026-11-17 |
| [ADR-0010](#adr-0010-in-memory-forks-copy-tool-call-data) | In-Memory Forks Copy Tool Call Data | Accepted | 2026-11-17 |
| [ADR-0011](#adr-0011-send-only-new-tool-call-statements-tracked-per-saver) | Send Only New Tool Call Statements, Tracked per Saver | Accepted | 2026-11-17 |
| [ADR-0012](#adr-0012-fair-ancestry-query-and-exact-file-cleanup-in-the-benchmark) | Fair Ancestry Query and Exact File Cleanup in the Benchmark | Accepted | 2026-11-17 |
| [ADR-0013](#adr-0013-clean-install-test-skips-only-when-the-package-index-is-unreachable) | Clean-Install Test Skips Only When the Package Index Is Unreachable | Accepted | 2026-11-17 |
| [ADR-0014](#adr-0014-tool-call-capture-commits-without-its-own-flush) | Tool Call Capture Commits Without Its Own Flush | Accepted | 2026-11-17 |
| [ADR-0015](#adr-0015-read-only-opening-for-inspection-commands) | Read-Only Opening for Inspection Commands | Accepted | 2026-11-18 |

---

## ADR-0001: Embedded Single-File Graph Storage with Relate by Neo4j

Date: 2026-11-12  
Status: Accepted

### Context

LangGraph workflows represent state transitions as directed execution graphs. Developers building autonomous agent systems frequently need to inspect state lineages, branch executions, and debug unexpected cycles or tool errors. Existing checkpointer implementations rely either on relational engines (such as SQLite or PostgreSQL) or remote database instances. Relational checkpointers flatten execution graphs into relational rows or JSON strings, making multi-hop path traversals, branching visualizations, and cycle detection computationally expensive and difficult to express in SQL. Conversely, running a full graph database server (like Neo4j Enterprise or Community in a Docker container) imposes heavy operational friction for local development and embedded edge deployments.

We need a persistent storage engine that offers native graph traversal capabilities (via Cypher) while operating as an embedded, single-file database that requires zero background servers or container setup.

### Decision

We decided to use Relate by Neo4j (`relate-db`) as the core embedded storage engine for `RelateSaver`. 

The engine writes graph data into a single `.rl` file on disk with Write-Ahead Logging (WAL) enabled for crash safety. It executes Cypher queries in-process, allowing direct path traversal and topology queries without network round-trips or server configuration.

### Alternatives considered

1. **PostgreSQL with recursive CTEs (`PostgresSaver`):** While robust and standard in production web applications, PostgreSQL requires hosting a background database server or Docker container, which creates unwanted operational friction for agent engineers working locally.
2. **SQLite (`SqliteSaver`):** Embedded and lightweight, but does not natively support graph data modeling or Cypher syntax. Executing multi-hop causal graphs, branching tree queries, and cycle analysis requires complex recursive SQL queries or slow in-memory graph reconstruction in Python.
3. **Neo4j Server via Docker:** Provides full Cypher capabilities but violates the embedded zero-infrastructure requirement, preventing seamless local testing and single-file portability.

### Consequences

- **Positive:** Developers can persist and query complex agent execution DAGs using native Cypher without spinning up a Docker daemon or database server.
- **Positive:** All state is self-contained in a single portable `.rl` file that can be committed, shared, or archived.
- **Negative:** Embedded single-file database engines require strict write serialization to prevent database lock contention and thread-affinity issues. This is addressed in [ADR-0002](#adr-0002-dedicated-actor-proxy-for-thread-affinity-and-single-writer-concurrency).

---

## ADR-0002: Dedicated Actor Proxy for Thread-Affinity and Single-Writer Concurrency

Date: 2026-11-12  
Status: Accepted  
Amended by: [ADR-0014](#adr-0014-tool-call-capture-commits-without-its-own-flush), [ADR-0015](#adr-0015-read-only-opening-for-inspection-commands)

### Context

LangGraph workflows frequently execute parallel node tasks or handle concurrent user requests using multi-threaded pools (such as `concurrent.futures.ThreadPoolExecutor`). Embedded database drivers often enforce thread affinity (meaning connection handles must only be accessed by the OS thread that created them) or use file-level locking mechanisms that raise immediate collision errors when concurrent threads attempt simultaneous writes.

If multiple worker threads invoke `saver.put()` or `saver.put_writes()` concurrently, direct database access causes thread-affinity exceptions or database lock timeout failures.

### Decision

We decided to implement the `RelateThreadProxy` actor module. 

`RelateThreadProxy` maintains a single, dedicated background worker thread that owns the embedded database connection handle. All write operations (`put`, `put_writes`, transactions) are submitted as command messages to a thread-safe actor queue (`queue.Queue`). The worker thread processes commands sequentially, commits transactions to disk through the Write-Ahead Log (WAL), and resolves thread-safe futures back to the calling threads.

### Alternatives considered

1. **Coarse-grained `threading.Lock` around DB calls:** Placing a mutex directly in `RelateSaver` ensures only one thread executes at a time, but it does not protect against drivers that enforce strict OS thread affinity on connection handles created on worker threads.
2. **Connection-per-thread pooling:** Opening a separate connection handle for each worker thread in the pool. In embedded file-based storage, multiple active connections still collide on WAL write locks and file descriptors, causing frequent retry storms and concurrency crashes.
3. **Multi-process manager:** Spawning an external Python multiprocessing daemon. This introduces heavy inter-process serialization overhead and process lifecycle complexity for a library intended to be embedded.

### Consequences

- **Positive:** Guaranteed immunity against database thread-affinity exceptions and file lock collisions during parallel multi-agent execution.
- **Positive:** Predictable, ordered transaction execution with Write-Ahead Log durability.
- **Negative:** Minor queue dispatch overhead (<1ms per transaction), which is negligible compared to agent LLM inference and node execution times.

---

## ADR-0003: Hybrid Graph-Blob Schema for Agent Execution States

Date: 2026-11-12  
Status: Accepted

### Context

LangGraph checkpoints contain two distinct categories of data:
1. **Topological metadata:** Step indices, thread identifiers, namespace qualifiers, parent checkpoint links, and tool invocation references that form the causal execution DAG.
2. **Channel state payloads:** Arbitrary, complex, and potentially large Python objects (such as Pydantic models, chat message histories, raw binary strings, or nested dictionaries).

If we attempt to decompose entire agent state payloads into individual property graph nodes and edges, the database graph would quickly experience node explosion, and arbitrary nested Python structures would be difficult to schema-validate in Cypher. Conversely, if we store the entire checkpoint as an opaque binary blob in a single table, we lose the ability to query topological paths, detect state cycles, or trace failure root causes using native Cypher.

### Decision

We decided to adopt a hybrid graph-blob data schema:
- **Graph Topology:** First-class nodes (`:Thread`, `:Checkpoint`, `:Write`, `:ToolCall`) and relationships (`[:NEXT]`, `[:FORKED_TO]`, `[:HAS_WRITE]`, `[:INVOKED]`) capture all structural lineage, steps, timestamps, and causal connections.
- **State Hashes:** Each `:Checkpoint` node records a `state_hash` (SHA-256 digest of channel state), allowing fast cycle and oscillation detection without decompressing state payloads.
- **Binary Payloads:** Full channel states and write payloads are serialized using LangGraph's standard `JsonPlusSerializer` (preserving custom types, UUIDs, datetimes, and Pydantic models) and stored as immutable byte arrays (`checkpoint_blob`, `write_blob`) directly on the respective nodes.

### Alternatives considered

1. **Fully decomposed property graph:** Mapping every state dictionary key and nested value to individual graph nodes and relationships. This leads to immense overhead, schema rigidity, and performance degradation when serializing complex custom Python types.
2. **Flat relational JSON table:** Storing checkpoints as flat JSON documents with simple foreign keys. This abandons Cypher's graph traversal strengths, requiring clumsy application-side graph reconstruction for tree visualizers and path analyses.

### Consequences

- **Positive:** High performance for graph traversals and forensics (Cypher queries operate directly on lightweight nodes and relationships without decompressing large state blobs).
- **Positive:** Full serialization fidelity for complex agent data structures through `JsonPlusSerializer`.
- **Negative:** State channel contents cannot be directly filtered inside Cypher queries without deserializing the blob; filtering is performed at the metadata and topological level.

---

## ADR-0004: In-Memory Graph Cloning for Zero-Side-Effect Time-Travel Sandboxing

Date: 2026-11-12  
Status: Accepted  
Amended by: [ADR-0010](#adr-0010-in-memory-forks-copy-tool-call-data), [ADR-0015](#adr-0015-read-only-opening-for-inspection-commands)

### Context

When debugging complex agent trajectories, engineers often need to travel back to a previous checkpoint, modify a state channel or prompt, and re-execute the workflow from that point to test an alternative trajectory. If this replay mutates the primary durable database (`checkpoints.rl`), it risks corrupting historical audit logs, invalidating subsequent steps, or polluting durable storage with speculative failed runs.

We need a mechanism that allows developers to branch and replay from any historical checkpoint with zero disk mutation and zero cleanup required.

### Decision

We decided to implement `TimeTravelEngine` with a `fork_in_memory()` operation.

When a developer targets a checkpoint for replay, `TimeTravelEngine` queries the source database for the historical subgraph up to that checkpoint, instantiates a fresh in-memory Relate database using the `:memory:` URI target, and clones the ancestor nodes and edges into the sandbox. The operation links the branched root via a `[:FORKED_TO]` relationship and returns an isolated `RelateSaver` instance. Any subsequent steps executed in the sandbox run entirely in memory.

### Alternatives considered

1. **Physical database file copy on disk:** Copying the physical `.rl` file to a temporary directory on disk. This incurs high file I/O overhead on large graphs and leaves orphaned temporary files on the developer's filesystem if the process terminates abruptly.
2. **In-place branch tagging in the main database:** Writing alternative execution branches directly into `checkpoints.rl` with branch tags. While supported by the graph schema (`[:FORKED_TO]`), speculative debug trials would permanently alter and inflate the durable database file.

### Consequences

- **Positive:** True zero-side-effect time-travel sandboxing. The primary `.rl` file remains bitwise untouched.
- **Positive:** Zero cleanup overhead; the in-memory database is instantly reclaimed by the OS and Python runtime when the object is closed or garbage-collected.
- **Negative:** Large execution histories cloned into memory will consume RAM proportional to the cloned subgraph. Cloned histories are scoped strictly to the selected thread and ancestor chain to keep memory consumption minimal.

---

## ADR-0005: Capture Tool Calls from Persisted LangChain Messages

Date: 2026-11-17  
Status: Accepted  
Amended by: [ADR-0011](#adr-0011-send-only-new-tool-call-statements-tracked-per-saver)

### Context

Root-cause analysis (`trace_root_cause()`, `relate-dag forensics trace-error`) looks for `:ToolCall` nodes linked from checkpoints by `[:INVOKED]`, but nothing in the package created those nodes; only a test inserted them by hand. So on a real run the forensics report could never name a failing tool. LangGraph does not tell a checkpointer "a tool was called". What it does hand over is the conversation itself: the model's `AIMessage` with its `tool_calls`, and each tool's `ToolMessage` with the matching `tool_call_id` and a `status` of `success` or `error`. Those messages pass through `put_writes()` (as the step's writes) and `put()` (inside the saved channel values).

### Decision

We record tool calls from those messages, inside the saver, in a separate small module (`tool_calls.py`):
- A pure extractor scans the values passed to `put_writes()` and the channel values passed to `put()`. Each tool call in an `AIMessage` becomes one `:ToolCall` node keyed by the call's own id (`call_id`), with `tool_name` and serialized arguments (`args_blob`). A later `ToolMessage` with the same id records the outcome on that node: `result_blob` on success, `error` on failure.
- Nodes are created with `MERGE` on `call_id`, so seeing the same call again (the message history is re-saved at every step) never creates a duplicate. The first outcome recorded wins.
- Each thread links the call once, from the first checkpoint at which it saw the call: the checkpoint named in the config for `put_writes()`, or the checkpoint being stored for `put()`.
- A result that matches no recorded call creates nothing.
- Capture runs as its own transaction through the same single-writer proxy, after the checkpoint or writes have committed. Any failure in capture is logged (without argument or result content) and ignored, so saving a checkpoint can never fail because of it.
- `RootCauseReport` gains a `root_tool_name` field so the report names the failing tool, not only its id.

### Alternatives considered

1. **Capture in the same transaction as the checkpoint:** Rejected; a database error in forensic capture would roll back the checkpoint itself.
2. **A LangChain callback handler that users attach to their tools:** Rejected; it needs extra user wiring and misses runs where it was not attached.
3. **Wrapping LangGraph's `ToolNode`:** Rejected; it only sees tools run through that node and couples the package to one prebuilt component.
4. **Prefixing `call_id` with the thread id to avoid collisions:** Rejected; the stored id would no longer equal the model's tool call id that users see and that results reference.
5. **Linking every checkpoint that contains the call:** Rejected; message histories accumulate, so edges would grow with the square of the conversation length.
6. **One `UNWIND` statement per save instead of one statement per call:** Rejected for now; fewer round trips, but the embedded engine's support is unverified and the per-statement pattern matches existing write code.

### Consequences

- **Positive:** Forensics works on data the run produced, with no change to how users build graphs.
- **Positive:** Parsing is tested without a database; capture problems cannot corrupt or block checkpointing.
- **Negative:** Every `put()` rescans the whole message history, sending one idempotent statement per tool call and per result seen so far. Cost grows with conversation length.
- **Negative:** A crash between the checkpoint commit and the capture transaction loses those nodes until the next `put()` in the thread re-records them, linked to that later checkpoint.
- **Negative:** Tool call ids reused across threads (seen only with hand-written fakes) share one node and its first outcome.

---

## ADR-0006: Async Checkpointer Methods Delegate to the Synchronous Methods on Worker Threads

Date: 2026-11-17  
Status: Accepted

### Context

LangGraph's `ainvoke` and `astream` call the checkpointer's async methods (`aget_tuple`, `alist`, `aput`, `aput_writes`). `RelateSaver` only had synchronous methods, so async graphs could not use it. The embedded Relate engine offers no async Python driver, and every database call already goes through one dedicated proxy thread.

### Decision

Each async method runs its synchronous counterpart with `asyncio.to_thread`, so database work never blocks the event loop and results are identical to synchronous calls. `alist` fetches the list on a worker thread and then yields the items. Writes still pass through `RelateThreadProxy`, so the single-writer guarantee is unchanged. `adelete_thread` is not added because there is no synchronous `delete_thread`; it must be added together with one if that ever changes. We do not build a native async driver.

### Alternatives considered

1. **A native async Relate driver:** Rejected; the engine has none, and building one is out of scope.
2. **An async queue inside the proxy:** Rejected; it duplicates the proxy's ordering logic in a second concurrency model.
3. **A dedicated thread pool owned by the saver:** Rejected; it adds lifecycle code for no gain while the proxy is the real bottleneck.
4. **Calling the synchronous methods directly inside the coroutines:** Rejected; it blocks the event loop on every database call.
5. **Yielding `alist` items one worker-thread hop at a time:** Rejected; one hop per item costs more than reading the whole page, which is a single query anyway.

### Consequences

- **Positive:** Async graphs work with no new dependency, and async and sync callers can share one saver.
- **Negative:** Each async call costs a thread hop, and concurrency is capped by the default executor size; throughput is still bounded by the single writer.
- **Negative:** Cancelling an awaiting task does not stop a write that has already started; it completes and commits.

---

## ADR-0007: Static Quality Gates Pass Through Configuration and Behavior-Neutral Fixes

Date: 2026-11-17  
Status: Accepted

### Context

The fast test command (`ruff check .`, `ruff format --check .`, `mypy src tests`, then the fast tests) did not pass. Ruff reported unused imports and an ambiguous variable name in `forensics.py` and a late import in `tests/integration/test_forensics.py`; 14 files were not formatted. Mypy stopped before type checking because it found `tests/fakes/fake_relate_driver.py` under two module names: `tests/` is not a package, yet tests import `tests.fakes.fake_relate_driver`.

### Decision

- **Mypy:** set `explicit_package_bases = true` and `mypy_path = "src"` in `[tool.mypy]`, so module names come from the repository root and `src`. No `__init__.py` files are added to `tests/`, so how pytest imports tests stays exactly as it is. Type errors that then appear are fixed with annotations. If test helpers need it, a `[[tool.mypy.overrides]]` block for `tests.*` may relax only `disallow_untyped_defs` and `disallow_incomplete_defs`, and a per-module `ignore_missing_imports` is allowed only for third-party packages without type information. No global `ignore_errors`, and every `# type: ignore` names its error code.
- **Ruff:** remove the unused imports, rename the ambiguous variable, and fix the late import by reordering, or with a line-level `# noqa: E402` where an import must follow path setup. Then run `ruff format .` once. No rule is disabled project-wide.
- All of these changes are behavior-neutral: no test is removed, skipped or marked as an expected failure, and the full suite must pass unchanged.

### Alternatives considered

1. **Add `__init__.py` to `tests/`, `tests/unit/` and `tests/integration/`:** Rejected; it changes how pytest names and imports every test module, a runtime change for a type-checker problem.
2. **Exclude `tests/` from mypy:** Rejected; the agreed gate is `mypy src tests`.
3. **Turn off mypy strict mode:** Rejected; it lowers checking of the library itself.
4. **Ignore the failing ruff rules in configuration:** Rejected; it hides future real problems of the same kind.

### Consequences

- **Positive:** One command tells a contributor whether the repository is healthy, and it now passes.
- **Negative:** The number of real type errors hidden behind the configuration error is unknown until mypy runs, so this fix may touch many files with annotations only.

---

## ADR-0008: Standalone Benchmark Script Against SqliteSaver

Date: 2026-11-17  
Status: Accepted  
Amended by: [ADR-0012](#adr-0012-fair-ancestry-query-and-exact-file-cleanup-in-the-benchmark)

### Context

Evaluators want to know what an embedded graph costs compared with LangGraph's standard local checkpointer, `SqliteSaver`. The answer must be measured on the same workload and published as measured, including where SQLite is faster. It must not slow down or destabilize the test suites.

### Decision

We add `benchmarks/bench_checkpointers.py`, a standalone script taking the checkpoint count N as `--checkpoints N`. It writes the same chain of N checkpoints, plus one small branch, with both savers to files in a temporary directory (RelateSaver twice: with the default per-write flush and with `wal_enabled=False`) and reports four metrics: put throughput, get latency (median and 95th percentile), on-disk size after the run, and the time of one ancestry query from the branch tip to the root. Each saver answers that query its own natural way: Cypher path traversal for Relate (anchored on the branch tip, because the unanchored direction is orders of magnitude slower in relate-db 1.0), a recursive SQL query for SQLite. Both must return the same count. Results go to standard output as a Markdown table with the environment, ready to paste into the README. `langgraph-checkpoint-sqlite` is added only to the `dev` extra. The script is not collected by pytest.

### Alternatives considered

1. **A pytest-benchmark suite:** Rejected; it adds a dependency and puts timing-sensitive code in the test suites.
2. **Comparing only through the public checkpointer API:** Rejected; that API has no ancestry query, so the most graph-relevant metric would be lost.
3. **In-memory databases:** Rejected; on-disk size is one of the required metrics, and file-backed runs are the realistic case.
4. **Making `langgraph-checkpoint-sqlite` a runtime dependency:** Rejected; users of the package never need it.

### Consequences

- **Positive:** Reproducible numbers that anyone can regenerate with one command.
- **Negative:** The SQLite ancestry query reads `SqliteSaver`'s internal `checkpoints` table and may need updating if that library changes its table layout.
- **Negative:** The numbers depend on the machine, so the README must state N and the environment with every published result.

---

## ADR-0009: Declare Every Directly Imported Runtime Dependency and Prove It with a Clean Install

Date: 2026-11-17  
Status: Accepted  
Amended by: [ADR-0013](#adr-0013-clean-install-test-skips-only-when-the-package-index-is-unreachable)

### Context

`relate-dag` imports `click`, but `pyproject.toml` does not declare it. It used to arrive indirectly through `typer`; current `typer` releases no longer depend on `click`, so on a clean install the command crashes with a missing-module error. The package also imports `langchain-core` directly (and the new tool call capture uses its message types) while relying on `langgraph` to bring it in.

### Decision

- Add `click>=8.1.0` and `langchain-core>=0.2.0` to `[project].dependencies`.
- A fast test in `tests/unit/test_packaging.py` reads `pyproject.toml` and checks that `click` is a runtime dependency (AC-34) and that `langgraph-checkpoint-sqlite` appears only in the `dev` extra (AC-57). It also scans the imports in `src/` and fails if a non-standard-library top-level module does not map to a declared dependency, using a fixed module-to-package table (`relate` to `relate-db`, `langchain_core` to `langchain-core`, and so on).
- A slow test, `tests/integration/test_clean_install.py`, proves AC-33 end to end: it creates a fresh virtual environment in a temporary directory, runs `pip install <repository root>` without extras, then runs that environment's `relate-dag --help`. It requires exit code 0, the subcommands in the output, and no `ImportError` or `ModuleNotFoundError`. It needs network access to the package index.

### Alternatives considered

1. **Remove the `click` import and use only `typer`:** Rejected; it is a code change with behavior risk, and `click` may still be needed for the custom group class.
2. **Document a manual clean-install command only:** Rejected; it would not catch the next undeclared import.
3. **Rely on transitive dependencies:** Rejected; this is exactly how the defect happened.

### Consequences

- **Positive:** A clean `pip install` gives a working CLI, and a future undeclared import fails a fast test.
- **Negative:** The full suite needs network access for the clean-install test, and that test takes tens of seconds.

---

## ADR-0010: In-Memory Forks Copy Tool Call Data

Date: 2026-11-17  
Status: Accepted

### Context

[ADR-0004](#adr-0004-in-memory-graph-cloning-for-zero-side-effect-time-travel-sandboxing) copies a thread's checkpoints, their writes and the edges between them into an in-memory database. Since [ADR-0005](#adr-0005-capture-tool-calls-from-persisted-langchain-messages), real runs also store `:ToolCall` nodes linked from checkpoints by `[:INVOKED]`, and root-cause analysis reads them to name the failing tool. A fork did not copy them, so `trace_root_cause()` on a fork of a failed run could not name the tool that the same call on the original run names.

### Decision

This amends ADR-0004; its other parts are unchanged.
- A fork also copies every `:ToolCall` node that has an `[:INVOKED]` edge from a copied checkpoint, with `call_id`, `tool_name`, `args_blob` and `created_at`, and the `[:INVOKED]` edges from copied checkpoints to those nodes. Tool calls linked only from checkpoints outside the copied history are not copied, nor are the reserved write-to-tool-call edges.
- A copied node keeps its stored outcome (`result_blob` or `error`) only if the tool's result message for that call is already in the channel values of the checkpoint the fork starts from. The fork finds those messages by deserializing that checkpoint and running the existing tool call extractor over its channel values; no new parsing is added. Otherwise the node is copied without an outcome, so a replay in the fork records its own first outcome.
- The copy is part of the fork's single population transaction. A failed read of the tool calls, or a fork-point checkpoint that cannot be deserialized, makes the fork fail with an error instead of returning an incomplete copy. The source database is only read.

Details: [`fork_in_memory` contract](ARCHITECTURE.md#fork_in_memory-contract).

### Alternatives considered

1. **Copy tool calls invoked from any checkpoint of the thread:** Rejected; it would bring in calls from after the fork point and from other branches.
2. **Recreate tool calls in the fork by re-running capture over the copied writes and checkpoints:** Rejected; it repeats the capture logic, and the copy would differ from the source whenever capture rules or crash gaps made the source differ from a fresh scan.
3. **Copy every outcome as stored:** Rejected by the project owner; an outcome recorded after the fork point would leak into the fork, and because the first outcome wins, a what-if replay could never replace it.
4. **Copy calls without any outcomes:** Rejected; forensics on a fork of a failed run could then never name the failing tool.

### Consequences

- **Positive:** Forensics on a fork taken at the thread's latest checkpoint, or at any checkpoint that already contains the failing tool's result, names the same failing tool and call id as on the source.
- **Positive:** A fork taken before a tool ran shows the call with no outcome, and a replay in the fork records the outcome it actually produces.
- **Negative:** A result that is only in the fork point's pending writes, not yet in its channel values, is not counted; the call shows no outcome until a replay records it.
- **Negative:** Forking now deserializes the fork-point checkpoint once, and fails if that checkpoint cannot be read back.
- **Negative:** Forks of tool-heavy runs use more memory, in proportion to the copied tool calls and their argument and result sizes.
- **Testing default:** the "source untouched" check compares the source file's sha256 before and after the fork. If relate-db proves to modify the main file even on read-only access, the test compares node and relationship counts instead and says why in a comment.

---

## ADR-0011: Send Only New Tool Call Statements, Tracked per Saver

Date: 2026-11-17  
Status: Accepted  
Amended by: [ADR-0014](#adr-0014-tool-call-capture-commits-without-its-own-flush) (capture transaction flush); clarified 2026-11-17 (see Amendment below)

### Context

Under [ADR-0005](#adr-0005-capture-tool-calls-from-persisted-langchain-messages), every `put()` scanned the whole message history and re-sent one statement for every tool call and every result seen so far, in a second transaction that was issued and flushed to disk even when nothing was new. On a tool-heavy 150-step run we measured that this more than doubles save time.

### Decision

This amends ADR-0005; capture still runs as its own best-effort transaction after the checkpoint or writes commit, and the stored graph rules are unchanged.
- Each `RelateSaver` keeps an in-memory record, `AppliedToolCalls`, of which tool call statements it has already committed, keyed by (thread id, call id), with separate marks for "call recorded" and "outcome recorded".
- On each save the saver still finds the tool events in the values, then keeps only those the record does not mark. If none are left, it issues no capture transaction at all. Otherwise it serializes and sends only the new ones.
- The record is updated only after the capture transaction commits. An outcome is marked only if its call was already recorded, or is recorded in the same transaction; otherwise it is sent again by later saves until it can be marked.
- The comparison runs on the caller's thread, under a lock held only for the record lookups and updates. The single database thread does no extra work, and the proxy's interface does not change.
- `put()` and `put_writes()` share the same record. A new saver, including one created after a restart and the saver returned by a fork, starts with an empty record; its first save on a thread re-sends what it finds, which the idempotent statements make harmless.
- The record is never pruned during the saver's lifetime.

### Alternatives considered

1. **Run the comparison inside the database thread:** Rejected; it would add serialization work to the single writer that every caller waits on, and change the proxy's interface, which is costly to change.
2. **Ask the database which calls already exist before each save:** Rejected; it replaces one transaction with one read per save, so a save with nothing new still pays a database round trip.
3. **Remember how many messages of each thread were already scanned and scan only the rest:** Rejected for now; message histories can be edited or trimmed between steps, so a position is not a reliable bookmark.
4. **A size-limited record that forgets the oldest entries:** Not needed yet. It would be safe, since a forgotten entry only causes one idempotent resend, and can be added later without changing stored data.

### Consequences

- **Positive:** In steady state each tool call costs one capture transaction when it first appears and one when its result appears; saves that add no tool activity make no capture transaction.
- **Positive:** After a restart the first save on a thread re-checks its whole history, so calls lost to an earlier failed capture are recorded then, as before.
- **Negative:** Memory grows with the number of distinct (thread, call) pairs a saver sees, roughly a few hundred bytes each, until the saver object is discarded.
- **Negative:** `put()` still walks the full message history in memory on every save; only serialization and database work are limited to new events.
- **Negative:** A tool result that matches no call is never marked, so each later save whose history contains it still issues one capture transaction. Real model runs do not produce such results.
- **Negative (corrected 2026-11-17, see Amendment):** If a save names a checkpoint that is not stored yet, the call's node is created without its `[:INVOKED]` edge; the call is resent by later saves until one names a stored checkpoint, and the edge attaches to that later checkpoint, not the one first named. Under concurrent `ainvoke` LangGraph does this in normal operation.
- **Negative (unchanged from ADR-0005, now stated in full):** If the process dies between a step's checkpoint commit and its capture transaction, that step's tool call data is missing until a later `put()` on the same thread rescans the history. If the thread never saves again, the data stays missing. See [Tool call capture ordering](ARCHITECTURE.md#tool-call-capture-ordering).

### Amendment (2026-11-17): invocations without an edge are not recorded

We first assumed LangGraph always saves a checkpoint before any `put_writes()` that names it. That is false under concurrent `ainvoke`: a task's writes, carrying the model's tool call, can commit before the checkpoint they name. The invocation statement then creates the `:ToolCall` node but finds no checkpoint to link it from. Recording that invocation as done would leave the call permanently without its `[:INVOKED]` edge, so root-cause analysis could never reach it. The rule above is refined, not replaced:
- The invocation statement returns, per call, whether the thread now has an edge to it (`attached`).
- An invocation that committed without an edge is not recorded. The next save that carries the call resends the idempotent statement, which attaches the edge to that save's checkpoint. Its outcome, once the node exists, is recorded as before.
- Details: [Applied tool call record](ARCHITECTURE.md#applied-tool-call-record).

---

## ADR-0012: Fair Ancestry Query and Exact File Cleanup in the Benchmark

Date: 2026-11-17  
Status: Accepted

### Context

The benchmark of [ADR-0008](#adr-0008-standalone-benchmark-script-against-sqlitesaver) compared an anchored Relate ancestry query with a SQLite recursive query whose join order made SQLite scan the whole `checkpoints` table at every step. At N=1000 that form took about 170 ms; with the join order fixed the same query took about 0.93 ms and returned the same count. The README's statement that the ancestry query was faster in Relate came from the unfair form. Separately, the script deleted every file in the working directory whose name merely started with one of its database names, which could remove a user's unrelated files when `--workdir` points at an existing directory.

### Decision

This amends ADR-0008; the workload, metrics and exit codes are unchanged.
- Every query in the benchmark is written in its best known form for its engine. The SQLite recursive member reads `FROM ancestry a CROSS JOIN checkpoints c ON c.checkpoint_id = a.checkpoint_id`, which makes SQLite walk from the current ancestor and look up one row by primary key per step. The Relate query stays anchored on the branch tip.
- The output table has four columns: Metric, RelateSaver with the default flush per write, RelateSaver with `wal_enabled=False`, and SqliteSaver.
- The script deletes only the files it creates, `relate.rl`, `relate-noflush.rl` and `sqlite.db`, and their `-wal`, `-shm` and `-journal` sidecars, and nothing else. On-disk size counts the same files.
- The README may say a saver is faster on a metric only when the published, fairly measured table shows it.

### Alternatives considered

1. **Keep the earlier SQLite query as "the natural way":** Rejected; a query with a known fix that is about 180 times slower is not a fair comparison.
2. **Compare only through the public checkpointer methods:** Rejected again, for the reason in ADR-0008: there is no ancestry method to compare.
3. **Refuse to run in a non-empty `--workdir`:** Rejected; re-running into the same directory is the normal way to inspect results.

### Consequences

- **Positive:** The published numbers can be defended: neither saver is handicapped by a query form we know to be slow.
- **Positive:** Pointing `--workdir` at an existing directory cannot delete unrelated files.
- **Negative:** On current measurements SQLite answers the ancestry query faster than Relate, and the README must say so.
- **Negative:** If relate-db creates sidecar files with other names, they are neither cleaned nor counted in on-disk size until this list is updated. We do not widen the pattern to catch them; whoever finds such a file reports its name so the list is changed deliberately.

---

## ADR-0013: Clean-Install Test Skips Only When the Package Index Is Unreachable

Date: 2026-11-17  
Status: Accepted

### Context

The clean-install test of [ADR-0009](#adr-0009-declare-every-directly-imported-runtime-dependency-and-prove-it-with-a-clean-install) installs the package from the public package index. On a machine without network access it failed, which made the full suite fail for a reason that says nothing about the code. Skipping it on any install failure would hide exactly the missing-dependency defects it exists to catch.

### Decision

This amends ADR-0009. The test skips, with a reason naming the failure, only when `pip install` fails and pip's own output contains one of a fixed list of connection-failure messages (connection refused or timed out, name resolution failure, unreachable network, proxy error). Every other failure fails the test, including "no matching distribution" without a connection message, build errors, SSL errors, timeouts of the test itself, and any failure of `relate-dag --help`. The decision is made by a small helper covered by a fast test with sample pip output. Details: [Clean-install test](ARCHITECTURE.md#clean-install-test-testsintegrationtest_clean_installpy).

### Alternatives considered

1. **Skip on any install failure:** Rejected; it turns real dependency defects into skips.
2. **Check connectivity to the index before installing:** Rejected; the check can succeed while the install still fails on the network, or fail behind a proxy pip can use.
3. **A command-line option or environment variable to skip the test:** Rejected; it relies on people remembering to set it, and a skipped proof looks the same as a passing one in quick reads.

### Consequences

- **Positive:** The full suite passes offline without hiding dependency defects, and the skip reason says the clean install was not verified.
- **Negative:** A network failure whose message is not on the list still fails the test; the list is updated when that happens.
- **Negative:** An offline run does not prove AC-33; it has to be re-run with network access.

---

## ADR-0014: Tool Call Capture Commits Without Its Own Flush

Date: 2026-11-17  
Status: Accepted

### Context

With the default `wal_enabled=True`, every write transaction ends with a flush to disk, and in relate-db 1.0 a flush takes time proportional to the size of the database file. Tool call capture runs as a second transaction after the checkpoint or writes save ([ADR-0005](#adr-0005-capture-tool-calls-from-persisted-langchain-messages), [ADR-0011](#adr-0011-send-only-new-tool-call-statements-tracked-per-saver)), so every save with new tool activity paid for two flushes. On a 150-step tool-heavy run against a file database we measured about 1.53 s per save at the end of the run with capture, against about 0.8 s with capture replaced by a stub. The difference is the capture transaction's own flush.

### Decision

This amends [ADR-0002](#adr-0002-dedicated-actor-proxy-for-thread-affinity-and-single-writer-concurrency), which flushed after every commit.
- `RelateThreadProxy.execute_batch` gains a keyword-only argument `flush: bool = True`. With `flush=False` the batch still commits or rolls back as one transaction before the caller gets its answer; only the flush after the commit is skipped. The default is unchanged, so every other caller keeps flushing.
- Only tool call capture passes `flush=False`. Checkpoint saves, writes saves, single statements, `transaction()` and the fork's population batch keep the default.
- The capture transaction becomes durable with the next flushed write on the same database (normally the next checkpoint or writes save) or when the database is closed. `close()` flushes explicitly before closing the connection, rather than relying on the engine to do so; `abort()` still does not flush.
- Durability statement: if the process is killed, data committed without a flush was kept in our tests. Whether it survives a power loss or operating system crash before the next flush is not verified. Tool call records are already best effort: a lost capture is handled like the crash between the two transactions described in [Tool call capture ordering](ARCHITECTURE.md#tool-call-capture-ordering).
- How we check it: a test spy shows the capture batch commits without a flush while checkpoint and writes saves still flush; tool call nodes, outcomes and edges are present after `close()` and reopening the file; and we re-run the same 150-step measurement and publish the new figure as measured.
- Measured result (2026-11-17, same 150-step tool-heavy run, file database, default flush, two runs): the average of the last 20 saves fell from about 1.53 s to about 0.94 s (total about 97 s to about 59 s), against about 0.8 s with capture disabled. Before ADR-0011 the same run averaged about 1.7 s. These are local measurements and vary from run to run.

### Alternatives considered

1. **Put capture in the same transaction as the checkpoint:** Rejected by ADR-0005 and AC-47; a capture error would roll back the checkpoint.
2. **Keep flushing the capture transaction:** Rejected; it nearly doubles save time late in tool-heavy runs, as measured above.
3. **Recommend `wal_enabled=False`:** Rejected; it also drops the flush for checkpoints, which are not best effort.
4. **Flush on a timer in the background:** Rejected; it adds a second source of database work and a shutdown race for little gain over flushing with the next save.

### Consequences

- **Positive:** A save with new tool activity pays for one flush instead of two, so capture no longer doubles save time.
- **Positive:** The interface change is one keyword with a safe default; no stored data changes.
- **Negative:** After the last save of a run, its tool call data is not flushed until `close()` or another write. A program that never closes the saver relies on the operating system to write it out, which held for a process kill in our tests but is unverified for power loss.
- **Negative:** Save time still grows with file size, because the checkpoint's own flush remains; this decision removes one flush, not the underlying cost.
- **Negative:** Another process reading the file between the capture commit and the next flush depends on relate-db making committed but unflushed data visible; we have not verified that.

---

## ADR-0015: Read-Only Opening for Inspection Commands

Date: 2026-11-18  
Status: Accepted

### Context

The repository commits a demo store, `examples/data/demo.rl`, so people can explore it with `relate-dag` right after cloning (AC-69 to AC-75). Every `relate-dag` command opened the file the same way a running agent does: a writable connection, the schema setup statements (`CREATE CONSTRAINT ... IF NOT EXISTS`, `CREATE INDEX ... IF NOT EXISTS`), and a flush on close. We measured that opening and closing a saver on a file without running any query rewrites about 8 KB of it (byte offsets 16 to 110591); the size stays the same and no sidecar files remain. So every inspection command left the committed store modified in `git status`, and AC-75 failed. It also means the claim in [ADR-0004](#adr-0004-in-memory-graph-cloning-for-zero-side-effect-time-travel-sandboxing) that the source file stays bitwise untouched did not hold for `relate-dag replay`, because the command opened the source writable before forking.

relate-db offers `relate.connect(path, read_only=True)`, which loads the store once and refuses writes and checkpoints. We checked it on relate-db 1.0.2: a `MATCH` query works, a `CREATE` raises `OperationalError`, the file's sha256 is unchanged after close, no sidecar files appear, and a missing store is not created.

The inspection code only reads. The tree view, loop detection, root-cause tracing and the fork all go through `RelateThreadProxy.execute_read`; the fork writes only into its own new `:memory:` saver, including the optional `[:FORKED_TO]` branch edge. No inspection path writes to the source today, so a read-only open needs no change to their queries.

### Decision

This amends [ADR-0002](#adr-0002-dedicated-actor-proxy-for-thread-affinity-and-single-writer-concurrency) (the proxy gains a read-only mode) and [ADR-0004](#adr-0004-in-memory-graph-cloning-for-zero-side-effect-time-travel-sandboxing) (the replay source is opened read-only, so its untouched-source consequence now holds). Neither decision is reversed.

1. **Public interface.** `RelateSaver` and `RelateThreadProxy` each gain a keyword-only argument `read_only: bool = False`. The default keeps today's writable behavior for every library user. A new exception, `RelateReadOnlyError`, subclass of `RelateCheckpointError`, is exported from the package.
2. **Opening.** A read-only proxy opens the file with `relate.connect(database=path, read_only=True)` on its single worker thread, as before. It does not create parent directories, and it never runs schema setup, whatever `auto_init_schema` says. `wal_enabled` has no effect in read-only mode.
3. **Never flushing.** A read-only proxy never flushes: not after a statement, not on `flush()` (which returns without doing anything), and not on `close()`, which only closes the connection.
4. **Refusing writes, loudly.** On a read-only proxy, `execute_write`, `execute`, `execute_batch` and `transaction` raise `RelateReadOnlyError` in the caller's thread before anything is queued. Write-flagged items that reach the queue another way (the dict and tuple message forms) are failed with the same error and never reach the driver. The driver's own refusal stays as a last line of defense. On a read-only saver, `put()` and `put_writes()` raise `RelateReadOnlyError` before serializing anything, and `aput()` and `aput_writes()` raise the same error from the `await`. Nothing is dropped silently. `get_tuple`, `list`, `aget_tuple`, `alist`, `detect_loops`, `trace_root_cause` and `fork_in_memory` work as before. `RelateSaver.flush()` does nothing, and `close()` closes without flushing.
5. **Invalid combinations fail at construction with `ValueError`.** These are `read_only=True` with `":memory:"` (an empty in-memory store has nothing to inspect) and `read_only=True` with a passed proxy that is writable. A saver built on a passed proxy takes its read-only state from that proxy. When a `connection_factory` is given, it is still called as `factory(db_path)` and must open the store read-only itself. The proxy and saver still refuse writes, but the byte-identical guarantee covers only the default driver.
6. **Stores without the current schema.** A read-only open neither adds nor upgrades constraints or indexes. All inspection queries are plain `MATCH` reads, which work without them; the engine scans instead of using an index. A store written before `:ToolCall` nodes existed returns no tool calls, so forensics reports no tool failure, as it would after a writable open. To add the current schema, open the store writable once (any `RelateSaver(path)`). A missing file raises `RelateConnectionError` from the constructor and is not created. The CLI keeps its existing "Database file not found" check before opening.
7. **Forks.** `fork_in_memory` keeps reading the source through `execute_read` only, so it works unchanged on a read-only source. The fork is always a writable `:memory:` saver and never inherits `read_only`.
8. **CLI.** `relate-dag tree`, `relate-dag forensics` (the combined summary), `forensics loops`, `forensics trace-error` and the source side of `relate-dag replay` construct `RelateSaver(db, read_only=True)`. They pass that saver to `DAGVisualizer`, `RelateForensics` and `TimeTravelEngine`, whose constructors are unchanged, and close it themselves. When `--db` is `:memory:`, they keep a writable saver, because an empty in-memory store has nothing to protect. No `relate-dag` command writes to the store.
9. **Dependency floor.** The declared requirement becomes `relate-db>=1.0.2`, the version on which we verified `read_only`.
10. **How we check it.** For each CLI command above, a test builds a closed file store, records its sha256 and the directory listing, runs the command, and requires both to be unchanged. Saver and proxy tests require each refused method to raise `RelateReadOnlyError` and the file hash to be unchanged afterwards. A fork test opens the source read-only and compares the source file's sha256 from before the open to after the close, so the open and the close are covered too, not only the fork.

### Alternatives considered

1. **A separate read-only reader class:** Rejected. `DAGVisualizer`, `RelateForensics` and the fork all take a `RelateSaver`. A second type would need either a shared protocol across three modules or a copy of the read queries, which is a larger change for the same guarantee.
2. **Copy the store to a temporary file and open the copy:** Rejected for the same reasons as in ADR-0004: input/output proportional to file size, and temporary files left behind if the process is killed.
3. **Keep the writable open but skip schema setup and the close flush:** Rejected. A writable open and close changes bytes even with no queries, and we could not show which step writes. Only the engine's read-only mode is verified to leave the file identical.
4. **Open read-only by default whenever the file already exists:** Rejected. It would break every program that reopens an existing store to continue a run.
5. **Tell evaluators to copy the store before exploring it:** Rejected by the project owner. The committed store should stay clean without extra steps.

### Consequences

- **Positive:** Inspection commands leave the store byte-identical with no sidecar files, so a committed store stays unmodified, and `relate-dag replay` now meets ADR-0004's untouched-source claim.
- **Positive:** The library default is unchanged. The change is one keyword, one exception type and a higher dependency floor.
- **Positive:** Writing through a read-only saver fails with a typed error instead of being lost or reaching the file.
- **Negative:** A read-only saver sees the store as it was when opened ("load once"). Runs saved later by another process appear only after reopening, and how a read-only open behaves while another process holds the file open for writing is not verified.
- **Negative:** Per relate-db's description, the read-only open loads the store once at open time. We have not measured the time or memory this takes on large stores.
- **Negative:** Inspection never adds missing indexes, so queries against an old store scan instead of using an index until the store is opened writable once.
- **Negative:** A custom `connection_factory` gets no read-only signal through its call. Callers who inject a driver must open it read-only themselves to keep the file unchanged.
