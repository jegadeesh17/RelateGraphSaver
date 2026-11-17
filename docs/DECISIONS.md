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
