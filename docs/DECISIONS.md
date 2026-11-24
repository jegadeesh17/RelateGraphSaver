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
| [ADR-0016](#adr-0016-tree-status-depends-on-whether-a-checkpoint-has-a-next-step) | Tree Status Depends on Whether a Checkpoint Has a Next Step | Accepted | 2026-11-18 |
| [ADR-0017](#adr-0017-replay-runs-user-graph-code-only-against-an-in-memory-sandbox) | Replay Runs User Graph Code Only Against an In-Memory Sandbox | Accepted | 2026-11-18 |
| [ADR-0018](#adr-0018-argument-provenance-by-string-containment-earliest-write-wins) | Argument Provenance by String Containment, Earliest Write Wins | Accepted | 2026-11-18 |
| [ADR-0019](#adr-0019-the-demo-store-holds-a-real-api-thread-and-its-build-needs-network-access) | The Demo Store Holds a Real-API Thread and Its Build Needs Network Access | Accepted | 2026-11-18 |
| [ADR-0020](#adr-0020-relate-db-version-policy-decided-by-the-test-result) | relate-db Version Policy Decided by the Test Result | Accepted | 2026-11-19 |
| [ADR-0021](#adr-0021-tool-call-root-cause-anchors-on-the-failure-checkpoint) | Tool-Call Root Cause Anchors on the Failure Checkpoint | Accepted | 2026-11-19 |
| [ADR-0022](#adr-0022-replay-input-errors-and-message-edit-safety) | Replay Input Errors and Message-Edit Safety | Accepted | 2026-11-20 |
| [ADR-0023](#adr-0023-replay-diff-compares-full-message-content) | Replay Diff Compares Full Message Content | Accepted | 2026-11-20 |
| [ADR-0024](#adr-0024-replay-diff-compares-channel-values-not-repr) | Replay Diff Compares Channel Values, Not Repr | Accepted | 2026-11-20 |
| [ADR-0025](#adr-0025-typed-channel-keys-write-level-origin-nodes-and-an-attribute-tag-for-update_state-errors) | Typed Channel Keys, Write-Level Origin Nodes and an Attribute Tag for update_state Errors | Accepted | 2026-11-21 |
| [ADR-0026](#adr-0026-latest-and-before-follow-checkpoint_id-order) | Latest and `before` Follow `checkpoint_id` Order | Accepted | 2026-11-22 |
| [ADR-0027](#adr-0027-flat-tree-layout-with-fork-indentation) | Flat Tree Layout with Fork Indentation | Accepted | 2026-11-22 |
| [ADR-0028](#adr-0028-relate_-environment-variable-configuration) | `RELATE_*` Environment Variable Configuration | Accepted | 2026-11-22 |
| [ADR-0029](#adr-0029-forensics-lineage-namespace-loop-detection-and-root-cause-priority) | Forensics Lineage, Namespace Loop Detection, and Root-Cause Priority | Accepted | 2026-11-22 |
| [ADR-0030](#adr-0030-the-failure-checkpoint-walk-follows-next-edges-only) | The Failure-Checkpoint Walk Follows `[:NEXT]` Edges Only | Accepted | 2026-11-23 |
| [ADR-0031](#adr-0031-writes-and-tool-calls-attach-only-to-a-checkpoint-of-their-own-thread) | Writes and Tool Calls Attach Only to a Checkpoint of Their Own Thread | Accepted | 2026-11-23 |
| [ADR-0032](#adr-0032-the-failure-checkpoint-walk-may-cross-one-forked_to-edge-after-a-put_writes-link) | The Failure-Checkpoint Walk May Cross One `[:FORKED_TO]` Edge After a `put_writes()` Link | Accepted | 2026-11-23 |
| [ADR-0033](#adr-0033-relate_-variables-are-parsed-only-when-used-none-means-unset-and-a-timeout-reports-whether-the-command-started) | RELATE_* variables are parsed only when used, None means unset, and a timeout reports whether the command started | Accepted | 2026-11-23 |
| [ADR-0034](#adr-0034-fork-reads-fail-loudly-a-committed-transaction-is-never-rolled-back-and-an-empty-relate_db_path-is-rejected) | Fork Reads Fail Loudly, a Committed Transaction Is Never Rolled Back, and an Empty RELATE_DB_PATH Is Rejected | Accepted | 2026-11-24 |

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

- **Positive:** True zero-side-effect time-travel sandboxing. The primary `.rl` file remains bitwise untouched (true only when the source is opened read-only; see ADR-0015).
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
Amended by: [ADR-0026](#adr-0026-latest-and-before-follow-checkpoint_id-order)  
Amended: 2026-11-24

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

`tests/unit/__init__.py` and `tests/integration/__init__.py` now exist and are required: `test_proxy.py` and `test_visualizer.py` exist in both folders, and pytest's default import mode needs those folders to be packages to keep same-named test modules apart. `explicit_package_bases` stays. The concern in alternative 1 still stands for the top-level `tests/` folder, which still has no `__init__.py`.

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

---

## ADR-0016: Tree Status Depends on Whether a Checkpoint Has a Next Step

Date: 2026-11-18  
Status: Accepted

### Context

`relate-dag tree` tagged every checkpoint that had a stored write as `[PENDING]`. LangGraph stores each task's writes on the checkpoint the task ran from, so almost every completed step has writes, and a finished run showed nearly all of its steps as pending. An evaluator reading the tree could not tell finished steps from steps that were really waiting.

### Decision

The status of a checkpoint is the first of these that applies: `[ERROR]` (detected as before), `[INTERRUPT]`, `[OK]` if the checkpoint has a `[:NEXT]` child, `[PENDING]` if it has stored writes, otherwise `[OK]`. Only `[:NEXT]` children count as a next step; a checkpoint whose only child is a `[:FORKED_TO]` branch is still a leaf of its own run. Writes under a checkpoint with a next step are labelled `write:`; writes on a leaf keep the label `pending write:`. Details: [tree status rule](ARCHITECTURE.md#tree-status-rule).

### Alternatives considered

1. **Keep the old rule:** Rejected; it reports finished work as pending.
2. **Ask LangGraph which tasks are still pending (`get_state(...).tasks`):** Rejected; the tree has no graph to ask, only the stored data.
3. **Count a `[:FORKED_TO]` child as a next step:** Rejected; a branch does not finish the original step, and the original run's last step would then look done when it was not.
4. **Add a `[TOOL ERROR]` tag for steps with a failing tool call:** Left out by the project owner; `forensics` reports tool failures.

### Consequences

- **Positive:** A finished run reads as finished; `[PENDING]` now means a last step that still holds writes, such as one waiting at the end of a run that stopped early.
- **Negative:** A step that stopped with unapplied writes but already has a next checkpoint shows `[OK]`; the stored writes are still listed under it.
- **Negative:** Output of `tree` changes for every stored run, so saved sample outputs must be refreshed.

---

## ADR-0017: Replay Runs User Graph Code Only Against an In-Memory Sandbox

Date: 2026-11-18  
Status: Accepted

### Context

`relate-dag replay` only cloned history into an in-memory store and printed a summary; nothing ran, so the command did not answer "what would have happened if this input had been different?". Re-running needs the user's graph, because the stored data holds state but not the code of the nodes. Running user code next to a user's real store risks writing speculative steps into it. [ADR-0004](#adr-0004-in-memory-graph-cloning-for-zero-side-effect-time-travel-sandboxing) and [ADR-0015](#adr-0015-read-only-opening-for-inspection-commands) already give us a read-only source and an in-memory fork.

### Decision

This builds on ADR-0004 and ADR-0015 and supersedes no earlier decision; no earlier ADR said that replay never runs code.

1. Replay is a library function, `replay_from_checkpoint`, with a CLI front end, `relate-dag replay --graph SPEC`. The user names the graph as `path/to/file.py:attr` or `package.module:attr`; the attribute is a factory taking `checkpointer=` or an uncompiled `StateGraph`. A file path is accepted because the examples folder is not a package and the installed `relate-dag` command does not put the working directory on the import path.
2. The graph is compiled with the fork's in-memory saver as its checkpointer. If the compiled graph's checkpointer is anything else, replay stops before any node runs. The source store is opened read-only and only read; the fork is always closed, on success and on error.
3. Edits are applied with LangGraph's own `update_state`. A message edit reuses the original message's id, so LangGraph replaces the message instead of appending a second copy.
4. The replayed run is compared with the recorded run (the latest leaf after the fork point). Tool calls are matched by tool name and position, because call ids are random in every run; messages by index and their first 120 characters; other channels by their printed value. Ids and timestamps are never compared.
5. Loading a bad graph spec raises `GraphSpecError`, a `ValueError` whose message contains the spec; the CLI prints it as an `Error:` line.

### Alternatives considered

1. **Replay in the CLI without user graph code:** Rejected; the stored data cannot tell the CLI what the nodes do.
2. **Run the user's graph against the real store, on a new branch:** Rejected for the reasons in ADR-0004; speculative runs would stay in the file.
3. **Run the replay in a subprocess or container:** Rejected; much more machinery, and it still could not stop the user's own tools from calling networks or files.
4. **Match tool calls by call id:** Rejected; ids differ in every run, so every call would appear changed.
5. **Import only `package.module:attr` specs:** Rejected; the examples cannot be imported that way from the installed command.

### Consequences

- **Positive:** `relate-dag replay --graph ... --set-message 0=...` shows, for example, `fetch_source #1  error -> ok` with the argument before and after, and the source file stays byte-identical.
- **Negative:** Replay imports and runs the user's code. Only the checkpointer is sandboxed; tools still have their real side effects (a replay of the dependency demo calls pypi.org). We document this and do not try to contain it.
- **Negative:** A graph whose factory ignores `checkpointer=` cannot be replayed; it is refused rather than allowed to write elsewhere.
- **Negative:** The diff is shallow by design: result text after 120 characters, message ids and tool calls trimmed from history are not compared.

---

## ADR-0018: Argument Provenance by String Containment, Earliest Write Wins

Date: 2026-11-18  
Status: Accepted  
Amended by: [ADR-0025](#adr-0025-typed-channel-keys-write-level-origin-nodes-and-an-attribute-tag-for-update_state-errors)

### Context

Root-cause reports named the failing tool call but not where its bad argument came from. In a typical agent the bad value is produced one or more steps earlier (a planner invents a package name; the model copies it into a tool call). The stored writes contain those earlier values, but LangGraph keeps no record of which value flowed into which argument.

### Decision

1. For each string argument of the failing call with at least 3 characters, we look for the earliest write on the same `[:NEXT]` path, up to the failure, whose value contains that string, case-sensitively. We search plain strings, message text, tool-call arguments and nested containers. The earliest matching write wins.
2. The node that made the write is the one whose `versions_seen` entry changes between the write's checkpoint and the next one.
3. Hops are counted from the checkpoint where the tool actually ran and failed, not from the earlier step that requested the call.
4. The result fills optional fields on `RootCauseReport` (`origin_argument`, `origin_node`, `origin_step`, `origin_hops`) and feeds a store-wide `failure_summary()`. Details: [argument provenance](ARCHITECTURE.md#9-argument-provenance-and-tool-failure-summary-forensicspy).

### Alternatives considered

1. **Exact data-flow tracking:** Rejected; it would need instrumenting every node, which a checkpointer cannot do.
2. **Ask a language model to explain the origin:** Rejected; needs a key, costs money and is not repeatable.
3. **Case-insensitive or fuzzy matching:** Rejected; in the demo the user's request contains `RelateGraphSaver` while the bad argument is `relategraphsaver`, and the planner, not the user, introduced the bad name.
4. **Latest matching write instead of earliest:** Rejected; it usually finds the model's own tool call, one step before the failure, which says nothing new.

### Consequences

- **Positive:** `trace-error` can say "url came from node planner (step N, H hops before failure)" from stored data alone, offline.
- **Negative:** It is a heuristic. A short or common string can match an unrelated earlier write; values transformed on the way (case changes, joined strings) are not found; non-string arguments are not traced.
- **Negative:** Several nodes running in one step are reported together, because the stored writes do not name their node.
- **Negative:** `failure_summary()` traces every failed call, so its cost grows with the number of failures in the store.

---

## ADR-0019: The Demo Store Holds a Real-API Thread and Its Build Needs Network Access

Date: 2026-11-18  
Status: Accepted

### Context

The committed demo store was built offline from scripted agents, so every failure in it was staged. Evaluators look for the graph explaining a real failure. We want a failure that a real public service produced, while keeping the explore commands key-free and offline for the person cloning the repository.

### Decision

1. `examples/build_demo_store.py` adds a fourth thread, `demo-real-api`, produced by `examples/dependency_agent.py`. Its deterministic planner adds the package name `relategraphsaver`, which does not exist on PyPI; the lookup against the key-free PyPI JSON API returns a real 404.
2. Building the store therefore needs network access to pypi.org. There is no offline or staged-data option for this thread. If the real lookups do not produce exactly the expected outcomes (a 404 for `relategraphsaver`, results for every other package), the build exits 1 and leaves the existing store unchanged.
3. Reading the store stays offline and key-free, as before.
4. The store is rebuilt once for this change; every rebuild writes new checkpoint ids, so the README and its tests are updated in the same commit.
5. An optional run with a real language model, if the project owner approves its model and cost, goes into a separate file, `examples/data/demo-llm.rl`, so the main store never needs a key.

This replaces the offline-build assumption for the demo store in the specification (AC-73, open question 8); no earlier ADR recorded it, so none is superseded.

### Alternatives considered

1. **Keep only staged failures:** Rejected; the demo would not show a failure the system did not stage.
2. **An offline flag that fakes the PyPI answers:** Rejected; staged data could return unnoticed.
3. **Record the language-model run in the main store:** Rejected; the main store must rebuild without a key.
4. **Use an API that needs a key:** Rejected; evaluators could not re-run the build.

### Consequences

- **Positive:** The demo shows a real 404 from a public service, traced back to the planner step that caused it.
- **Negative:** Rebuilding needs pypi.org, and package versions and summaries drift, so tests compare outcomes, not result text.
- **Negative:** The demo depends on `relategraphsaver` staying absent from PyPI; if someone publishes that name, the build fails its own check and the demo must change.

---

## ADR-0020: relate-db Version Policy Decided by the Test Result

Date: 2026-11-19  
Status: Accepted

### Context

The package required `relate-db>=1.0.2`, the version on which we verified read-only opening ([ADR-0015](#adr-0015-read-only-opening-for-inspection-commands)). relate-db 1.1.0 is available. The README's benchmark numbers and the sentence that the flush cost grows with file size were measured on 1.0. Declaring a floor we have not tested, or keeping an open upper range we have not tested, can both mislead users.

### Decision

We ran the full test suite and `python benchmarks/bench_checkpointers.py --checkpoints 1000` in a fresh environment with relate-db 1.1.0, and the result decides:

1. **If everything passes:** the requirement becomes `relate-db>=1.1.0`, the README benchmark table is replaced with numbers measured on 1.1.0, and the statement about flush cost is re-checked against them.
2. **If anything fails:** the requirement becomes `relate-db>=1.0.2,<1.1`, and the README numbers and statements stay as they are.

Outcome (2026-11-19): the requirement is now `relate-db>=1.1.0`. In a fresh virtual environment with relate-db 1.1.0 and only the `dev` extra installed, 604 tests passed and the benchmark ran to completion. The one failure was the mypy gate: strict mypy could not resolve the optional `langchain_openai` import used by `examples/openrouter_agent.py`. That defect does not involve relate-db and would fail on 1.0.2 as well. It was fixed by a mypy override that ignores the missing `langchain_openai` module, so a `dev`-only install passes without the `llm` extra, and a slow test now proves it in a fresh environment. With that fix the full suite passes on 1.1.0. The README benchmark block and table were replaced with the output of that run. The flush cost still grows with the size of the database file on 1.1.0: in a separate timing run of 1000 puts, the median put took about four times longer for the last 100 checkpoints (about 109 ms) than for the first 100 (about 26 ms), and 1.0.2 behaved the same (about 114 ms and 26 ms). The README statement was reworded to say this on 1.1.0.

Either outcome keeps the README's version statements matching the requirement (AC-143).

### Alternatives considered

1. **Keep `>=1.0.2` with no upper bound and no test:** Rejected; users would get 1.1.0 by default without our having run anything on it.
2. **Require 1.1.0 without testing:** Rejected for the same reason, and it would invalidate the published numbers.
3. **Pin one exact version:** Rejected; it blocks bug-fix releases for users of a library.

### Consequences

- **Positive:** The declared range is always one we have run the suite on.
- **Negative:** Users on relate-db 1.0.x can no longer install this package; the floor moved to 1.1.0 because that is the version the suite and benchmark ran on.
- **Negative:** Future relate-db releases are not covered by this decision; whoever raises the floor or adds an upper bound must update this ADR, the requirement and the README together.

## ADR-0021: Tool-Call Root Cause Anchors on the Failure Checkpoint

Date: 2026-11-19  
Status: Accepted

### Context

The saver links each tool call to one checkpoint with a `(:Checkpoint)-[:INVOKED]->(:ToolCall)` edge, attached by whichever save sees the call first: `put_writes()` for the step that requested the call (the edge lands on that step) or `put()` for the next checkpoint (the edge lands on the next step). LangGraph runs these two saves on different threads, so the same agent run can store the edge on either step. `trace_root_cause()` reported the `[:INVOKED]` checkpoint as `error_checkpoint_id` but started the argument-origin trace from the checkpoint where the tool failed. On runs where the edge landed on the requesting step, `origin_hops` no longer matched what `trace_argument_origin(thread_id, report.error_checkpoint_id, args)` returned, so the consistency promised by AC-119 held on some runs and not on others.

### Decision

When the error comes from a failing tool call, `trace_root_cause()` reports the failure checkpoint as `error_checkpoint_id`: the first checkpoint at or after the `[:INVOKED]` checkpoint on the `[:NEXT]` path whose writes hold the `ToolMessage` for the call, falling back to the `[:INVOKED]` checkpoint when none is found. That checkpoint is the same under both edge layouts, and the origin trace starts from it, so `origin_hops` and the reported error checkpoint always agree (AC-119, AC-129). Errors found through an `__error__` write or checkpoint metadata keep the checkpoint they were found on. The saver's edge placement is unchanged.

### Alternatives considered

1. **Make the saver's edge placement deterministic:** Rejected; it would change which checkpoint already stored runs attach to, which is an expensive-to-change contract, and it needs an ordering between the two saves that LangGraph does not guarantee.
2. **Start the origin trace from the `[:INVOKED]` checkpoint instead:** Rejected; hops would then count from the step before the tool ran on some runs, which breaks AC-129.

### Consequences

- **Positive:** `error_checkpoint_id`, `origin_hops` and `trace_argument_origin()` agree on every run, whichever step the edge landed on.
- **Positive:** No stored data changes; the committed demo store produces the same README output as before.
- **Negative:** `causal_path_length` for tool-call failures still depends on where the `[:INVOKED]` edge attached, so it can differ by one step between two runs of the same agent.
- **Negative:** Edge placement still decides which tool calls forensics and `failure_summary()` find, so it stays an expensive-to-change contract.

---

## ADR-0022: Replay Input Errors and Message-Edit Safety

Date: 2026-11-20  
Status: Accepted

### Context

[ADR-0017](#adr-0017-replay-runs-user-graph-code-only-against-an-in-memory-sandbox) lets replay run a user's graph with edits. Three gaps showed up. Invalid replay inputs were documented as `ValueError`, but the CLI could not tell them apart from an exception raised inside user graph code, so a crashing node and a bad argument printed the same way. A factory that does not accept `checkpointer=` failed with a bare `TypeError` instead of the spec error users already get for other bad graphs. And a `set_messages` edit relies on LangGraph's `add_messages` reducer to replace a message by id; on a graph whose `messages` channel has another reducer, or none, the same edit would append a duplicate or overwrite the whole history. We also needed to know whether a replay with no edit really re-runs the step at the fork point, because `fork_in_memory` copies that checkpoint's stored writes into the sandbox.

### Decision

1. Invalid replay inputs raise `ReplayInputError`, which subclasses both `RelateCheckpointError` and `ValueError`, so the documented `ValueError` contract still holds (AC-151, AC-155). The CLI prints its message as a plain `Error:` line; any other exception from user graph code prints `Error: replay failed: <Type>: <message>`.
2. Before calling a factory, replay checks that its signature accepts `checkpointer=` (`inspect.signature(factory).bind(checkpointer=...)`). Only a failed bind raises `GraphSpecError` naming `checkpointer`, before any node runs (AC-155). A `TypeError` raised inside the factory body is the user's own error: it propagates and the CLI prints `Error: replay failed: TypeError: <message>`. A factory whose signature cannot be inspected is called directly.
3. Before applying `set_messages`, replay checks that the compiled graph's `messages` channel is a `BinaryOperatorAggregate` reduced by LangGraph's `add_messages`. A `functools.partial` wrapper, such as the one `add_messages(format=...)` returns, is unwrapped first; the function then matches by identity, or by module `langgraph.graph.message` and name for LangGraph's wrapped reducer. A function with the same name from another module does not match, because it would append instead of replacing by id. Otherwise it raises `ReplayInputError` mentioning `add_messages`, and the graph is not run (AC-156).
4. No code change for no-edit replays: we verified on LangGraph 1.2.22 that `invoke` from the fork checkpoint re-executes the fork step even though its writes were copied, and `updated_checkpoint_id` equals `fork_checkpoint_id`. AC-154 guards this behaviour so a LangGraph upgrade that changes it fails the suite.

### Alternatives considered

1. **Send the full edited message list instead of only the edited messages:** Rejected; under `add_messages` it duplicates every unedited message, and without `add_messages` it silently truncates or replaces history.
2. **Delete the fork checkpoint's writes in the sandbox before a no-edit replay:** Rejected; LangGraph already re-runs the step, so it adds code for no observed benefit.
3. **Use `update_state(..., as_node="__copy__")` to force a fresh step:** Rejected; it needs a LangGraph floor above our declared `>=0.2.0`.
4. **Keep plain `ValueError` for input errors:** Rejected; the CLI could not separate input errors from failures inside user code.

### Consequences

- **Positive:** Code that catches `ValueError` keeps working, and the CLI now says whether the user's input or the user's graph failed.
- **Positive:** A message edit can no longer silently corrupt the replayed history on a graph without `add_messages`.
- **Negative:** `ReplayInputError` is also a `RelateCheckpointError`, so handlers must catch it before that base class, or it is reported as a database error.
- **Negative:** Graphs with a custom message reducer cannot use `set_messages`; they must use `update`.
- **Negative:** The no-edit behaviour depends on LangGraph internals; AC-154 detects a change but does not prevent it.

---

## ADR-0023: Replay Diff Compares Full Message Content

Date: 2026-11-20  
Status: Accepted  
Amended by: [ADR-0024](#adr-0024-replay-diff-compares-channel-values-not-repr)

### Context

[ADR-0017](#adr-0017-replay-runs-user-graph-code-only-against-an-in-memory-sandbox) compared messages by index and their first 120 characters. Two replays whose messages differed only after character 120, or only in an AI message's tool-call names or arguments, were reported as identical, so the diff could hide the change the user was looking for.

### Decision

This replaces the message rule in decision 4 of ADR-0017; the rest of ADR-0017 stands. Messages are compared by index with `==`, never by `repr`: two messages are equal when they have the same type, the same content and the same sequence of tool-call `(name, arguments)` pairs (AC-159). String content is compared as is. List content is compared block by block with every `id` key removed at any nesting depth, because content blocks get fresh ids in every run. Dict key order and object reprs therefore never produce a change. Message ids and tool-call ids are never compared, because they are random in every run (AC-106); a tool argument that is itself named `id` is still compared. The text shown for a changed message stays the first 120 characters plus `...` when longer; the marker is never part of the comparison.

### Alternatives considered

1. **Keep the 120-character comparison:** Rejected; it hides late differences in long messages and every tool-call argument change.
2. **Compare whole messages with `==`:** Rejected; message and tool-call ids differ in every run, so every message would appear changed.
3. **Show the full text of changed messages:** Rejected; long messages make the CLI table unreadable.

### Consequences

- **Positive:** Any change in message content or requested tool calls appears in the diff.
- **Negative:** Replays that differ only late in long messages, such as a timestamp in model output, now show a change where they showed none before.
- **Negative:** Two messages can show the same truncated text and still be reported as `changed`; the user must inspect the stored state to see where they differ.

---

## ADR-0024: Replay Diff Compares Channel Values, Not Repr

Date: 2026-11-20  
Status: Accepted  
Amended by: [ADR-0025](#adr-0025-typed-channel-keys-write-level-origin-nodes-and-an-attribute-tag-for-update_state-errors)

### Context

AC-106 requires comparing run snapshots such that runs producing equivalent state report an empty diff. However, [ADR-0023](#adr-0023-replay-diff-compares-full-message-content) stated "Other channels compare `repr(value)`". In Python and LangGraph, non-message channels can contain nested dictionaries, lists, or custom payloads (including LangChain messages stored outside the main `messages` channel). Comparing string representations (`repr`) creates spurious diffs due to arbitrary dictionary key iteration order across runs or Python versions, or due to random message IDs embedded in messages placed in auxiliary channels.

Additionally, when replaying a graph with an invalid state update or an unrecognized writer node, LangGraph's `update_state` raises `InvalidUpdateError`. An identical `InvalidUpdateError` can also be raised during graph execution if a graph node itself calls `update_state` incorrectly or encounters an invalid update during `graph.invoke`. The CLI previously appended a hint (`Hint: pass --as-node with the node that should be treated as the writer.`, AC-116) for any `InvalidUpdateError`. When an internal node failed during graph execution, displaying the `--as-node` CLI hint was misleading because `--as-node` only applies to the initial replay state update.

### Decision

This amends ADR-0023's channel comparison rule and clarifies replay exception propagation:

1. **Value-based channel comparison via `_channel_key(value)`:** Non-message channels are compared by value representation using a recursive key mapper:
   - `BaseMessage` instances are normalized via `_message_key(value)` (matching type, content with nested `id` keys removed from list parts, and tool-call names and arguments), ensuring random message IDs do not trigger diffs.
   - `list` and `tuple` map to a `list` of recursively mapped channel keys.
   - `dict` maps to `{k: _channel_key(v)}` for all key-value pairs; because Python dictionary equality compares key-value pairs independently of insertion order, dictionary key order is ignored under `==`. Dict `id` keys outside messages are preserved (not stripped).
   - Primitive scalar types (`str`, `int`, `float`, `bool`, `None`) map to the values themselves.
   - Any other arbitrary object falls back to `repr(value)`.
2. **Channel diff rendering:** `_diff_channels` compares mapped channel keys using `==` to determine whether a change occurred. When a channel is added, removed, or changed, `before` and `after` strings in `ChannelChange` are rendered using `repr(value)` of the original channel value for clear diagnostic inspection.
3. **Tagging `update_state` exceptions:** In `replay_from_checkpoint`, when `graph.update_state` raises `InvalidUpdateError`, the exception is tagged using Python's `exc.add_note(...)` before re-raising. The exception type and message propagate unchanged (AC-101), preserving the public API contract for SDK callers. The CLI inspects the exception's notes to determine whether the failure originated directly from `update_state` (displaying the `--as-node` hint, AC-116) or occurred later during node execution in `graph.invoke` (printing `Error: replay failed: InvalidUpdateError: <message>` without the hint, AC-166).

### Alternatives considered

1. **Compare non-message channels with raw `==`:** Rejected; channels containing LangChain messages would compare random message IDs and report false diffs, and unhashable or custom objects without clean equality semantics could raise exceptions or behave inconsistently.
2. **Keep `repr(value)` for non-message channels:** Rejected; dictionary key order or embedded message instances produce false-positive changes between functionally identical runs.
3. **Strip `id` keys from all dictionaries in all channels:** Rejected; while message content blocks generate random `id` keys that must be ignored, user data dictionaries in arbitrary channels may treat `id` as meaningful domain data (such as database primary keys).
4. **Wrap `InvalidUpdateError` in a custom `ReplayStateUpdateError`:** Rejected; wrapping changes the exception hierarchy, breaking existing callers and acceptance criteria expecting `InvalidUpdateError` to propagate directly.
5. **String-match exception message in CLI to detect `update_state` failures:** Rejected; matching internal message strings is brittle against changes in LangGraph error formatting across minor versions.

### Consequences

- **Positive:** Replay diffs across runs with dictionary state or auxiliary message channels compare semantic value without spurious diffs from key order or message IDs (AC-106).
- **Positive:** Diagnostic output retains human-readable `repr(value)` in the before/after columns of `ChannelChange`.
- **Positive:** The CLI accurately distinguishes replay setup errors from graph execution errors without modifying LangGraph exception types or public API contracts (AC-166).
- **Negative:** Non-primitive objects not covered by `_channel_key` (e.g. custom classes) still fall back to `repr()`.
- **Negative:** Exception notes require Python 3.11+, which is already the minimum supported Python version for this project.

---

## ADR-0025: Typed Channel Keys, Write-Level Origin Nodes and an Attribute Tag for update_state Errors

Date: 2026-11-21  
Status: Accepted

### Context

Hardening of replay and forensics revealed three subtle edge cases:
1. Under ADR-0024, primitive scalar values mapped directly to themselves in `_channel_key`. In Python, values of different types can compare equal under `==` (for example, `0 == False`, `1 == True`, and `1 == 1.0`). When a replay execution changed a channel value across these types, `_diff_channels` reported no change. In addition, `float('nan') == float('nan')` evaluates to `False`, causing identical NaN values across runs to be reported as spurious channel changes.
2. In fan-out workflows where concurrent nodes run in parallel, attributing argument origins solely to `versions_seen` differences between checkpoints (ADR-0018) attributed the write to all nodes that updated versions in that step, even if only one branch wrote the argument payload.
3. Attaching diagnostic information to `update_state` exceptions via `add_note` (ADR-0024) caused notes to appear in tracebacks during standard application execution and testing, even though the annotation was intended only for CLI scoping.

### Decision

This amends ADR-0018 and ADR-0024:

1. **Typed channel keys in replay diffs:**
   - Primitive scalar values (`int`, `float`, `bool`, `str`, `None`) map to `(type(value).__name__, value)` tuples (applied recursively through lists and dicts).
   - Because `bool` is a subclass of `int` in Python, distinguishing by type name ensures `("bool", False) != ("int", 0)` and `("int", 1) != ("float", 1.0)`.
   - `float('nan')` is normalized to `("float", "nan")`, ensuring two NaN values compare equal under `==` and produce no false diff.
2. **Write-level origin node resolution:**
   - Origin nodes are determined by taking the `task_path` node of each matching write in the earliest matching checkpoint and intersecting them with the nodes whose `versions_seen` changed between that checkpoint and the next.
   - When `task_path` is empty or the intersection is empty, it falls back to the `versions_seen` nodes.
3. **Attribute tagging for `update_state` errors:**
   - `update_state` failures are tagged using a private attribute `_relate_update_state_error` instead of calling `add_note`.
   - The CLI and helpers check this attribute via `is_update_state_error` to determine whether to display the `--as-node` hint without polluting exception tracebacks with note text.

### Alternatives considered

1. **Raw repr comparison:** Rejected; Python dictionary key ordering differences would produce false diffs for nested dicts.
2. **math.isclose comparison:** Rejected; state diffs must be exact, and diffing should not treat distinct numbers as equivalent.
3. **UNION Cypher query:** Rejected; `UNION` support in relate-db Cypher engine is unverified; filtering and intersecting writes in Python keeps database queries simple and robust.
4. **add_note with readable sentence:** Rejected; exception notes are unconditionally printed in tracebacks and error reports.

### Consequences

- **Positive:** Type shifts between equal-under-`==` values (e.g. `0` vs `False`, `1` vs `1.0`) are reliably detected in channel diffs (AC-168).
- **Positive:** Identical `float('nan')` values across runs compare as equal without false diffs (AC-169).
- **Positive:** Argument provenance narrows down to the specific node that wrote the argument in concurrent fan-out steps (AC-172).
- **Positive:** `InvalidUpdateError` carries no added note in tracebacks while preserving CLI hint detection (AC-172).
- **Negative:** `_channel_key` produces slightly larger tuples in memory during replay diffing.

---

## ADR-0026: Latest and `before` Follow `checkpoint_id` Order

Date: 2026-11-22  
Status: Accepted  
Supersedes: Step-based ordering in [ADR-0007](#adr-0007-static-quality-gates-pass-through-configuration-and-behavior-neutral-fixes) and AC-06

### Context

`RelateSaver.get_tuple` (when retrieving the latest checkpoint for a thread without an explicit `checkpoint_id`) and `RelateSaver.list` (and pagination with `before`) originally sorted checkpoints by integer step number descending (`ORDER BY c.step DESC`).

LangGraph's reference checkpointer implementation (`SqliteSaver`) and the LangGraph runtime assume that checkpoint sorting and reverse-chronological listing use time-ordered UUIDv6 `checkpoint_id` strings (`ORDER BY c.checkpoint_id DESC`, `c.checkpoint_id < $before`). When workflows branch or when `update_state` is invoked on an earlier historical checkpoint, the forked state writes a new checkpoint that may share or have a lower `step` number than existing checkpoints on the main branch. Ordering by `step` causes `get_tuple()` to return a stale checkpoint from a different branch rather than the latest write, and causes `before` pagination to skip or mishandle forked checkpoints.

### Decision

We update checkpoint sorting and filtering across `saver.py` (`get_tuple`, `list`, and `before` filter) to order strictly by `c.checkpoint_id DESC` and filter with `c.checkpoint_id < $before`.

Checkpoint IDs emitted by LangGraph use UUIDv6 (`id.uuid6()`), which embeds a 60-bit timestamp in big-endian byte order, guaranteeing that lexicographical string comparison sorts in chronological creation order. This supersedes the previous step-based ordering in AC-06. Synthetic or hand-crafted checkpoint IDs in test suites that do not use UUIDv6 must sort chronologically in ASCII string order.

### Alternatives considered

1. **Composite ordering by `(step DESC, created_at DESC)`:** Rejected; `step` numbers diverge or collide across branches and forks, and ISO-8601 string timestamps can collide during high-frequency execution in tight loops.
2. **Retain step-based ordering with branch tracking:** Rejected; diverges from standard LangGraph `SqliteSaver` semantics and adds unnecessary query complexity when UUIDv6 already provides native chronological ordering.
3. **Cypher graph traversal following `NEXT` backwards:** Rejected; cannot resolve unlinked forks or disjoint branch heads without traversing multiple paths.

### Consequences

- **Positive:** Full parity with LangGraph runtime expectations and `SqliteSaver` semantics (AC-06, AC-175).
- **Positive:** Correctly resolves latest checkpoints and pagination for `update_state` forks and branching workflows.
- **Negative:** Hand-crafted synthetic test IDs in legacy unit tests must sort chronologically in ASCII order.

---

## ADR-0027: Flat Tree Layout with Fork Indentation

Date: 2026-11-22  
Status: Accepted

### Context

`DAGVisualizer.render_tree` and `relate-dag tree` originally traversed execution DAGs using recursive depth-first tree traversal, rendering each subsequent `[:NEXT]` child checkpoint at an increased indentation level.

In long-running agent workflows or benchmarking runs exceeding 1,000 to 1,500 checkpoints, recursive rendering encounters Python's default call stack recursion limit (`sys.getrecursionlimit()`), raising `RecursionError` and preventing trace visualization. Furthermore, linear execution steps connected by `[:NEXT]` do not represent hierarchical parent-child relationships; indenting each step produces severe rightward terminal drift and broken line wraps. Only branch points—specifically `[:FORKED_TO]` edges created during time travel or branching—represent true conceptual sub-trees.

### Decision

We re-architect `DAGVisualizer` to use an iterative sibling rendering layout:
1. Linear `[:NEXT]` children are rendered sequentially as siblings at the same depth and indentation level, constructed iteratively with explicit loops and no call-stack recursion.
2. Only `[:FORKED_TO]` relationships nest child branches into indented sub-trees.
3. README sample outputs and visualizer expectations (`tests/unit/test_readme.py`) are updated to reflect the flat, non-drifting tree layout.

### Alternatives considered

1. **Increase recursion limit via `sys.setrecursionlimit(10000)`:** Rejected; fragile workaround that risks hard segmentation faults / stack overflows in the C runtime and fails to resolve terminal line-wrapping from rightward drift.
2. **Paginate or truncate tree output at N steps:** Rejected; developers inspecting agent traces require full lineage visibility without arbitrary truncation.
3. **Graphviz or Mermaid export only:** Rejected; the terminal-native tree CLI (`relate-dag tree`) is a core feature for instant local developer inspection.

### Consequences

- **Positive:** Eliminates recursion limits, allowing `relate-dag tree` to render 1,500+ checkpoint runs without `RecursionError` (AC-185).
- **Positive:** Terminal output remains readable and clean without rightward drift for linear step sequences.
- **Negative:** Visualizer tests and README example outputs must match the updated sibling layout format.

---

## ADR-0028: `RELATE_*` Environment Variable Configuration

Date: 2026-11-22  
Status: Accepted

### Context

Configuration values for `RelateSaver` (such as the database path, WAL flush behavior, proxy timeout, and log level) were previously documented inconsistently (e.g., `"checkpoints.rl"` was mentioned as the default path in some documentation while code used `":memory:"`). In 12-factor application architectures, serverless workflows, and CLI environments, configuring checkpointer settings via standardized environment variables is essential.

Additionally, configuration priority must be explicit: constructor arguments must always take precedence over environment variables, and environment variables must take precedence over defaults. When an invalid environment variable value is supplied, silently falling back to a default can mask critical configuration defects.

### Decision

We define a strict three-tier configuration resolution order:
1. **Explicit constructor argument:** Used whenever provided (not `None` or sentinel).
2. **`RELATE_*` environment variable:**
   - `RELATE_DB_PATH` (str): target database file path or `":memory:"`.
   - `RELATE_WAL_ENABLED` (bool): parsed from `"1"`, `"true"`, `"yes"`, `"on"` / `"0"`, `"false"`, `"no"`, `"off"` (case-insensitive).
   - `RELATE_PROXY_TIMEOUT_SEC` (float): positive float numeric string.
   - `RELATE_LOG_LEVEL` (str): `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`.
3. **Default value:** `db_path=":memory:"`, `wal_enabled=True`, `default_timeout=30.0`, `log_level="INFO"`.

If an environment variable contains an unparseable or invalid value, initialization raises `ValueError` explicitly naming the offending variable (e.g. `ValueError("Invalid value for RELATE_PROXY_TIMEOUT_SEC: ...")`). All documentation and signatures are corrected to declare `":memory:"` as the default `db_path`.

### Alternatives considered

1. **Default `db_path` to `"checkpoints.rl"`:** Rejected; writing files to disk by default causes unexpected disk side-effects during testing and ephemeral runs; `":memory:"` is safe and zero-side-effect by default.
2. **Silent fallback on invalid environment variables:** Rejected; masking configuration errors leads to silent misbehavior in production (e.g., disabling WAL or misconfiguring timeouts).
3. **Adopting `pydantic-settings`:** Rejected; adds an unnecessary heavyweight external runtime dependency for four environment variables.

### Consequences

- **Positive:** Deterministic configuration resolution with transparent priority (AC-178).
- **Positive:** Fast, clear failure when environment variables contain typos or invalid values.
- **Positive:** Code and documentation agree that `":memory:"` is the default database path.

---

## ADR-0029: Forensics Lineage, Namespace Loop Detection, and Root-Cause Priority

Date: 2026-11-22  
Status: Accepted

### Context

Forensic analysis in `RelateForensics` (`trace_root_cause`, `detect_loops`, and argument provenance) presented subtle edge cases in complex multi-agent graphs:
1. Lineage traversals across checkpoints could bleed into sibling threads or cross-thread forks when relationship patterns were unconstrained by thread identity.
2. `detect_loops` constructed checkpoint sequences by sorting all thread checkpoints by `step` across all namespaces, interleaving subgraph steps and obscuring the actual chronological path.
3. Poison pill detection in `trace_root_cause` inspected serialized state payloads for substring matches of `"poison"` and `"fault"`. Legitimate user messages or prompts containing these words were falsely flagged as poisoned state failures.
4. On the `__error__` path (when a graph step writes an error to the `__error__` channel), antecedent tool calls far back in history could be blamed as root cause even when the failure was completely unrelated to the tool.

### Decision

We establish five forensic traversal and attribution rules:
1. **Thread-constrained lineage traversal:** Lineage walks follow `[:NEXT|FORKED_TO*0..]` strictly restricted to checkpoints within the specified `thread_id` (AC-181).
2. **Namespace-isolated root-to-leaf loop detection:** `detect_loops` constructs execution sequences by walking `[:NEXT]` edges from root to leaf separately per `checkpoint_ns`, sorting candidate node names and preserving true path order. A `threshold < 2` raises `ValueError` (AC-182, AC-188).
3. **Channel-name only poison detection:** Poison analysis matches exclusively against the channel name (e.g., channel named `poison`); substring matching for `"poison"` or `"fault"` within serialized value text is removed (AC-183).
4. **Root-cause priority on `__error__` path:** When execution terminates along an `__error__` path, an antecedent tool-call error is accepted as root cause only when `distance_to_error <= 1`. Causal queries break ties with `ORDER BY distance_to_error, c.checkpoint_id, tool_call_id`, and tree writes order by `w.task_id, w.idx`.
5. **Read-only by default:** `RelateForensics` and `DAGVisualizer` open database stores with `read_only=True` by default (AC-187).

### Alternatives considered

1. **Global step sorting for cycle detection:** Rejected; subgraph and concurrent fan-out executions interleave step indices, producing spurious cycle warnings or missing repeating loops.
2. **Retain regex heuristics on serialized state text:** Rejected; parsing serialized binary blobs for English words causes false-positive failure diagnoses on benign user payloads.
3. **Unbounded tool error attribution:** Rejected; attributing a crash to a tool that failed 10 steps earlier obscures the immediate error cause on the `__error__` path.

### Consequences

- **Positive:** Accurate loop and cycle detection across nested subgraphs and branched threads.
- **Positive:** Eliminates false-positive poison reports on legitimate conversational content.
- **Positive:** Deterministic, localized root-cause attribution prioritizing proximate tool failures (`distance_to_error <= 1`).
- **Positive:** Diagnostic tools preserve database file immutability by defaulting to `read_only=True`.

---

## ADR-0030: The Failure-Checkpoint Walk Follows `[:NEXT]` Edges Only

Date: 2026-11-23  
Status: Accepted  
Amended by: [ADR-0032](#adr-0032-the-failure-checkpoint-walk-may-cross-one-forked_to-edge-after-a-put_writes-link)

### Context

When a tool call fails, `trace_root_cause()` reports the *failure checkpoint*: the first checkpoint, from the one linked to the call by `[:INVOKED]` onward, whose writes hold the tool's result message ([ADR-0021](#adr-0021-tool-call-root-cause-anchors-on-the-failure-checkpoint)). ADR-0021 and AC-129 define that walk along `[:NEXT]` edges, the main line of the run. The implementation reused the lineage walk from [ADR-0029](#adr-0029-forensics-lineage-namespace-loop-detection-and-root-cause-priority), which also follows `[:FORKED_TO]` edges. When someone resumes a thread from an earlier checkpoint, the tool runs again on the branch with the same call id, so the main line and the branch can both hold a result for that call at the same distance. The walk then picked whichever row the database returned first, because it ordered only by distance. The same stored data could give a different `error_checkpoint_id` and `origin_hops` from one run of the command to the next, and could name a checkpoint on a branch that did not produce the recorded outcome.

### Decision

- The failure-checkpoint walk starts at the `[:INVOKED]` checkpoint (distance 0) and follows outgoing `[:NEXT]` edges between checkpoints of the same thread only. It never follows `[:FORKED_TO]`.
- Checkpoints are visited in `(distance, checkpoint_id)` order. The first one whose writes (skipping channels that start with `__` and empty blobs) hold a `ToolMessage` for the call is the failure checkpoint. If none does, or the `[:INVOKED]` checkpoint is not in the thread, the `[:INVOKED]` checkpoint is used, as ADR-0021 already says.
- AC-181's rule (follow `[:NEXT]` and `[:FORKED_TO]`, stay in the thread) applies to walks toward ancestors: the causal query and argument provenance. It does not apply to this forward walk.
- The thread's edges are read once per public call, with each edge's type, and shared by the ancestor walks and this forward walk. Nothing is cached between calls.
- For the record, the causal query's tie-break that ADR-0029 calls `tool_call_id` is the pair `direct_call_id, linked_call_id`: the call reached through the checkpoint's own `[:INVOKED]` edge, then one reached through a write.

### Alternatives considered

1. **Keep following `[:FORKED_TO]` and only add a tie-break on `checkpoint_id`:** Rejected; the answer would be repeatable but could still name a branch checkpoint, which contradicts ADR-0021 and AC-129.
2. **Walk the main line first and fall back to branches when it holds no result:** Rejected for now; it adds a second ordering rule for an uncommon case, and the existing fallback to the `[:INVOKED]` checkpoint already gives a stable answer.
3. **Record the behavior as a known limitation:** Rejected; it is a determinism defect in a public report, and the fix restores the documented contract without changing stored data.

### Consequences

- **Positive:** `trace_root_cause()`, `failure_summary()` and the `trace-error` command give the same answer on every run for the same file.
- **Positive:** Threads without branches, including every thread of the committed demo store, report exactly what they reported before.
- **Positive:** Reading edges once per call removes two of the three edge reads on long threads.
- **Negative:** When a tool's only result was recorded on a branch, the failure checkpoint is the `[:INVOKED]` checkpoint, so `origin_hops` counts from there rather than from the branch checkpoint where the tool ran.

---

## ADR-0031: Writes and Tool Calls Attach Only to a Checkpoint of Their Own Thread

Date: 2026-11-23  
Status: Accepted

### Context

`put_writes()` stores each write with the `thread_id`, `checkpoint_ns` and `checkpoint_id` from its config, then links it to a checkpoint with `[:HAS_WRITE]`, finding that checkpoint by `checkpoint_id` alone. A later `put()` links any earlier write that has the new checkpoint's id, again by id alone. Tool-call capture in `put_writes()` finds the checkpoint for the `[:INVOKED]` edge the same way.

Checkpoint ids are unique across the whole file, and that is exactly why this goes wrong: there is only one checkpoint with that id, and the id-only match finds it whichever thread owns it. LangGraph's own run loop always pairs a thread with that thread's checkpoint, so normal runs are not affected. But `put_writes()` and `aput_writes()` are public methods, and a script, test helper or migration tool that calls them with thread B in the config and a checkpoint id from thread A attaches the write, and any tool call in it, to thread A. `get_tuple()` for thread A then returns thread B's write as a pending write, so resuming thread A would apply data from another conversation, and forensics on thread A would count it. LangGraph's SQLite saver keys writes by thread, namespace and checkpoint id together, so such a write is simply never seen by thread A.

### Decision

- `put_writes()` creates `[:HAS_WRITE]` only to a checkpoint whose `thread_id`, `checkpoint_ns` and `checkpoint_id` all equal the write's: `OPTIONAL MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id}) WHERE c.thread_id = $thread_id AND c.checkpoint_ns = $checkpoint_ns`. The id stays in the node map so the lookup uses the unique constraint.
- `put()` links earlier writes only when they belong to the same thread and namespace: `OPTIONAL MATCH (w:Write {checkpoint_id: $checkpoint_id}) WHERE w.thread_id = $thread_id AND w.checkpoint_ns = $checkpoint_ns`. Writes that legitimately arrive before their checkpoint are still linked.
- The tool-call invocation statement matches its checkpoint with `WHERE c.thread_id = $thread_id AND c.checkpoint_ns = $checkpoint_ns`, taking both values from the config passed to `put()` or `put_writes()`. A mismatched call is then treated like any call whose checkpoint is not stored yet: it is not linked and is sent again by the next save that names a stored checkpoint of its own thread and namespace.
- Note: the namespace guard applies to `[:INVOKED]` as well as `[:HAS_WRITE]`, as AC-190 requires; the rule of one `[:INVOKED]` edge per thread and call is unchanged and stays per thread.
- `get_tuple()` returns as pending writes only linked writes whose `thread_id` and `checkpoint_ns` equal the checkpoint's, so files that already contain such a cross-thread link stop leaking it into a resume.
- A mismatched write is stored, not rejected. Writes may arrive before their checkpoint is stored, so the saver cannot tell a mistake from a race without an extra read on every save, and the other LangGraph savers do not reject it either.

### Alternatives considered

1. **Leave it, because checkpoint ids are unique:** Rejected; uniqueness is what makes the id-only match find the other thread's checkpoint. It does not make the match safe.
2. **Raise an error when the named checkpoint belongs to another thread:** Rejected; it needs an extra read on every `put_writes()`, and no other LangGraph saver raises here.
3. **Migrate existing files to remove cross-thread links:** Rejected; such links come only from mismatched direct calls, and the `get_tuple()` filter already stops them from affecting a resume.

### Consequences

- **Positive:** One thread's resume state can no longer contain another thread's writes, matching the other LangGraph savers.
- **Positive:** No change for writes and tool calls saved by LangGraph's own run loop, so stored runs and the demo store report the same results.
- **Negative:** A cross-thread link already stored in an older file is still visible to forensics and is still copied into in-memory forks; only `get_tuple()` filters it.
- **Negative:** A mismatched write is kept in the file with no link, where nothing reads it.

---

## ADR-0032: The Failure-Checkpoint Walk May Cross One `[:FORKED_TO]` Edge After a `put_writes()` Link

Date: 2026-11-23  
Status: Accepted

### Context

[ADR-0021](#adr-0021-tool-call-root-cause-anchors-on-the-failure-checkpoint) requires that forensics report the same failure checkpoint whichever save linked the tool call: `put_writes()` puts the `[:INVOKED]` edge on the checkpoint P that the requesting step ran from, and `put()` puts it on the next checkpoint Y. LangGraph runs these two saves on different threads, so either can win. [ADR-0030](#adr-0030-the-failure-checkpoint-walk-follows-next-edges-only) then limited the forward walk to `[:NEXT]` edges so that the same file always gives the same answer.

That left one case where the answer still depends on the race. When someone resumes a thread from an earlier checkpoint P that already has a `[:NEXT]` child, `put()` links the new checkpoint as `P -[:FORKED_TO]-> Y`, and the tool's result is written on Y. If `put()` won, the edge is on Y and the walk reports Y. If `put_writes()` won, the edge is on P, the `[:NEXT]`-only walk goes down the old main line, finds no result for this call and falls back to P. We reproduced both layouts by calling the saver directly in each order. Two recordings of the same run then report a different `error_checkpoint_id` and `origin_hops` (AC-119).

Which save placed the edge can be read from data the walk already loads. `put_writes()` stores a step's writes on the checkpoint the step ran from and links the call to that same checkpoint, so when it placed the edge, the `[:INVOKED]` checkpoint's own writes hold the `AIMessage` that requested the call. `put()` links the call to the checkpoint it creates. That checkpoint's channel values hold the `AIMessage`, but its writes come from the step that runs next (the tool and its `ToolMessage`), so they do not hold it.

### Options Considered

1. **Allow one `[:FORKED_TO]` hop as the first step, only when `put_writes()` placed the edge.** The branch the run actually created becomes reachable in exactly the layout that hid it. The other layout is unchanged.
2. **Walk the main line first and fall back to branches when it holds no result** (alternative 2 in ADR-0030). This reopens a decision already made. When `put()` placed the edge it can name a branch that a later, unrelated resume created, and it adds a second ordering rule.
3. **Record a known limitation.** No code change, but forensics stays race-dependent in exactly the resume (time-travel) case that the inspection tools are meant for.

How to tell that `put_writes()` placed the edge:

- a. **Look for the requesting `AIMessage` in the `[:INVOKED]` checkpoint's own writes.** The walk already reads and deserializes these rows at distance 0.
- b. **Check that the `[:INVOKED]` checkpoint's channel values do not hold the `AIMessage`.** This needs an extra read of `checkpoint_blob` and deserializing the whole state.
- c. **Store which save placed the edge as a property on `[:INVOKED]`.** This changes the write path and stored data (a tool call recording rule, see "Expensive to change" in the architecture), and it does nothing for files already written.

### User's Chosen Architecture

On 2026-11-23 the project owner chose option 1 with check (a). The owner's reasoning: it keeps the same-file determinism of ADR-0030, and it is the only option that restores race independence across recordings (ADR-0021).

- **Start.** The walk starts at the `[:INVOKED]` checkpoint S at distance 0. If S is not a checkpoint of the thread, S is the answer, as before.
- **Gate: "`put_writes()` placed the edge".** The gate is true when one of S's own writes (skipping channels that start with `__`, and empty or undeserializable blobs) is an `AIMessage`, or a list or tuple holding one, whose `tool_calls` has an entry with `id` equal to the failing call's `call_id`. These are the same message shapes that tool-call capture reads from written values (ADR-0005). Only S's writes are checked. S's `checkpoint_blob` is never read for this.
- **Candidates.** The `[:NEXT]` chain from S, with distances as in ADR-0030. When the gate is true, every `[:FORKED_TO]` child F of S whose two ends are both checkpoints of the thread is also a candidate at distance 1, followed by the `[:NEXT]` chain from F at distance 1 plus its `[:NEXT]` hops. No other `[:FORKED_TO]` edge is ever followed. When S has several `[:FORKED_TO]` children, all of them sit at distance 1 alongside S's `[:NEXT]` child. A checkpoint reachable by more than one route keeps its smallest distance. Today every checkpoint has one parent, so this cannot happen.
- **Selection** stays as in ADR-0030. Candidates are visited in `(distance, checkpoint_id)` order. The first whose writes hold a `ToolMessage` for the call is the failure checkpoint. If none does, the answer is S.
- **Gate false** covers an edge placed by `put()`, or a call attached to a later checkpoint because its named checkpoint was not stored yet (ADR-0011). The walk is then exactly the ADR-0030 walk.
- **Reads.** The thread's edges, with their types, are still read once per public call and shared by every walk in that call, and nothing is cached between calls. The branch candidates are computed in Python from those same edges. One writes read covers the `[:NEXT]` candidates and the branch candidates together. Distance-0 rows sort first, so the gate is settled before any distance-1 row is considered. Branch rows are skipped when the gate is false.
- **Defense in depth.** The failure-checkpoint writes read, the causal query and the argument-origin writes read now also match the checkpoint's thread. The id stays in the node map, so the lookup uses the unique constraint, and the thread is checked in a `WHERE` clause (`... {checkpoint_id: pr.id}) WHERE c.thread_id = $thread_id`) rather than in the node map, which would probe the thread index on every row. Checkpoint ids are unique across the file today, and [ADR-0031](#adr-0031-writes-and-tool-calls-attach-only-to-a-checkpoint-of-their-own-thread) relies on that, so results do not change. The filter stops a lineage row from ever matching another thread's checkpoint if that uniqueness is ever lost, for example after a hand-edited or merged file.
- **AC-189** still describes the walk when the gate is false. When the gate is true and both S's `[:NEXT]` child and a branch hold the result, the lower `checkpoint_id` wins. When only a branch holds it, that branch is reported instead of S.

### Explicit Trade-offs Accepted

- **Positive:** The resume case reports Y under both layouts. If `put()` won, Y is found at distance 0. If `put_writes()` won, the gate is true and Y is found at distance 1. `error_checkpoint_id` and `origin_hops` therefore no longer depend on the race (ADR-0021, AC-119).
- **Positive:** One file still gives one answer on every run, because the ordering is unchanged.
- **Positive:** Threads whose `[:INVOKED]` checkpoint has no `[:FORKED_TO]` child, including every thread of the committed demo store, report exactly what they reported before.
- **Negative:** If the same call id holds a result on both the main line and a branch at distance 1 (a deterministic model, or a resume that re-runs the same call), the lower `checkpoint_id` wins, not necessarily the branch the recorded run produced. LangGraph checkpoint ids are time-ordered, so in practice the older main-line child wins.
- **Negative:** The gate trusts message content. A node that writes back an `AIMessage` it did not produce, for example by returning the whole message history, makes the gate true for an edge that `put()` placed. This matters only when S's own writes hold no `ToolMessage` for the call and S has a branch.
- **Negative:** When S has branches and the gate is false, the branch writes are still read in the single writes read and then discarded.
- **Negative:** The thread filter applies to checkpoints, not writes. A cross-thread `[:HAS_WRITE]` link already stored in an older file (ADR-0031) is still read.
- **Negative:** ADR-0030's remaining gap gets narrower but does not close. When the gate is false and the only result is on a branch, S is still reported.

---

## ADR-0033: RELATE_* variables are parsed only when used, None means unset, and a timeout reports whether the command started

Date: 2026-11-23
Status: Accepted

This amends [ADR-0028](#adr-0028-relate_-environment-variable-configuration), AC-178 (configuration) and AC-179 (proxy timeout).

### Context

Four defects came out of a review of how `RelateSaver` and `RelateThreadProxy` handle configuration and timeouts:

- The README says a `RELATE_*` variable is read only when it is used. The saver parsed all four on every construction, so an invalid `RELATE_LOG_LEVEL` broke a saver whose caller had passed `log_level` explicitly.
- `RelateSaver(db_path=None)` created a database file named `None` instead of using `RELATE_DB_PATH` or the default.
- Every `RelateSaver` set the level of the `langgraph_checkpoint_relate` logger, even when nothing was configured, which overwrote the level the application had chosen for that logger.
- AC-179 cancelled the future when the caller's timeout expired. Cancelling cannot stop a command the worker has already started, so that command ran and committed while the caller was told it had failed.

### Decision

- **`None` means unset.** For `db_path`, `wal_enabled`, `default_timeout` and `log_level`, an argument that is `None` counts as not passed, and resolution continues with the `RELATE_*` variable and then the default (AC-194).
- **Parse when used.** A `RELATE_*` variable is parsed only when its argument is not passed. An invalid value raises `ValueError` naming the variable only in that case (AC-194). The message texts are unchanged.
- **Logger level only when configured.** The package logger's level is set only when `log_level` or `RELATE_LOG_LEVEL` is given. Otherwise it is left unchanged. `RELATE_LOG_LEVEL` accepts `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`.
- **A timeout reports whether the command started.** When a caller's timeout expires for a single execute, a batch or a flush, the proxy tries to cancel the command's future. If that succeeds, the worker has not started the command and never will, and `RelateConcurrencyError` says it was not executed. If it fails, the worker has started the command and it runs to completion, and `RelateConcurrencyError` says the operation timed out while running and may have committed (AC-195).

### Options Considered

1. **Keep eager parsing.** Rejected: it contradicts the README environment-variables section.
2. **Keep `default_timeout=None` meaning "no timeout".** Rejected: nothing in `src`, tests, examples or benchmarks uses it, and the architecture already says `None` falls back to the environment variable or default.
3. **Abort a running write on timeout.** Rejected: the single writer thread cannot interrupt `conn.execute` safely.

### Explicit Trade-offs Accepted

- **Positive:** An explicit argument fully replaces its environment variable, so an unrelated bad variable cannot break a caller that did not need it.
- **Positive:** The application's own level for the package logger survives creating a saver.
- **Positive:** A caller that sees "not executed" can retry safely. "May have committed" tells it to check the store before retrying.
- **Negative:** `default_timeout=None` now means the environment or default timeout (30.0 s), not no timeout. Code that passed `None` to wait forever now times out.
- **Negative:** An invalid `RELATE_*` variable that the caller's arguments cover is no longer reported at all.
- **Negative:** After a timeout on a running command, the caller cannot tell whether it committed without reading the store.

---

## ADR-0034: Fork Reads Fail Loudly, a Committed Transaction Is Never Rolled Back, and an Empty RELATE_DB_PATH Is Rejected

Date: 2026-11-24  
Status: Accepted

This amends [ADR-0010](#adr-0010-in-memory-forks-copy-tool-call-data), [ADR-0014](#adr-0014-tool-call-capture-commits-without-its-own-flush) and [ADR-0033](#adr-0033-relate_-variables-are-parsed-only-when-used-none-means-unset-and-a-timeout-reports-whether-the-command-started).

### Context

- `fork_in_memory()` caught `RelateCheckpointError` on three source reads (the parent lineage lookup, the `[:NEXT]`/`[:FORKED_TO]` edge query and the thread metadata query) and carried on, so a fork could silently miss ancestors, edges or metadata.
- When the WAL flush failed after a successful commit, the proxy's single-command and batch handlers called `rollback()` on a transaction that had already committed.
- `RELATE_DB_PATH=""` was accepted, although `.env.example` says empty values are rejected and the other three `RELATE_*` variables already reject them.

### Decision

1. **Fork reads fail loudly.** A `RelateCheckpointError` from any source read in `fork_in_memory()` propagates, and no fork is returned (AC-197). An empty result is not an error: missing edges are still rebuilt from `parent_checkpoint_id`, and a missing thread `created_at` still uses the current time.
2. **A committed transaction is never rolled back.** When the flush fails after a commit, the proxy does not call `rollback()`, and the committed data stays readable. The flush error still reaches the caller, because durability was not confirmed (AC-198).
3. **An empty `RELATE_DB_PATH` is rejected.** When `db_path` is not passed and `RELATE_DB_PATH` is empty or only whitespace, `RelateSaver()` raises `ValueError` naming `RELATE_DB_PATH` (AC-199).

### Alternatives considered

1. **Keep the fallbacks.** Rejected: the fork would silently differ from the source.
2. **Wrap the flush error in a new error type.** Rejected: it changes the exception hierarchy.
3. **Reject an empty explicit `db_path` too.** Rejected for now: it is outside this change.

### Consequences

- **Positive:** A fork either matches its source lineage or fails, so replay and forensics on a fork never run on partial data.
- **Positive:** A flush failure no longer issues a rollback against a transaction that is already committed, so the proxy's behavior no longer depends on how the driver treats that rollback.
- **Positive:** The documented rule for empty `RELATE_*` values now holds for all four variables.
- **Negative:** A transient read error during a fork now fails the fork instead of returning a partial one.
- **Negative:** After a flush error the caller cannot tell from the error alone whether the data will survive a crash; it is committed but not confirmed durable.
- **Negative:** An environment that set `RELATE_DB_PATH=""` to get the `:memory:` default now fails at construction.
- **Negative:** When the flush in `put()` fails, the checkpoint is stored but `put()` raises before tool-call capture; retrying `put()` with the same `checkpoint_id` fails the `checkpoint_id` uniqueness constraint. The tool calls are captured by the next save that sees the same messages ([ADR-0021](#adr-0021-tool-call-root-cause-anchors-on-the-failure-checkpoint)).
