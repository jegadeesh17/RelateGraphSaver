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
