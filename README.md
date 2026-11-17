# RelateGraphSaver

Embedded graph-native checkpoint and time-travel persistence engine for LangGraph, powered by Relate by Neo4j.

## Features

- **Embedded Single-File WAL Persistence:** Persists checkpoints, intermediate channel updates, and execution metadata into a zero-configuration `.rl` database file with atomic Write-Ahead Logging.
- **Native Cypher DAG Modeling:** Represents agent execution state as a property graph with `:Thread`, `:Checkpoint`, `:Write`, and `:ToolCall` nodes linked by causal `:NEXT`, `:FORKED_TO`, and `:HAS_WRITE` relationships.
- **Sandboxed In-Memory Time Travel:** Clones historical execution subgraphs up to any targeted checkpoint into an isolated `:memory:` database with `fork_in_memory()`, allowing speculative replays without altering production files.
- **Terminal Execution Trees:** Inspect multi-agent branching structures, pending writes, and error states directly in the terminal using `relate-dag tree`.
- **Cypher Forensic Diagnostics:** Detect cyclical state oscillations (`relate-dag forensics loops`) and trace backward causal paths from failures to offending antecedent writes or tool calls (`relate-dag forensics trace-error`).
- **Human-in-the-Loop Interruption:** Native persistence for LangGraph `interrupt()` checkpoints and pending writes, supporting seamless execution pauses and resumptions.
- **Thread-Affinity Actor Concurrency:** Dedicated actor thread proxy (`RelateThreadProxy`) eliminates thread-affinity violations and database file lock collisions across concurrent worker threads.

## Quick start

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### Installation

Install from source:

```bash
git clone https://github.com/jegadeesh17/RelateGraphSaver.git
cd RelateGraphSaver
pip install -e ".[dev]"
```

Or install via `pip`:

```bash
pip install langgraph-checkpoint-relate
```

## Usage

Integrate `RelateSaver` directly into any compiled LangGraph workflow:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph_checkpoint_relate import RelateSaver


class AgentState(TypedDict):
    query: str
    response: str


def plan_step(state: AgentState) -> dict[str, str]:
    return {"response": f"Processed: {state['query']}"}


# Build a standard StateGraph
builder = StateGraph(AgentState)
builder.add_node("planner", plan_step)
builder.add_edge(START, "planner")
builder.add_edge("planner", END)

# Persist to local Relate graph database
with RelateSaver("checkpoints.rl") as checkpointer:
    app = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session-42"}}

    # Execute workflow
    result = app.invoke({"query": "Graph-native persistence"}, config=config)
    print("Execution output:", result)

    # Retrieve checkpoint history
    history = list(checkpointer.list(config))
    print(f"Recorded {len(history)} checkpoints in thread.")
```

### Command-line interface

Inspect and debug execution graphs using the bundled `relate-dag` CLI:

```bash
# Render execution DAG as an interactive Unicode terminal tree
relate-dag tree session-42 --db checkpoints.rl

# Fork history from a historical checkpoint into an isolated in-memory sandbox
relate-dag replay session-42 <checkpoint_id> --db checkpoints.rl

# Detect state oscillations and infinite loops
relate-dag forensics loops session-42 --threshold 2 --db checkpoints.rl

# Trace root cause of an execution failure
relate-dag forensics trace-error session-42 --db checkpoints.rl
```

### Running the multi-agent demo

Run the complete multi-agent research demo (Planner, Researcher, Human Review interrupt, and Critic):

```bash
python examples/research_agent.py --topic "Autonomous Agent Architecture"
```

### Running the tool-calling demo

`examples/tool_agent.py` needs no API key. A deterministic agent asks a `ToolNode` to run one tool, and the tool fails on purpose. RelateSaver records the call as a `:ToolCall` node, so the root cause can be traced afterwards from the database file alone (ADR-0005):

```bash
# Run the agent; it prints the thread ID and a root-cause analysis
python examples/tool_agent.py --db tool_agent.rl --thread-id demo-1

# Trace the failure again later, from the file only
relate-dag forensics trace-error demo-1 --db tool_agent.rl
```

Pass `--no-fail` to let the tool succeed.

### Async usage

`RelateSaver` implements the async checkpointer methods (`aget_tuple`, `alist`, `aput`, `aput_writes`), so compiled graphs work with `ainvoke` and `astream`. The synchronous methods run on worker threads, off the event loop, and writes still go through the single writer thread (ADR-0006).

## How we use Relate, and why

Relate is an embedded graph database from Neo4j. We use these capabilities, and nothing beyond them:

- **Embedded single-file database.** Each saver opens one `.rl` file in-process. Checkpoints need no server, container or credentials, and the whole execution history of an agent is one file you can copy, attach to a bug report or delete (ADR-0001).
- **Write-ahead logging (WAL).** Every checkpoint and every writes save is committed through the WAL and flushed, so a committed checkpoint survives a process restart and an interrupted write is rolled back on reopen. An agent that resumes a thread after a crash needs exactly that guarantee. Tool-call records are committed without their own flush and become durable with the next flushed save or when `close()` is called; they are best effort, and a lost record does not affect the checkpoint (ADR-0014).
- **Native Cypher queries.** Agent execution is a graph, so we query it as one. `relate-dag forensics trace-error` walks `[:NEXT*0..]` backwards from a failed checkpoint across `[:HAS_WRITE]` and `[:INVOKED]` edges to the write or tool call that caused it. This is a single path query, where a relational schema needs a recursive join (ADR-0003).
- **Connected schema.** `:Thread`, `:Checkpoint`, `:Write` and `:ToolCall` nodes are linked by `NEXT` (the causal order), `FORKED_TO` (a branch created by replay or a fork), `HAS_WRITE` (pending and channel writes) and `INVOKED` (a tool call made at a checkpoint). Forks and tool calls are edges in the data, so the terminal tree and the forensics reports read them directly instead of reconstructing them.
- **Uniqueness constraints and indexes.** Constraints on thread, checkpoint, write and tool-call identifiers make writes idempotent and reject duplicates; indexes on thread, namespace, step and state hash keep lookups by thread and by checkpoint from scanning the graph.
- **`:memory:` databases.** `fork_in_memory()` copies the history up to a chosen checkpoint into a throwaway `:memory:` database, so you can replay and experiment without touching the production file and without cleanup afterwards (ADR-0004).

### Limits

RelateSaver is a single-process engine. Use one `RelateSaver` per process and per database file. Inside that process it is safe to share across threads and async tasks, because all writes are serialized through one writer thread (ADR-0002), but a second process opening the same file is not supported. For multi-worker or multi-pod deployments, use PostgresSaver (`langgraph-checkpoint-postgres`) instead.

## Benchmark

`benchmarks/bench_checkpointers.py` writes N checkpoints with RelateSaver and with LangGraph's SqliteSaver, then measures put throughput, `get_tuple` latency, on-disk size and a branch ancestry query (ADR-0008). Reproduce it from the project virtual environment:

```bash
python benchmarks/bench_checkpointers.py --checkpoints 1000
```

These are the values that command printed, unedited. They are one run on one machine; numbers vary from run to run on the same machine, so expect different ones on yours.

- Checkpoints (N): 1000
- Payload bytes (B): 256
- OS: Windows-11-10.0.26340-SP0
- CPU: AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD
- Python: 3.13.2
- relate-db: 1.0.2
- langgraph-checkpoint: 4.3.2
- langgraph-checkpoint-sqlite: 3.1.3
- SQLite: 3.45.3

| Metric | RelateSaver (flush per write, default) | RelateSaver (wal_enabled=False) | SqliteSaver |
| :--- | ---: | ---: | ---: |
| Put throughput (checkpoints/s) | 14.5 | 309.1 | 1204.1 |
| Get p50 (ms) | 1.208 | 1.442 | 0.036 |
| Get p95 (ms) | 1.773 | 1.792 | 0.040 |
| On-disk size (KiB) | 1648.0 | 11532.0 | 724.0 |
| Ancestry query, median of 5 (ms) | 6.210 | 6.200 | 0.920 |

What the numbers say:

- SqliteSaver is faster than RelateSaver on every speed metric here: writes, point reads (`get_tuple`) and the ancestry query. Its file is also smaller. Each saver's queries are written in the best form we know for its engine (ADR-0012): the ancestry query is anchored on the branch tip in Relate and walks parents by primary key in SQLite, and both return the same count.
- With the default settings RelateSaver flushes after every write, which is what makes a committed checkpoint durable. In relate-db 1.0 the cost of that flush grows with the size of the database file, so save latency grows as a run gets longer. We keep it as the default because it is the safe choice (ADR-0001).
- `wal_enabled=False` skips the per-write flush and keeps put latency flat, but its file is larger, and we have not verified that it survives power loss without the flush. It is not recommended for data you cannot lose.
- Absolute numbers will differ on your machine. Run the command above to get your own.

We choose Relate for what the connected graph lets you ask and do, not for speed. Forks, tool calls and causal order are edges, so `relate-dag` can trace a failure to the write or tool call behind it with one path query; `fork_in_memory()` gives you a sandboxed copy of the history to replay without touching the production file; and the whole history stays in a single file with no server. If write throughput or multi-process access matters more than those, use SqliteSaver or PostgresSaver.

## Running tests

Run the automated test suite using `pytest`:

```bash
# Run the complete test suite (unit and integration tests)
python -m pytest -q

# Run fast test suite
python -m pytest -q -m "not slow"

# Run a specific test file
python -m pytest -q tests/unit/test_saver.py
```

## Project structure

```text
RelateGraphSaver/
├── src/
│   └── langgraph_checkpoint_relate/
│       ├── __init__.py         # Package entrypoint and public exports
│       ├── cli.py              # relate-dag CLI (tree, replay, forensics)
│       ├── errors.py           # Exception hierarchy and taxonomy
│       ├── forensics.py        # RelateForensics (loop detection & root cause)
│       ├── models.py           # DTO dataclasses (WriteNode, Forensic reports)
│       ├── proxy.py            # RelateThreadProxy actor queue
│       ├── saver.py            # RelateSaver implementing BaseCheckpointSaver
│       ├── schema.py           # Cypher DDL schema constraints and indexes
│       ├── serde.py            # Typed state serialization and hashing
│       ├── time_travel.py      # TimeTravelEngine in-memory sandbox
│       └── visualizer.py       # DAGVisualizer terminal tree renderer
├── tests/
│   ├── conftest.py             # Shared pytest fixtures and configurations
│   ├── fakes/                  # In-memory mock Relate database drivers
│   ├── integration/            # End-to-end multi-agent and CLI integration tests
│   └── unit/                   # Component-level unit test suites
├── benchmarks/
│   └── bench_checkpointers.py  # RelateSaver vs SqliteSaver benchmark
├── examples/
│   ├── research_agent.py       # Multi-agent research pipeline demo
│   └── tool_agent.py           # Key-free tool-calling demo with a failing tool
├── docs/                       # Living architectural documentation
├── pyproject.toml              # Build specifications, dependencies, and metadata
├── CHANGELOG.md                # Release history and unreleased changes
└── README.md                   # Project documentation and quickstart
```

## Documentation

- [docs/README.md](docs/README.md): Index of living technical documentation.
- [docs/SPEC.md](docs/SPEC.md): User journeys, behavioral requirements, and Given/When/Then acceptance criteria.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Technical architecture, Cypher graph schema, seams, and concurrency patterns.
- [docs/DECISIONS.md](docs/DECISIONS.md): Architecture Decision Records (ADRs) capturing rationale and trade-offs.
- [CHANGELOG.md](CHANGELOG.md): Complete release history.

## How this was built

Designed and built by jegadeesh17 with [SoftwareEngineeringAgents](https://github.com/jegadeesh17/SoftwareEngineeringAgents), a multi-agent workflow for Claude Code.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
