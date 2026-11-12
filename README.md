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
├── examples/
│   └── research_agent.py       # Multi-agent research pipeline demo
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
