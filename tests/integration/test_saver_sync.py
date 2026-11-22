"""Integration tests for synchronous RelateSaver get_tuple and list namespace semantics.

Verifies AC-173:
- Given a config without checkpoint_ns (or None), get_tuple() defaults checkpoint_ns to ""
  (matching LangGraph SqliteSaver behavior), resolving root namespace checkpoints and
  ignoring subgraph checkpoints.
- Given a list() call without checkpoint_ns, results remain unfiltered by namespace.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint

from langgraph_checkpoint_relate.saver import RelateSaver


def _make_checkpoint(checkpoint_id: str, value: str = "val") -> Checkpoint:
    base: dict[str, Any] = dict(empty_checkpoint())
    base["id"] = checkpoint_id
    base["channel_values"] = {"data": value}
    base["channel_versions"] = {"data": "1.0"}
    return cast(Checkpoint, base)


def test_get_tuple_omitted_checkpoint_ns_defaults_to_root_ac_173() -> None:
    """AC-173: get_tuple with omitted checkpoint_ns resolves root namespace ("") checkpoint tuple.

    Does not return a subgraph namespace checkpoint even if the subgraph checkpoint was
    created later or has a higher step.
    """
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "test-thread-ac173"

        # 1. Put a root namespace checkpoint ("") at step 1
        cfg_root: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        saver.put(cfg_root, _make_checkpoint("cp-root-01", "root_step_1"), {"step": 1}, {})

        # 2. Put a subgraph checkpoint with higher step / created later
        cfg_subgraph: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "subgraph_node",
            }
        }
        saver.put(cfg_subgraph, _make_checkpoint("cp-subgraph-01", "subgraph_step_2"), {"step": 2}, {})

        # 3. Call get_tuple with omitted checkpoint_ns
        cfg_omitted: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        tuple_omitted = saver.get_tuple(cfg_omitted)
        assert tuple_omitted is not None
        assert tuple_omitted.config["configurable"]["checkpoint_id"] == "cp-root-01"
        assert tuple_omitted.config["configurable"]["checkpoint_ns"] == ""
        assert tuple_omitted.checkpoint["channel_values"]["data"] == "root_step_1"

        # 4. Call get_tuple with checkpoint_ns explicitly set to None
        cfg_none: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": None,
            }
        }
        tuple_none = saver.get_tuple(cfg_none)
        assert tuple_none is not None
        assert tuple_none.config["configurable"]["checkpoint_id"] == "cp-root-01"
        assert tuple_none.config["configurable"]["checkpoint_ns"] == ""
        assert tuple_none.checkpoint["channel_values"]["data"] == "root_step_1"

        # 5. Targeted get_tuple with checkpoint_id pointing to subgraph checkpoint
        # but omitted checkpoint_ns: must return None because it searches in "" namespace
        cfg_targeted_subgraph_no_ns: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "cp-subgraph-01",
            }
        }
        assert saver.get_tuple(cfg_targeted_subgraph_no_ns) is None

        # 6. Targeted get_tuple with checkpoint_id pointing to root checkpoint
        # with omitted checkpoint_ns: resolves successfully
        cfg_targeted_root_no_ns: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "cp-root-01",
            }
        }
        tuple_targeted_root = saver.get_tuple(cfg_targeted_root_no_ns)
        assert tuple_targeted_root is not None
        assert tuple_targeted_root.config["configurable"]["checkpoint_id"] == "cp-root-01"
        assert tuple_targeted_root.config["configurable"]["checkpoint_ns"] == ""

        # 7. Querying explicit subgraph namespace resolves the subgraph checkpoint
        tuple_subgraph = saver.get_tuple(cfg_subgraph)
        assert tuple_subgraph is not None
        assert tuple_subgraph.config["configurable"]["checkpoint_id"] == "cp-subgraph-01"
        assert tuple_subgraph.config["configurable"]["checkpoint_ns"] == "subgraph_node"


def test_list_omitted_checkpoint_ns_returns_all_namespaces_ac_173() -> None:
    """AC-173: list() called without checkpoint_ns yields checkpoints across all namespaces."""
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "test-thread-list-ac173"

        cfg_root: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        cfg_sub: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": "sub_agent"}
        }

        saver.put(cfg_root, _make_checkpoint("cp-root-1", "val1"), {"step": 1}, {})
        saver.put(cfg_sub, _make_checkpoint("cp-sub-1", "val2"), {"step": 2}, {})

        # list without checkpoint_ns returns both checkpoints
        cfg_no_ns: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        results = list(saver.list(cfg_no_ns))
        assert len(results) == 2
        result_ids = {t.checkpoint["id"] for t in results}
        assert result_ids == {"cp-root-1", "cp-sub-1"}

        # list with explicit checkpoint_ns filter only returns the matching namespace
        results_sub = list(saver.list(cfg_sub))
        assert len(results_sub) == 1
        assert results_sub[0].checkpoint["id"] == "cp-sub-1"


def test_get_tuple_invalid_input_boundaries_ac_173() -> None:
    """AC-173: Invalid or empty configuration returns None at boundary without error."""
    with RelateSaver(db_path=":memory:") as saver:
        assert saver.get_tuple({}) is None
        assert saver.get_tuple({"configurable": {}}) is None
        assert saver.get_tuple({"configurable": {"thread_id": ""}}) is None
        assert saver.get_tuple({"configurable": {"thread_id": None}}) is None
        assert saver.get_tuple({"configurable": {"thread_id": 12345}}) is None
        assert saver.get_tuple({"configurable": {"thread_id": "nonexistent_thread"}}) is None
