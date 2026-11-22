"""Integration tests for synchronous RelateSaver get_tuple and list namespace semantics.

Verifies AC-173:
- Given a config without checkpoint_ns (or None), get_tuple() defaults checkpoint_ns to ""
  (matching LangGraph SqliteSaver behavior), resolving root namespace checkpoints and
  ignoring subgraph checkpoints.
- Given a list() call without checkpoint_ns, results remain unfiltered by namespace.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_checkpoint_relate.errors import (
    RelateCheckpointError,
    RelateConcurrencyError,
    RelateConnectionError,
)
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.serde import RelateStateSerializer


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
        saver.put(
            cfg_subgraph, _make_checkpoint("cp-subgraph-01", "subgraph_step_2"), {"step": 2}, {}
        )

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

        cfg_root: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
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


def test_get_tuple_latest_ordered_by_checkpoint_id_descending_ac_175() -> None:
    """AC-175, ADR-0026: get_tuple() resolves latest checkpoint by checkpoint_id descending.

    Proves that even when an earlier checkpoint has a higher step number due to a workflow
    fork or update_state call, the checkpoint with the highest checkpoint_id (UUIDv6)
    is returned as the latest checkpoint.
    """
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "test-thread-ac175-latest"
        cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Main branch executes up to step 5
        cid_main = "1ef82a3c-b034-6c2e-8000-000000000001"
        saver.put(cfg, _make_checkpoint(cid_main, "main_step_5"), {"step": 5}, {})

        # update_state forks from an earlier state, created later with lower step (step 2)
        cid_fork = "1ef82a3c-b034-6c2e-8000-000000000002"
        saver.put(cfg, _make_checkpoint(cid_fork, "fork_step_2"), {"step": 2}, {})

        # get_tuple without checkpoint_id resolves the latest write by checkpoint_id descending
        latest = saver.get_tuple(cfg)
        assert latest is not None
        assert latest.config["configurable"]["checkpoint_id"] == cid_fork
        assert latest.checkpoint["channel_values"]["data"] == "fork_step_2"
        assert latest.metadata.get("step") == 2


def test_list_ordered_by_checkpoint_id_descending_ac_175() -> None:
    """AC-175, ADR-0026: list() yields checkpoints ordered reverse-chronologically by checkpoint_id."""
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "test-thread-ac175-list"
        cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Checkpoints with divergent step numbers
        cids = [
            ("1ef82a3c-b034-6c2e-8000-000000000001", 10, "val_1"),
            ("1ef82a3c-b034-6c2e-8000-000000000002", 3, "val_2"),
            ("1ef82a3c-b034-6c2e-8000-000000000003", 8, "val_3"),
        ]
        for cid, step, val in cids:
            saver.put(cfg, _make_checkpoint(cid, val), {"step": step}, {})

        records = list(saver.list(cfg))
        assert len(records) == 3
        yielded_ids = [r.checkpoint["id"] for r in records]
        assert yielded_ids == [
            "1ef82a3c-b034-6c2e-8000-000000000003",
            "1ef82a3c-b034-6c2e-8000-000000000002",
            "1ef82a3c-b034-6c2e-8000-000000000001",
        ]
        # Steps are yielded in checkpoint_id descending order, not step order
        assert [r.metadata["step"] for r in records] == [8, 3, 10]


def test_list_before_pagination_by_checkpoint_id_ac_175() -> None:
    """AC-175, ADR-0026: list(before=...) filters checkpoints where checkpoint_id < before."""
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "test-thread-ac175-before"
        cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        cids = [
            "1ef82a3c-b034-6c2e-8000-000000000001",
            "1ef82a3c-b034-6c2e-8000-000000000002",
            "1ef82a3c-b034-6c2e-8000-000000000003",
            "1ef82a3c-b034-6c2e-8000-000000000004",
        ]
        for i, cid in enumerate(cids, start=1):
            saver.put(cfg, _make_checkpoint(cid, f"v_{i}"), {"step": i}, {})

        # (a) Filter before cid 0003 with config dict: returns 0002, 0001
        before_cfg: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": cids[2]}
        }
        res_cfg = list(saver.list(cfg, before=before_cfg))
        assert [r.checkpoint["id"] for r in res_cfg] == [cids[1], cids[0]]

        # (b) Filter before cid 0003 with string and limit=1: returns 0002
        res_str = list(saver.list(cfg, before=cids[2], limit=1))
        assert [r.checkpoint["id"] for r in res_str] == [cids[1]]

        # (c) Filter before earliest checkpoint: returns empty list
        before_earliest: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": cids[0]}
        }
        res_empty = list(saver.list(cfg, before=before_earliest))
        assert res_empty == []

        # (d) Invalid input boundary: empty string before returns all checkpoints
        res_invalid_str = list(saver.list(cfg, before="   "))
        assert len(res_invalid_str) == 4


def test_get_tuple_reraises_connection_error_ac_176() -> None:
    """AC-176: get_tuple() re-raises RelateConnectionError from proxy.execute_read."""
    with RelateSaver(db_path=":memory:") as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": "thread-conn-err"}}
        with patch.object(
            saver.proxy, "execute_read", side_effect=RelateConnectionError("db disconnected")
        ):
            with pytest.raises(RelateConnectionError, match="db disconnected"):
                saver.get_tuple(cfg)


def test_get_tuple_reraises_concurrency_error_ac_176() -> None:
    """AC-176: get_tuple() re-raises RelateConcurrencyError from proxy.execute_read."""
    with RelateSaver(db_path=":memory:") as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": "thread-conc-err"}}
        with patch.object(
            saver.proxy, "execute_read", side_effect=RelateConcurrencyError("write contention")
        ):
            with pytest.raises(RelateConcurrencyError, match="write contention"):
                saver.get_tuple(cfg)


def test_list_reraises_connection_error_ac_176() -> None:
    """AC-176: list() re-raises RelateConnectionError from proxy.execute_read."""
    with RelateSaver(db_path=":memory:") as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": "thread-list-conn-err"}}
        with patch.object(
            saver.proxy, "execute_read", side_effect=RelateConnectionError("db disconnected")
        ):
            with pytest.raises(RelateConnectionError, match="db disconnected"):
                list(saver.list(cfg))


def test_list_reraises_concurrency_error_ac_176() -> None:
    """AC-176: list() re-raises RelateConcurrencyError from proxy.execute_read."""
    with RelateSaver(db_path=":memory:") as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": "thread-list-conc-err"}}
        with patch.object(
            saver.proxy, "execute_read", side_effect=RelateConcurrencyError("lock timeout")
        ):
            with pytest.raises(RelateConcurrencyError, match="lock timeout"):
                list(saver.list(cfg))


def test_generic_checkpoint_error_returns_none_or_empty_ac_176() -> None:
    """AC-176: Generic RelateCheckpointError continues to return None or empty iterator."""
    with RelateSaver(db_path=":memory:") as saver:
        cfg: RunnableConfig = {"configurable": {"thread_id": "thread-generic-err"}}
        with patch.object(
            saver.proxy, "execute_read", side_effect=RelateCheckpointError("generic error")
        ):
            assert saver.get_tuple(cfg) is None
            assert list(saver.list(cfg)) == []


def test_saver_auto_wraps_serializer_protocol_and_roundtrips_state_ac_177() -> None:
    """AC-177: RelateSaver wraps custom SerializerProtocol and preserves state across put/get_tuple."""
    custom_serde = JsonPlusSerializer()
    with RelateSaver(db_path=":memory:", serde=custom_serde) as saver:
        # Verify auto-wrapping
        assert isinstance(saver.serde, RelateStateSerializer)
        assert saver.serde.inner_serializer is custom_serde

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-serde-ac177",
                "checkpoint_ns": "",
            }
        }
        chk = _make_checkpoint("chk-serde-1", "persisted_value")
        chk["channel_values"]["nested"] = {"sub_key": [1, 2, 3], "flag": True}

        saved_config = saver.put(config, chk, {"source": "input", "step": 1}, {})
        retrieved = saver.get_tuple(saved_config)

        assert retrieved is not None
        assert retrieved.checkpoint["id"] == "chk-serde-1"
        assert retrieved.checkpoint["channel_values"]["data"] == "persisted_value"
        assert retrieved.checkpoint["channel_values"]["nested"] == {
            "sub_key": [1, 2, 3],
            "flag": True,
        }


def test_saver_retains_existing_relate_serializer_ac_177() -> None:
    """AC-177: RelateSaver initialized with existing RelateStateSerializer preserves it without wrapping."""
    existing_serde = RelateStateSerializer()
    with RelateSaver(db_path=":memory:", serde=existing_serde) as saver:
        assert saver.serde is existing_serde


