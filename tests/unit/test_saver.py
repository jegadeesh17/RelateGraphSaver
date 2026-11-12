"""Unit tests for RelateSaver checkpointer conforming to LangGraph BaseCheckpointSaver."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.errors import CheckpointSerializationError
from langgraph_checkpoint_relate.proxy import RelateThreadProxy
from langgraph_checkpoint_relate.saver import RelateSaver


def _make_checkpoint(
    checkpoint_id: str,
    channel_values: dict[str, Any] | None = None,
    parent_checkpoint_id: str | None = None,
) -> Checkpoint:
    """Helper to create a structured Checkpoint dictionary."""
    base: dict[str, Any] = dict(empty_checkpoint())
    base["id"] = checkpoint_id
    if parent_checkpoint_id is not None:
        base["parent_checkpoint_id"] = parent_checkpoint_id
    if channel_values:
        base["channel_values"] = channel_values
        base["channel_versions"] = dict.fromkeys(channel_values, "1.0")
    return cast(Checkpoint, base)


class TestRelateSaverPut:
    """AC-02: Unit tests for RelateSaver.put operation and lineage edge creation."""

    def test_put_persists_checkpoint_node_ac_02(self) -> None:
        """AC-02: Persist a checkpoint node with valid config and verify state_hash and ID."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {"configurable": {"thread_id": "thread-ac02-01"}}
            checkpoint = _make_checkpoint("cp-001", {"count": 42})
            metadata: CheckpointMetadata = {"step": 1, "source": "input"}
            new_versions: ChannelVersions = {"count": "1.0"}

            returned_config = saver.put(config, checkpoint, metadata, new_versions)

            assert returned_config["configurable"]["thread_id"] == "thread-ac02-01"
            assert returned_config["configurable"]["checkpoint_id"] == "cp-001"
            assert returned_config["configurable"]["checkpoint_ns"] == ""

            # Verify persisted node in Relate database
            records = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "RETURN c.thread_id AS thread_id, c.step AS step, c.state_hash AS state_hash",
                {"cid": "cp-001"},
            )
            assert len(records) == 1
            assert records[0]["thread_id"] == "thread-ac02-01"
            assert records[0]["step"] == 1
            assert isinstance(records[0]["state_hash"], str)
            assert len(records[0]["state_hash"]) == 64

    def test_put_generates_checkpoint_id_if_omitted_ac_02(self) -> None:
        """AC-02: When checkpoint dictionary has no id, generate a unique UUID."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {"configurable": {"thread_id": "thread-ac02-02"}}
            checkpoint = _make_checkpoint("", {"msg": "hello"})
            metadata: CheckpointMetadata = {"step": 0}

            returned = saver.put(config, checkpoint, metadata, {})
            generated_id = returned["configurable"]["checkpoint_id"]
            assert generated_id is not None
            assert len(generated_id) > 0

            # Verify node can be retrieved
            retrieved = saver.get_tuple(returned)
            assert retrieved is not None
            assert retrieved.checkpoint["id"] == generated_id

    def test_put_creates_next_lineage_edge_ac_02(self) -> None:
        """AC-02: Sequential put operations create direct causal [:NEXT] relationship."""
        with RelateSaver(db_path=":memory:") as saver:
            config1: RunnableConfig = {"configurable": {"thread_id": "thread-lineage"}}
            cp1 = _make_checkpoint("step-1-id", {"status": "started"})
            res1 = saver.put(config1, cp1, {"step": 1}, {})

            # Step 2 with parent_checkpoint_id pointing to step 1
            cp2 = _make_checkpoint(
                "step-2-id", {"status": "in_progress"}, parent_checkpoint_id="step-1-id"
            )
            res2 = saver.put(res1, cp2, {"step": 2}, {})

            # Step 3 chained from res2
            cp3 = _make_checkpoint(
                "step-3-id", {"status": "done"}, parent_checkpoint_id="step-2-id"
            )
            saver.put(res2, cp3, {"step": 3}, {})

            # Query causal NEXT edges in Cypher
            edges = saver.proxy.execute_read(
                "MATCH (p:Checkpoint)-[:NEXT]->(c:Checkpoint) "
                "RETURN p.checkpoint_id AS parent, c.checkpoint_id AS child "
                "ORDER BY p.step ASC"
            )
            assert len(edges) == 2
            assert edges[0] == {"parent": "step-1-id", "child": "step-2-id"}
            assert edges[1] == {"parent": "step-2-id", "child": "step-3-id"}

    def test_put_missing_thread_id_raises_value_error_ac_02(self) -> None:
        """AC-02: Boundary validation: missing or blank thread_id raises ValueError."""
        with RelateSaver(db_path=":memory:") as saver:
            cp = _make_checkpoint("cp-err")
            meta: CheckpointMetadata = {"step": 1}

            with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
                saver.put({}, cp, meta, {})

            with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
                saver.put({"configurable": {}}, cp, meta, {})

            with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
                saver.put({"configurable": {"thread_id": "   "}}, cp, meta, {})

    def test_put_unserializable_state_raises_error_ac_02(self) -> None:
        """AC-02: Persisting un-serializable payload raises CheckpointSerializationError."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {"configurable": {"thread_id": "thread-unserializable"}}
            # Open file object cannot be serialized by JsonPlus
            with open(__file__, "rb") as open_file:
                cp = _make_checkpoint("cp-unserializable", {"file_handle": open_file})
                meta: CheckpointMetadata = {"step": 1}

                with pytest.raises(CheckpointSerializationError):
                    saver.put(config, cp, meta, {})


class TestRelateSaverGetTuple:
    """AC-03, AC-04, AC-05: Unit tests for RelateSaver.get_tuple."""

    def test_get_tuple_latest_checkpoint_ac_03(self) -> None:
        """AC-03: Given thread with multiple checkpoints, get_tuple returns latest checkpoint."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {"configurable": {"thread_id": "thread-latest"}}
            cp1 = _make_checkpoint("cp-step-1", {"val": 10})
            saver.put(config, cp1, {"step": 1}, {})

            cp2 = _make_checkpoint("cp-step-2", {"val": 20}, parent_checkpoint_id="cp-step-1")
            saver.put(config, cp2, {"step": 2}, {})

            cp3 = _make_checkpoint("cp-step-3", {"val": 30}, parent_checkpoint_id="cp-step-2")
            saver.put(config, cp3, {"step": 3}, {})

            latest = saver.get_tuple({"configurable": {"thread_id": "thread-latest"}})
            assert latest is not None
            assert latest.checkpoint["id"] == "cp-step-3"
            assert latest.checkpoint["channel_values"]["val"] == 30
            assert latest.metadata["step"] == 3
            assert latest.parent_config is not None
            assert latest.parent_config["configurable"]["checkpoint_id"] == "cp-step-2"

    def test_get_tuple_targeted_checkpoint_id_ac_04(self) -> None:
        """AC-04: get_tuple with a specific checkpoint_id returns exact matching CheckpointTuple."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {"configurable": {"thread_id": "thread-targeted"}}
            saver.put(config, _make_checkpoint("cp-a", {"v": "alpha"}), {"step": 1}, {})
            saver.put(config, _make_checkpoint("cp-b", {"v": "beta"}, "cp-a"), {"step": 2}, {})
            saver.put(config, _make_checkpoint("cp-c", {"v": "gamma"}, "cp-b"), {"step": 3}, {})

            # Query historical checkpoint cp-b directly
            target_cfg: RunnableConfig = {
                "configurable": {"thread_id": "thread-targeted", "checkpoint_id": "cp-b"}
            }
            tuple_b = saver.get_tuple(target_cfg)
            assert tuple_b is not None
            assert tuple_b.checkpoint["id"] == "cp-b"
            assert tuple_b.checkpoint["channel_values"]["v"] == "beta"
            assert tuple_b.metadata["step"] == 2
            assert tuple_b.parent_config is not None
            assert tuple_b.parent_config["configurable"]["checkpoint_id"] == "cp-a"

    def test_get_tuple_nonexistent_returns_none_ac_05(self) -> None:
        """AC-05: Non-existent thread_id or checkpoint_id returns None without exception."""
        with RelateSaver(db_path=":memory:") as saver:
            # Thread does not exist
            assert saver.get_tuple({"configurable": {"thread_id": "ghost-thread"}}) is None

            # Thread exists but checkpoint_id does not
            saver.put(
                {"configurable": {"thread_id": "existing-thread"}},
                _make_checkpoint("real-cp", {"a": 1}),
                {"step": 1},
                {},
            )
            cfg_ghost_cp: RunnableConfig = {
                "configurable": {"thread_id": "existing-thread", "checkpoint_id": "ghost-cp"}
            }
            assert saver.get_tuple(cfg_ghost_cp) is None

    def test_get_tuple_empty_or_none_config_returns_none_ac_05(self) -> None:
        """AC-05: None or empty config returns None gracefully."""
        with RelateSaver(db_path=":memory:") as saver:
            assert saver.get_tuple(None) is None  # type: ignore[arg-type]
            assert saver.get_tuple({}) is None
            assert saver.get_tuple({"configurable": {}}) is None

    def test_get_tuple_retrieves_pending_writes_ac_03(self) -> None:
        """AC-03: get_tuple correctly populates pending_writes stored via put_writes."""
        with RelateSaver(db_path=":memory:") as saver:
            cfg: RunnableConfig = {
                "configurable": {"thread_id": "thread-writes", "checkpoint_id": "cp-w1"}
            }
            saver.put(cfg, _make_checkpoint("cp-w1"), {"step": 1}, {})

            # Store pending writes
            saver.put_writes(cfg, [("channel_a", "val_a"), ("channel_b", 42)], task_id="task-001")

            tup = saver.get_tuple(cfg)
            assert tup is not None
            assert tup.pending_writes is not None
            assert len(tup.pending_writes) == 2
            assert tup.pending_writes[0] == ("task-001", "channel_a", "val_a")
            assert tup.pending_writes[1] == ("task-001", "channel_b", 42)


class TestRelateSaverList:
    """AC-06, AC-07: Unit tests for RelateSaver.list ordering and filtering."""

    def test_list_reverse_chronological_ordering_ac_06(self) -> None:
        """AC-06: list() yields records in reverse-chronological order by step."""
        with RelateSaver(db_path=":memory:") as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": "thread-order"}}
            saver.put(cfg, _make_checkpoint("cp-1", {"step_name": "one"}), {"step": 1}, {})
            saver.put(cfg, _make_checkpoint("cp-2", {"step_name": "two"}), {"step": 2}, {})
            saver.put(cfg, _make_checkpoint("cp-3", {"step_name": "three"}), {"step": 3}, {})

            results = list(saver.list(cfg))
            assert len(results) == 3
            assert [r.checkpoint["id"] for r in results] == ["cp-3", "cp-2", "cp-1"]
            assert [r.metadata["step"] for r in results] == [3, 2, 1]

    def test_list_with_before_and_limit_ac_07(self) -> None:
        """AC-07: list() with before and limit yields at most limit records prior to target."""
        with RelateSaver(db_path=":memory:") as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": "thread-pagination"}}
            for i in range(1, 6):
                saver.put(
                    cfg,
                    _make_checkpoint(f"cp-{i}", {"i": i}, f"cp-{i - 1}" if i > 1 else None),
                    {"step": i},
                    {},
                )

            # Before cp-4 with limit 2: should return [cp-3, cp-2]
            before_cfg: RunnableConfig = {
                "configurable": {"thread_id": "thread-pagination", "checkpoint_id": "cp-4"}
            }
            results = list(saver.list(cfg, before=before_cfg, limit=2))
            assert len(results) == 2
            assert [r.checkpoint["id"] for r in results] == ["cp-3", "cp-2"]

            # Before cp-4 with limit 1: should return [cp-3]
            results_lim1 = list(saver.list(cfg, before=before_cfg, limit=1))
            assert len(results_lim1) == 1
            assert results_lim1[0].checkpoint["id"] == "cp-3"

            # Before as string checkpoint_id
            results_str = list(saver.list(cfg, before="cp-3", limit=2))
            assert len(results_str) == 2
            assert [r.checkpoint["id"] for r in results_str] == ["cp-2", "cp-1"]

    def test_list_with_metadata_filter_ac_06(self) -> None:
        """AC-06: list() with filter matches metadata key-value properties."""
        with RelateSaver(db_path=":memory:") as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": "thread-filter"}}
            meta_a: dict[str, Any] = {"step": 1, "node": "agent", "run": 1}
            meta_b: dict[str, Any] = {"step": 2, "node": "tools", "run": 1}
            meta_c: dict[str, Any] = {"step": 3, "node": "agent", "run": 2}
            saver.put(cfg, _make_checkpoint("cp-a"), cast(CheckpointMetadata, meta_a), {})
            saver.put(cfg, _make_checkpoint("cp-b"), cast(CheckpointMetadata, meta_b), {})
            saver.put(cfg, _make_checkpoint("cp-c"), cast(CheckpointMetadata, meta_c), {})

            agent_runs = list(saver.list(cfg, filter={"node": "agent"}))
            assert len(agent_runs) == 2
            assert [r.checkpoint["id"] for r in agent_runs] == ["cp-c", "cp-a"]

            run1_tools = list(saver.list(cfg, filter={"node": "tools", "run": 1}))
            assert len(run1_tools) == 1
            assert run1_tools[0].checkpoint["id"] == "cp-b"

    def test_list_zero_or_negative_limit_ac_07(self) -> None:
        """AC-07: list() with limit <= 0 returns empty generator immediately."""
        with RelateSaver(db_path=":memory:") as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": "thread-zero-lim"}}
            saver.put(cfg, _make_checkpoint("cp-1"), {"step": 1}, {})

            assert list(saver.list(cfg, limit=0)) == []
            assert list(saver.list(cfg, limit=-5)) == []

    def test_list_all_threads_when_config_none_ac_06(self) -> None:
        """AC-06: list(None) yields checkpoints across multiple threads."""
        with RelateSaver(db_path=":memory:") as saver:
            saver.put(
                {"configurable": {"thread_id": "t-1"}}, _make_checkpoint("c-t1"), {"step": 1}, {}
            )
            saver.put(
                {"configurable": {"thread_id": "t-2"}}, _make_checkpoint("c-t2"), {"step": 2}, {}
            )

            all_results = list(saver.list(None))
            assert len(all_results) == 2
            ids = {r.checkpoint["id"] for r in all_results}
            assert ids == {"c-t1", "c-t2"}


class TestRelateSaverLangGraphIntegration:
    """Integration verification: compile and invoke real LangGraph StateGraph workflows."""

    def test_stategraph_compile_and_invoke_integration_ac_02_ac_03(self) -> None:
        """AC-02, AC-03: Real StateGraph workflow execution persists full step DAG."""

        class PipelineState(TypedDict):
            counter: int
            log: list[str]

        def step_increment(state: PipelineState) -> dict[str, Any]:
            return {
                "counter": state["counter"] + 1,
                "log": state["log"] + ["incremented"],
            }

        def step_multiply(state: PipelineState) -> dict[str, Any]:
            return {
                "counter": state["counter"] * 3,
                "log": state["log"] + ["multiplied"],
            }

        builder = StateGraph(PipelineState)
        builder.add_node("increment", step_increment)
        builder.add_node("multiply", step_multiply)
        builder.add_edge(START, "increment")
        builder.add_edge("increment", "multiply")
        builder.add_edge("multiply", END)

        with RelateSaver(db_path=":memory:") as saver:
            graph = builder.compile(checkpointer=saver)
            config: RunnableConfig = {"configurable": {"thread_id": "graph-exec-01"}}

            # Execute pipeline
            final_state = graph.invoke({"counter": 5, "log": ["init"]}, config)
            # counter: (5 + 1) * 3 = 18
            assert final_state["counter"] == 18
            assert final_state["log"] == ["init", "incremented", "multiplied"]

            # Verify latest checkpoint tuple matches final state
            latest = saver.get_tuple(config)
            assert latest is not None
            assert latest.checkpoint["channel_values"]["counter"] == 18

            # Verify complete execution history (START -> increment -> multiply -> END)
            history = list(saver.list(config))
            assert len(history) == 4

            # Verify DAG structure in Neo4j-compatible graph
            edges = saver.proxy.execute_read(
                "MATCH (p:Checkpoint)-[:NEXT]->(c:Checkpoint) "
                "WHERE p.thread_id = $tid "
                "RETURN p.step AS from_step, c.step AS to_step "
                "ORDER BY from_step ASC",
                {"tid": "graph-exec-01"},
            )
            assert len(edges) == 3
            # Steps flow: -1 (start) -> 0 -> 1 -> 2 (end)
            assert edges[0]["from_step"] == -1
            assert edges[0]["to_step"] == 0
            assert edges[1]["from_step"] == 0
            assert edges[1]["to_step"] == 1
            assert edges[2]["from_step"] == 1
            assert edges[2]["to_step"] == 2


class TestRelateSaverLifecycleAndHelpers:
    """Tests for RelateSaver initialization options and helper methods."""

    def test_saver_with_existing_proxy_lifecycle(self) -> None:
        """Verify RelateSaver accepts an existing RelateThreadProxy without taking ownership."""
        proxy = RelateThreadProxy(db_path=":memory:")
        saver = RelateSaver(db_path=proxy)
        assert saver.proxy is proxy
        assert not saver._owns_proxy

        # Closing saver does not terminate external proxy
        saver.close()
        assert proxy.is_alive()
        proxy.close()

    def test_get_next_version_monotonicity(self) -> None:
        """Verify get_next_version produces monotonically increasing version strings."""
        with RelateSaver(db_path=":memory:") as saver:
            v0 = saver.get_next_version(None)
            v1 = saver.get_next_version(v0)
            v2 = saver.get_next_version(v1)

            assert int(v0.split(".")[0]) == 1
            assert int(v1.split(".")[0]) == 2
            assert int(v2.split(".")[0]) == 3


class TestRelateSaverPutWrites:
    """AC-14: Persistence of pending task channel writes and graph linking."""

    def test_put_writes_persists_and_links_via_has_write_ac_14(self) -> None:
        """AC-14: Persist pending channel writes and verify [:HAS_WRITE] relationship."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac14-01",
                    "checkpoint_ns": "sub_ns",
                    "checkpoint_id": "cp-ac14-01",
                }
            }
            saver.put(config, _make_checkpoint("cp-ac14-01"), {"step": 1}, {})

            writes = [
                ("channel_alpha", {"action": "review", "score": 98}),
                ("channel_beta", ["item_1", "item_2"]),
            ]
            saver.put_writes(config, writes, task_id="task-evaluator", task_path="main/sub_eval")

            # Verify [:HAS_WRITE] relationship and node properties in Relate graph
            records = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid})-[:HAS_WRITE]->(w:Write) "
                "RETURN w.write_id AS write_id, "
                "       w.task_id AS task_id, "
                "       w.task_path AS task_path, "
                "       w.channel AS channel, "
                "       w.idx AS idx "
                "ORDER BY w.idx ASC",
                {"cid": "cp-ac14-01"},
            )
            assert len(records) == 2
            assert records[0]["task_id"] == "task-evaluator"
            assert records[0]["task_path"] == "main/sub_eval"
            assert records[0]["channel"] == "channel_alpha"
            assert records[0]["idx"] == 0
            assert records[0]["write_id"] == "thread-ac14-01:sub_ns:cp-ac14-01:task-evaluator:0"

            assert records[1]["task_id"] == "task-evaluator"
            assert records[1]["task_path"] == "main/sub_eval"
            assert records[1]["channel"] == "channel_beta"
            assert records[1]["idx"] == 1
            assert records[1]["write_id"] == "thread-ac14-01:sub_ns:cp-ac14-01:task-evaluator:1"

    def test_put_writes_special_channel_indexing_ac_14(self) -> None:
        """AC-14: Verify special channels receive mapped negative indices from WRITES_IDX_MAP."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac14-special",
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-special",
                }
            }
            saver.put(config, _make_checkpoint("cp-special"), {"step": 1}, {})

            # Writes with both special channels and a normal channel
            writes = [
                ("__interrupt__", {"reason": "human_review"}),
                ("__resume__", {"decision": "approved"}),
                ("__error__", "execution_timeout"),
                ("__scheduled__", "task-99"),
                ("normal_ch", "standard_payload"),
            ]
            saver.put_writes(config, writes, task_id="task-special-01", task_path="node_a")

            records = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid})-[:HAS_WRITE]->(w:Write) "
                "RETURN w.channel AS channel, w.idx AS idx, w.write_id AS write_id "
                "ORDER BY w.idx ASC",
                {"cid": "cp-special"},
            )
            assert len(records) == 5
            # Negative indices for special channels
            channel_map = {r["channel"]: r for r in records}
            assert channel_map["__resume__"]["idx"] == -4
            assert (
                channel_map["__resume__"]["write_id"]
                == "thread-ac14-special::cp-special:task-special-01:-4"
            )

            assert channel_map["__interrupt__"]["idx"] == -3
            assert (
                channel_map["__interrupt__"]["write_id"]
                == "thread-ac14-special::cp-special:task-special-01:-3"
            )

            assert channel_map["__scheduled__"]["idx"] == -2
            assert (
                channel_map["__scheduled__"]["write_id"]
                == "thread-ac14-special::cp-special:task-special-01:-2"
            )

            assert channel_map["__error__"]["idx"] == -1
            assert (
                channel_map["__error__"]["write_id"]
                == "thread-ac14-special::cp-special:task-special-01:-1"
            )

            # Normal channel gets enumeration index (idx 4 in the writes sequence)
            assert channel_map["normal_ch"]["idx"] == 4
            assert (
                channel_map["normal_ch"]["write_id"]
                == "thread-ac14-special::cp-special:task-special-01:4"
            )

    def test_put_writes_idempotency_ac_14(self) -> None:
        """AC-14: Re-putting writes with identical task_id and channel does not duplicate nodes or edges."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac14-idempotent",
                    "checkpoint_id": "cp-idem-01",
                }
            }
            saver.put(config, _make_checkpoint("cp-idem-01"), {"step": 1}, {})

            writes = [("status_channel", "pending_review")]

            # First write call
            saver.put_writes(config, writes, task_id="task-idem", task_path="step_node")

            # Second identical write call
            saver.put_writes(config, writes, task_id="task-idem", task_path="step_node")

            # Third call with updated value on same channel and task
            saver.put_writes(
                config,
                [("status_channel", "updated_review")],
                task_id="task-idem",
                task_path="step_node",
            )

            # Verify only one :Write node exists
            write_records = saver.proxy.execute_read(
                "MATCH (w:Write {thread_id: $tid, checkpoint_id: $cid}) "
                "RETURN w.write_id AS write_id, count(w) AS cnt",
                {"tid": "thread-ac14-idempotent", "cid": "cp-idem-01"},
            )
            assert len(write_records) == 1

            # Verify only one [:HAS_WRITE] edge exists
            rel_records = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid})-[r:HAS_WRITE]->(w:Write) "
                "RETURN count(r) AS rel_count",
                {"cid": "cp-idem-01"},
            )
            assert rel_records[0]["rel_count"] == 1

    def test_put_writes_boundary_validations_ac_14(self) -> None:
        """AC-14: Boundary validations for missing or empty parameters."""
        with RelateSaver(db_path=":memory:") as saver:
            # Empty writes list returns early without raising
            saver.put_writes(
                {"configurable": {"thread_id": "t", "checkpoint_id": "c"}}, [], task_id="t1"
            )

            # Missing thread_id raises ValueError
            with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
                saver.put_writes(
                    {"configurable": {"checkpoint_id": "c"}}, [("ch", "v")], task_id="t1"
                )

            # Missing checkpoint_id raises ValueError
            with pytest.raises(ValueError, match="checkpoint_id is required"):
                saver.put_writes({"configurable": {"thread_id": "t"}}, [("ch", "v")], task_id="t1")


class TestRelateSaverPendingWritesOrdering:
    """AC-15: Reconstruction of pending writes in CheckpointTuple with correct index ordering."""

    def test_get_tuple_reconstructs_pending_writes_in_index_order_ac_15(self) -> None:
        """AC-15: get_tuple orders pending_writes by w.idx ASC, placing special channels before normal channels."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac15-order",
                    "checkpoint_id": "cp-ac15-01",
                }
            }
            saver.put(config, _make_checkpoint("cp-ac15-01"), {"step": 1}, {})

            # Writes added in reverse or mixed order
            writes = [
                ("output_channel", {"result": "success"}),
                ("__resume__", "resume_payload"),
                ("__interrupt__", "interrupt_payload"),
            ]
            saver.put_writes(config, writes, task_id="task-order-01")

            tup = saver.get_tuple(config)
            assert tup is not None
            assert tup.pending_writes is not None
            assert len(tup.pending_writes) == 3

            # Ordering by w.idx ASC:
            # __resume__ has idx -4
            # __interrupt__ has idx -3
            # output_channel has idx 0
            assert tup.pending_writes[0] == ("task-order-01", "__resume__", "resume_payload")
            assert tup.pending_writes[1] == ("task-order-01", "__interrupt__", "interrupt_payload")
            assert tup.pending_writes[2] == (
                "task-order-01",
                "output_channel",
                {"result": "success"},
            )

    def test_get_tuple_no_writes_returns_empty_list_ac_15(self) -> None:
        """AC-15: Checkpoint without pending writes returns an empty list for pending_writes."""
        with RelateSaver(db_path=":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac15-nowrites",
                    "checkpoint_id": "cp-ac15-nowrites",
                }
            }
            saver.put(config, _make_checkpoint("cp-ac15-nowrites"), {"step": 1}, {})

            tup = saver.get_tuple(config)
            assert tup is not None
            assert tup.pending_writes == []


class TestRelateSaverResumeCausalLineage:
    """AC-16: Resuming from an interrupted checkpoint connects causal edges correctly."""

    def test_resume_from_interrupted_checkpoint_links_parent_ac_16(self) -> None:
        """AC-16: Resumed checkpoint records interrupted checkpoint as parent and creates [:NEXT] edge."""
        with RelateSaver(db_path=":memory:") as saver:
            # Step 1: Interrupted execution
            cfg1: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac16-lineage",
                    "checkpoint_id": "cp-interrupted",
                }
            }
            saver.put(
                cfg1,
                _make_checkpoint("cp-interrupted", {"state": "waiting_approval"}),
                {"step": 1},
                {},
            )
            saver.put_writes(cfg1, [("__interrupt__", "approval_needed")], task_id="human_task")

            # Verify pending write is stored
            interrupted_tup = saver.get_tuple(cfg1)
            assert interrupted_tup is not None
            assert interrupted_tup.pending_writes is not None
            assert len(interrupted_tup.pending_writes) == 1
            assert interrupted_tup.pending_writes[0][1] == "__interrupt__"

            # Step 2: Resume workflow with put() referencing cp-interrupted as parent
            cfg2: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac16-lineage",
                    "checkpoint_id": "cp-resumed",
                }
            }
            cp_resumed = _make_checkpoint(
                "cp-resumed",
                {"state": "approved_and_executed"},
                parent_checkpoint_id="cp-interrupted",
            )
            saver.put(cfg2, cp_resumed, {"step": 2}, {})

            # Verify parent_config in reconstructed tuple
            resumed_tup = saver.get_tuple(cfg2)
            assert resumed_tup is not None
            assert resumed_tup.parent_config is not None
            assert resumed_tup.parent_config["configurable"]["checkpoint_id"] == "cp-interrupted"

            # Verify [:NEXT] edge in Relate property graph
            edges = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {checkpoint_id: 'cp-interrupted'})-[r:NEXT]->(c:Checkpoint {checkpoint_id: 'cp-resumed'}) "
                "RETURN type(r) AS rel_type"
            )
            assert len(edges) == 1
            assert edges[0]["rel_type"] == "NEXT"

    def test_resume_alternate_branch_creates_forked_to_edge_ac_16(self) -> None:
        """AC-16: Replaying or branching from an interrupted checkpoint creates [:FORKED_TO] edge."""
        with RelateSaver(db_path=":memory:") as saver:
            cfg1: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-ac16-fork",
                    "checkpoint_id": "cp-base",
                }
            }
            saver.put(cfg1, _make_checkpoint("cp-base"), {"step": 1}, {})
            saver.put_writes(cfg1, [("__interrupt__", "fork_prompt")], task_id="prompt_task")

            # Branch A creates primary NEXT child
            cp_a = _make_checkpoint("cp-branch-a", parent_checkpoint_id="cp-base")
            saver.put(
                {"configurable": {"thread_id": "thread-ac16-fork", "checkpoint_id": "cp-branch-a"}},
                cp_a,
                {"step": 2},
                {},
            )

            # Branch B resumes from cp-base again (creates FORKED_TO edge because cp-base already has a NEXT child)
            cp_b = _make_checkpoint("cp-branch-b", parent_checkpoint_id="cp-base")
            saver.put(
                {"configurable": {"thread_id": "thread-ac16-fork", "checkpoint_id": "cp-branch-b"}},
                cp_b,
                {"step": 2},
                {},
            )

            fork_edges = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {checkpoint_id: 'cp-base'})-[r:FORKED_TO]->(c:Checkpoint {checkpoint_id: 'cp-branch-b'}) "
                "RETURN type(r) AS rel_type"
            )
            assert len(fork_edges) == 1
            assert fork_edges[0]["rel_type"] == "FORKED_TO"

    def test_langgraph_human_in_the_loop_interrupt_resume_flow_ac_14_ac_15_ac_16(self) -> None:
        """AC-14, AC-15, AC-16: End-to-end LangGraph StateGraph interrupt and Command resume execution."""

        class ApprovalState(TypedDict):
            query: str
            approval: str
            result: str

        def step_ask_human(state: ApprovalState) -> dict[str, Any]:
            decision = interrupt("Please approve or reject the query.")
            return {"approval": decision}

        def step_process(state: ApprovalState) -> dict[str, Any]:
            return {"result": f"processed:{state['query']}:{state['approval']}"}

        builder = StateGraph(ApprovalState)
        builder.add_node("ask_human", step_ask_human)
        builder.add_node("process", step_process)
        builder.add_edge(START, "ask_human")
        builder.add_edge("ask_human", "process")
        builder.add_edge("process", END)

        with RelateSaver(db_path=":memory:") as saver:
            graph = builder.compile(checkpointer=saver)
            config: RunnableConfig = {"configurable": {"thread_id": "thread-hitl-flow"}}

            # 1. Run until interrupt
            initial_output = graph.invoke(
                {"query": "transfer_funds", "approval": "", "result": ""}, config
            )
            assert "__interrupt__" in initial_output

            # AC-15: Verify get_tuple contains pending __interrupt__ write
            interrupted_tuple = saver.get_tuple(config)
            assert interrupted_tuple is not None
            assert interrupted_tuple.pending_writes is not None
            assert len(interrupted_tuple.pending_writes) >= 1
            interrupt_channels = [pw[1] for pw in interrupted_tuple.pending_writes]
            assert "__interrupt__" in interrupt_channels

            # AC-14: Verify :HAS_WRITE relationship exists in Relate graph
            interrupted_cid = interrupted_tuple.checkpoint["id"]
            write_edges = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid})-[:HAS_WRITE]->(w:Write) "
                "WHERE w.channel = '__interrupt__' "
                "RETURN w.idx AS idx, w.task_path AS task_path",
                {"cid": interrupted_cid},
            )
            assert len(write_edges) == 1
            assert write_edges[0]["idx"] == -3

            # 2. Resume execution with human approval
            final_output = graph.invoke(Command(resume="approved_by_user"), config)
            assert final_output["approval"] == "approved_by_user"
            assert final_output["result"] == "processed:transfer_funds:approved_by_user"

            # AC-16: Verify causal path links interrupted checkpoint to subsequent execution checkpoints
            resumed_tuple = saver.get_tuple(config)
            assert resumed_tuple is not None
            resumed_cid = resumed_tuple.checkpoint["id"]

            # Interrupted checkpoint connects directly to the resumed step via [:NEXT]
            direct_children = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {checkpoint_id: $p_cid})-[:NEXT]->(c:Checkpoint) "
                "RETURN c.checkpoint_id AS child_id, c.parent_checkpoint_id AS parent_id",
                {"p_cid": interrupted_cid},
            )
            assert len(direct_children) == 1
            assert direct_children[0]["parent_id"] == interrupted_cid

            # And a causal transitive path connects the interrupted checkpoint to the final checkpoint
            causal_path = saver.proxy.execute_read(
                "MATCH path = (p:Checkpoint {checkpoint_id: $p_cid})-[:NEXT*]->(c:Checkpoint {checkpoint_id: $res_cid}) "
                "RETURN length(path) AS path_len",
                {"p_cid": interrupted_cid, "res_cid": resumed_cid},
            )
            assert len(causal_path) == 1
            assert causal_path[0]["path_len"] >= 1
