"""Integration tests for the async checkpointer methods (ADR-0006).

Run against real ``:memory:`` savers. Async code is driven with ``asyncio.run`` inside
ordinary test functions, so no pytest async plugin is needed.

- AC-50: async methods return results equal to their synchronous counterparts.
- AC-51: ``ainvoke`` persists checkpoints retrievable with ``get_tuple``.
- AC-52: ``astream`` yields every step and persists the final checkpoint.
- AC-53: concurrent ``ainvoke`` runs on distinct threads stay isolated and complete.
- AC-54: the synchronous methods run off the event loop thread, writes via the proxy.
- AC-55: ``adelete_thread`` is defined if and only if ``delete_thread`` is.
"""

from __future__ import annotations

import asyncio
import operator
import threading
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from langgraph_checkpoint_relate.saver import RelateSaver


class _State(TypedDict):
    tag: str
    log: Annotated[list[str], operator.add]


def _node_a(state: _State) -> dict[str, Any]:
    return {"log": [f"{state['tag']}:a"]}


def _node_b(state: _State) -> dict[str, Any]:
    return {"log": [f"{state['tag']}:b"]}


def _node_c(state: _State) -> dict[str, Any]:
    return {"log": [f"{state['tag']}:c"]}


def _build_graph(saver: RelateSaver) -> CompiledStateGraph[Any, Any, Any, Any]:
    builder = StateGraph(_State)
    builder.add_node("a", _node_a)
    builder.add_node("b", _node_b)
    builder.add_node("c", _node_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    return builder.compile(checkpointer=saver)


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint(checkpoint_id: str) -> Any:
    cp: Any = dict(empty_checkpoint())
    cp["id"] = checkpoint_id
    return cp


async def _collect(saver: RelateSaver, config: RunnableConfig | None) -> list[CheckpointTuple]:
    return [tup async for tup in saver.alist(config)]


class TestAsyncMethodsMatchSync:
    """AC-50: async results equal the synchronous counterparts."""

    def test_async_methods_equal_sync_ac_50(self) -> None:
        sync_saver = RelateSaver(":memory:")
        async_saver = RelateSaver(":memory:")
        try:
            thread_id = "ac50-thread"
            writes = [("messages", {"role": "user", "content": "hi"}), ("status", "pending")]
            checkpoints = [_checkpoint(f"cp-{idx}") for idx in range(3)]
            sync_cfgs: list[RunnableConfig] = []
            async_cfgs: list[RunnableConfig] = []

            async def drive_async() -> None:
                parent: str | None = None
                for idx in range(3):
                    cid = f"cp-{idx}"
                    base = _cfg(thread_id, parent) if parent else _cfg(thread_id)
                    async_cfgs.append(
                        await async_saver.aput(base, checkpoints[idx], {"step": idx}, {})
                    )
                    parent = cid
                await async_saver.aput_writes(_cfg(thread_id, "cp-2"), writes, "task-1", "path")

            asyncio.run(drive_async())

            parent_id: str | None = None
            for idx in range(3):
                cid = f"cp-{idx}"
                base = _cfg(thread_id, parent_id) if parent_id else _cfg(thread_id)
                sync_cfgs.append(sync_saver.put(base, checkpoints[idx], {"step": idx}, {}))
                parent_id = cid
            sync_saver.put_writes(_cfg(thread_id, "cp-2"), writes, "task-1", "path")

            assert async_cfgs == sync_cfgs

            async def read_async() -> tuple[CheckpointTuple | None, list[CheckpointTuple]]:
                latest = await async_saver.aget_tuple(_cfg(thread_id))
                history = await _collect(async_saver, _cfg(thread_id))
                return latest, history

            a_latest, a_history = asyncio.run(read_async())
            s_latest = sync_saver.get_tuple(_cfg(thread_id))
            s_history = list(sync_saver.list(_cfg(thread_id)))

            assert a_latest is not None
            assert a_latest == s_latest
            assert [w[1:] for w in a_latest.pending_writes or []] == [
                ("messages", {"role": "user", "content": "hi"}),
                ("status", "pending"),
            ]
            assert len(a_history) == 3
            assert a_history == s_history

            async def read_limited() -> list[CheckpointTuple]:
                return [tup async for tup in async_saver.alist(_cfg(thread_id), limit=2)]

            assert asyncio.run(read_limited()) == list(sync_saver.list(_cfg(thread_id), limit=2))
        finally:
            sync_saver.close()
            async_saver.close()

    def test_aget_tuple_missing_thread_returns_none_ac_50(self) -> None:
        with RelateSaver(":memory:") as saver:
            assert asyncio.run(saver.aget_tuple(_cfg("no-such-thread"))) is None
            assert asyncio.run(_collect(saver, _cfg("no-such-thread"))) == []

    def test_aput_invalid_config_raises_same_type_as_sync_ac_50(self) -> None:
        with RelateSaver(":memory:") as saver:
            bad_config: RunnableConfig = {"configurable": {}}
            with pytest.raises(Exception) as sync_exc:
                saver.put(bad_config, _checkpoint("x"), {}, {})
            with pytest.raises(type(sync_exc.value)):
                asyncio.run(saver.aput(bad_config, _checkpoint("x"), {}, {}))


class TestAsyncGraphExecution:
    """AC-51, AC-52, AC-53: real graphs run through ainvoke and astream."""

    def test_ainvoke_persists_final_checkpoint_ac_51(self) -> None:
        with RelateSaver(":memory:") as saver:
            graph = _build_graph(saver)
            config = _cfg("ac51")
            result = asyncio.run(graph.ainvoke({"tag": "t", "log": []}, config))
            assert result["log"] == ["t:a", "t:b", "t:c"]

            latest = saver.get_tuple(config)
            assert latest is not None
            assert latest.checkpoint["channel_values"]["log"] == ["t:a", "t:b", "t:c"]
            assert latest.metadata is not None
            assert latest.metadata["step"] == 3

    def test_astream_yields_every_step_ac_52(self) -> None:
        with RelateSaver(":memory:") as saver:
            graph = _build_graph(saver)
            config = _cfg("ac52")

            async def stream() -> list[dict[str, Any]]:
                return [
                    chunk
                    async for chunk in graph.astream(
                        {"tag": "s", "log": []}, config, stream_mode="updates"
                    )
                ]

            chunks = asyncio.run(stream())
            assert chunks == [
                {"a": {"log": ["s:a"]}},
                {"b": {"log": ["s:b"]}},
                {"c": {"log": ["s:c"]}},
            ]
            latest = saver.get_tuple(config)
            assert latest is not None
            assert latest.checkpoint["channel_values"]["log"] == ["s:a", "s:b", "s:c"]

    def test_concurrent_ainvoke_isolated_per_thread_ac_53(self) -> None:
        run_count = 6
        with RelateSaver(":memory:") as saver:
            graph = _build_graph(saver)
            thread_ids = [f"ac53-{idx}" for idx in range(run_count)]

            async def run_all() -> list[dict[str, Any]]:
                return list(
                    await asyncio.gather(
                        *(graph.ainvoke({"tag": tid, "log": []}, _cfg(tid)) for tid in thread_ids)
                    )
                )

            results = asyncio.run(run_all())
            assert len(results) == run_count

            expected_len = len(list(saver.list(_cfg(thread_ids[0]))))
            assert expected_len >= 4
            for tid in thread_ids:
                history = list(saver.list(_cfg(tid)))
                assert len(history) == expected_len
                assert history[0].checkpoint["channel_values"]["log"] == [
                    f"{tid}:a",
                    f"{tid}:b",
                    f"{tid}:c",
                ]
                for tup in history:
                    assert tup.config["configurable"]["thread_id"] == tid
                    for entry in tup.checkpoint["channel_values"].get("log", []):
                        assert entry.startswith(f"{tid}:")


class TestAsyncRunsOffLoop:
    """AC-54: synchronous work leaves the event loop thread; writes use the proxy."""

    @staticmethod
    def _spy(target: Any, name: str, idents: list[int], calls: list[str] | None = None) -> None:
        original = getattr(target, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            idents.append(threading.get_ident())
            if calls is not None:
                calls.append(name)
            return original(*args, **kwargs)

        setattr(target, name, wrapper)

    def test_sync_methods_run_off_event_loop_thread_ac_54(self) -> None:
        with RelateSaver(":memory:") as saver:
            sync_idents: list[int] = []
            for name in ("get_tuple", "list", "put", "put_writes"):
                self._spy(saver, name, sync_idents)

            proxy_idents: list[int] = []
            proxy_calls: list[str] = []
            for name in ("execute_write", "execute_batch"):
                self._spy(saver.proxy, name, proxy_idents, proxy_calls)

            async def drive() -> int:
                loop_ident = threading.get_ident()
                await saver.aput(_cfg("ac54"), _checkpoint("cp-0"), {"step": 0}, {})
                await saver.aput_writes(_cfg("ac54", "cp-0"), [("status", "ok")], "task-1")
                assert await saver.aget_tuple(_cfg("ac54")) is not None
                assert len(await _collect(saver, _cfg("ac54"))) == 1
                return loop_ident

            loop_ident = asyncio.run(drive())

            # put, put_writes, get_tuple and list each ran once on a non-loop thread.
            assert len(sync_idents) == 4
            assert all(ident != loop_ident for ident in sync_idents)
            # Writes still went through the proxy, also off the loop thread.
            assert "execute_write" in proxy_calls
            assert "execute_batch" in proxy_calls
            assert all(ident != loop_ident for ident in proxy_idents)


class TestDeleteThreadParity:
    """AC-55: adelete_thread is overridden if and only if delete_thread is."""

    def test_adelete_thread_defined_iff_delete_thread_defined_ac_55(self) -> None:
        assert ("delete_thread" in vars(RelateSaver)) == ("adelete_thread" in vars(RelateSaver))
