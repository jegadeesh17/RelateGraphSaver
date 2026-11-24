"""Integration tests for synchronous RelateSaver get_tuple and list namespace semantics.

Verifies AC-173:
- Given a config without checkpoint_ns (or None), get_tuple() defaults checkpoint_ns to ""
  (matching LangGraph SqliteSaver behavior), resolving root namespace checkpoints and
  ignoring subgraph checkpoints.
- Given a list() call without checkpoint_ns, results remain unfiltered by namespace.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
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


def test_explicit_db_path_takes_precedence_over_env_var_ac_178(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC-178: Explicit db_path takes precedence over RELATE_DB_PATH environment variable."""
    env_path = str(tmp_path / "from_env.rl")
    monkeypatch.setenv("RELATE_DB_PATH", env_path)

    # Explicit :memory: overrides env var
    with RelateSaver(db_path=":memory:") as saver:
        assert saver.db_path == ":memory:"
        assert saver.proxy.db_path == ":memory:"


def test_env_var_db_path_takes_precedence_over_default_ac_178(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC-178: RELATE_DB_PATH takes precedence over default ':memory:' when db_path is omitted."""
    env_path = str(tmp_path / "from_env.rl")
    monkeypatch.setenv("RELATE_DB_PATH", env_path)

    with RelateSaver() as saver:
        assert saver.db_path == env_path
        assert saver.proxy.db_path == env_path

    # When RELATE_DB_PATH is cleared, default is :memory:
    monkeypatch.delenv("RELATE_DB_PATH", raising=False)
    with RelateSaver() as saver_default:
        assert saver_default.db_path == ":memory:"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_relate_db_path_is_rejected_ac_199(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """AC-199: an empty or whitespace-only RELATE_DB_PATH is rejected when db_path is not passed."""
    monkeypatch.setenv("RELATE_DB_PATH", value)
    with pytest.raises(ValueError, match="RELATE_DB_PATH"):
        RelateSaver()


def test_explicit_wal_enabled_takes_precedence_over_env_var_ac_178(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-178: Explicit wal_enabled argument overrides RELATE_WAL_ENABLED env var."""
    # Env says 0 (False), explicit says True
    monkeypatch.setenv("RELATE_WAL_ENABLED", "0")
    with RelateSaver(db_path=":memory:", wal_enabled=True) as saver:
        assert saver.wal_enabled is True
        assert saver.proxy.wal_enabled is True

    # Env says 1 (True), explicit says False
    monkeypatch.setenv("RELATE_WAL_ENABLED", "1")
    with RelateSaver(db_path=":memory:", wal_enabled=False) as saver:
        assert saver.wal_enabled is False
        assert saver.proxy.wal_enabled is False


def test_relate_wal_enabled_parses_boolean_strings_properly_ac_178(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-178: RELATE_WAL_ENABLED correctly parses truthy and falsy string variations."""
    for truthy in ("1", "true", "yes", "on", "TRUE", "Yes", "On"):
        monkeypatch.setenv("RELATE_WAL_ENABLED", truthy)
        with RelateSaver(db_path=":memory:") as saver:
            assert saver.wal_enabled is True
            assert saver.proxy.wal_enabled is True

    for falsy in ("0", "false", "no", "off", "FALSE", "No", "Off"):
        monkeypatch.setenv("RELATE_WAL_ENABLED", falsy)
        with RelateSaver(db_path=":memory:") as saver:
            assert saver.wal_enabled is False
            assert saver.proxy.wal_enabled is False


def test_invalid_relate_wal_enabled_raises_value_error_ac_178(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-178: Invalid RELATE_WAL_ENABLED raises ValueError explicitly naming the variable."""
    for bad in ("invalid", "2", "yes_please", "", "  "):
        monkeypatch.setenv("RELATE_WAL_ENABLED", bad)
        with pytest.raises(ValueError, match="RELATE_WAL_ENABLED"):
            RelateSaver(db_path=":memory:")


def test_invalid_relate_proxy_timeout_sec_raises_value_error_ac_178(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-178: Invalid RELATE_PROXY_TIMEOUT_SEC raises ValueError explicitly naming the variable."""
    for bad in ("abc", "0", "-5", "-0.1", "nan", "inf"):
        monkeypatch.setenv("RELATE_PROXY_TIMEOUT_SEC", bad)
        with pytest.raises(ValueError, match="RELATE_PROXY_TIMEOUT_SEC"):
            RelateSaver(db_path=":memory:")

    # Valid timeout parses correctly
    monkeypatch.setenv("RELATE_PROXY_TIMEOUT_SEC", "12.5")
    with RelateSaver(db_path=":memory:") as saver:
        assert saver.default_timeout == 12.5
        assert saver.proxy.default_timeout == 12.5


def test_invalid_relate_log_level_raises_value_error_ac_178(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-178: Invalid RELATE_LOG_LEVEL raises ValueError explicitly naming the variable."""
    for bad in ("VERBOSE", "TRACE", "warn", "not_a_level", ""):
        monkeypatch.setenv("RELATE_LOG_LEVEL", bad)
        with pytest.raises(ValueError, match="RELATE_LOG_LEVEL"):
            RelateSaver(db_path=":memory:")

    # Valid log level parses case-insensitively
    monkeypatch.setenv("RELATE_LOG_LEVEL", "debug")
    with RelateSaver(db_path=":memory:") as saver:
        assert saver.log_level == "DEBUG"


_PACKAGE_LOGGER = "langgraph_checkpoint_relate"


@pytest.fixture
def clean_relate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Clear every RELATE_* variable and run in an empty working directory."""
    for name in (
        "RELATE_DB_PATH",
        "RELATE_WAL_ENABLED",
        "RELATE_PROXY_TIMEOUT_SEC",
        "RELATE_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def restore_package_logger() -> Iterator[logging.Logger]:
    """Save the package logger and saver module logger levels and restore them after the test."""
    package_logger = logging.getLogger(_PACKAGE_LOGGER)
    saver_logger = logging.getLogger(f"{_PACKAGE_LOGGER}.saver")
    saved = (package_logger.level, saver_logger.level)
    yield package_logger
    package_logger.setLevel(saved[0])
    saver_logger.setLevel(saved[1])


def test_ac194_db_path_none_uses_default_and_creates_no_file(clean_relate_env: Path) -> None:
    """AC-194: db_path=None counts as not passed, so no file named None is created."""
    with RelateSaver(db_path=None) as saver:
        assert saver.db_path == ":memory:"
    assert list(clean_relate_env.iterdir()) == []


def test_ac194_db_path_none_uses_env_path(
    clean_relate_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-194: db_path=None falls back to RELATE_DB_PATH."""
    env_path = str(clean_relate_env / "from_env.rl")
    monkeypatch.setenv("RELATE_DB_PATH", env_path)
    with RelateSaver(db_path=None) as saver:
        assert saver.db_path == env_path
    assert not (clean_relate_env / "None").exists()


def test_ac194_default_timeout_none_uses_default_or_env(
    clean_relate_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-194: default_timeout=None gives 30.0, or RELATE_PROXY_TIMEOUT_SEC when it is set."""
    with RelateSaver(db_path=":memory:", default_timeout=None) as saver:
        assert saver.default_timeout == 30.0
        assert saver.proxy.default_timeout == 30.0
    monkeypatch.setenv("RELATE_PROXY_TIMEOUT_SEC", "5")
    with RelateSaver(db_path=":memory:", default_timeout=None) as saver:
        assert saver.default_timeout == 5.0
        assert saver.proxy.default_timeout == 5.0


def test_ac194_wal_enabled_none_follows_env(
    clean_relate_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-194: wal_enabled=None follows RELATE_WAL_ENABLED, then defaults to True."""
    with RelateSaver(db_path=":memory:", wal_enabled=None) as saver:
        assert saver.wal_enabled is True
    monkeypatch.setenv("RELATE_WAL_ENABLED", "false")
    with RelateSaver(db_path=":memory:", wal_enabled=None) as saver:
        assert saver.wal_enabled is False


def test_ac194_log_level_none_follows_env(
    clean_relate_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_package_logger: logging.Logger,
) -> None:
    """AC-194: log_level=None follows RELATE_LOG_LEVEL, then reports INFO."""
    with RelateSaver(db_path=":memory:", log_level=None) as saver:
        assert saver.log_level == "INFO"
    monkeypatch.setenv("RELATE_LOG_LEVEL", "error")
    with RelateSaver(db_path=":memory:", log_level=None) as saver:
        assert saver.log_level == "ERROR"


@pytest.mark.parametrize(
    ("variable", "bad", "kwargs"),
    [
        ("RELATE_WAL_ENABLED", "maybe", {"wal_enabled": True}),
        ("RELATE_PROXY_TIMEOUT_SEC", "abc", {"default_timeout": 7.0}),
        ("RELATE_LOG_LEVEL", "verbose", {"log_level": "WARNING"}),
    ],
)
def test_ac194_invalid_env_raises_only_when_its_argument_is_not_passed(
    clean_relate_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_package_logger: logging.Logger,
    variable: str,
    bad: str,
    kwargs: dict[str, Any],
) -> None:
    """AC-194: an invalid RELATE_* value is parsed, and raises, only when its argument is unset."""
    monkeypatch.setenv(variable, bad)
    with RelateSaver(db_path=":memory:", **kwargs):
        pass
    with pytest.raises(ValueError, match=variable):
        RelateSaver(db_path=":memory:")
    for key in kwargs:
        none_kwargs: dict[str, Any] = {key: None}
        with pytest.raises(ValueError, match=variable):
            RelateSaver(db_path=":memory:", **none_kwargs)


def test_ac194_logger_level_untouched_unless_configured(
    clean_relate_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_package_logger: logging.Logger,
) -> None:
    """AC-194: the package logger level is set only by log_level or RELATE_LOG_LEVEL."""
    package_logger = restore_package_logger
    package_logger.setLevel(logging.DEBUG)

    with RelateSaver(db_path=":memory:") as saver:
        assert saver.log_level == "INFO"
        assert package_logger.level == logging.DEBUG

    with RelateSaver(db_path=":memory:", log_level="WARNING"):
        assert package_logger.level == logging.WARNING

    package_logger.setLevel(logging.DEBUG)
    monkeypatch.setenv("RELATE_LOG_LEVEL", "ERROR")
    with RelateSaver(db_path=":memory:"):
        assert package_logger.level == logging.ERROR


def test_put_writes_regular_channel_first_write_wins_ac_180() -> None:
    """AC-180: For regular channels, duplicate writes with the same task_id and index do not overwrite (first write wins)."""
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "thread-ac180-regular"
        checkpoint_id = "cp-ac180-regular"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        saver.put(config, _make_checkpoint(checkpoint_id), {"step": 1}, {})

        # First write to regular channel
        saver.put_writes(config, [("channel_a", "first_value")], task_id="task-1")

        # Second write with same task_id and index to same channel
        saver.put_writes(config, [("channel_a", "second_value")], task_id="task-1")

        tup = saver.get_tuple(config)
        assert tup is not None
        assert tup.pending_writes is not None
        assert len(tup.pending_writes) == 1
        task_id, channel, val = tup.pending_writes[0]
        assert task_id == "task-1"
        assert channel == "channel_a"
        assert val == "first_value"


def test_put_writes_special_channel_overwrites_earlier_value_ac_180() -> None:
    """AC-180: For special channels (WRITES_IDX_MAP), subsequent writes overwrite earlier values."""
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "thread-ac180-special"
        checkpoint_id = "cp-ac180-special"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        saver.put(config, _make_checkpoint(checkpoint_id), {"step": 1}, {})

        # Initial write to special channels
        saver.put_writes(
            config,
            [
                ("__error__", "initial_error"),
                ("__interrupt__", {"status": "paused_1"}),
            ],
            task_id="task-special",
        )

        # Overwrite with subsequent writes on the same special channels
        saver.put_writes(
            config,
            [
                ("__error__", "updated_error"),
                ("__interrupt__", {"status": "paused_2"}),
            ],
            task_id="task-special",
        )

        tup = saver.get_tuple(config)
        assert tup is not None
        assert tup.pending_writes is not None
        assert len(tup.pending_writes) == 2
        writes_dict = {ch: val for _, ch, val in tup.pending_writes}
        assert writes_dict["__error__"] == "updated_error"
        assert writes_dict["__interrupt__"] == {"status": "paused_2"}


def test_checkpoint_tuple_pending_writes_ordered_by_task_id_then_idx_ac_180() -> None:
    """AC-180: CheckpointTuple pending_writes are ordered deterministically by w.task_id, then w.idx."""
    with RelateSaver(db_path=":memory:") as saver:
        thread_id = "thread-ac180-order"
        checkpoint_id = "cp-ac180-order"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        saver.put(config, _make_checkpoint(checkpoint_id), {"step": 1}, {})

        # Insert writes out of task_id and idx order
        # task_z: idx 0 (ch_z), idx -1 (__error__)
        saver.put_writes(
            config,
            [("ch_z", "val_z"), ("__error__", "err_z")],
            task_id="task-z",
        )
        # task_m: idx 0 (ch_m)
        saver.put_writes(
            config,
            [("ch_m", "val_m")],
            task_id="task-m",
        )
        # task_a: idx 0 (ch_a1), idx 1 (ch_a2), idx -3 (__interrupt__)
        saver.put_writes(
            config,
            [("ch_a1", "val_a1"), ("ch_a2", "val_a2"), ("__interrupt__", "int_a")],
            task_id="task-a",
        )

        tup = saver.get_tuple(config)
        assert tup is not None
        assert tup.pending_writes is not None

        # Expected order:
        # task-a:
        #   idx -3: __interrupt__
        #   idx 0:  ch_a1
        #   idx 1:  ch_a2
        # task-m:
        #   idx 0:  ch_m
        # task-z:
        #   idx -1: __error__
        #   idx 0:  ch_z
        expected = [
            ("task-a", "__interrupt__", "int_a"),
            ("task-a", "ch_a1", "val_a1"),
            ("task-a", "ch_a2", "val_a2"),
            ("task-m", "ch_m", "val_m"),
            ("task-z", "__error__", "err_z"),
            ("task-z", "ch_z", "val_z"),
        ]
        assert tup.pending_writes == expected
