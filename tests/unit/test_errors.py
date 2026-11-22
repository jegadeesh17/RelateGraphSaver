"""Unit tests for Relate checkpoint exception taxonomy (AC-01)."""

import concurrent.futures

import pytest

from langgraph_checkpoint_relate.errors import (
    CheckpointNotFoundError,
    CheckpointSerializationError,
    RelateCheckpointError,
    RelateCheckpointNotFoundError,
    RelateConcurrencyError,
    RelateConnectionError,
    RelateSerializationError,
)
from langgraph_checkpoint_relate.proxy import _wrap_exception

SUBCLASS_TYPES = [
    RelateConnectionError,
    RelateSerializationError,
    RelateConcurrencyError,
    RelateCheckpointNotFoundError,
]


def test_base_exception_inheritance_ac_01() -> None:
    """AC-01: Verify RelateCheckpointError inherits from standard Python Exception."""
    assert issubclass(RelateCheckpointError, Exception)
    error = RelateCheckpointError("Base error occurred")
    assert isinstance(error, Exception)
    assert str(error) == "Base error occurred"
    assert error.message == "Base error occurred"


@pytest.mark.parametrize("exc_cls", SUBCLASS_TYPES)
def test_subclass_inheritance_ac_01(exc_cls: type[RelateCheckpointError]) -> None:
    """AC-01: Verify all specialized checkpoint errors inherit from RelateCheckpointError."""
    assert issubclass(exc_cls, RelateCheckpointError)
    instance = exc_cls("Operation failed")
    assert isinstance(instance, RelateCheckpointError)
    assert isinstance(instance, Exception)


def test_specification_aliases_ac_01() -> None:
    """AC-01: Verify specification aliases resolve to canonical exception classes."""
    assert CheckpointSerializationError is RelateSerializationError
    assert CheckpointNotFoundError is RelateCheckpointNotFoundError

    serialization_err = CheckpointSerializationError("Serialization failure")
    assert isinstance(serialization_err, RelateSerializationError)

    not_found_err = CheckpointNotFoundError("Checkpoint not found")
    assert isinstance(not_found_err, RelateCheckpointNotFoundError)


def test_error_message_and_str_representation_ac_01(sample_error_message: str) -> None:
    """AC-01: Verify string representation preserves exact error messaging across all types."""
    for exc_cls in [RelateCheckpointError, *SUBCLASS_TYPES]:
        err = exc_cls(sample_error_message)
        assert str(err) == sample_error_message
        assert err.message == sample_error_message


def test_error_with_multiple_args_ac_01() -> None:
    """AC-01: Verify exception initialization with multiple arguments preserves standard behavior."""
    err = RelateSerializationError("Failed serialization", "state_key: agent_summary", 400)
    assert err.message == "Failed serialization"
    assert "Failed serialization" in str(err)
    assert "state_key: agent_summary" in str(err)


def test_error_empty_and_default_input_ac_01() -> None:
    """AC-01: Verify exception handling for empty/default inputs at boundaries."""
    # Empty string input
    empty_err = RelateCheckpointError("")
    assert empty_err.message == ""
    assert str(empty_err) == ""

    # No argument input
    default_err = RelateCheckpointError()
    assert default_err.message == ""
    assert str(default_err) == ""

    # Subclass default input
    conn_err = RelateConnectionError()
    assert conn_err.message == ""
    assert str(conn_err) == ""


def test_error_exception_chaining_ac_01() -> None:
    """AC-01: Verify exception chaining preserves root cause information."""
    original_cause = TypeError("Object of type socket is not JSON serializable")
    try:
        try:
            raise original_cause
        except TypeError as cause:
            raise RelateSerializationError("Serialization failed for state") from cause
    except RelateSerializationError as caught:
        assert caught.__cause__ is original_cause
        assert "Serialization failed" in str(caught)
        assert caught.__cause__ is not None
        assert "socket is not JSON serializable" in str(caught.__cause__)


def test_polymorphic_catch_ac_01() -> None:
    """AC-01: Verify all subclasses can be caught polymorphically with RelateCheckpointError."""
    for exc_cls in SUBCLASS_TYPES:
        with pytest.raises(RelateCheckpointError) as exc_info:
            raise exc_cls("Dispatched error")
        assert isinstance(exc_info.value, RelateCheckpointError)
        assert exc_info.value.message == "Dispatched error"


def test_wrap_exception_word_boundary_avoids_false_positive_concurrency_ac_01() -> None:
    """AC-01: Substring matches like 'unlocked', 'clock', 'block', or 'timeout_ms' do not trigger concurrency errors."""
    false_match_messages = [
        "node state is unlocked",
        "system clock drift detected",
        "failed to parse block statement",
        "invalid parameter timeout_ms provided",
    ]
    for msg in false_match_messages:
        exc = RuntimeError(msg)
        wrapped = _wrap_exception(exc)
        assert isinstance(wrapped, RelateCheckpointError)
        assert not isinstance(wrapped, RelateConcurrencyError)
        assert not isinstance(wrapped, RelateConnectionError)
        assert wrapped.__cause__ is exc
        assert msg in str(wrapped)


def test_wrap_exception_accurate_concurrency_classification_ac_01() -> None:
    """AC-01: Accurate classification of concurrency and lock errors using word-boundary regex."""
    concurrency_messages = [
        "database is locked",
        "write lock contention",
        "query timed out",
        "resource deadlock detected",
        "driver busy with another query",
        "hash collision detected in worker",
    ]
    for msg in concurrency_messages:
        exc = RuntimeError(msg)
        wrapped = _wrap_exception(exc)
        assert isinstance(wrapped, RelateConcurrencyError)
        assert wrapped.__cause__ is exc
        assert msg in str(wrapped)

    std_timeout = TimeoutError("socket timeout")
    wrapped_std = _wrap_exception(std_timeout)
    assert isinstance(wrapped_std, RelateConcurrencyError)
    assert wrapped_std.__cause__ is std_timeout

    fut_timeout = concurrent.futures.TimeoutError()
    wrapped_fut = _wrap_exception(fut_timeout)
    assert isinstance(wrapped_fut, RelateConcurrencyError)
    assert wrapped_fut.__cause__ is fut_timeout


def test_wrap_exception_accurate_connection_classification_ac_01() -> None:
    """AC-01: Accurate classification of connection errors using word-boundary regex and class names."""
    connection_messages = [
        "connection closed",
        "active connection reset by peer",
        "relate database connection terminated",
    ]
    for msg in connection_messages:
        exc = RuntimeError(msg)
        wrapped = _wrap_exception(exc)
        assert isinstance(wrapped, RelateConnectionError)
        assert wrapped.__cause__ is exc
        assert msg in str(wrapped)

    class InterfaceError(Exception):
        pass

    iface_err = InterfaceError("Underlying driver interface error")
    wrapped_iface = _wrap_exception(iface_err)
    assert isinstance(wrapped_iface, RelateConnectionError)
    assert wrapped_iface.__cause__ is iface_err

    # Substrings like 'unclosed' or 'interconnection' must not trigger connection error
    for non_conn_msg in ["unclosed file descriptor", "interconnection routing"]:
        exc = RuntimeError(non_conn_msg)
        wrapped_non_conn = _wrap_exception(exc)
        assert not isinstance(wrapped_non_conn, RelateConnectionError)

