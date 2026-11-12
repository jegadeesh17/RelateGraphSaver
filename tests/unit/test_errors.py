"""Unit tests for Relate checkpoint exception taxonomy (AC-01)."""

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
