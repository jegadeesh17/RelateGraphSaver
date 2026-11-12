"""Shared test fixtures for Relate checkpointer test suite."""

import pytest


@pytest.fixture
def sample_thread_id() -> str:
    """Provide a standard thread identifier for testing."""
    return "test-thread-001"


@pytest.fixture
def sample_checkpoint_id() -> str:
    """Provide a standard checkpoint identifier for testing."""
    return "1ef82a3c-b034-6c2e-8000-000000000001"


@pytest.fixture
def sample_error_message() -> str:
    """Provide a standard error message string for exception tests."""
    return "Database connection failed unexpectedly"
