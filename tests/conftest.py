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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect ``network`` tests unless the ``-m`` expression names the marker (AC-131).

    A default ``-m`` in ``addopts`` would be replaced by ``-m "not slow"``, so the marker is
    filtered here instead.
    """
    if "network" in str(config.getoption("markexpr", default="") or ""):
        return
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        (deselected if item.get_closest_marker("network") else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
