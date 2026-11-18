"""Exception hierarchy for Relate checkpoint saver operations."""

from typing import Any


class RelateCheckpointError(Exception):
    """Base exception for all Relate checkpoint errors."""

    def __init__(self, message: str = "", *args: Any) -> None:
        if args:
            super().__init__(message, *args)
        elif message:
            super().__init__(message)
        else:
            super().__init__()
        self.message = message


class RelateConnectionError(RelateCheckpointError):
    """Raised when establishing, accessing, or maintaining a connection to Relate DB fails."""


class RelateSerializationError(RelateCheckpointError):
    """Raised when checkpoint serialization or deserialization fails."""


class RelateConcurrencyError(RelateCheckpointError):
    """Raised when concurrent operations collide, deadlock, or encounter write contention."""


class RelateCheckpointNotFoundError(RelateCheckpointError):
    """Raised when a specified thread or checkpoint cannot be located."""


class RelateReadOnlyError(RelateCheckpointError):
    """Raised when a write is attempted through a read-only saver or proxy."""


# Specification parity aliases
CheckpointSerializationError = RelateSerializationError
CheckpointNotFoundError = RelateCheckpointNotFoundError

__all__ = [
    "CheckpointNotFoundError",
    "CheckpointSerializationError",
    "RelateCheckpointError",
    "RelateCheckpointNotFoundError",
    "RelateConcurrencyError",
    "RelateConnectionError",
    "RelateReadOnlyError",
    "RelateSerializationError",
]
