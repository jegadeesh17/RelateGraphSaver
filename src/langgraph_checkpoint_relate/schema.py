"""Cypher DDL constraints, indexes, and schema initialization logic for Relate."""

from __future__ import annotations

import logging
from typing import Any

from langgraph_checkpoint_relate.errors import RelateConnectionError

logger = logging.getLogger(__name__)

# Constraint DDL statements
CONSTRAINT_THREAD_UNIQUE: str = (
    "CREATE CONSTRAINT thread_id_unique IF NOT EXISTS\n"
    "FOR (t:Thread) REQUIRE t.thread_id IS UNIQUE;"
)

CONSTRAINT_CHECKPOINT_UNIQUE: str = (
    "CREATE CONSTRAINT checkpoint_id_unique IF NOT EXISTS\n"
    "FOR (c:Checkpoint) REQUIRE c.checkpoint_id IS UNIQUE;"
)

CONSTRAINT_WRITE_UNIQUE: str = (
    "CREATE CONSTRAINT write_id_unique IF NOT EXISTS\n"
    "FOR (w:Write) REQUIRE w.write_id IS UNIQUE;"
)

CONSTRAINT_TOOL_CALL_UNIQUE: str = (
    "CREATE CONSTRAINT tool_call_id_unique IF NOT EXISTS\n"
    "FOR (tc:ToolCall) REQUIRE tc.call_id IS UNIQUE;"
)

# Index DDL statements
INDEX_CHECKPOINT_THREAD: str = (
    "CREATE INDEX checkpoint_thread_idx IF NOT EXISTS\n"
    "FOR (c:Checkpoint) ON (c.thread_id, c.checkpoint_ns, c.step);"
)

INDEX_CHECKPOINT_HASH: str = (
    "CREATE INDEX checkpoint_hash_idx IF NOT EXISTS\n"
    "FOR (c:Checkpoint) ON (c.state_hash);"
)

SCHEMA_CONSTRAINTS: tuple[str, ...] = (
    CONSTRAINT_THREAD_UNIQUE,
    CONSTRAINT_CHECKPOINT_UNIQUE,
    CONSTRAINT_WRITE_UNIQUE,
    CONSTRAINT_TOOL_CALL_UNIQUE,
)

SCHEMA_INDEXES: tuple[str, ...] = (
    INDEX_CHECKPOINT_THREAD,
    INDEX_CHECKPOINT_HASH,
)

ALL_SCHEMA_STATEMENTS: tuple[str, ...] = SCHEMA_CONSTRAINTS + SCHEMA_INDEXES


def initialize_schema(conn: Any) -> None:
    """Initialize Relate graph database schema with constraints and indexes.

    Executes all uniqueness constraints and performance indexes idempotently
    against the target Relate connection.

    Args:
        conn: A Relate Connection or compatible database connection object
            supporting the PEP 249 execute() interface.

    Raises:
        RelateConnectionError: If connection is None, invalid, or query execution fails.
    """
    if conn is None or not hasattr(conn, "execute"):
        raise RelateConnectionError(
            "Valid database connection required to initialize schema (got None or missing execute)"
        )

    try:
        for ddl_statement in ALL_SCHEMA_STATEMENTS:
            conn.execute(ddl_statement)

        if hasattr(conn, "commit") and not getattr(conn, "autocommit", True):
            conn.commit()

        logger.debug("Relate graph schema successfully initialized (%d statements)", len(ALL_SCHEMA_STATEMENTS))
    except RelateConnectionError:
        raise
    except Exception as exc:
        raise RelateConnectionError(f"Failed to execute schema initialization statement: {exc}") from exc


__all__ = [
    "ALL_SCHEMA_STATEMENTS",
    "CONSTRAINT_CHECKPOINT_UNIQUE",
    "CONSTRAINT_THREAD_UNIQUE",
    "CONSTRAINT_TOOL_CALL_UNIQUE",
    "CONSTRAINT_WRITE_UNIQUE",
    "INDEX_CHECKPOINT_HASH",
    "INDEX_CHECKPOINT_THREAD",
    "SCHEMA_CONSTRAINTS",
    "SCHEMA_INDEXES",
    "initialize_schema",
]
