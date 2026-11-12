"""In-memory dictionary and graph fake driver mimicking relate-db for fast contract testing."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

try:
    from relate.exceptions import (
        DatabaseError,
        IntegrityError,
        InterfaceError,
        ProgrammingError,
        ResultNotSingleError,
    )
except ImportError:  # pragma: no cover
    class DatabaseError(Exception):  # type: ignore[no-redef]
        """Database error fallback."""

    class IntegrityError(DatabaseError):  # type: ignore[no-redef]
        """Integrity error fallback."""

    class InterfaceError(Exception):  # type: ignore[no-redef]
        """Interface error fallback."""

    class ProgrammingError(DatabaseError):  # type: ignore[no-redef]
        """Programming error fallback."""

    class ResultNotSingleError(ProgrammingError):  # type: ignore[no-redef]
        """Result not single error fallback."""

logger = logging.getLogger(__name__)


class FakeRecord(dict[str, Any]):
    """PEP 249 / Neo4j Record fake supporting dictionary and attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'FakeRecord' object has no attribute '{name}'") from None

    def data(self) -> dict[str, Any]:
        """Return raw data dictionary excluding internal fake metadata."""
        return {k: v for k, v in self.items() if not k.startswith("__")}


class FakeCursor:
    """Cursor fake supporting DB-API 2.0 and Relate-specific cursor methods."""

    def __init__(self, records: Sequence[Mapping[str, Any]] | None = None, connection: FakeConnection | None = None) -> None:
        self._records: list[FakeRecord] = [FakeRecord(r) for r in (records or [])]
        self._index: int = 0
        self._connection = connection
        self.is_closed: bool = False
        self.rowcount: int = len(self._records)

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> FakeCursor:
        """Execute a query through the parent connection or driver."""
        if self.is_closed:
            raise InterfaceError("Cannot execute on a closed cursor")
        if self._connection is not None:
            cur = self._connection.execute(query, parameters)
            self._records = cur._records
            self._index = 0
            self.rowcount = cur.rowcount
            return self
        return self

    def fetchone(self) -> FakeRecord | None:
        """Fetch the next row of a query result set."""
        if self.is_closed:
            raise InterfaceError("Cannot fetch from a closed cursor")
        if self._index < len(self._records):
            rec = self._records[self._index]
            self._index += 1
            return rec
        return None

    def fetchall(self) -> list[FakeRecord]:
        """Fetch all remaining rows of a query result set."""
        if self.is_closed:
            raise InterfaceError("Cannot fetch from a closed cursor")
        remaining = self._records[self._index :]
        self._index = len(self._records)
        return remaining

    def single(self) -> FakeRecord:
        """Return the single record, raising ResultNotSingleError if count != 1."""
        if self.is_closed:
            raise InterfaceError("Cannot fetch from a closed cursor")
        if len(self._records) != 1:
            raise ResultNotSingleError(
                f"expected exactly one row, got {len(self._records)}"
            )
        return self._records[0]

    def peek(self) -> FakeRecord | None:
        """Return the next row without advancing the cursor."""
        if self.is_closed:
            raise InterfaceError("Cannot peek a closed cursor")
        if self._index < len(self._records):
            return self._records[self._index]
        return None

    def close(self) -> None:
        """Close the cursor."""
        self.is_closed = True

    def __iter__(self) -> Iterator[FakeRecord]:
        if self.is_closed:
            raise InterfaceError("Cannot iterate over a closed cursor")
        while self._index < len(self._records):
            yield self._records[self._index]
            self._index += 1

    def __len__(self) -> int:
        return len(self._records)


class FakeConnection:
    """Database connection fake conforming to relate.Connection."""

    def __init__(self, driver: FakeRelateDriver, autocommit: bool = True) -> None:
        self.driver = driver
        self.autocommit = autocommit
        self.is_closed = False

    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> FakeCursor:
        """Execute a Cypher query against the fake driver."""
        if self.is_closed:
            raise InterfaceError("Cannot execute on a closed connection")
        return self.driver.execute(query, parameters, connection=self)

    def cursor(self) -> FakeCursor:
        """Create a new cursor associated with this connection."""
        if self.is_closed:
            raise InterfaceError("Cannot create cursor on a closed connection")
        return FakeCursor(connection=self)

    def commit(self) -> None:
        """Commit current transaction."""
        if self.is_closed:
            raise InterfaceError("Cannot commit a closed connection")

    def rollback(self) -> None:
        """Roll back current transaction."""
        if self.is_closed:
            raise InterfaceError("Cannot rollback a closed connection")

    def close(self) -> None:
        """Close connection."""
        self.is_closed = True


class FakeRelateDriver:
    """In-memory dictionary and graph engine fake mimicking Relate DB."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.constraints: set[str] = set()
        self.indexes: set[str] = set()
        self.unique_rules: dict[str, str] = {}  # label -> property
        self.executed_queries: list[tuple[str, dict[str, Any] | None]] = []
        self._handlers: list[
            tuple[re.Pattern[str], Callable[[re.Match[str], dict[str, Any] | None], list[dict[str, Any]]]]
        ] = []

    def connect(self, database: str = ":memory:", **kwargs: Any) -> FakeConnection:
        """Create a new FakeConnection bound to this driver."""
        return FakeConnection(driver=self)

    def reset(self) -> None:
        """Reset all in-memory graph state, constraints, and execution logs."""
        self.nodes.clear()
        self.edges.clear()
        self.constraints.clear()
        self.indexes.clear()
        self.unique_rules.clear()
        self.executed_queries.clear()

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[FakeRecord]:
        """Execute query and return list of FakeRecord dictionaries."""
        cursor = self.execute(query, parameters)
        return cursor.fetchall()

    def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        connection: FakeConnection | None = None,
    ) -> FakeCursor:
        """Execute a Cypher query with parameter binding, enforcing constraints."""
        clean_query = query.strip()
        self.executed_queries.append((clean_query, parameters))

        # Check custom handlers first
        for pattern, handler in self._handlers:
            match = pattern.search(clean_query)
            if match:
                rows = handler(match, parameters)
                return FakeCursor(rows, connection=connection)

        # 1. Handle CREATE CONSTRAINT
        constraint_match = re.search(
            r"CREATE\s+CONSTRAINT\s+(\w+)\s+(?:IF\s+NOT\s+EXISTS\s+)?FOR\s+\(\w+:(\w+)\)\s+REQUIRE\s+\w+\.(\w+)\s+IS\s+UNIQUE",
            clean_query,
            re.IGNORECASE,
        )
        if constraint_match:
            c_name, label, prop = constraint_match.groups()
            if c_name in self.constraints and "IF NOT EXISTS" not in clean_query.upper():
                raise IntegrityError(f"Constraint `{c_name}` already exists")
            self.constraints.add(c_name)
            self.unique_rules[label] = prop
            return FakeCursor([], connection=connection)

        # 2. Handle CREATE INDEX
        index_match = re.search(
            r"CREATE\s+INDEX\s+(\w+)\s+(?:IF\s+NOT\s+EXISTS\s+)?FOR\s+\(\w+:(\w+)\)\s+ON\s+\(([^)]+)\)",
            clean_query,
            re.IGNORECASE,
        )
        if index_match:
            idx_name, label, cols = index_match.groups()
            if idx_name in self.indexes and "IF NOT EXISTS" not in clean_query.upper():
                raise DatabaseError(f"Index `{idx_name}` already exists")
            self.indexes.add(idx_name)
            return FakeCursor([], connection=connection)

        # 3. Handle CREATE (:Label {props}) or MERGE
        create_node_match = re.search(
            r"(?:CREATE|MERGE)\s+\(\s*(?:\w+)?\s*:(\w+)\s*(?:\{([^}]+)\})?\s*\)",
            clean_query,
            re.IGNORECASE,
        )
        if create_node_match:
            label, props_str = create_node_match.groups()
            resolved_props: dict[str, Any] = {}
            if props_str:
                resolved_props = self._parse_and_bind_properties(props_str, parameters)

            # Check uniqueness constraints
            if label in self.unique_rules:
                unique_prop = self.unique_rules[label]
                unique_val = resolved_props.get(unique_prop)
                if unique_val is not None:
                    for existing_node in self.nodes.values():
                        if (
                            existing_node.get("__label__") == label
                            and existing_node.get(unique_prop) == unique_val
                        ):
                            raise IntegrityError(
                                f"ConstraintValidationFailed: `{label}_{unique_prop}_unique` uniqueness constraint violated for {unique_val}"
                            )

            node_key = f"{label}:{resolved_props.get(self.unique_rules.get(label, 'id'), len(self.nodes))}"
            node_record = {"__label__": label, **resolved_props}
            self.nodes[node_key] = node_record

            # Check if there is also an edge creation in the same query
            edge_match = re.search(r"-\[:(\w+)\]->", clean_query)
            if edge_match:
                rel_type = edge_match.group(1)
                self.edges.append((node_key, rel_type, ""))

            return FakeCursor([], connection=connection)

        # 4. Handle MATCH (n:Label {match_props}) RETURN ...
        match_node = re.search(
            r"MATCH\s+\((\w+):(\w+)(?:\s*\{([^}]+)\})?\)\s+RETURN\s+([^;]+)",
            clean_query,
            re.IGNORECASE,
        )
        if match_node:
            var_name, label, match_props_str, return_expr = match_node.groups()
            match_props = self._parse_and_bind_properties(match_props_str, parameters) if match_props_str else {}

            matching_nodes = []
            for n in self.nodes.values():
                if n.get("__label__") != label:
                    continue
                matches = True
                for k, v in match_props.items():
                    if n.get(k) != v:
                        matches = False
                        break
                if matches:
                    matching_nodes.append(n)

            # Return records
            records: list[dict[str, Any]] = []
            return_expr_clean = return_expr.strip()
            for n in matching_nodes:
                clean_n = {k: v for k, v in n.items() if not k.startswith("__")}
                if return_expr_clean == var_name:
                    records.append(clean_n)
                elif " AS " in return_expr_clean.upper():
                    # e.g., p.name AS name, p.age AS age
                    rec: dict[str, Any] = {}
                    parts = [p.strip() for p in return_expr_clean.split(",")]
                    for part in parts:
                        subparts = re.split(r"\s+AS\s+", part, flags=re.IGNORECASE)
                        if len(subparts) == 2:
                            prop_expr, alias = subparts[0].strip(), subparts[1].strip()
                            prop_key = prop_expr.split(".", 1)[-1] if "." in prop_expr else prop_expr
                            rec[alias] = clean_n.get(prop_key)
                        else:
                            prop_key = part.split(".", 1)[-1] if "." in part else part
                            rec[prop_key] = clean_n.get(prop_key)
                    records.append(rec)
                else:
                    records.append(clean_n)

            return FakeCursor(records, connection=connection)

        # Default fallback: return empty cursor
        return FakeCursor([], connection=connection)

    def _parse_and_bind_properties(self, props_str: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
        """Parse property string key: $param or key: literal, and bind parameters."""
        resolved: dict[str, Any] = {}
        pairs = [p.strip() for p in props_str.split(",") if p.strip()]
        for pair in pairs:
            if ":" not in pair:
                continue
            k, raw_v = [x.strip() for x in pair.split(":", 1)]
            if raw_v.startswith("$"):
                param_name = raw_v[1:]
                if parameters and param_name in parameters:
                    resolved[k] = parameters[param_name]
                else:
                    resolved[k] = None
            elif (raw_v.startswith("'") and raw_v.endswith("'")) or (raw_v.startswith('"') and raw_v.endswith('"')):
                resolved[k] = raw_v[1:-1]
            elif raw_v.isdigit():
                resolved[k] = int(raw_v)
            else:
                resolved[k] = raw_v
        return resolved


# Specification parity alias
InMemoryRelateDriver = FakeRelateDriver

__all__ = [
    "FakeConnection",
    "FakeCursor",
    "FakeRecord",
    "FakeRelateDriver",
    "InMemoryRelateDriver",
]
