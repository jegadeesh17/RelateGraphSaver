"""Tool call extraction and Cypher statement building (ADR-0005).

Pure functions with no database access. ``extract_tool_events`` finds tool calls made by
the model and tool results in values written to a checkpointer; ``build_tool_call_statements``
turns them into idempotent ``:ToolCall`` Cypher statements for the caller to execute.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

logger = logging.getLogger(__name__)

EMPTY_ERROR_TEXT = "Tool call returned status 'error' with no message"

_INVOCATION_CYPHER = (
    "MERGE (tc:ToolCall {call_id: $call_id})\n"
    "ON CREATE SET tc.tool_name = $tool_name, tc.args_blob = $args_blob, "
    "tc.created_at = $created_at\n"
    "WITH tc\n"
    "OPTIONAL MATCH (c:Checkpoint {checkpoint_id: $checkpoint_id})\n"
    "OPTIONAL MATCH (linked:Checkpoint {thread_id: $thread_id})-[:INVOKED]->(tc)\n"
    "WITH tc, c, count(linked) AS linked_count\n"
    "FOREACH (_ IN CASE WHEN c IS NOT NULL AND linked_count = 0 THEN [1] ELSE [] END |\n"
    "  MERGE (c)-[:INVOKED]->(tc))"
)

_SUCCESS_OUTCOME_CYPHER = (
    "MATCH (tc:ToolCall {call_id: $call_id})\n"
    "WHERE tc.result_blob IS NULL AND tc.error IS NULL\n"
    "SET tc.result_blob = $result_blob"
)

_ERROR_OUTCOME_CYPHER = (
    "MATCH (tc:ToolCall {call_id: $call_id})\n"
    "WHERE tc.result_blob IS NULL AND tc.error IS NULL\n"
    "SET tc.error = $error"
)


class BlobSerializer(Protocol):
    """Anything that can turn a Python object into bytes."""

    def dumps(self, obj: Any) -> bytes: ...


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call requested by the model."""

    call_id: str
    tool_name: str
    args: Any


@dataclass(frozen=True)
class ToolOutcome:
    """The result (or error) of one tool call."""

    call_id: str
    is_error: bool
    content: Any


@dataclass(frozen=True)
class ToolEvents:
    """Tool calls and tool results found in one set of written values."""

    invocations: tuple[ToolInvocation, ...]
    outcomes: tuple[ToolOutcome, ...]


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def _messages_in(value: Any) -> list[Any]:
    """Return the messages a single written value contributes (no recursion)."""
    if isinstance(value, BaseMessage):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, BaseMessage)]
    return []


def _invocations_from(message: AIMessage) -> list[ToolInvocation]:
    found: list[ToolInvocation] = []
    for entry in message.tool_calls:
        try:
            call_id: Any = entry["id"]
            tool_name: Any = entry["name"]
            if not (_non_empty_str(call_id) and _non_empty_str(tool_name)):
                continue
            found.append(ToolInvocation(call_id, tool_name, entry.get("args", {})))
        except Exception as exc:
            logger.warning("Skipping malformed tool call entry (%s)", type(exc).__name__)
    return found


def _outcome_from(message: ToolMessage) -> ToolOutcome | None:
    call_id = message.tool_call_id
    if not _non_empty_str(call_id):
        return None
    return ToolOutcome(
        call_id=call_id,
        is_error=message.status == "error",
        content=message.content,
    )


def extract_tool_events(values: Iterable[Any]) -> ToolEvents:
    """Collect tool calls and tool results from written values. Never raises.

    A ``BaseMessage`` value is a one-message list; ``list``/``tuple`` values contribute
    their ``BaseMessage`` elements; anything else contributes nothing. Malformed entries are
    skipped. Duplicates by ``call_id`` are dropped, keeping the first invocation and the first
    outcome.
    """
    invocations: dict[str, ToolInvocation] = {}
    outcomes: dict[str, ToolOutcome] = {}
    try:
        iterator = iter(values)
    except Exception as exc:
        logger.warning("Tool call extraction received non-iterable values (%s)", type(exc).__name__)
        return ToolEvents((), ())

    while True:
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            logger.warning("Tool call extraction stopped early (%s)", type(exc).__name__)
            break
        try:
            messages = _messages_in(value)
        except Exception as exc:
            logger.warning("Skipping unreadable value (%s)", type(exc).__name__)
            continue
        for message in messages:
            try:
                if isinstance(message, AIMessage):
                    for invocation in _invocations_from(message):
                        invocations.setdefault(invocation.call_id, invocation)
                elif isinstance(message, ToolMessage):
                    outcome = _outcome_from(message)
                    if outcome is not None:
                        outcomes.setdefault(outcome.call_id, outcome)
            except Exception as exc:
                logger.warning("Skipping unreadable message (%s)", type(exc).__name__)
    return ToolEvents(tuple(invocations.values()), tuple(outcomes.values()))


def _error_text(content: Any) -> str:
    """Render tool error content as text; never empty."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        text = "\n".join(parts)
    else:
        text = str(content)
    return text if text else EMPTY_ERROR_TEXT


def build_tool_call_statements(
    events: ToolEvents,
    *,
    thread_id: str,
    checkpoint_id: str,
    serde: BlobSerializer,
    created_at: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Build Cypher statements: invocation statements first, then outcome statements.

    An entry whose serialization fails is skipped; the others are still built.
    """
    statements: list[tuple[str, dict[str, Any]]] = []
    for invocation in events.invocations:
        try:
            args_blob = serde.dumps(invocation.args)
        except Exception as exc:
            logger.warning("Skipping tool call arguments (%s)", type(exc).__name__)
            continue
        statements.append(
            (
                _INVOCATION_CYPHER,
                {
                    "call_id": invocation.call_id,
                    "tool_name": invocation.tool_name,
                    "args_blob": args_blob,
                    "created_at": created_at,
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                },
            )
        )
    for outcome in events.outcomes:
        try:
            if outcome.is_error:
                statements.append(
                    (
                        _ERROR_OUTCOME_CYPHER,
                        {"call_id": outcome.call_id, "error": _error_text(outcome.content)},
                    )
                )
            else:
                statements.append(
                    (
                        _SUCCESS_OUTCOME_CYPHER,
                        {
                            "call_id": outcome.call_id,
                            "result_blob": serde.dumps(outcome.content),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("Skipping tool call outcome (%s)", type(exc).__name__)
    return statements
