"""Replay a stored run through the user's graph in an in-memory sandbox (ADR-0017).

``load_graph_factory`` turns a ``module:attribute`` or ``path.py:attribute`` spec into a graph
factory (AC-94..AC-97). ``replay_from_checkpoint`` forks the stored thread into memory with
``TimeTravelEngine.fork_in_memory``, optionally edits state, resumes the graph there and
compares the result with the recorded run (AC-98..AC-103, AC-107). The source store is only
read. ``snapshot_run`` and ``diff_runs`` produce and compare the comparable content of a run
(AC-104..AC-106).
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel import Pregel

from langgraph_checkpoint_relate.errors import (
    CheckpointNotFoundError,
    GraphSpecError,
    ReplayInputError,
)
from langgraph_checkpoint_relate.models import (
    ChannelChange,
    ReplayResult,
    RunDiff,
    RunSnapshot,
    ToolCallChange,
    ToolCallRecord,
)
from langgraph_checkpoint_relate.time_travel import TimeTravelEngine
from langgraph_checkpoint_relate.tool_calls import _error_text, extract_tool_events

if TYPE_CHECKING:
    from langgraph_checkpoint_relate.saver import RelateSaver

logger = logging.getLogger(__name__)

GraphFactory = Callable[..., Any] | StateGraph[Any, Any, Any, Any]

_DETAIL_LIMIT = 120
_MESSAGE_LIMIT = 120


# ---------------------------------------------------------------------------
# Graph spec loading (AC-94..AC-97)
# ---------------------------------------------------------------------------


def _is_file_spec(module_part: str) -> bool:
    return module_part.endswith(".py") or "/" in module_part or "\\" in module_part


def _load_module_from_file(spec: str, module_part: str) -> Any:
    path = Path(module_part)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise GraphSpecError(f"Graph spec '{spec}': file '{resolved}' does not exist")

    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    module_name = f"_relate_replay_{digest}"
    module_spec = importlib.util.spec_from_file_location(module_name, resolved)
    if module_spec is None or module_spec.loader is None:
        raise GraphSpecError(f"Graph spec '{spec}': '{resolved}' is not a loadable Python file")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise GraphSpecError(
            f"Graph spec '{spec}': importing '{resolved}' raised {type(exc).__name__}: {exc}"
        ) from exc
    return module


def load_graph_factory(spec: str) -> GraphFactory:
    """Load the graph factory or uncompiled ``StateGraph`` named by ``spec``.

    The spec is ``module:attribute`` or ``path/to/file.py:attribute``, split on the last ``:``.
    A file path is resolved against the current directory and loaded without changing
    ``sys.path``. The attribute must be an uncompiled ``StateGraph`` or a callable that is not
    a compiled graph (a compiled graph carries its own checkpointer).

    Raises:
        GraphSpecError: For every failure; the message contains the spec and the error is
            chained from the underlying cause when there is one.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise GraphSpecError(
            f"Graph spec '{spec}' must look like 'module:attribute' or 'path/to/file.py:attribute'"
        )
    module_part, _, attribute = spec.rpartition(":")
    if not module_part.strip() or not attribute.strip():
        raise GraphSpecError(
            f"Graph spec '{spec}' must have a non-empty module or file path and attribute name"
        )
    module_part = module_part.strip()
    attribute = attribute.strip()

    if _is_file_spec(module_part):
        module = _load_module_from_file(spec, module_part)
    else:
        try:
            module = importlib.import_module(module_part)
        except Exception as exc:
            raise GraphSpecError(
                f"Graph spec '{spec}': importing module '{module_part}' failed with "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    try:
        candidate = getattr(module, attribute)
    except AttributeError as exc:
        raise GraphSpecError(
            f"Graph spec '{spec}': '{module_part}' has no attribute '{attribute}'"
        ) from exc

    if isinstance(candidate, Pregel):
        raise GraphSpecError(
            f"Graph spec '{spec}': '{attribute}' is an already compiled graph, which carries its "
            "own checkpointer; point to a factory that accepts checkpointer= or to an "
            "uncompiled StateGraph"
        )
    if isinstance(candidate, StateGraph) or callable(candidate):
        return candidate  # type: ignore[no-any-return]
    raise GraphSpecError(
        f"Graph spec '{spec}': '{attribute}' is a {type(candidate).__name__}, expected a graph "
        "factory or an uncompiled StateGraph"
    )


# ---------------------------------------------------------------------------
# Snapshot and diff (AC-104..AC-106)
# ---------------------------------------------------------------------------


def _is_ignored_channel(name: str) -> bool:
    return name.startswith("__") or name.startswith("branch:")


def _build_tool_calls(messages: tuple[Any, ...]) -> tuple[ToolCallRecord, ...]:
    events = extract_tool_events(messages)
    outcomes = {outcome.call_id: outcome for outcome in events.outcomes}
    ordinals: dict[str, int] = {}
    records: list[ToolCallRecord] = []
    for invocation in events.invocations:
        ordinal = ordinals.get(invocation.tool_name, 0) + 1
        ordinals[invocation.tool_name] = ordinal
        outcome = outcomes.get(invocation.call_id)
        if outcome is None:
            status = "no result"
            detail = None
        elif outcome.is_error:
            status = "error"
            detail = _error_text(outcome.content)[:_DETAIL_LIMIT]
        else:
            status = "ok"
            detail = str(outcome.content)[:_DETAIL_LIMIT]
        records.append(
            ToolCallRecord(
                tool_name=invocation.tool_name,
                ordinal=ordinal,
                args=invocation.args,
                status=status,
                call_id=invocation.call_id,
                detail=detail,
            )
        )
    return tuple(records)


def snapshot_run(saver: RelateSaver, thread_id: str, checkpoint_id: str) -> RunSnapshot:
    """Read the comparable content of one checkpoint: messages, other channels, tool calls.

    Tool calls are derived from the messages (AC-104); calls trimmed from the message history
    are absent.

    Raises:
        CheckpointNotFoundError: If the checkpoint does not exist for the thread.
    """
    found = saver.get_tuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }
    )
    if found is None:
        raise CheckpointNotFoundError(
            f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'"
        )
    values = found.checkpoint.get("channel_values") or {}
    raw_messages = values.get("messages")
    messages: tuple[Any, ...] = (
        tuple(raw_messages) if isinstance(raw_messages, (list, tuple)) else ()
    )
    channels = {
        name: value
        for name, value in values.items()
        if name != "messages" and not _is_ignored_channel(name)
    }
    metadata = found.metadata or {}
    step_value = metadata.get("step", 0)
    step = step_value if isinstance(step_value, int) else 0
    return RunSnapshot(
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
        step=step,
        messages=messages,
        channels=channels,
        tool_calls=_build_tool_calls(messages),
    )


# ``(type, content, ((tool name, args), ...))``. The parts hold dicts and lists, so the key is
# compared with ``!=`` only and never hashed.
_MessageKey = tuple[str, Any, tuple[tuple[Any, Any], ...]]


def _normalize(value: Any) -> Any:
    """Return ``value`` with every dict key named ``id`` dropped, recursively.

    Dicts keep their other items (compared by ``==``, so key order does not matter), lists and
    tuples map elementwise, and any other value is returned as it is.
    """
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items() if key != "id"}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _message_preview(message: Any) -> tuple[_MessageKey, str]:
    """Return ``(key, shown)``: what decides equality and the text to display.

    ``key`` holds the message type, the full content and the ``(name, args)`` of each tool call
    of an AI message, compared by ``==`` and never by ``repr``. String content and tool-call
    arguments are compared as they are (an argument named ``id`` is a real argument); list
    content is normalized so that ``id`` keys inside content blocks are ignored. Message ids and
    tool-call ids are left out. ``shown`` is ``"<type>: <first _MESSAGE_LIMIT
    characters>"``, with ``...`` appended when the content was longer (AC-106, AC-159, ADR-0023).
    """
    kind = str(getattr(message, "type", type(message).__name__))
    content = getattr(message, "content", message)
    full = content if isinstance(content, str) else _normalize(content)
    tool_calls = tuple(
        (call.get("name"), call.get("args"))
        for call in (getattr(message, "tool_calls", None) or [])
    )
    text = str(content)
    shown = f"{kind}: {text[:_MESSAGE_LIMIT]}"
    if len(text) > _MESSAGE_LIMIT:
        shown += "..."
    return (kind, full, tool_calls), shown


def _diff_tool_calls(recorded: RunSnapshot, replayed: RunSnapshot) -> tuple[ToolCallChange, ...]:
    replay_by_key = {(c.tool_name, c.ordinal): c for c in replayed.tool_calls}
    recorded_keys = {(c.tool_name, c.ordinal) for c in recorded.tool_calls}
    changes: list[ToolCallChange] = []
    for before in recorded.tool_calls:
        after = replay_by_key.get((before.tool_name, before.ordinal))
        if after is None:
            changes.append(
                ToolCallChange(before.tool_name, before.ordinal, "removed", before, None)
            )
        elif before.args != after.args or before.status != after.status:
            changes.append(
                ToolCallChange(before.tool_name, before.ordinal, "changed", before, after)
            )
    for after in replayed.tool_calls:
        if (after.tool_name, after.ordinal) not in recorded_keys:
            changes.append(ToolCallChange(after.tool_name, after.ordinal, "added", None, after))
    return tuple(changes)


def _diff_messages(recorded: RunSnapshot, replayed: RunSnapshot) -> tuple[ChannelChange, ...]:
    changes: list[ChannelChange] = []
    for index in range(max(len(recorded.messages), len(replayed.messages))):
        before = (
            _message_preview(recorded.messages[index]) if index < len(recorded.messages) else None
        )
        after = (
            _message_preview(replayed.messages[index]) if index < len(replayed.messages) else None
        )
        label = f"messages[{index}]"
        if before is None and after is not None:
            changes.append(ChannelChange(label, "added", None, after[1]))
        elif after is None and before is not None:
            changes.append(ChannelChange(label, "removed", before[1], None))
        elif before is not None and after is not None and before[0] != after[0]:
            changes.append(ChannelChange(label, "changed", before[1], after[1]))
    return tuple(changes)


def _diff_channels(recorded: RunSnapshot, replayed: RunSnapshot) -> tuple[ChannelChange, ...]:
    before_channels = {k: v for k, v in recorded.channels.items() if not _is_ignored_channel(k)}
    after_channels = {k: v for k, v in replayed.channels.items() if not _is_ignored_channel(k)}
    changes: list[ChannelChange] = []
    for name in sorted(set(before_channels) | set(after_channels)):
        in_before = name in before_channels
        in_after = name in after_channels
        before = repr(before_channels[name]) if in_before else None
        after = repr(after_channels[name]) if in_after else None
        if not in_before:
            changes.append(ChannelChange(name, "added", None, after))
        elif not in_after:
            changes.append(ChannelChange(name, "removed", before, None))
        elif before != after:
            changes.append(ChannelChange(name, "changed", before, after))
    return tuple(changes)


def diff_runs(recorded: RunSnapshot, replayed: RunSnapshot) -> RunDiff:
    """Compare a recorded run with its replay.

    Tool calls are matched by ``(tool_name, ordinal)``, never by call id; messages by index, by
    type, full content and the names and arguments of an AI message's tool calls (AC-106,
    AC-159, ADR-0023); other channels by ``repr``. A changed message is shown as
    ``"<type>: <first 120 characters of content>"``, with ``...`` appended when the content is
    longer, for display only. Checkpoint ids, timestamps, message ids, tool-call ids and
    ``__*`` / ``branch:*`` channels are never compared, so a replay that behaves like the
    recording gives an empty diff (AC-104..AC-106).
    """
    return RunDiff(
        tool_calls=_diff_tool_calls(recorded, replayed),
        messages=_diff_messages(recorded, replayed),
        channels=_diff_channels(recorded, replayed),
    )


# ---------------------------------------------------------------------------
# Replay (AC-98..AC-103, AC-107)
# ---------------------------------------------------------------------------


def _validate_inputs(update: dict[str, Any] | None, set_messages: Mapping[int, str] | None) -> None:
    if update is not None and not isinstance(update, dict):
        raise ReplayInputError(f"update must be a dict, got {type(update).__name__}")
    if set_messages is not None:
        if not isinstance(set_messages, Mapping):
            raise ReplayInputError(
                f"set_messages must be a mapping, got {type(set_messages).__name__}"
            )
        for key, text in set_messages.items():
            if isinstance(key, bool) or not isinstance(key, int) or key < 0:
                raise ReplayInputError(
                    f"set_messages keys must be non-negative integers, got {key!r}"
                )
            if not isinstance(text, str):
                raise ReplayInputError(
                    f"set_messages[{key}] must be a string, got {type(text).__name__}"
                )
        if update is not None and "messages" in update:
            raise ReplayInputError("update may not contain 'messages' when set_messages is given")


def _check_accepts_checkpointer(factory: Any, sandbox: RelateSaver) -> None:
    """Raise ``GraphSpecError`` when the factory cannot take ``checkpointer=`` (AC-155).

    Binding the signature, instead of catching ``TypeError`` around the call, keeps a
    ``TypeError`` raised inside the factory body visible as the user's own error.
    """
    try:
        signature = inspect.signature(factory)
    except (ValueError, TypeError):
        return  # no introspectable signature (some builtins): the call itself will report it
    try:
        signature.bind(checkpointer=sandbox)
    except TypeError as exc:
        name = getattr(factory, "__qualname__", type(factory).__name__)
        raise GraphSpecError(
            f"The graph factory {name} could not be called with checkpointer=...: {exc}"
        ) from exc


def _compile(factory: GraphFactory, sandbox: RelateSaver) -> Any:
    if isinstance(factory, StateGraph):
        graph = factory.compile(checkpointer=sandbox)
    else:
        _check_accepts_checkpointer(factory, sandbox)
        graph = factory(checkpointer=sandbox)
    missing = [name for name in ("invoke", "get_state", "update_state") if not hasattr(graph, name)]
    if missing:
        raise GraphSpecError(
            f"The graph factory returned a {type(graph).__name__} without {', '.join(missing)}; "
            "it must return a compiled LangGraph graph"
        )
    if getattr(graph, "checkpointer", None) is not sandbox:
        raise GraphSpecError(
            "The graph factory ignored the checkpointer argument; it must compile the graph "
            "with checkpointer=<the checkpointer it is given>"
        )
    return graph


def _next_checkpoint_hint(source_saver: RelateSaver, checkpoint_id: str) -> str:
    try:
        rows = source_saver.proxy.execute_read(
            "MATCH (:Checkpoint {checkpoint_id: $cid})-[:NEXT]->(n:Checkpoint) "
            "RETURN n.checkpoint_id AS checkpoint_id ORDER BY n.created_at ASC LIMIT 1",
            {"cid": checkpoint_id},
        )
    except Exception as exc:  # the hint is optional; the error itself is still raised
        logger.debug("Could not look up the next checkpoint (%s)", type(exc).__name__)
        return ""
    if rows:
        return f" (checkpoint '{rows[0]['checkpoint_id']}')"
    return ""


def _is_add_messages(operator: Any) -> bool:
    # ``add_messages(format=...)`` returns a functools.partial of LangGraph's reducer.
    while isinstance(operator, functools.partial):
        operator = operator.func
    if operator is add_messages:
        return True
    # LangGraph wraps its reducer, so the wrapped function is not the same object; match it by
    # name and module, never by name alone: a user's own add_messages would append, not replace.
    return getattr(operator, "__module__", None) == add_messages.__module__ and getattr(
        operator, "__name__", None
    ) in {add_messages.__name__, "add_messages"}


def _require_add_messages(graph: Any) -> None:
    """Reject graphs whose ``messages`` channel would overwrite instead of merge by id (AC-156)."""
    channel = getattr(graph, "channels", {}).get("messages")
    if isinstance(channel, BinaryOperatorAggregate) and _is_add_messages(
        getattr(channel, "operator", None)
    ):
        return
    raise ReplayInputError(
        "set_messages needs a 'messages' channel reduced by add_messages, so edited messages "
        "replace the originals by id"
    )


def _edited_messages(
    messages: list[Any],
    set_messages: Mapping[int, str],
    source_saver: RelateSaver,
    checkpoint_id: str,
) -> list[Any]:
    if not messages:
        raise ReplayInputError(
            f"Checkpoint '{checkpoint_id}' holds no messages yet, so set_messages cannot edit "
            f"it; use the next (step 0) checkpoint{_next_checkpoint_hint(source_saver, checkpoint_id)}"
        )
    edited: list[Any] = []
    for index in sorted(set_messages):
        if index >= len(messages):
            raise ReplayInputError(
                f"set_messages index {index} is out of range: the checkpoint holds "
                f"{len(messages)} message(s)"
            )
        original = messages[index]
        if not getattr(original, "id", None):
            raise ReplayInputError(f"Message {index} has no id, so it cannot be replaced")
        edited.append(original.model_copy(update={"content": set_messages[index]}))
    return edited


def _recorded_leaf(source_saver: RelateSaver, thread_id: str, fork_id: str) -> str:
    rows = source_saver.proxy.execute_read(
        "MATCH (f:Checkpoint {thread_id: $thread_id, checkpoint_id: $fork_id})"
        "-[:NEXT*0..]->(c:Checkpoint) "
        "OPTIONAL MATCH (c)-[n:NEXT]->(:Checkpoint) "
        "WITH c, count(n) AS outgoing "
        "WHERE outgoing = 0 "
        "RETURN c.checkpoint_id AS checkpoint_id, c.created_at AS created_at, c.step AS step "
        "ORDER BY c.created_at DESC, c.step DESC LIMIT 1",
        {"thread_id": thread_id, "fork_id": fork_id},
    )
    if not rows:
        return fork_id
    return str(rows[0]["checkpoint_id"])


def replay_from_checkpoint(
    source_saver: RelateSaver,
    thread_id: str,
    checkpoint_id: str,
    factory: GraphFactory,
    *,
    update: dict[str, Any] | None = None,
    as_node: str | None = None,
    set_messages: Mapping[int, str] | None = None,
) -> ReplayResult:
    """Re-run the graph from a stored checkpoint in an in-memory sandbox and diff the result.

    Args:
        source_saver: Saver holding the recorded thread; it is only read (AC-103).
        thread_id: Thread to replay.
        checkpoint_id: Checkpoint to fork from.
        factory: Graph factory accepting ``checkpointer=``, or an uncompiled ``StateGraph``.
        update: Optional channel values to write into the fork before resuming (AC-101).
        as_node: Node the update is attributed to; inferred when omitted.
        set_messages: Optional ``{index: text}`` replacing the content of existing messages
            while keeping their ids (AC-100). The checkpoint must already hold messages.

    Returns:
        A ``ReplayResult`` whose ``diff`` compares the recorded run with the replay.

    Raises:
        ReplayInputError: A ``ValueError`` subclass, for invalid inputs or when ``set_messages``
            targets a checkpoint that holds no messages yet, or the graph's ``messages`` channel is
            not reduced by ``add_messages`` (AC-151, AC-155, AC-156).
        CheckpointNotFoundError: If the checkpoint does not exist for the thread.
        GraphSpecError: If the factory does not return a usable graph on the sandbox.
    """
    _validate_inputs(update, set_messages)
    sandbox = TimeTravelEngine.fork_in_memory(source_saver, thread_id, checkpoint_id)
    try:
        graph = _compile(factory, sandbox)
        fork_cfg: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }
        updated_checkpoint_id = checkpoint_id
        if update or set_messages:
            fork_state = graph.get_state(fork_cfg)
            values: dict[str, Any] = {**(update or {})}
            if set_messages:
                _require_add_messages(graph)
                messages = list(fork_state.values.get("messages", []))
                values["messages"] = _edited_messages(
                    messages, set_messages, source_saver, checkpoint_id
                )
            node = as_node
            if node is None and not update:
                step = (fork_state.metadata or {}).get("step", 0)
                if isinstance(step, int) and step <= 0:
                    node = "__start__"
            new_cfg = graph.update_state(fork_cfg, values, as_node=node)
            updated_checkpoint_id = str(new_cfg["configurable"]["checkpoint_id"])

        graph.invoke(
            None,
            {"configurable": {"thread_id": thread_id, "checkpoint_id": updated_checkpoint_id}},
        )
        final_state = graph.get_state({"configurable": {"thread_id": thread_id}})
        final_checkpoint_id = str(final_state.config["configurable"]["checkpoint_id"])
        interrupted = bool(final_state.next)

        recorded_checkpoint_id = _recorded_leaf(source_saver, thread_id, checkpoint_id)
        diff = diff_runs(
            snapshot_run(source_saver, thread_id, recorded_checkpoint_id),
            snapshot_run(sandbox, thread_id, final_checkpoint_id),
        )
    finally:
        sandbox.close()

    return ReplayResult(
        fork_checkpoint_id=checkpoint_id,
        updated_checkpoint_id=updated_checkpoint_id,
        final_checkpoint_id=final_checkpoint_id,
        interrupted=interrupted,
        diff=diff,
        recorded_checkpoint_id=recorded_checkpoint_id,
    )


__all__ = [
    "GraphFactory",
    "diff_runs",
    "load_graph_factory",
    "replay_from_checkpoint",
    "snapshot_run",
]
