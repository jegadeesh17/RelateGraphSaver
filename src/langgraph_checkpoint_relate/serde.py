"""State serialization and deserialization engine for Relate checkpointer.

Provides fidelity-preserving binary and base64 serialization of LangGraph state channels,
standard LangChain message objects, datetimes, UUIDs, and Pydantic models with
deterministic SHA-256 state hashing and precise error reporting.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_checkpoint_relate.errors import CheckpointSerializationError

KNOWN_TYPES: frozenset[str] = frozenset({"msgpack", "json", "bytes", "bytearray", "null", "pickle"})


class RelateStateSerializer(SerializerProtocol):
    """Serialization and deserialization engine for Relate graph checkpointer.

    Wraps LangGraph's JsonPlusSerializer to provide:
    - Typed tuple serialization (type, bytes) conforming to SerializerProtocol.
    - Self-describing byte-string serialization (dumps/loads).
    - Base64 string serialization (dumps_base64/loads_base64).
    - Deterministic SHA-256 state hashing for cycle and oscillation detection.
    - Precise error attribution identifying offending keys on serialization failure.
    """

    def __init__(
        self,
        *,
        inner_serializer: JsonPlusSerializer | None = None,
        pickle_fallback: bool = False,
    ) -> None:
        """Initialize RelateStateSerializer.

        Args:
            inner_serializer: Optional custom JsonPlusSerializer instance.
            pickle_fallback: Whether to fall back to pickle for unsupported types.
        """
        self._serializer = inner_serializer or JsonPlusSerializer(pickle_fallback=pickle_fallback)

    def _find_offending_key(
        self,
        obj: Any,
        path: str = "",
        visited: set[int] | None = None,
    ) -> str | None:
        """Recursively search an object structure to find the offending key."""
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return path or None
        visited.add(obj_id)

        if isinstance(obj, dict):
            # Prioritize channel_values if present (standard LangGraph Checkpoint)
            if "channel_values" in obj and isinstance(obj["channel_values"], dict) and not path:
                res = self._find_offending_key(obj["channel_values"], "channel_values", visited)
                if res:
                    return res

            for k, v in obj.items():
                curr = f"{path}.{k}" if path else str(k)
                try:
                    self._serializer.dumps_typed(v)
                except Exception:
                    nested = self._find_offending_key(v, curr, visited)
                    return nested or curr

        elif isinstance(obj, (list, tuple, set)):
            for idx, item in enumerate(obj):
                curr = f"{path}[{idx}]"
                try:
                    self._serializer.dumps_typed(item)
                except Exception:
                    nested = self._find_offending_key(item, curr, visited)
                    return nested or curr

        elif hasattr(obj, "__dict__"):
            try:
                attrs = vars(obj)
            except Exception:
                attrs = {}
            for k, v in attrs.items():
                curr = f"{path}.{k}" if path else str(k)
                try:
                    self._serializer.dumps_typed(v)
                except Exception:
                    nested = self._find_offending_key(v, curr, visited)
                    return nested or curr

        return path or None

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize an object into a (type, bytes) tuple.

        Args:
            obj: Python object to serialize.

        Returns:
            Tuple of (type_str, raw_bytes).

        Raises:
            CheckpointSerializationError: If the payload cannot be serialized.
        """
        try:
            return self._serializer.dumps_typed(obj)
        except CheckpointSerializationError:
            raise
        except Exception as exc:
            offending_key = self._find_offending_key(obj)
            if offending_key:
                leaf_key = offending_key.rsplit(".", 1)[-1].split("[")[0]
                msg = (
                    f"Failed to serialize checkpoint: offending key '{offending_key}' "
                    f"(leaf key: '{leaf_key}') is not serializable. Error: {exc}"
                )
            else:
                msg = (
                    f"Failed to serialize checkpoint: object of type '{type(obj).__name__}' "
                    f"is not serializable. Error: {exc}"
                )
            raise CheckpointSerializationError(msg) from exc

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize an object from a (type, bytes) tuple.

        Args:
            data: Tuple of (type_str, raw_bytes).

        Returns:
            Original Python object.

        Raises:
            CheckpointSerializationError: If deserialization fails.
        """
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise CheckpointSerializationError(
                f"Malformed typed payload: expected (type, bytes) tuple, got {type(data).__name__}"
            )
        try:
            return self._serializer.loads_typed(data)
        except CheckpointSerializationError:
            raise
        except Exception as exc:
            raise CheckpointSerializationError(
                f"Failed to deserialize typed checkpoint payload: {exc}"
            ) from exc

    def dumps(self, obj: Any) -> bytes:
        """Serialize an object to a self-describing byte string.

        Prepends a 1-byte length prefix and ASCII type descriptor before payload bytes.

        Args:
            obj: Python object to serialize.

        Returns:
            Serialized byte string.

        Raises:
            CheckpointSerializationError: If serialization fails.
        """
        type_str, payload_bytes = self.dumps_typed(obj)
        type_bytes = type_str.encode("utf-8")
        if len(type_bytes) > 255:
            raise CheckpointSerializationError(
                f"Serialization type string exceeds 255 bytes: {type_str}"
            )
        return bytes([len(type_bytes)]) + type_bytes + payload_bytes

    def loads(self, data: bytes | tuple[str, bytes] | str) -> Any:
        """Deserialize an object from a byte string, base64 string, or typed tuple.

        Args:
            data: Serialized byte string, base64 string, or (type, bytes) tuple.

        Returns:
            Deserialized Python object.

        Raises:
            CheckpointSerializationError: If deserialization fails.
        """
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], str)
            and isinstance(data[1], (bytes, bytearray))
        ):
            return self.loads_typed((data[0], bytes(data[1])))

        if isinstance(data, str):
            return self.loads_base64(data)

        if not isinstance(data, (bytes, bytearray)):
            raise CheckpointSerializationError(
                f"Expected bytes, str, or (type, bytes) tuple for deserialization, got {type(data).__name__}"
            )

        raw = bytes(data)
        if len(raw) >= 5:
            t_len = raw[0]
            if 4 <= t_len <= 32 and 1 + t_len <= len(raw):
                try:
                    candidate = raw[1 : 1 + t_len].decode("utf-8")
                    if candidate in KNOWN_TYPES:
                        return self.loads_typed((candidate, raw[1 + t_len :]))
                except CheckpointSerializationError:
                    raise
                except Exception as exc:
                    raise CheckpointSerializationError(
                        f"Failed to deserialize binary checkpoint payload: {exc}"
                    ) from exc

        raise CheckpointSerializationError(
            "Failed to deserialize binary checkpoint payload: missing or corrupted envelope header"
        )

    def dumps_base64(self, obj: Any) -> str:
        """Serialize an object to a base64-encoded ASCII string.

        Args:
            obj: Python object to serialize.

        Returns:
            Base64 encoded string.

        Raises:
            CheckpointSerializationError: If serialization fails.
        """
        byte_data = self.dumps(obj)
        return base64.b64encode(byte_data).decode("ascii")

    def loads_base64(self, data: str) -> Any:
        """Deserialize an object from a base64-encoded ASCII string.

        Args:
            data: Base64 string to deserialize.

        Returns:
            Deserialized Python object.

        Raises:
            CheckpointSerializationError: If decoding or deserialization fails.
        """
        if not isinstance(data, str):
            raise CheckpointSerializationError(f"Expected base64 str, got {type(data).__name__}")
        try:
            byte_data = base64.b64decode(data.encode("ascii"))
        except (binascii.Error, ValueError) as exc:
            raise CheckpointSerializationError(f"Failed to decode base64 payload: {exc}") from exc

        return self.loads(byte_data)

    # Convenience aliases
    dumps_b64 = dumps_base64
    loads_b64 = loads_base64

    def _canonicalize(self, obj: Any) -> Any:
        """Recursively canonicalize dictionary keys and sets for deterministic hashing."""
        if isinstance(obj, dict):
            return {
                str(k): self._canonicalize(v)
                for k, v in sorted(obj.items(), key=lambda item: str(item[0]))
            }
        elif isinstance(obj, list):
            return [self._canonicalize(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self._canonicalize(x) for x in obj)
        elif isinstance(obj, (set, frozenset)):
            return [
                self._canonicalize(x)
                for x in sorted(obj, key=lambda item: (type(item).__name__, repr(item)))
            ]
        return obj

    def compute_state_hash(self, state: Any) -> str:
        """Compute a deterministic SHA-256 state hash over serialized channel values.

        Used for O(1) loop and oscillation detection in the execution DAG.

        Args:
            state: Channel values dictionary, Checkpoint dictionary, or raw bytes.

        Returns:
            64-character lowercase hexadecimal SHA-256 digest string.

        Raises:
            CheckpointSerializationError: If state cannot be serialized.
        """
        if isinstance(state, (bytes, bytearray)):
            return hashlib.sha256(state).hexdigest()

        # If a full Checkpoint dict is passed, extract channel_values
        if (
            isinstance(state, dict)
            and "channel_values" in state
            and isinstance(state["channel_values"], dict)
        ):
            target = state["channel_values"]
        else:
            target = state

        canonicalized = self._canonicalize(target)
        _, payload = self.dumps_typed(canonicalized)
        return hashlib.sha256(payload).hexdigest()

    state_hash = compute_state_hash


default_serializer = RelateStateSerializer()

__all__ = [
    "KNOWN_TYPES",
    "RelateStateSerializer",
    "default_serializer",
]
