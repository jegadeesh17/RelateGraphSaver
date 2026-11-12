"""Hatchling custom build and metadata hook.

Enables custom domain classifiers during wheel build and packaging validation.
"""

from __future__ import annotations

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    """Custom metadata hook registering project classifiers."""

    def update(self, metadata: dict[str, Any]) -> None:
        metadata["description"] = (
            "Embedded Neo4j-compatible Cypher graph checkpointer for LangGraph agent workflows"
        )

    def get_known_classifiers(self) -> list[str]:
        return ["Framework :: LangGraph"]
