"""Offline stand-ins for ``fetch_pypi_json`` used by the dependency agent tests."""

from __future__ import annotations

import urllib.error

MISSING_PACKAGE = "relategraphsaver"


class FakePyPI:
    """Fetcher with a fixed answer per package; ``relategraphsaver`` raises a 404 LookupError."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, name: str) -> dict[str, str]:
        self.calls.append(name)
        if name == MISSING_PACKAGE:
            raise LookupError(f"PyPI returned HTTP 404: package '{name}' does not exist")
        return {"name": name, "version": "1.2.3", "summary": f"Fake summary of {name}"}


class UnreachablePyPI:
    """Fetcher that fails like a network outage for every name."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, name: str) -> dict[str, str]:
        self.calls.append(name)
        raise urllib.error.URLError("pypi.org is unreachable (simulated)")
