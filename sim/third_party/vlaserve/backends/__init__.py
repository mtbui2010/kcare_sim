from __future__ import annotations

from pathlib import Path

from .base import ActionChunk, Backend, Observation

__all__ = ["ActionChunk", "Backend", "Observation", "load_backend"]


def load_backend(kind: str, path: Path, device: str, options: dict | None = None) -> Backend:
    options = options or {}
    if kind == "lerobot":
        from .lerobot_backend import LeRobotBackend
        return LeRobotBackend(path, device=device, **options)
    raise ValueError(f"unknown backend '{kind}' (available: lerobot)")
