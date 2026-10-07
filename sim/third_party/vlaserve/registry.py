"""Turn a user-facing model name into a checkpoint directory.

A name is resolved, in order, as:
  1. an alias in the registry (the bundled `registry/models.yaml`, then the user's
     `~/.config/vla-serve/models.yaml`, which wins on conflict);
  2. a local directory containing a LeRobot `config.json`;
  3. a Hugging Face repo id (`org/name`), fetched through the normal HF cache, so
     `HF_HOME` / `HF_HUB_OFFLINE` behave exactly as they do for LeRobot itself.

So any LeRobot-format checkpoint works without a registry entry. The registry only adds a short
name, a pinned revision, and a licence when the checkpoint does not carry one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ModelNotFound
from .licenses import LicenseInfo, resolve as resolve_license

USER_REGISTRY = Path(os.environ.get("VLASERVE_REGISTRY",
                                    Path.home() / ".config" / "vla-serve" / "models.yaml"))


@dataclass
class Entry:
    name: str
    source: str                       # HF repo id or local path
    revision: str | None = None
    backend: str = "lerobot"
    license: str | None = None
    description: str = ""
    profile: str | None = None        # default robot profile name
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedModel:
    name: str
    path: Path
    entry: Entry
    license: LicenseInfo


def _load_yaml(text: str, origin: str) -> dict[str, Entry]:
    data = yaml.safe_load(text) or {}
    out = {}
    for name, body in (data.get("models") or {}).items():
        if not isinstance(body, dict) or "source" not in body:
            raise ValueError(f"{origin}: model '{name}' needs a 'source'")
        known = {"source", "revision", "backend", "license", "description", "profile", "options"}
        unknown = set(body) - known
        if unknown:
            raise ValueError(f"{origin}: model '{name}' has unknown keys {sorted(unknown)}")
        out[name] = Entry(name=name, **body)
    return out


def load_registry(user_file: Path | None = None) -> dict[str, Entry]:
    builtin = resources.files("vlaserve").joinpath("data/models.yaml").read_text()
    entries = _load_yaml(builtin, "bundled registry")
    user_file = USER_REGISTRY if user_file is None else user_file
    if user_file.is_file():
        entries.update(_load_yaml(user_file.read_text(), str(user_file)))
    return entries


def _download(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    try:
        return Path(snapshot_download(repo, revision=revision))
    except (LocalEntryNotFoundError, HfHubHTTPError, ValueError) as e:
        hint = " (HF_HUB_OFFLINE is set and it is not cached)" if os.environ.get("HF_HUB_OFFLINE") else ""
        raise ModelNotFound(f"cannot fetch '{repo}'{hint}: {e}") from e


def resolve(name: str, registry: dict[str, Entry] | None = None) -> ResolvedModel:
    registry = load_registry() if registry is None else registry
    if name in registry:
        entry = registry[name]
    else:
        entry = Entry(name=name, source=name)

    local = Path(entry.source).expanduser()
    if local.is_dir():
        path = local
    elif "/" in entry.source and not entry.source.startswith((".", "/", "~")):
        path = _download(entry.source, entry.revision)
    else:
        raise ModelNotFound(f"'{name}' is not a registry alias, a local checkpoint or an HF repo id. "
                            f"Known aliases: {sorted(registry)}")
    cfg_file = path / "config.json"
    cfg = json.loads(cfg_file.read_text()) if cfg_file.is_file() else {}
    return ResolvedModel(name=name, path=path, entry=entry,
                         license=resolve_license(cfg, entry.license))
