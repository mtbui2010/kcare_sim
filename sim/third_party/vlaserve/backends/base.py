"""The one interface every backend implements. The server and CLI only ever see this."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from ..spec import PolicySpec


@dataclass
class Observation:
    images: dict[str, np.ndarray]          # short camera name -> HxWxC uint8
    state: np.ndarray | None = None        # (state_dim,) float
    task: str | None = None


@dataclass
class ActionChunk:
    actions: np.ndarray                    # (H, action_dim) float32, absolute, robot units
    timings_ms: dict[str, float] = field(default_factory=dict)


class Backend(abc.ABC):
    spec: PolicySpec

    @abc.abstractmethod
    def predict(self, obs: Observation, seed: int | None = None) -> ActionChunk:
        """One observation in, one action chunk out. Must be safe to call from one thread at a time."""

    def reset(self) -> None:
        """Clear per-episode state (observation history, action queues). Default: nothing."""

    def close(self) -> None:
        """Release GPU memory."""
