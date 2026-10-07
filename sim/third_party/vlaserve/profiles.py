"""Robot profiles: what a robot provides, kept separate from what a policy needs.

A checkpoint says which cameras and state/action sizes it was trained with. A profile says what a
particular robot setup produces and under which names. `check()` compares the two before the
robot moves; the mismatches it catches (a camera called `wrist` vs `image2`, a 14-d arm on a
16-d policy, joints in a different order) are the usual way a working checkpoint fails on a new
setup without any error.

    name: openarm-bimanual
    fps: 30
    cameras:            # policy camera name -> robot camera name
      base: cam_high
      left_wrist: cam_left_wrist
    state_dim: 16
    action_names: [right_joint_1.pos, ...]   # optional; checked in order if both sides have it
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .errors import ProfileMismatch
from .spec import PolicySpec


@dataclass
class RobotProfile:
    name: str
    fps: float
    cameras: dict[str, str]
    state_dim: int | None = None
    action_dim: int | None = None
    action_names: list[str] | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str) -> "RobotProfile":
        p = Path(name_or_path).expanduser()
        if p.is_file():
            text = p.read_text()
        else:
            res = resources.files("vlaserve").joinpath(f"data/profiles/{name_or_path}.yaml")
            if not res.is_file():
                raise FileNotFoundError(f"no profile file '{name_or_path}' and no bundled profile of that name")
            text = res.read_text()
        d = yaml.safe_load(text) or {}
        known = {"name", "fps", "cameras", "state_dim", "action_dim", "action_names", "notes"}
        return cls(**{k: v for k, v in d.items() if k in known},
                   extra={k: v for k, v in d.items() if k not in known})

    def check(self, spec: PolicySpec) -> list[str]:
        """Every disagreement between this robot and the policy. Empty means compatible."""
        problems = []
        missing = [c for c in spec.required_cameras if c not in self.cameras]
        if missing:
            problems.append(f"policy needs camera(s) {missing}; profile maps {sorted(self.cameras)}")
        extra = [c for c in self.cameras if c not in spec.cameras]
        if extra:
            problems.append(f"profile maps camera(s) {extra} the policy does not have "
                            f"(policy: {sorted(spec.cameras)})")
        if spec.state_dim is not None and self.state_dim is not None and self.state_dim != spec.state_dim:
            problems.append(f"state_dim: robot {self.state_dim}, policy {spec.state_dim}")
        if self.action_dim is not None and self.action_dim != spec.action_dim:
            problems.append(f"action_dim: robot {self.action_dim}, policy {spec.action_dim}")
        if self.action_names and spec.action_names and list(self.action_names) != list(spec.action_names):
            diff = [i for i, (a, b) in enumerate(zip(self.action_names, spec.action_names)) if a != b]
            problems.append(f"action_names differ (first mismatch at index {diff[0] if diff else 'length'}): "
                            "joint order must match training exactly")
        if spec.n_action_steps and self.fps:
            chunk_seconds = spec.n_action_steps / self.fps
            if chunk_seconds < 0.1:
                problems.append(f"a chunk of {spec.n_action_steps} steps lasts only "
                                f"{1e3 * chunk_seconds:.0f} ms at {self.fps} fps; inference must beat that")
        return problems

    def require(self, spec: PolicySpec) -> None:
        problems = self.check(spec)
        if problems:
            raise ProfileMismatch(f"profile '{self.name}' vs policy: " + "; ".join(problems))

    def to_policy_images(self, robot_images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Rename robot camera frames to the policy's camera names."""
        out = {}
        for policy_cam, robot_cam in self.cameras.items():
            if robot_cam not in robot_images:
                raise ProfileMismatch(f"robot frame '{robot_cam}' (for policy camera '{policy_cam}') "
                                      f"missing; got {sorted(robot_images)}")
            out[policy_cam] = robot_images[robot_cam]
        return out
