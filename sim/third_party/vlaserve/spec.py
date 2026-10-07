"""What a policy consumes and produces, read from checkpoint metadata without loading weights.

Reading the spec never imports torch, so `vla-serve inspect` is instant even for an 8 GB
checkpoint on a network drive, and works in the client-only install.

Where the truth lives. `config.json`'s `input_features` can be stale: a policy fine-tuned from
`lerobot/smolvla_base` keeps the base model's `camera1..3` (256x256) and 6-d state, while the
dataset it was trained on had `head`, `wrist` and an 8-d state, bridged by a `rename_map` in the
preprocessor (seen on kaair_sim's open-drawer SmolVLA). The normaliser statistics, by contrast,
are computed from the training dataset itself and use the dataset's own names. So, in order:

  cameras / state / action size   normaliser stats  >  config.json
  camera names clients send       dataset names (stats keys), i.e. *before* the rename step

Image resolution is deliberately not part of the contract: frames are passed through at the
resolution the client sends, exactly as during training, and the policy resizes them itself
(pi05 `_preprocess_images`, smolvla `resize_with_pad`). The declared shape is kept only as a hint.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ObservationError, UnsupportedCheckpoint

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
# LeRobot's convention for padding cameras: the policy fills a missing one with a blank image.
PADDING_CAMERA_PREFIX = "empty_camera"
MAX_SAFETENSORS_HEADER = 16 * 1024 * 1024


def safetensors_shapes(path: Path) -> dict[str, list[int]]:
    """Tensor name -> shape from a .safetensors header, without reading the tensors.

    Format: 8-byte little-endian header length, then that many bytes of JSON.
    """
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise UnsupportedCheckpoint(f"{path}: truncated safetensors file")
        (n,) = struct.unpack("<Q", head)
        if n > MAX_SAFETENSORS_HEADER:
            raise UnsupportedCheckpoint(f"{path}: safetensors header of {n} bytes is implausible")
        meta = json.loads(f.read(n))
    return {k: v["shape"] for k, v in meta.items() if k != "__metadata__"}


# How each policy family turns a camera frame into its network input, read from LeRobot 0.6.1:
#   smolvla      common/vla_utils.resize_with_pad, target = resize_imgs_with_padding stored as
#                (WIDTH, HEIGHT) (modeling_smolvla swaps it), padding on the LEFT and TOP;
#   xvla         the same function, but resize_imgs_with_padding passed as (HEIGHT, WIDTH);
#   pi0/pi05/pi0_fast   common/vla_utils.resize_with_pad_torch, target = image_resolution (H, W),
#                padding split evenly (centred).
# All use bilinear, align_corners=False, no antialias, black (0) padding, and all skip the resize
# when the frame already has the target size. So a client that sends frames already resized this
# way gets exactly the network input, and the policy does not resize again. Families not listed
# get no client-side resize: their frames are sent as captured.
def image_input_rule(cfg: dict) -> dict | None:
    t = cfg.get("type")
    rip = cfg.get("resize_imgs_with_padding")
    if t == "smolvla" and rip:
        return {"size": [int(rip[1]), int(rip[0])], "pad": "top_left"}
    if t == "xvla" and rip:
        return {"size": [int(rip[0]), int(rip[1])], "pad": "top_left"}
    if t in ("pi0", "pi05", "pi0_fast") and cfg.get("image_resolution"):
        r = cfg["image_resolution"]
        return {"size": [int(r[0]), int(r[1])], "pad": "center"}
    return None


def _preprocessor(path: Path) -> list[dict]:
    pre = path / "policy_preprocessor.json"
    return json.loads(pre.read_text()).get("steps", []) if pre.is_file() else []


@dataclass
class PolicySpec:
    policy_type: str
    cameras: dict[str, tuple[int, int, int] | None]   # name a client sends -> declared (C,H,W) hint
    state_dim: int | None
    action_dim: int
    chunk_size: int | None
    n_action_steps: int | None
    action_names: list[str] | None = None
    policy_fills: list[str] = field(default_factory=list)  # policy cameras nobody sends (padded)
    language: bool = True        # whether the processor pipeline tokenises a task string
    source: str = "config"       # "stats" or "config": where cameras / dims came from
    image_input: dict | None = None   # {"size": [H, W], "pad": "top_left"|"center"}: the network input
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def required_cameras(self) -> list[str]:
        return list(self.cameras)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cameras"] = {k: (list(v) if v else None) for k, v in self.cameras.items()}
        d["required_cameras"] = self.required_cameras
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicySpec":
        """Inverse of to_dict (what GET /v1/spec returns), for clients."""
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}   # tolerate newer/older servers
        d["cameras"] = {k: (tuple(v) if v else None) for k, v in d["cameras"].items()}
        return cls(**d)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "PolicySpec":
        path = Path(path)
        cfg_file = path / "config.json"
        if not cfg_file.is_file():
            raise UnsupportedCheckpoint(f"{path} has no config.json; not a LeRobot policy checkpoint")
        cfg = json.loads(cfg_file.read_text())
        if "type" not in cfg or "input_features" not in cfg or "output_features" not in cfg:
            raise UnsupportedCheckpoint(f"{cfg_file} lacks type/input_features/output_features")

        steps = _preprocessor(path)
        rename: dict[str, str] = {}
        stats: dict[str, list[int]] = {}
        for step in steps:
            if step.get("registry_name") == "rename_observations_processor":
                rename = (step.get("config") or {}).get("rename_map") or {}
            if step.get("registry_name") == "normalizer_processor" and step.get("state_file"):
                f = path / step["state_file"]
                if f.is_file():
                    stats = safetensors_shapes(f)

        declared = {k[len(IMAGE_PREFIX):]: tuple(int(x) for x in v["shape"])
                    for k, v in cfg["input_features"].items() if k.startswith(IMAGE_PREFIX)}
        policy_cams = list(declared)
        stat_cams = sorted({k[len(IMAGE_PREFIX):].rsplit(".", 1)[0]
                            for k in stats if k.startswith(IMAGE_PREFIX)})

        if stat_cams:
            source = "stats"
            cameras = {}
            for cam in stat_cams:
                target = rename.get(IMAGE_PREFIX + cam, IMAGE_PREFIX + cam)[len(IMAGE_PREFIX):]
                # The declared shape only describes this camera if it was not renamed into a
                # feature inherited from a base model.
                cameras[cam] = declared.get(cam) if target == cam else None
            reached = {rename.get(IMAGE_PREFIX + c, IMAGE_PREFIX + c)[len(IMAGE_PREFIX):] for c in cameras}
            fills = [c for c in policy_cams if c not in reached]
        else:
            source = "config"
            cameras = {c: s for c, s in declared.items() if not c.startswith(PADDING_CAMERA_PREFIX)}
            fills = [c for c in policy_cams if c.startswith(PADDING_CAMERA_PREFIX)]

        def dim(stat_key: str, cfg_feats: dict, key: str) -> int | None:
            if stat_key in stats:
                return int(stats[stat_key][0])
            return int(cfg_feats[key]["shape"][0]) if key in cfg_feats else None

        state_dim = dim(f"{STATE_KEY}.mean", cfg["input_features"], STATE_KEY)
        action_dim = dim("action.mean", cfg["output_features"], "action")
        if action_dim is None:
            raise UnsupportedCheckpoint(f"{cfg_file}: no 'action' output feature")

        language = any("tokenizer" in s.get("registry_name", "") for s in steps)
        extra = {k: cfg[k] for k in ("paligemma_variant", "vlm_model_name", "dtype",
                                     "num_inference_steps", "use_relative_actions") if k in cfg}
        if rename:
            extra["rename_map"] = rename
        return cls(
            policy_type=cfg["type"],
            cameras=cameras,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=cfg.get("chunk_size"),
            n_action_steps=cfg.get("n_action_steps"),
            action_names=cfg.get("action_feature_names"),
            policy_fills=fills,
            language=language,
            source=source,
            image_input=image_input_rule(cfg),
            extra=extra,
        )

    # -- observation validation ------------------------------------------------------------
    def validate(self, images: dict[str, np.ndarray], state: np.ndarray | None,
                 task: str | None) -> None:
        """Reject an observation the policy cannot consume, before it reaches the GPU.

        Images are HWC uint8 (what a camera driver gives you), at any resolution: the policy
        resizes them itself, as it did during training.
        """
        missing = [c for c in self.required_cameras if c not in images]
        if missing:
            raise ObservationError(f"missing camera(s) {missing}; policy expects "
                                   f"{self.required_cameras}")
        unknown = [c for c in images if c not in self.cameras]
        if unknown:
            raise ObservationError(f"unknown camera(s) {unknown}; policy expects {list(self.cameras)}")
        for name, img in images.items():
            channels = self.cameras[name][0] if self.cameras[name] else 3
            if img.ndim != 3 or img.shape[2] != channels:
                raise ObservationError(f"camera '{name}': expected HxWx{channels} "
                                       f"image, got shape {tuple(img.shape)}")
            if img.shape[0] > 8192 or img.shape[1] > 8192:
                raise ObservationError(f"camera '{name}': {img.shape[:2]} exceeds 8192 px")
            if img.dtype != np.uint8:
                raise ObservationError(f"camera '{name}': expected uint8, got {img.dtype}")
        if self.state_dim is not None:
            if state is None:
                raise ObservationError(f"policy needs a state vector of length {self.state_dim}")
            if state.shape != (self.state_dim,):
                raise ObservationError(f"state: expected shape ({self.state_dim},), got {state.shape}")
            if not all(math.isfinite(float(x)) for x in state):
                raise ObservationError("state contains NaN or Inf")
        if self.language and not task:
            raise ObservationError("policy is language-conditioned; send a non-empty 'task'")
