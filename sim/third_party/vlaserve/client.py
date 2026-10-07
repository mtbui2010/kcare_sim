"""Client side: call the server, and run a robot at a fixed rate on top of chunked inference.

    from vlaserve.client import PolicyClient, ChunkExecutor

    client = PolicyClient("http://127.0.0.1:11450", model="pi05-libero")
    chunk = client.act(images={"image": rgb, "image2": wrist}, state=q, task="pick up the bowl")

    ex = ChunkExecutor(client, get_observation=read_robot, send_action=robot.command, fps=15)
    ex.run(task="pick up the bowl", seconds=30)

Why the executor exists: a VLA returns H actions per call and takes longer than one control
period to do it (pi0.5 ~100-400 ms vs 33-66 ms per step). Calling it synchronously stalls the
robot every chunk. The executor keeps a queue of future actions, asks for the next chunk while
the current one is still executing, and when the new chunk arrives drops the actions whose time
has already passed during inference, so the robot never replays stale commands.

Merge rule: latest chunk wins (the new chunk replaces what is left of the old one). It is the
simplest correct rule. Smoother blends (temporal ensembling, real-time chunking) can replace
`ActionQueue.on_chunk` without touching anything else.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx
import numpy as np

from . import codec

log = logging.getLogger(__name__)


class ServerError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status


@dataclass
class Chunk:
    actions: np.ndarray                 # (H, D)
    action_names: list[str] | None
    n_action_steps: int | None
    timings_ms: dict[str, float]
    roundtrip_ms: float


class PolicyClient:
    """HTTP client for one vla-serve server.

    Frames are prepared on this side before they are sent (both defaults, both switchable):

      resize="auto"   resize-with-pad each frame to the policy's network input, exactly as the
                      policy itself would (imaging.resize_with_pad), so the server and the policy
                      pass it through untouched. The rule comes from GET /v1/spec, once per model.
                      Policy families without a known rule are sent as captured. "off" disables.
      encoding="jpeg" JPEG at `jpeg_quality` (95), 4:4:4 chroma, on the wire (~27x smaller than raw
                      for two 480x848 frames; actions within 0.12x of the seed-to-seed spread on
                      the measured SmolVLA). "raw" sends exact uint8 arrays.
    """

    def __init__(self, url: str = "http://127.0.0.1:11450", model: str | None = None,
                 timeout: float = 30.0, resize: str = "auto", encoding: str = "jpeg",
                 jpeg_quality: int = 95):
        if resize not in ("auto", "off"):
            raise ValueError("resize must be 'auto' or 'off'")
        if encoding not in ("jpeg", "raw"):
            raise ValueError("encoding must be 'jpeg' or 'raw'")
        if not (1 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be in 1..100")
        self.url = url.rstrip("/")
        self.model = model
        self.resize, self.encoding, self.jpeg_quality = resize, encoding, jpeg_quality
        self._http = httpx.Client(timeout=timeout)
        self._image_input: dict[str, dict | None] = {}

    def image_input(self, model: str | None = None) -> dict | None:
        """The policy's network-input rule ({"size": [H, W], "pad": ...}), fetched once per model."""
        name = model or self.model
        if name not in self._image_input:
            self._image_input[name] = self.spec(name)["spec"].get("image_input")
        return self._image_input[name]

    def prepare(self, img: np.ndarray, model: str | None = None):
        """One camera frame -> what goes on the wire, after the configured resize and encoding."""
        from . import imaging
        img = np.ascontiguousarray(img, dtype=np.uint8)
        if self.resize == "auto" and img.ndim == 3 and img.shape[2] == 3:
            rule = self.image_input(model)
            if rule:
                img = imaging.resize_with_pad(img, *rule["size"], pad=rule["pad"])
        if self.encoding == "jpeg" and img.ndim == 3 and img.shape[2] == 3:
            return codec.jpeg(imaging.encode_jpeg(img, self.jpeg_quality))
        return img

    def _check(self, r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ServerError(r.status_code, detail)
        return r

    def load(self, model: str | None = None, device: str | None = None) -> dict:
        body = {"model": model or self.model}
        if device:
            body["device"] = device
        return self._check(self._http.post(f"{self.url}/v1/models/load", json=body, timeout=None)).json()

    def spec(self, model: str | None = None) -> dict:
        return self._check(self._http.get(f"{self.url}/v1/spec",
                                          params={"model": model or self.model})).json()

    def reset(self, model: str | None = None) -> None:
        self._check(self._http.post(f"{self.url}/v1/models/reset", json={"model": model or self.model}))

    def act(self, images: dict[str, np.ndarray], state=None, task: str | None = None,
            seed: int | None = None, model: str | None = None) -> Chunk:
        obs = {"images": {k: self.prepare(v, model) for k, v in images.items()}}
        if state is not None:
            obs["state"] = np.asarray(state, dtype=np.float32)
        if task is not None:
            obs["task"] = task
        body = {"model": model or self.model, "observation": obs}
        if seed is not None:
            body["seed"] = seed
        t0 = time.perf_counter()
        r = self._check(self._http.post(f"{self.url}/v1/act", content=codec.packb(body),
                                        headers={"content-type": "application/msgpack"}))
        out = codec.unpackb(r.content)
        return Chunk(actions=np.array(out["actions"]), action_names=out.get("action_names"),
                     n_action_steps=out.get("n_action_steps"), timings_ms=out.get("timings_ms", {}),
                     roundtrip_ms=1e3 * (time.perf_counter() - t0))

    def close(self) -> None:
        self._http.close()


# -- chunk execution ---------------------------------------------------------------------------

class ActionQueue:
    """Pure logic, no clocks, so it is unit-testable.

    Time is counted in *executed actions*, not control ticks. A chunk planned from an observation
    taken when `executed_at_obs` actions had run starts with actions for moves the robot has since
    made; those `executed - executed_at_obs` actions are dropped and the rest replace whatever is
    queued. Ticks where the robot stood still (waiting for the first chunk, or a stall) do not
    age the plan: the robot did not move, so the observation still describes it.
    """

    def __init__(self, execute_steps: int | None = None):
        self.execute_steps = execute_steps
        self._q: list[np.ndarray] = []
        self.executed = 0
        self.dropped_stale = 0

    def __len__(self) -> int:
        return len(self._q)

    def on_chunk(self, actions: np.ndarray, executed_at_obs: int) -> None:
        usable = actions[: self.execute_steps]
        stale = max(0, self.executed - executed_at_obs)
        self.dropped_stale += min(stale, len(usable))
        self._q = list(usable[stale:])

    def pop(self) -> np.ndarray | None:
        if not self._q:
            return None
        self.executed += 1
        return self._q.pop(0)


@dataclass
class RunStats:
    steps: int = 0
    stalls: int = 0                 # ticks with an empty queue after the first chunk arrived
    startup_ticks: int = 0          # ticks spent waiting for the first chunk
    chunks: int = 0
    dropped_stale: int = 0
    overruns: int = 0               # ticks where send_action itself took longer than a period
    roundtrip_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        rt = np.array(self.roundtrip_ms) if self.roundtrip_ms else np.array([np.nan])
        return {"steps": self.steps, "stalls": self.stalls, "startup_ticks": self.startup_ticks,
                "chunks": self.chunks,
                "dropped_stale": self.dropped_stale, "overruns": self.overruns,
                "roundtrip_ms_median": float(np.median(rt)),
                "roundtrip_ms_p95": float(np.percentile(rt, 95))}


class ChunkExecutor:
    """Run `send_action` at `fps`, fed by asynchronous chunk requests.

    get_observation() -> dict(images=..., state=...)   called on the inference thread
    send_action(np.ndarray)                             called on the control thread, every tick
    refill_at: request the next chunk when this many actions (or fewer) remain queued.
               Default: half the executed chunk. Set it above inference latency x fps, or the
               robot will stall waiting.
    """

    def __init__(self, client: PolicyClient, get_observation: Callable[[], dict],
                 send_action: Callable[[np.ndarray], None], fps: float,
                 execute_steps: int | None = None, refill_at: int | None = None,
                 on_stall: Callable[[], None] | None = None):
        if not (0 < fps <= 1000):
            raise ValueError("fps must be in (0, 1000]")
        self.client, self.get_observation, self.send_action = client, get_observation, send_action
        self.fps = fps
        self.execute_steps = execute_steps
        self.refill_at = refill_at
        self.on_stall = on_stall
        self.stats = RunStats()

    def run(self, task: str | None = None, seconds: float | None = None,
            stop: threading.Event | None = None) -> RunStats:
        stop = stop or threading.Event()
        period = 1.0 / self.fps
        lock = threading.Lock()
        queue = ActionQueue(execute_steps=self.execute_steps)   # size fixed by the first chunk
        started = False                                          # first chunk received
        in_flight = threading.Event()
        error: list[BaseException] = []

        def infer():
            nonlocal started
            try:
                with lock:
                    executed_at_obs = queue.executed             # read with the observation
                obs = self.get_observation()
                chunk = self.client.act(task=task, **obs)
                with lock:
                    if queue.execute_steps is None:
                        queue.execute_steps = min(chunk.n_action_steps or len(chunk.actions),
                                                  len(chunk.actions))
                    if self.refill_at is None:
                        self.refill_at = max(1, queue.execute_steps // 2)
                    queue.on_chunk(chunk.actions, executed_at_obs)
                    started = True
                    self.stats.chunks += 1
                    self.stats.dropped_stale = queue.dropped_stale
                    self.stats.roundtrip_ms.append(chunk.roundtrip_ms)
            except BaseException as e:  # surfaced on the control thread
                error.append(e)
                stop.set()
            finally:
                in_flight.clear()

        deadline = None if seconds is None else time.monotonic() + seconds
        next_tick = time.monotonic()
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            with lock:
                need = not started or len(queue) <= (self.refill_at or 0)
                action = queue.pop()
                waiting_first = not started
            if need and not in_flight.is_set():
                in_flight.set()
                threading.Thread(target=infer, daemon=True).start()
            if action is None:
                if waiting_first:
                    self.stats.startup_ticks += 1      # waiting for the very first chunk
                else:
                    self.stats.stalls += 1
                if self.on_stall:
                    self.on_stall()
            else:
                self.send_action(action)
            self.stats.steps += 1
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self.stats.overruns += 1
                next_tick = time.monotonic()
        stop.set()
        if error:
            raise error[0]
        return self.stats
