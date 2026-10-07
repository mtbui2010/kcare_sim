"""HTTP server. JSON for control endpoints, msgpack for /v1/act.

    GET  /health
    GET  /v1/models                    loaded models: spec, licence, warning
    POST /v1/models/load    {"model": name, "device"?: "cuda:0"}
    POST /v1/models/unload  {"model": name}
    POST /v1/models/reset   {"model": name}          start of a new episode
    GET  /v1/spec?model=name                         spec from config.json, no weights loaded
    POST /v1/act            msgpack {"model", "observation": {"images", "state", "task"}, "seed"?}
                         -> msgpack {"actions": (H, D) float32, "action_names", "n_action_steps",
                                     "timings_ms", "model"}

One inference at a time per model (a per-model lock, held only while the GPU works), run in a
worker thread so the event loop keeps answering /health during a 300 ms pi0.5 call.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from . import codec
from .backends import Backend, Observation, load_backend
from .errors import (ModelNotFound, ModelNotLoaded, ObservationError, ProfileMismatch,
                     UnsupportedCheckpoint, VLAServeError)
from .registry import ResolvedModel, resolve
from .spec import PolicySpec

log = logging.getLogger(__name__)
MSGPACK = "application/msgpack"

STATUS = {ObservationError: 400, ProfileMismatch: 400, UnsupportedCheckpoint: 422,
          ModelNotFound: 404, ModelNotLoaded: 409}


@dataclass
class Loaded:
    resolved: ResolvedModel
    backend: Backend
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    loaded_at: float = field(default_factory=time.time)
    calls: int = 0

    def describe(self) -> dict:
        lic = self.resolved.license
        return {"model": self.resolved.name, "path": str(self.resolved.path), "device": self.device,
                "backend": self.resolved.entry.backend, "spec": self.backend.spec.to_dict(),
                "license": lic.license, "license_source": lic.source, "license_warning": lic.warning,
                "calls": self.calls}


BackendFactory = Callable[[ResolvedModel, str], Backend]


def _default_factory(resolved: ResolvedModel, device: str) -> Backend:
    return load_backend(resolved.entry.backend, resolved.path, device, resolved.entry.options)


class ModelManager:
    def __init__(self, default_device: str = "cuda",
                 factory: BackendFactory = _default_factory,
                 resolver: Callable[[str], ResolvedModel] = resolve):
        self.default_device = default_device
        self._factory = factory
        self._resolve = resolver
        self._models: dict[str, Loaded] = {}
        self._load_lock = threading.Lock()   # loads are rare and huge; serialise them

    def load(self, name: str, device: str | None = None) -> Loaded:
        device = device or self.default_device
        with self._load_lock:
            if name in self._models:
                return self._models[name]
            resolved = self._resolve(name)
            if resolved.license.warning:
                log.warning("%s: %s", name, resolved.license.warning)
            t0 = time.perf_counter()
            backend = self._factory(resolved, device)
            log.info("loaded %s on %s in %.1fs", name, device, time.perf_counter() - t0)
            self._models[name] = Loaded(resolved, backend, device)
            return self._models[name]

    def resolve(self, name: str) -> ResolvedModel:
        return self._resolve(name)

    def get(self, name: str) -> Loaded:
        if name not in self._models:
            raise ModelNotLoaded(f"'{name}' is not loaded; POST /v1/models/load first. "
                                 f"Loaded: {sorted(self._models)}")
        return self._models[name]

    def unload(self, name: str) -> None:
        with self._load_lock:
            m = self._models.pop(name, None)
        if m is None:
            raise ModelNotLoaded(f"'{name}' is not loaded")
        with m.lock:                         # wait for an in-flight call to finish
            m.backend.close()

    def all(self) -> list[Loaded]:
        return list(self._models.values())


def _parse_observation(body: dict, spec: PolicySpec) -> tuple[Observation, int | None]:
    if not isinstance(body, dict) or not isinstance(body.get("observation"), dict):
        raise ObservationError("body must be a map with an 'observation' map")
    o = body["observation"]
    images = o.get("images") or {}
    if not isinstance(images, dict) or not all(isinstance(v, np.ndarray) for v in images.values()):
        raise ObservationError("'images' must map camera name -> uint8 HxWxC array")
    state = o.get("state")
    if state is not None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
    task = o.get("task")
    if task is not None and not isinstance(task, str):
        raise ObservationError("'task' must be a string")
    seed = body.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ObservationError("'seed' must be an integer")
    obs = Observation(images=images, state=state, task=task)
    spec.validate(obs.images, obs.state, obs.task)
    return obs, seed


def create_app(manager: ModelManager) -> FastAPI:
    app = FastAPI(title="vla-serve")

    # Registered on VLAServeError, not Exception: Starlette treats an `Exception` handler as the
    # 500 handler and re-raises after it, so caller mistakes would surface as server errors.
    @app.exception_handler(VLAServeError)
    async def _errors(request: Request, exc: VLAServeError):
        status = next((s for cls, s in STATUS.items() if isinstance(exc, cls)), 500)
        if status == 500:
            log.error("unmapped %s on %s: %s", type(exc).__name__, request.url.path, exc)
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc)}, status)

    async def _json(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception as e:
            raise ObservationError(f"body must be JSON: {e}") from e
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            raise ObservationError("body must be a JSON object with a string 'model'")
        return body

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": [m.resolved.name for m in manager.all()]}

    @app.get("/v1/models")
    async def models():
        return {"models": [m.describe() for m in manager.all()]}

    @app.get("/v1/spec")
    async def spec(model: str):
        r = await run_in_threadpool(manager.resolve, model)
        return {"model": model, "path": str(r.path), "spec": PolicySpec.from_checkpoint(r.path).to_dict(),
                "license": r.license.license, "license_warning": r.license.warning}

    @app.post("/v1/models/load")
    async def load(request: Request):
        body = await _json(request)
        m = await run_in_threadpool(manager.load, body["model"], body.get("device"))
        return m.describe()

    @app.post("/v1/models/unload")
    async def unload(request: Request):
        body = await _json(request)
        await run_in_threadpool(manager.unload, body["model"])
        return {"unloaded": body["model"]}

    @app.post("/v1/models/reset")
    async def reset(request: Request):
        m = manager.get((await _json(request))["model"])

        def _do():
            with m.lock:
                m.backend.reset()
        await run_in_threadpool(_do)
        return {"reset": m.resolved.name}

    @app.post("/v1/act")
    async def act(request: Request):
        body = codec.unpackb(await request.body())
        name = body.get("model") if isinstance(body, dict) else None
        if not isinstance(name, str):
            raise ObservationError("msgpack body needs a string 'model'")
        m = manager.get(name)
        obs, seed = _parse_observation(body, m.backend.spec)
        if await request.is_disconnected():
            return Response(status_code=499)   # never start GPU work for a client that left

        def _do():
            with m.lock:
                m.calls += 1
                return m.backend.predict(obs, seed=seed)
        chunk = await run_in_threadpool(_do)
        spec = m.backend.spec
        return Response(codec.packb({
            "model": name, "actions": chunk.actions.astype(np.float32),
            "action_names": spec.action_names, "n_action_steps": spec.n_action_steps,
            "timings_ms": chunk.timings_ms}), media_type=MSGPACK)

    return app
