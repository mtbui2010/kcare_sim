"""vla-serve command line.

    vla-serve list                                   aliases and robot profiles
    vla-serve inspect MODEL                          spec + licence, from config.json only
    vla-serve check MODEL --profile PROFILE          robot vs policy, before anything moves
    vla-serve serve [MODEL ...] [--port 11450]       start the server, optionally preloading
    vla-serve bench MODEL [-n 20]                    in-process latency with synthetic inputs

MODEL is a registry alias, a local checkpoint directory, or an HF repo id.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from .errors import VLAServeError

DEFAULT_PORT = 11450


def _spec(model: str):
    from .registry import resolve
    from .spec import PolicySpec
    r = resolve(model)
    return r, PolicySpec.from_checkpoint(r.path)


def cmd_list(args) -> int:
    from importlib import resources

    from .registry import USER_REGISTRY, load_registry
    print("models (bundled + %s):" % USER_REGISTRY)
    for name, e in sorted(load_registry().items()):
        print(f"  {name:32s} {e.source}")
        if e.description:
            print(f"  {'':32s} {' '.join(e.description.split())[:100]}")
    print("\nrobot profiles:")
    for f in sorted(resources.files("vlaserve").joinpath("data/profiles").iterdir()):
        if f.name.endswith(".yaml"):
            print(f"  {f.name[:-5]}")
    print("\nAny LeRobot checkpoint also works by HF repo id or local path.")
    return 0


def cmd_inspect(args) -> int:
    r, spec = _spec(args.model)
    out = {"model": args.model, "path": str(r.path), "license": r.license.license,
           "license_source": r.license.source, "license_warning": r.license.warning,
           "spec": spec.to_dict()}
    print(json.dumps(out, indent=2))
    return 0


def cmd_check(args) -> int:
    from .profiles import RobotProfile
    _, spec = _spec(args.model)
    profile = RobotProfile.load(args.profile)
    problems = profile.check(spec)
    if not problems:
        print(f"OK: profile '{profile.name}' is compatible with {args.model}")
        return 0
    print(f"{len(problems)} problem(s) between profile '{profile.name}' and {args.model}:")
    for p in problems:
        print(f"  - {p}")
    return 1


def cmd_serve(args) -> int:
    import uvicorn

    from .server import ModelManager, create_app
    manager = ModelManager(default_device=args.device)
    for m in args.models:
        manager.load(m)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning("listening on %s: anyone who can reach this port can drive the robot", args.host)
    uvicorn.run(create_app(manager), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_bench(args) -> int:
    from .backends import Observation, load_backend
    from .registry import resolve
    r = resolve(args.model)
    if r.license.warning:
        print(f"licence: {r.license.warning}", file=sys.stderr)
    t0 = time.perf_counter()
    backend = load_backend(r.entry.backend, r.path, args.device, r.entry.options)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s on {args.device}", file=sys.stderr)   # stdout stays pure JSON
    spec = backend.spec
    rng = np.random.default_rng(0)
    obs = Observation(
        images={c: rng.integers(0, 256, (shape[1], shape[2], shape[0]) if shape else (480, 640, 3),
                                dtype=np.uint8)
                for c, shape in spec.cameras.items()},
        state=rng.standard_normal(spec.state_dim).astype(np.float32) if spec.state_dim else None,
        task="pick up the object" if spec.language else None)
    for _ in range(args.warmup):
        backend.predict(obs, seed=0)
    rows = [backend.predict(obs, seed=0).timings_ms for _ in range(args.n)]
    out = {"model": args.model, "device": args.device, "n": args.n, "load_s": round(load_s, 1),
           "chunk_shape": list(backend.predict(obs, seed=0).actions.shape)}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {"median_ms": round(float(np.median(v)), 2), "p95_ms": round(float(np.percentile(v, 95)), 2)}
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vla-serve", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("inspect"); s.add_argument("model"); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("check"); s.add_argument("model"); s.add_argument("--profile", required=True)
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("serve"); s.add_argument("models", nargs="*")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--device", default="cuda"); s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("bench"); s.add_argument("model"); s.add_argument("-n", type=int, default=20)
    s.add_argument("--warmup", type=int, default=3); s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    if args.cmd in ("serve", "bench"):
        import importlib.util

        def absent(mod: str) -> bool:
            try:
                return importlib.util.find_spec(mod) is None
            except (ImportError, ValueError):
                return True
        missing = [m for m in ("torch", "lerobot", "fastapi", "uvicorn") if absent(m)]
        if missing:
            print(f"error: '{args.cmd}' needs the server dependencies (missing: {', '.join(missing)}).\n"
                  "  pip install 'vla-serve[server]'   (install the torch build for your CUDA first)\n"
                  "  or run the container: see deploy/README.md", file=sys.stderr)
            return 2
    logging.basicConfig(level=logging.INFO if args.verbose or args.cmd == "serve" else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except (VLAServeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
