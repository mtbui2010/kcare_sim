"""msgpack with numpy arrays: the wire format for /v1/act.

Arrays travel raw ({dtype, shape, bytes}) or, for camera frames, as JPEG ({"__jpeg__": true,
"data": bytes}), decoded here into an HxWx3 uint8 array. The client sends JPEG by default (see
imaging.py and PolicyClient(encoding=...)); raw keeps frames bit-exact.

Every length read from the wire is checked before memory is spent on it.
"""
from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np

from .errors import ObservationError

MAX_BODY_BYTES = 64 * 1024 * 1024
ALLOWED_DTYPES = {"uint8", "float32", "float64", "int32", "int64"}
_TAG = "__nd__"
_JPEG = "__jpeg__"


def jpeg(data: bytes) -> dict:
    """Wrap JPEG bytes for the wire; decoded server-side into an HxWx3 uint8 array."""
    return {_JPEG: True, "data": data}


def _default(obj: Any):
    if isinstance(obj, np.ndarray):
        if obj.dtype.name not in ALLOWED_DTYPES:
            raise TypeError(f"cannot encode dtype {obj.dtype}")
        return {_TAG: True, "dtype": obj.dtype.name, "shape": list(obj.shape),
                "data": np.ascontiguousarray(obj).tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot encode {type(obj).__name__}")


def _hook(d: dict):
    if d.get(_JPEG) is True:
        data = d.get("data")
        if not isinstance(data, bytes):
            raise ObservationError("jpeg data must be bytes")
        from .imaging import decode_jpeg
        try:
            return decode_jpeg(data)
        except Exception as e:
            raise ObservationError(f"bad JPEG frame: {e}") from e
    if d.get(_TAG) is not True:
        return d
    dtype, shape, data = d.get("dtype"), d.get("shape"), d.get("data")
    if dtype not in ALLOWED_DTYPES:
        raise ObservationError(f"array dtype {dtype!r} not allowed ({sorted(ALLOWED_DTYPES)})")
    if (not isinstance(shape, list) or len(shape) > 4
            or not all(isinstance(s, int) and 0 <= s <= 1 << 16 for s in shape)):
        raise ObservationError(f"bad array shape {shape!r}")
    if not isinstance(data, bytes):
        raise ObservationError("array data must be bytes")
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if expected != len(data):
        raise ObservationError(f"array shape {shape} {dtype} needs {expected} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    if len(data) > MAX_BODY_BYTES:
        raise ObservationError(f"body is {len(data)} bytes, limit {MAX_BODY_BYTES}")
    try:
        return msgpack.unpackb(data, object_hook=_hook, raw=False,
                               max_bin_len=MAX_BODY_BYTES, max_str_len=1 << 20)
    except ObservationError:
        raise
    except Exception as e:  # msgpack raises several unrelated types on garbage
        raise ObservationError(f"body is not valid msgpack: {e}") from e
