"""Client-side frame preparation: resize to the network input, and JPEG for the wire.

numpy + Pillow only (part of the light client install).

Resize. `resize_with_pad` reproduces LeRobot's own (common/vla_utils.resize_with_pad and
resize_with_pad_torch): aspect-preserving bilinear resize with align_corners=False and no
antialiasing, then black padding to the target size, on the left/top (SmolVLA, X-VLA) or split
evenly (pi0 family). The policy skips its own resize when a frame already has the target size, so
the frame sent is exactly what the network sees, up to rounding to uint8 (< 0.5 grey level).

JPEG. Quality 95 with 4:4:4 chroma (no chroma subsampling) by default. Measured on the office-scene
SmolVLA, two 480x848 frames resized to its 512x512 input: 89 kB instead of 2.44 MB raw, ~11 ms to
encode + decode, and the action chunk moves by at most 0.0045 rad for the same seed — 0.12x the
spread across noise seeds. JPEG's default 4:2:0 chroma subsampling is what hurts: at quality 90
it moved actions by 0.22 rad, as much as a different noise draw. `encoding="raw"` stays bit-exact.
"""
from __future__ import annotations

import io

import numpy as np

MAX_SIDE = 8192


def _axis_weights(n_in: int, n_out: int):
    """Source indices and weights for one axis, as torch's area_pixel_compute_source_index."""
    # float32 throughout, like torch's kernel for float inputs: computing these in float64 shifts
    # the weights enough to move some pixels by > 0.5 grey level.
    f32 = np.float32
    scale = f32(n_in) / f32(n_out)
    src = (np.arange(n_out, dtype=f32) + f32(0.5)) * scale - f32(0.5)
    src = np.maximum(src, f32(0.0))
    i0 = np.floor(src).astype(np.int64)
    i0 = np.minimum(i0, n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = (src - i0.astype(f32)).astype(f32)
    return i0, i1, (f32(1.0) - w1), w1


def bilinear(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """HxWxC uint8 -> height x width x C uint8; torch bilinear, align_corners=False, no antialias."""
    if img.shape[:2] == (height, width):
        return img
    y0, y1, wy0, wy1 = _axis_weights(img.shape[0], height)
    x0, x1, wx0, wx1 = _axis_weights(img.shape[1], width)
    # Separable, float32, x first then y: the same order of operations as torch's kernel
    # (h0λ·(w0λ·p00 + w1λ·p01) + h1λ·(w0λ·p10 + w1λ·p11)), and ~10x faster than gathering 2-D.
    rows = np.unique(np.concatenate([y0, y1]))
    f = img[rows].astype(np.float32)
    tmp = f[:, x0] * wx0[None, :, None] + f[:, x1] * wx1[None, :, None]
    pos = np.searchsorted(rows, np.arange(img.shape[0]))      # row index -> position in `rows`
    out = tmp[pos[y0]] * wy0[:, None, None] + tmp[pos[y1]] * wy1[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def resize_with_pad(img: np.ndarray, height: int, width: int, pad: str) -> np.ndarray:
    """The policy's own resize-with-pad, done before sending. `pad`: "top_left" or "center"."""
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)          # int() truncation, exactly as LeRobot
    resized = bilinear(img, rh, rw)
    out = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    if pad == "top_left":
        oy, ox = height - rh, width - rw
    elif pad == "center":
        oy, ox = (height - rh) // 2, (width - rw) // 2
    else:
        raise ValueError(f"unknown pad mode {pad!r}")
    out[oy:oy + rh, ox:ox + rw] = resized
    return out


def encode_jpeg(img: np.ndarray, quality: int = 95, subsampling: int = 0) -> bytes:
    """subsampling 0 = 4:4:4 (keep full-resolution colour; see the module docstring), 2 = 4:2:0."""
    from PIL import Image
    if not (1 <= quality <= 100):
        raise ValueError("jpeg quality must be in 1..100")
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img), "RGB").save(buf, format="JPEG", quality=quality,
                                                           subsampling=subsampling)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> HxWx3 uint8. The header is checked before any pixel is decoded."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if im.format != "JPEG":
        raise ValueError(f"expected JPEG, got {im.format}")
    w, h = im.size
    if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE):
        raise ValueError(f"JPEG of {w}x{h} exceeds {MAX_SIDE} px")
    return np.array(im.convert("RGB"))          # a writable copy (torch.from_numpy warns otherwise)
