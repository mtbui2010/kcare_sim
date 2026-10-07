#!/usr/bin/env python3
"""Render a narrated tutorial video from a storyboard: PIL frames -> ffmpeg (H.264 + AAC).

    python render_video.py storyboard.json [--out video.mp4] [--stills DIR] [--only id1,id2]

  --stills DIR   write one PNG per scene (at 60 % of the scene) instead of encoding: fast layout QC
  --only ids     render only these scenes (with --stills, or to preview a section)

Paths inside the storyboard are relative to the storyboard file. The narration comes from
tts.py: <voice.dir>/<scene_id>.wav and <voice.dir>/timing.json. Each scene lasts
lead + narration + tail seconds (a video scene may be longer, see below), subtitles show the
sentence being spoken, and the audio track is assembled to match.

Scene types (full field list: references/storyboard.md):
  title         big title, subtitle lines, a small note
  two_column    problem -> answer rows that appear one by one
  diagram       boxes with titled text and arrows that appear at given times
  terminal      commands that type out, then their (real, captured) output
  code_terminal a code panel above a terminal
  image         one picture (a chart, a screenshot), fitted
  video         a recorded clip (optionally split into two camera views) with overlays
  results       an optional highlight box + success-rate rows with 95 % Wilson intervals
  end           closing card with key/value links

Only Pillow + numpy are needed (plus ffmpeg/ffprobe on PATH). Fonts: DejaVu from matplotlib if
installed, else fc-match, else the storyboard's "fonts" block.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------------------------------
# configuration


THEME = {
    "bg": [252, 252, 251], "ink": [11, 11, 11], "ink2": [82, 81, 78], "muted": [137, 135, 129],
    "term": [26, 26, 25], "term_ink": [236, 236, 232], "term_dim": [160, 158, 150],
    "accent": [42, 120, 214], "accent2": [235, 104, 52], "good": [27, 175, 122], "grid": [225, 224, 217],
    "prompt": [110, 180, 255], "ok": [120, 220, 150], "bad": [255, 150, 120], "panel": [245, 244, 240],
    "accent_soft": [232, 240, 251], "good_soft": [232, 246, 240],
}


def find_fonts(cfg):
    if cfg.get("fonts"):
        return cfg["fonts"]
    try:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if (d / "DejaVuSans.ttf").exists():
            return {"sans": str(d / "DejaVuSans.ttf"), "bold": str(d / "DejaVuSans-Bold.ttf"),
                    "mono": str(d / "DejaVuSansMono.ttf"), "monob": str(d / "DejaVuSansMono-Bold.ttf")}
    except ImportError:
        pass
    out = {}
    for k, q in {"sans": "DejaVu Sans", "bold": "DejaVu Sans:bold", "mono": "DejaVu Sans Mono",
                 "monob": "DejaVu Sans Mono:bold"}.items():
        r = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True)
        out[k] = r.stdout.strip()
    if not all(out.values()):
        sys.exit("no fonts found: install matplotlib, fontconfig+DejaVu, or set storyboard 'fonts'")
    return out


class Ctx:
    def __init__(self, sb_path: Path):
        self.root = sb_path.parent
        self.sb = json.loads(sb_path.read_text())
        v = self.sb.get("video", {})
        self.W, self.H, self.FPS = v.get("width", 1280), v.get("height", 720), v.get("fps", 30)
        self.lead, self.tail, self.fade = v.get("lead", 0.4), v.get("tail", 0.7), v.get("fade", 0.3)
        self.theme = {**THEME, **{k: list(x) for k, x in v.get("theme", {}).items()}}
        self.T = {k: tuple(x) for k, x in self.theme.items()}
        self.fonts = find_fonts(self.sb)
        self._font_cache = {}
        voice = self.sb.get("voice", {"dir": "voice"})
        self.voice_dir = self.root / voice["dir"]
        tf = self.voice_dir / "timing.json"
        self.timing = json.loads(tf.read_text()) if tf.exists() else {}
        self.display_map = self.sb.get("display_map", [])
        self.replace = self.sb.get("replace", {})       # text replacements for terminal output
        self.subtitle_cfg = {"size": 22, "max_lines": 3, **self.sb.get("subtitles", {})}

    def font(self, kind, size):
        key = (kind, size)
        if key not in self._font_cache:
            self._font_cache[key] = ImageFont.truetype(self.fonts[kind], size)
        return self._font_cache[key]

    def path(self, p):
        return (self.root / p).resolve()

    def narration(self, sid):
        return self.timing.get(sid, {}).get("duration", 0.0)


# ------------------------------------------------------------------------------------------
# drawing primitives


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def header(c, d, step, title):
    if step:
        d.text((48, 30), step.upper(), font=c.font("bold", 18), fill=c.T["accent"])
    if title:
        d.text((48, 54), title, font=c.font("bold", 34), fill=c.T["ink"])


def arrow(d, a, b, color, width=4):
    d.line([tuple(a), tuple(b)], fill=color, width=width)
    v = np.array(b, float) - np.array(a, float); v /= max(np.linalg.norm(v), 1e-6)
    n = np.array([-v[1], v[0]]); p = np.array(b, float)
    d.polygon([tuple(p), tuple(p - 16 * v + 8 * n), tuple(p - 16 * v - 8 * n)], fill=color)


def color(c, name):
    return c.T.get(name, c.T["accent"]) if isinstance(name, str) else tuple(name)


def display(c, text):
    for spoken, shown in c.display_map:
        text = text.replace(spoken, shown)
    return text


def subtitles(c, im, sid, t):
    tt = t - c.lead
    cur = next((s for s, a, b in c.timing.get(sid, {}).get("sentences", []) if a - 0.05 <= tt <= b + 0.15), None)
    if not cur:
        return
    cur = display(c, cur)
    d = ImageDraw.Draw(im, "RGBA")
    f = c.font("sans", c.subtitle_cfg["size"])
    lines = wrap(d, cur, f, c.W - 160)[: c.subtitle_cfg["max_lines"]]
    lh = int(c.subtitle_cfg["size"] * 1.36)
    h = lh * len(lines) + 18
    y0 = c.H - h - 12
    d.rounded_rectangle((60, y0, c.W - 60, c.H - 12), 10, fill=(20, 20, 20, 200))
    for i, line in enumerate(lines):
        w = d.textlength(line, font=f)
        d.text(((c.W - w) / 2, y0 + 9 + lh * i), line, font=f, fill=(255, 255, 255))


def text_lines(c, spec):
    """A terminal output: inline list, or a captured file (with `replace` applied)."""
    if "output" in spec:
        lines = spec["output"]
    elif "output_file" in spec:
        lines = c.path(spec["output_file"]).read_text().splitlines()
        if "lines" in spec:                         # [start, end] slice of the file
            a, b = spec["lines"]
            lines = lines[a:b]
    else:
        lines = []
    out = []
    for l in lines:
        for k, v in c.replace.items():
            l = l.replace(k, v)
        out.append(l.rstrip())
    return out


def terminal(c, d, cmds, t, box, size=17, pace=1.0, ok_words=(), bad_words=()):
    x0, y0, x1, y1 = box
    T = c.T
    d.rounded_rectangle(box, 12, fill=T["term"])
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((x0 + 16 + 20 * i, y0 + 14, x0 + 28 + 20 * i, y0 + 26), fill=col)
    f, fb = c.font("mono", size), c.font("monob", size)
    lh = int(size * 1.45)
    y, clock = y0 + 44, 0.0
    maxw = max(10, int((x1 - x0 - 40) / d.textlength("m", font=f)))
    ok_words = tuple(ok_words) + ("OK:", '"status":"ok"', "passed")
    bad_words = tuple(bad_words) + ("Error", "error:", "FAIL", "problem")
    for spec in cmds:
        cmd, out = spec["cmd"], text_lines(c, spec)
        if t < clock:
            break
        n = len(cmd)
        typed = int(np.clip((t - clock) / (0.9 * pace) * n, 0, n))
        for k, part in enumerate([cmd[i:i + maxw - 2] for i in range(0, max(n, 1), maxw - 2)]):
            shown = part[: max(0, typed - k * (maxw - 2))]
            if k == 0:
                d.text((x0 + 20, y), "$", font=fb, fill=T["prompt"])
            d.text((x0 + 20 + d.textlength("$ ", font=f), y), shown, font=fb, fill=T["term_ink"])
            y += lh
        clock += 1.1 * pace
        for j, line in enumerate(out):
            if t < clock + 0.06 * j:
                return
            for piece in [line[i:i + maxw] for i in range(0, max(len(line), 1), maxw)] or [""]:
                if y > y1 - lh:
                    return
                col = T["term_dim"] if line.lstrip().startswith(("#", "INFO", "WARNING")) else T["term_ink"]
                if any(w in line for w in ok_words):
                    col = T["ok"]
                if any(w in line for w in bad_words):
                    col = T["bad"]
                d.text((x0 + 20, y), piece, font=f, fill=col)
                y += lh
        clock += (0.06 * len(out) + 1.2) * pace
        y += 6


def code_panel(c, d, lines, box, size=16):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, 12, fill=c.T["panel"], outline=c.T["grid"], width=2)
    f = c.font("mono", size)
    for i, l in enumerate(lines):
        col = c.T["muted"] if l.lstrip().startswith(("#", "//")) else c.T["ink"]
        d.text((x0 + 18, y0 + 14 + int(size * 1.45) * i), l, font=f, fill=col)


def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n; ce = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (ce - r) / dd, 100 * (ce + r) / dd


# ------------------------------------------------------------------------------------------
# scene types: each returns (draw_fn(t) -> Image, duration)


def canvas(c):
    im = Image.new("RGB", (c.W, c.H), c.T["bg"])
    return im, ImageDraw.Draw(im)


def base_duration(c, sc):
    return sc.get("duration", c.narration(sc["id"]) + c.lead + c.tail)


def scene_title(c, sc):
    def f(t):
        im, d = canvas(c)
        y = sc.get("y", 210)
        d.text((48, y), sc["title"], font=c.font("bold", sc.get("title_size", 84)), fill=c.T["ink"])
        for i, l in enumerate(sc.get("subtitle_lines", [])):
            d.text((52, y + 110 + 44 * i), l, font=c.font("sans", 34), fill=c.T["ink2"])
        if sc.get("note"):
            d.text((52, y + 110 + 44 * len(sc.get("subtitle_lines", [])) + 40), sc["note"],
                   font=c.font("sans", 19), fill=c.T["muted"])
        return im
    return f, base_duration(c, sc)


def scene_two_column(c, sc):
    dur = base_duration(c, sc)
    rows = sc["rows"]
    reveal_end = sc.get("reveal_end", max(1.0, c.narration(sc["id"]) - 3))

    def f(t):
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        d.text((70, 128), sc.get("left_label", ""), font=c.font("bold", 20), fill=c.T["muted"])
        d.text((700, 128), sc.get("right_label", ""), font=c.font("bold", 20), fill=c.T["muted"])
        for i, (left, right) in enumerate(rows):
            if t < c.lead + 1.5 + i * reveal_end / len(rows):
                break
            y = 172 + 78 * i
            d.rounded_rectangle((60, y, 620, y + 60), 10, fill=(255, 255, 255), outline=c.T["grid"], width=2)
            d.text((80, y + 17), left, font=c.font("sans", 20), fill=c.T["ink"])
            arrow(d, (630, y + 30), (686, y + 30), c.T["accent"], 3)
            d.rounded_rectangle((696, y, c.W - 58, y + 60), 10, fill=c.T["accent_soft"])
            d.text((716, y + 17), right, font=c.font("sans", 20), fill=c.T["accent"])
        return im
    return f, dur


def scene_diagram(c, sc):
    def f(t):
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        for b in sc.get("boxes", []):
            x0, y0, x1, y1 = b["rect"]
            d.rounded_rectangle((x0, y0, x1, y1), 16, outline=c.T["grid"], width=3, fill=(255, 255, 255))
            d.text((x0 + 28, y0 + 24), b.get("title", ""), font=c.font("bold", 28), fill=c.T["ink"])
            for i, l in enumerate(b.get("lines", [])):
                mono = l.startswith("$ ") or b.get("mono_lines", False)
                d.text((x0 + 28, y0 + 80 + 34 * i), l.removeprefix("$ ") if mono else l,
                       font=c.font("mono" if mono else "sans", 20), fill=c.T["ink2"])
        for a in sc.get("arrows", []):
            if t < a.get("at", 0):
                continue
            col = color(c, a.get("color", "accent"))
            arrow(d, a["from"], a["to"], col)
            if a.get("label"):
                lx, ly = a.get("label_xy", (min(a["from"][0], a["to"][0]) + 18, a["from"][1] - 44))
                d.text((lx, ly), a["label"], font=c.font("sans", 18), fill=col)
        return im
    return f, base_duration(c, sc)


def scene_terminal(c, sc):
    box = tuple(sc.get("box", (48, 112, c.W - 48, 590)))

    def f(t):
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        terminal(c, d, sc["commands"], t - c.lead, box, sc.get("size", 17), sc.get("pace", 1.0),
                 sc.get("ok_words", ()), sc.get("bad_words", ()))
        return im
    return f, base_duration(c, sc)


def scene_code_terminal(c, sc):
    code = sc["code"] if isinstance(sc["code"], list) else c.path(sc["code"]).read_text().splitlines()
    split = sc.get("split_y", 300)

    def f(t):
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        code_panel(c, d, code, (48, 112, c.W - 48, split), sc.get("code_size", 16))
        terminal(c, d, sc["commands"], t - c.lead - sc.get("terminal_delay", 2.0),
                 (48, split + 16, c.W - 48, 590), sc.get("size", 16), sc.get("pace", 1.6),
                 sc.get("ok_words", ()), sc.get("bad_words", ()))
        return im
    return f, base_duration(c, sc)


def scene_image(c, sc):
    img = Image.open(c.path(sc["image"])).convert("RGB")
    box = sc.get("box", (48, 120, c.W - 48, 600))
    img.thumbnail((box[2] - box[0], box[3] - box[1]))

    def f(t):
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        im.paste(img, (box[0] + (box[2] - box[0] - img.width) // 2, box[1] + 10))
        return im
    return f, base_duration(c, sc)


def read_video(path: Path):
    """All frames of a clip as RGB arrays, via ffmpeg (no OpenCV needed)."""
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    w, h = probe["width"], probe["height"]
    num, den = map(int, probe["r_frame_rate"].split("/"))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return frames, num / den


def scene_video(c, sc):
    frames, src_fps = read_video(c.path(sc["video"]))
    n = len(frames)
    clip_s = n / src_fps
    narr = c.narration(sc["id"])
    dur = sc.get("duration", max(narr + c.lead + c.tail, clip_s + 2.0))
    speed = sc.get("speed") or min(1.0, clip_s / max(0.5, dur - c.lead - 1.5))   # slow down to fit narration
    data = json.loads(c.path(sc["data"]).read_text()) if sc.get("data") else {}
    layout = sc.get("layout", "split")              # "split": left/right halves = two cameras; "full"
    panel_x = 830

    def f(t):
        i = int(np.clip((t - c.lead) * src_fps * speed, 0, n - 1))
        ts = i / src_fps
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        fr = frames[i]
        if layout == "split":
            a = Image.fromarray(fr[:, : fr.shape[1] // 2]).resize((768, 435))
            b = Image.fromarray(fr[:, fr.shape[1] // 2:]).resize((420, 238))
            im.paste(a, (48, 110)); im.paste(b, (panel_x, 110))
            names = sc.get("view_names", ["camera 1", "camera 2"])
            d.text((56, 116), names[0], font=c.font("bold", 16), fill=(255, 255, 255))
            d.text((panel_x + 8, 116), names[1], font=c.font("bold", 16), fill=(255, 255, 255))
            info_x, info_y, info_w = panel_x, 362, 420
        elif layout == "side":                    # big clip on the left (e.g. a 2x2 camera grid), info panel right
            v = Image.fromarray(fr); v.thumbnail((860, 490))
            im.paste(v, (40, 108))
            info_x, info_y, info_w = 40 + v.width + 24, 116, c.W - (40 + v.width + 24) - 32
        else:
            v = Image.fromarray(fr); v.thumbnail((c.W - 96, 470))
            im.paste(v, ((c.W - v.width) // 2, 110))
            info_x, info_y, info_w = 48, 110 + v.height + 12, c.W - 96
        if speed < 0.99:
            d.text((48 + 768 - 50 if layout == "split" else c.W - 120, 116), f"{speed:.1f}×",
                   font=c.font("bold", 16), fill=(255, 255, 255))
        y = info_y
        for line in sc.get("info", []):             # static lines; {t} = clip time, {step} = frame
            d.text((info_x, y), line.format(t=ts, step=i), font=c.font("mono", 18 if layout != "side" else 14), fill=c.T["ink"]); y += 30 if layout != "side" else 24
        ev = data.get("events")                     # [{"frame": k, "text": "..."}]: the latest one shows
        if ev:
            cur = [e for e in ev if e["frame"] <= i]
            if cur:
                fresh = i - cur[-1]["frame"] < sc.get("flash_frames", 8)
                d.rounded_rectangle((info_x, y + 2, info_x + info_w, y + 36), 8,
                                    fill=c.T["accent"] if fresh else c.T["accent_soft"])
                d.text((info_x + 10, y + 9), cur[-1]["text"], font=c.font("monob" if fresh else "mono", 18 if layout != "side" else 13),
                       fill=(255, 255, 255) if fresh else c.T["accent"])
            y += 48
        pr = data.get("progress")                   # {"values": [...per frame], "label", "max", "goal", "goal_label"}
        if pr:
            vals = pr["values"]
            v = max(0.0, float(vals[min(i, len(vals) - 1)]))
            d.text((info_x, y), pr.get("label", "{v:.1f}").format(v=v), font=c.font("mono", 18 if layout != "side" else 14), fill=c.T["ink"])
            d.rounded_rectangle((info_x, y + 30, info_x + info_w, y + 50), 6, fill=(240, 239, 236))
            d.rounded_rectangle((info_x, y + 30, info_x + max(12, int(info_w * min(v, pr["max"]) / pr["max"])), y + 50),
                                6, fill=c.T["accent2"])
            if pr.get("goal") is not None:
                gx = info_x + int(info_w * pr["goal"] / pr["max"])
                d.line([(gx, y + 24), (gx, y + 56)], fill=c.T["ink"], width=2)
                d.text((gx - 30, y + 58), pr.get("goal_label", ""), font=c.font("sans", 14), fill=c.T["muted"])
        badge = data.get("badge")                   # {"from_frame": k, "text": "✓ success"}
        if badge and i >= badge["from_frame"]:
            bx = 48 if layout != "side" else info_x
            by = 110 if layout != "side" else y + 90
            d.rounded_rectangle((bx, by, bx + 14 * len(badge["text"]) + 30, by + 42), 8, fill=c.T["good"])
            d.text((bx + 12, by + 7), badge["text"], font=c.font("bold", 22), fill=(255, 255, 255))
        return im
    return f, dur


def scene_results(c, sc):
    def f(t):
        im, d = canvas(c)
        header(c, d, sc.get("step"), sc.get("heading"))
        y = 112
        hb = sc.get("highlight_box")
        if hb:
            d.rounded_rectangle((48, y, c.W - 48, y + 84), 12, fill=c.T["good_soft"])
            d.text((70, y + 14), hb["title"], font=c.font("bold", 20), fill=c.T["ink"])
            d.text((70, y + 46), hb["text"], font=c.font("sans", 18), fill=c.T["ink2"])
            y += 102
        if sc.get("rows_title"):
            d.text((70, y), sc["rows_title"], font=c.font("bold", 20), fill=c.T["ink"]); y += 44
        if t < c.lead + sc.get("rows_at", 0):
            return im
        lo_ax = sc.get("axis_min", 60)
        x0, x1 = 600, c.W - 100
        X = lambda v: x0 + (x1 - x0) * (v - lo_ax) / (100 - lo_ax)
        rows = sc["rows"]
        for j, r in enumerate(rows):
            yy = y + 64 * j
            d.text((70, yy), r["label"], font=c.font("sans", 19), fill=c.T["ink"])
            d.text((70, yy + 24), r.get("sub", ""), font=c.font("sans", 14), fill=c.T["muted"])
            d.line([(x0, yy + 18), (x1, yy + 18)], fill=c.T["grid"], width=2)
            lo, hi = wilson(r["k"], r["n"])
            d.line([(X(max(lo, lo_ax)), yy + 18), (X(hi), yy + 18)], fill=(195, 194, 183), width=8)
            p = 100 * r["k"] / r["n"]
            col = c.T["accent"] if r.get("highlight", True) else c.T["muted"]
            d.ellipse((X(p) - 9, yy + 9, X(p) + 9, yy + 27), fill=col, outline=c.T["bg"], width=3)
            d.text((X(p) - 20, yy - 14), f'{r["k"]}/{r["n"]}', font=c.font("bold", 16), fill=c.T["ink"])
        yb = y + 64 * len(rows)
        for v in range(lo_ax, 101, 10):
            d.text((X(v) - 14, yb - 14), f"{v}%", font=c.font("sans", 14), fill=c.T["muted"])
        d.text((x0, yb + 6), sc.get("axis_note", "dot = success rate · bar = 95 % interval"),
               font=c.font("sans", 13), fill=c.T["muted"])
        return im
    return f, base_duration(c, sc)


def scene_end(c, sc):
    def f(t):
        im, d = canvas(c)
        d.text((48, 190), sc.get("title", "Try it"), font=c.font("bold", 56), fill=c.T["ink"])
        for i, (k, v) in enumerate(sc.get("links", [])):
            d.text((52, 300 + 62 * i), k, font=c.font("sans", 24), fill=c.T["muted"])
            d.text((380, 298 + 62 * i), v, font=c.font("monob", 26), fill=c.T["accent"])
        return im
    return f, base_duration(c, sc)


SCENES = {"title": scene_title, "two_column": scene_two_column, "diagram": scene_diagram,
          "terminal": scene_terminal, "code_terminal": scene_code_terminal, "image": scene_image,
          "video": scene_video, "results": scene_results, "end": scene_end}


# ------------------------------------------------------------------------------------------
# assembly


def read_wav(path: Path, sr: int) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != sr or w.getnchannels() != 1:
            sys.exit(f"{path}: expected mono {sr} Hz")
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype, scale = {2: (np.int16, 32768.0), 4: (np.int32, 2147483648.0)}[width]
    return np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("storyboard")
    ap.add_argument("--out")
    ap.add_argument("--stills")
    ap.add_argument("--only")
    a = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")
    sb_path = Path(a.storyboard).resolve()
    c = Ctx(sb_path)
    only = set(a.only.split(",")) if a.only else None
    built = []
    for sc in c.sb["scenes"]:
        if only and sc["id"] not in only:
            continue
        if sc["type"] not in SCENES:
            sys.exit(f"scene {sc['id']}: unknown type {sc['type']!r} (one of {sorted(SCENES)})")
        if sc["id"] not in c.timing and "duration" not in sc:
            print(f"warning: scene {sc['id']} has no narration and no duration; it will last {c.lead + c.tail:.1f} s")
        fn, dur = SCENES[sc["type"]](c, sc)
        built.append((sc["id"], fn, dur))
    total = sum(dur for _, _, dur in built)
    print(f"{len(built)} scenes, {total:.1f} s")

    if a.stills:
        out = Path(a.stills); out.mkdir(parents=True, exist_ok=True)
        for sid, fn, dur in built:
            t = sc_t = 0.6 * dur
            im = fn(sc_t); subtitles(c, im, sid, t)
            im.save(out / f"{sid}.png")
        print("wrote stills to", out)
        return

    sr = c.sb.get("voice", {}).get("sample_rate", 24000)
    out = Path(a.out or c.sb.get("output", "video.mp4"))
    if not out.is_absolute():
        out = c.root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    track = np.zeros(int(total * sr) + sr, np.float32)
    t0 = 0.0
    for sid, _, dur in built:
        wav = c.voice_dir / f"{sid}.wav"
        if wav.exists():
            s = read_wav(wav, sr)
            i = int((t0 + c.lead) * sr)
            track[i:i + len(s)] = s[: max(0, len(track) - i)]
        t0 += dur
    wav_out = out.with_suffix(".tmp.wav")
    with wave.open(str(wav_out), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    v = c.sb.get("video", {})
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{c.W}x{c.H}", "-r", str(c.FPS), "-i", "-", "-i", str(wav_out),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(v.get("crf", 26)), "-preset", "slow",
                           "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    bg = Image.new("RGB", (c.W, c.H), c.T["bg"])
    for sid, fn, dur in built:
        for k in range(int(dur * c.FPS)):
            t = k / c.FPS
            im = fn(t)
            subtitles(c, im, sid, t)
            alpha = min(1.0, t / c.fade, (dur - t) / c.fade)
            if alpha < 1:
                im = Image.blend(bg, im, max(alpha, 0.0))
            ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    wav_out.unlink()
    if ff.returncode:
        sys.exit("ffmpeg failed")
    starts, t0 = {}, 0.0
    for sid, _, dur in built:
        starts[sid] = round(t0, 2); t0 += dur
    (out.with_suffix(".scenes.json")).write_text(json.dumps(starts, indent=1))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB;", "scene starts in", out.with_suffix(".scenes.json").name)


if __name__ == "__main__":
    main()
