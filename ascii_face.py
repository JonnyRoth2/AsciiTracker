#!/usr/bin/env python3
"""Webcam face -> live ASCII art in the terminal.

Face detection uses OpenCV's bundled Haar cascade -- a trained Viola-Jones
classifier (learned from thousands of face images) that ships inside the
opencv package itself. Fully offline: no downloads, no API calls.

Run it in a real terminal (Windows Terminal recommended). Shrink the terminal
font (Ctrl+minus) to get more character cells and therefore more detail.
"""

import argparse
import contextlib
import itertools
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np

RAMP = np.array(list(
    " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"))
BG_RAMP = np.array(list(" .:-=+*#%@"))

CHAR_ASPECT = 0.5    # terminal glyph width / height
PAD_TOP = 0.35       # extend the detected box up for forehead/hair
PAD_BOTTOM = 0.08
PAD_SIDE = 0.10
SMOOTH = 0.35        # how fast the smoothed box chases the detection
HOLD_S = 1.2         # keep the last box this long after losing the face
DETECT_W = 320       # detection runs on a frame downscaled to this width
EDGE_FADE = 0.18     # outer fraction of the head oval that fades to blank
FPS_CAP = 30

FACE_SGR = "\x1b[92m"   # bright green
BG_SGR = "\x1b[90m"     # dim gray
RESET = "\x1b[0m"

if os.name == "nt":
    import msvcrt

    @contextlib.contextmanager
    def raw_terminal():
        yield

    def read_keys():
        keys = []
        while msvcrt.kbhit():
            keys.append(msvcrt.getwch().lower())
        return keys
else:
    import select
    import termios
    import tty

    @contextlib.contextmanager
    def raw_terminal():
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            old = None  # stdin isn't a tty; keys just won't work
        if old is not None:
            tty.setcbreak(fd)
        try:
            yield
        finally:
            if old is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def read_keys():
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.append(ch.lower())
        return keys


def detect_face(cascade, gray):
    """Largest detected face as [x, y, w, h] floats in full-frame coords."""
    scale = DETECT_W / gray.shape[1]
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    faces = cascade.detectMultiScale(small, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return np.array([x, y, w, h], dtype=np.float64) / scale


def detect_expression(eye_cascade, smile_cascade, gray, box):
    """Expression signals for the avatar: (eyes_found, smile_found,
    mouth_openness). Eyes and smile come from two more trained cascades that
    ship inside the opencv package; mouth openness is the fraction of dark
    "cavity" pixels in the mouth region, so talking animates continuously.
    All three are noisy per frame -- callers should smooth over time."""
    vh, vw = gray.shape
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1 = min(vw, int(box[0] + box[2]))
    y1 = min(vh, int(box[1] + box[3]))
    roi = gray[y0:y1, x0:x1]
    h, w = roi.shape
    if h < 24 or w < 24:
        return True, False, 0.0
    eyes = eye_cascade.detectMultiScale(
        roi[: h // 2], 1.1, 5, minSize=(w // 12, w // 12))
    smiles = smile_cascade.detectMultiScale(
        roi[h // 2 :], 1.7, 20, minSize=(w // 4, h // 8))

    mreg = roi[int(h * 0.62):int(h * 0.88), int(w * 0.30):int(w * 0.70)]
    mouth = 0.0
    if mreg.size:
        dark = (mreg < np.median(roi) * 0.55).mean()
        mouth = float(np.clip((dark - 0.04) / 0.30, 0.0, 1.0))
    return len(eyes) > 0, len(smiles) > 0, mouth


LIGHT = (-0.45, -0.55, 0.70)      # light direction for the avatar shading


def draw_head(chars, layer, gx0, gy0, fgw, fgh, tex, u, v, rr, mask,
              eyes_open, smiling, mouth):
    """Puppet a shaded 3d ascii head over the grid region: an ellipsoid lit
    from the upper left, its surface modulated by the real face texture so
    the drawing wraps around the actual head. Features are drawn on top in
    cleared patches; `mouth` (0..1) animates talking."""
    rows, cols = chars.shape

    # lambert shading of an ellipsoid: surface normal is (u, v, nz)
    nz = np.sqrt(np.clip(1.0 - u * u - v * v, 0.0, 1.0))
    lam = np.clip(LIGHT[0] * u + LIGHT[1] * v + LIGHT[2] * nz, 0.0, 1.0)
    shade = (0.18 + 0.82 * lam) * (0.45 + 0.55 * tex)
    fill = RAMP[np.rint(np.clip(shade, 0.0, 1.0) * (len(RAMP) - 1)).astype(int)]
    chars[gy0:gy0 + fgh, gx0:gx0 + fgw][mask] = fill[mask]
    layer[gy0:gy0 + fgh, gx0:gx0 + fgw][mask] = fill[mask] != " "

    # rim: walk a parametric ellipse, pick the char by tangent direction
    a, b = (fgw - 1) / 2.0, (fgh - 1) / 2.0
    cx, cy = gx0 + a, gy0 + b
    for i in range(max(24, 4 * (fgw + fgh))):
        th = 2 * math.pi * i / max(24, 4 * (fgw + fgh))
        c = int(round(cx + a * math.cos(th)))
        r = int(round(cy + b * math.sin(th)))
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        ang = math.degrees(math.atan2(b * math.cos(th),
                                      -a * math.sin(th))) % 180
        chars[r, c] = ("-" if ang < 22.5 or ang >= 157.5 else
                       "\\" if ang < 67.5 else
                       "|" if ang < 112.5 else "/")
        layer[r, c] = True

    if fgw < 12 or fgh < 7:
        return  # too small for features, shaded ball only

    def put_row(r, u0, s, pad=1):
        # write s centered at column for u0, clearing `pad` cells each side;
        # spaces inside s clear too, so open mouths read as dark cavities
        c0 = gx0 + int(round((u0 + 1) / 2 * (fgw - 1))) - len(s) // 2
        if not 0 <= r < rows:
            return
        for i in range(-pad, len(s) + pad):
            c = c0 + i
            if 0 <= c < cols:
                ch = s[i] if 0 <= i < len(s) else " "
                chars[r, c] = ch
                layer[r, c] = ch != " "

    def put_text(u0, v0, s, pad=1):
        put_row(gy0 + int(round((v0 + 1) / 2 * (fgh - 1))), u0, s, pad)

    n = max(1, fgw // 30)                        # feature width scales up
    eye = "(" + "O" * n + ")" if eyes_open else "-" * (n + 2)
    put_text(-0.4, -0.25, eye)
    put_text(0.4, -0.25, eye)
    if fgh >= 12:
        bv = -0.5 - (0.08 if smiling else 0.0)   # brows lift with a smile
        put_text(-0.4, bv, "~" * (n + 2))
        put_text(0.4, bv, "~" * (n + 2))
        put_text(0, 0.0, "|", pad=0)
        put_text(0, 0.16, "|", pad=0)

    # mouth: openness drives talking; a smile curves the closed mouth
    w = max(3, int(fgw * 0.20))
    if mouth < 0.15:
        rws = ["\\" + "_" * w + "/"] if smiling else ["-" * (w + 2)]
    elif mouth < 0.45:
        rws = ["(" + "_" * w + ")"]
    elif mouth < 0.75:
        rws = ["/" + " " * w + "\\",
               "\\" + "_" * w + "/"]
    else:
        rws = ["/" + "-" * w + "\\",
               "|" + " " * w + "|",
               "\\" + "_" * w + "/"]
    r0 = gy0 + int(round((0.52 + 1) / 2 * (fgh - 1))) - (len(rws) - 1) // 2
    for j, s in enumerate(rws):
        put_row(r0 + j, 0, s)


def pad_box(box, vw, vh):
    """Padded head box as clamped integer pixel bounds x0, y0, x1, y1."""
    x, y, w, h = box
    x -= w * PAD_SIDE
    y -= h * PAD_TOP
    w *= 1 + 2 * PAD_SIDE
    h *= 1 + PAD_TOP + PAD_BOTTOM
    return (max(0, int(x)), max(0, int(y)),
            min(vw, int(x + w)), min(vh, int(y + h)))


def fit_grid(max_cols, max_rows, w, h):
    """Largest character grid matching the w:h aspect, given ~2:1 cells."""
    aspect = w / h
    cols = max_cols
    rows = round(cols * CHAR_ASPECT / aspect)
    if rows > max_rows:
        rows = max_rows
        cols = min(max_cols, round(rows * aspect / CHAR_ASPECT))
    return max(cols, 2), max(rows, 2)


def render(gray, box, cols, rows, show_bg, zoom, track=True,
           avatar=False, eyes_open=True, smiling=False, mouth=0.0):
    """Build the character grid and a face/background layer mask."""
    vh, vw = gray.shape
    chars = np.full((rows, cols), " ", dtype="<U1")
    layer = np.zeros((rows, cols), dtype=bool)

    if not track:
        # tracking off: the whole frame gets the full-detail bright ramp
        small = cv2.resize(gray, (cols, rows),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        lo = np.percentile(small, 2.0)
        hi = np.percentile(small, 98.0)
        norm = np.clip((small - lo) / max(hi - lo, 16.0), 0.0, 1.0)
        chars[:] = RAMP[np.rint(norm * (len(RAMP) - 1)).astype(int)]
        layer[:] = chars != " "
        return chars, layer

    if show_bg and not zoom:
        small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
        idx = np.rint(small / 255 * (len(BG_RAMP) - 1)).astype(int)
        chars[:] = BG_RAMP[idx]

    if box is None:
        return chars, layer

    x0, y0, x1, y1 = pad_box(box, vw, vh)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return chars, layer

    if zoom:
        fgw, fgh = fit_grid(cols, rows, x1 - x0, y1 - y0)
        gx0, gy0 = (cols - fgw) // 2, (rows - fgh) // 2
    else:
        fgw = min(cols, max(2, round((x1 - x0) / vw * cols)))
        fgh = min(rows, max(2, round((y1 - y0) / vh * rows)))
        gx0 = min(max(round(x0 / vw * cols), 0), cols - fgw)
        gy0 = min(max(round(y0 / vh * rows), 0), rows - fgh)

    face = cv2.resize(crop, (fgw, fgh),
                      interpolation=cv2.INTER_AREA).astype(np.float32)

    # oval head mask; contrast-stretch within it so facial features pop
    yy, xx = np.mgrid[0:fgh, 0:fgw]
    u = (xx + 0.5) / fgw * 2 - 1
    v = (yy + 0.5) / fgh * 2 - 1
    rr = np.sqrt(u * u + v * v)
    mask = rr <= 1.0
    lo = np.percentile(face[mask], 2.0)
    hi = np.percentile(face[mask], 98.0)
    norm = np.clip((face - lo) / max(hi - lo, 16.0), 0.0, 1.0)

    if avatar:
        # clear the head's oval, then puppet a 3d-shaded head there,
        # textured with the real face and animated by the expressions
        chars[gy0:gy0 + fgh, gx0:gx0 + fgw][mask] = " "
        layer[gy0:gy0 + fgh, gx0:gx0 + fgw][mask] = False
        draw_head(chars, layer, gx0, gy0, fgw, fgh, norm, u, v, rr, mask,
                  eyes_open, smiling, mouth)
        return chars, layer

    # feathered edge: brightness fades to blank near the rim, so there is
    # no visible boundary shape around the head
    fade = np.clip((1.0 - rr) / EDGE_FADE, 0.0, 1.0)
    fchars = RAMP[np.rint(norm * fade * (len(RAMP) - 1)).astype(int)]

    # overlay only where the face produced ink; the dim background runs
    # underneath, so nothing outlines the head
    put = mask & (fchars != " ")
    chars[gy0:gy0 + fgh, gx0:gx0 + fgw][put] = fchars[put]
    layer[gy0:gy0 + fgh, gx0:gx0 + fgw][put] = True
    return chars, layer


def draw(chars, layer, ox, oy, status):
    out = []
    for r in range(chars.shape[0]):
        out.append("\x1b[%d;%dH" % (oy + r + 1, ox + 1))
        for face, run in itertools.groupby(
                zip(layer[r], chars[r]), key=lambda t: t[0]):
            out.append(FACE_SGR if face else BG_SGR)
            out.append("".join(ch for _, ch in run))
    term = shutil.get_terminal_size()
    out.append("\x1b[%d;1H%s%s" % (
        term.lines, BG_SGR, status[:term.columns - 1].ljust(term.columns - 1)))
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(
        description="webcam face -> live ascii art in the terminal")
    ap.add_argument("--source", default="0",
                    help="camera index or video file path (default 0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(
            "could not open camera/source %r\n"
            "note: WSL2 has no webcam access -- run this with Windows Python\n"
            "(or attach the camera to WSL with usbipd-win)" % (source,))
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml")

    os.system("")  # nudge legacy Windows consoles into ANSI mode
    sys.stdout.write("\x1b[?25l\x1b[?7l\x1b[2J")  # hide cursor, no wrap, clear

    box = None
    last_seen = 0.0
    show_bg = True
    zoom = False
    track = True
    avatar = False
    eye_score, smile_score, mouth_score = 1.0, 0.0, 0.0
    fps = 0.0
    last_size = None

    try:
        with raw_terminal():
            while True:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # selfie mirror
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                det = detect_face(cascade, gray) if track else None
                if det is not None:
                    box = det if box is None else box + (det - box) * SMOOTH
                    last_seen = t0
                elif box is not None and t0 - last_seen > HOLD_S:
                    box = None

                term = shutil.get_terminal_size()
                if term != last_size:
                    sys.stdout.write("\x1b[2J")
                    last_size = term
                cols, rows = fit_grid(term.columns, max(term.lines - 1, 2),
                                      gray.shape[1], gray.shape[0])
                ox = (term.columns - cols) // 2
                oy = (term.lines - 1 - rows) // 2

                eyes_open, smiling, mouth = True, False, 0.0
                if avatar and box is not None:
                    ef, sf, mo = detect_expression(eye_cascade, smile_cascade,
                                                   gray, box)
                    eye_score = 0.6 * eye_score + 0.4 * ef
                    smile_score = 0.7 * smile_score + 0.3 * sf
                    mouth_score = 0.5 * mouth_score + 0.5 * mo
                    eyes_open = eye_score > 0.4
                    smiling = smile_score > 0.5
                    mouth = mouth_score

                chars, layer = render(gray, box, cols, rows, show_bg, zoom,
                                      track, avatar, eyes_open, smiling,
                                      mouth)
                state = ("tracking off" if not track
                         else "no face" if box is None
                         else "avatar" if avatar else "tracking")
                status = ("[q]uit  [f] background  [z] zoom  [t] tracking"
                          "  [a] avatar   fps %4.1f   %s" % (fps, state))
                draw(chars, layer, ox, oy, status)

                for key in read_keys():
                    if key == "q":
                        return
                    elif key == "f":
                        show_bg = not show_bg
                    elif key == "z":
                        zoom = not zoom
                    elif key == "t":
                        track = not track
                        box = None
                    elif key == "a":
                        avatar = not avatar

                dt = time.monotonic() - t0
                if dt < 1 / FPS_CAP:
                    time.sleep(1 / FPS_CAP - dt)
                fps = 0.9 * fps + 0.1 / max(time.monotonic() - t0, 1e-6)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sys.stdout.write(RESET + "\x1b[?25h\x1b[?7h\x1b[2J\x1b[H")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
