#!/usr/bin/env python3
"""Flask backend for the browser version of ascii face.

The page (index.html) captures webcam frames with getUserMedia and POSTs
each one here as a JPEG. This backend runs the exact same offline pipeline
as the terminal version -- OpenCV's bundled Haar cascade for detection,
ascii_face.render() for the character grid -- and returns the grid as JSON
for the page to display. Nothing external is fetched by either side.
"""

import argparse
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

import ascii_face as af

app = Flask(__name__)

_lock = threading.Lock()
_box = None
_last_seen = 0.0
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.get("/")
def page():
    return send_file("index.html")


@app.post("/frame")
def frame():
    global _box, _last_seen

    buf = np.frombuffer(request.get_data(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="could not decode frame"), 400
    gray = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2GRAY)  # selfie mirror

    max_cols = min(int(request.args.get("cols", 160)), 400)
    max_rows = min(int(request.args.get("rows", 90)), 200)
    show_bg = request.args.get("bg", "1") == "1"
    zoom = request.args.get("zoom", "0") == "1"
    track = request.args.get("track", "1") == "1"

    with _lock:
        det = af.detect_face(_cascade, gray) if track else None
        now = time.monotonic()
        if det is not None:
            _box = det if _box is None else _box + (det - _box) * af.SMOOTH
            _last_seen = now
        elif _box is not None and (not track or now - _last_seen > af.HOLD_S):
            _box = None
        box = None if _box is None else _box.copy()

    cols, rows = af.fit_grid(max_cols, max_rows, gray.shape[1], gray.shape[0])
    chars, layer = af.render(gray, box, cols, rows, show_bg, zoom, track)

    # split into two aligned text layers so the page can color them:
    # dim background chars and bright face chars never share a cell
    bg_rows = ["".join(np.where(layer[r], " ", chars[r])) for r in range(rows)]
    face_rows = ["".join(np.where(layer[r], chars[r], " ")) for r in range(rows)]
    return jsonify(bg=bg_rows, face=face_rows, tracking=box is not None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print("open http://localhost:%d in your Windows browser" % args.port)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
