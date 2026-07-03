#!/usr/bin/env python3
"""Flask backend for the browser version of ascii face.

(A WebSocket is the closest thing browser JavaScript has to a UDP-style
link: one connection, no per-frame handshakes or HTTP headers. Raw UDP
sockets aren't available to web pages.)
"""

import argparse
import json
import time

import cv2
import numpy as np
from flask import Flask, send_file
from flask_sock import Sock

import ascii_face as af

app = Flask(__name__)
sock = Sock(app)

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.get("/")
def page():
    return send_file("index.html")


@sock.route("/stream")
def stream(ws):
    """JSON text messages update render options; binary messages are JPEG
    frames, each answered with the rendered ascii grid. Tracking state is
    per-connection."""
    opts = {"cols": 160, "rows": 90, "bg": True, "zoom": False, "track": True}
    box = None
    last_seen = 0.0

    while True:
        msg = ws.receive()
        if msg is None:
            break
        if isinstance(msg, str):
            opts.update(json.loads(msg))
            continue

        bgr = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2GRAY)  # mirror

        det = af.detect_face(_cascade, gray) if opts["track"] else None
        now = time.monotonic()
        if det is not None:
            box = det if box is None else box + (det - box) * af.SMOOTH
            last_seen = now
        elif box is not None and (not opts["track"]
                                  or now - last_seen > af.HOLD_S):
            box = None

        cols, rows = af.fit_grid(min(int(opts["cols"]), 400),
                                 min(int(opts["rows"]), 200),
                                 gray.shape[1], gray.shape[0])
        chars, layer = af.render(gray, box, cols, rows, bool(opts["bg"]),
                                 bool(opts["zoom"]), bool(opts["track"]))

        # two aligned text layers so the page can color them independently
        bg_rows = ["".join(np.where(layer[r], " ", chars[r]))
                   for r in range(rows)]
        face_rows = ["".join(np.where(layer[r], chars[r], " "))
                     for r in range(rows)]
        ws.send(json.dumps({"bg": bg_rows, "face": face_rows,
                            "tracking": box is not None}))


def main():
    ap = argparse.ArgumentParser(
        description="websocket backend for the browser ascii face")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    # keep-alive for page loads; the stream itself is one long connection
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"

    print("open http://localhost:%d in your Windows browser" % args.port)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
