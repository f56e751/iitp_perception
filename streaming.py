"""HTTP transport for the live pipeline (stdlib only, no torch/pyrealsense2).

Serves, on a single port:
  GET /  , /stream            -> MJPEG video (multipart/x-mixed-replace)
  GET /detections             -> latest detection record (application/json)
  GET /detections/stream      -> live NDJSON stream (one JSON record per line,
                                 flushed as each new frame is produced)

main.py feeds this module via publish_frame() / publish_detections(); the
detection stream pushes the freshest record whenever the frame sequence
advances (a slow client skips intermediate frames, like the MJPEG stream).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_state = {"jpeg": None, "record": None, "seq": 0}

# Set by start_server(); used to pace the MJPEG stream.
_fps = 30

# Bump when the detection record layout changes incompatibly. Consumers can
# read record["schema_version"] to detect a producer/consumer mismatch.
SCHEMA_VERSION = 1


def build_record(timestamp, elapsed_s, positions, class_names, confidences):
    """Serialize one frame's detections into a JSON-ready dict.

    positions / class_names / confidences are parallel arrays (same order,
    one entry per kept detection).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": float(timestamp),
        "elapsed_s": float(elapsed_s),
        "positions": [[float(v) for v in p] for p in positions],
        "class_names": list(class_names),
        "confidences": [float(c) for c in confidences],
    }


def publish_frame(jpeg_bytes):
    with _lock:
        _state["jpeg"] = jpeg_bytes


def publish_detections(record):
    with _lock:
        _state["record"] = record
        _state["seq"] += 1


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a, **_kw):
        pass

    def do_GET(self):
        if self.path in ("/", "/stream"):
            self._serve_mjpeg()
        elif self.path == "/detections":
            self._serve_detection_latest()
        elif self.path == "/detections/stream":
            self._serve_detection_stream()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _lock:
                    data = _state["jpeg"]
                if data is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / _fps)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_detection_latest(self):
        with _lock:
            record = _state["record"]
        body = json.dumps(record if record is not None else {}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_detection_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        last_seq = -1
        try:
            while True:
                with _lock:
                    seq = _state["seq"]
                    record = _state["record"]
                if record is None or seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq
                self.wfile.write((json.dumps(record) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_server(port, fps=30):
    """Start the HTTP server on a daemon thread and return it.

    Pass port=0 to bind an ephemeral port (read it back via
    server.server_address[1]); used by tests.
    """
    global _fps
    _fps = fps
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
