"""HTTP transport for the live pipeline (stdlib only, no torch/pyrealsense2).

Serves, on a single port:
  GET /  , /stream            -> MJPEG video (multipart/x-mixed-replace)
  GET /detections             -> latest detection record (application/json)
  GET /detections/stream      -> live NDJSON stream (one JSON record per line,
                                 flushed as each new frame is produced)
  GET /latency                -> server receive/send timestamps for NTP-style
                                 clock-offset and network-delay measurement

main.py feeds this module via publish_frame() / publish_detections(); the
detection stream pushes the freshest record whenever the frame sequence
advances (a slow client skips intermediate frames, like the MJPEG stream).
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_state = {"jpeg": None, "record": None, "seq": 0}

# Set by start_server(); used to pace the MJPEG stream.
_fps = 30

# Bump when the detection record layout changes incompatibly. Consumers can
# read record["schema_version"] to detect a producer/consumer mismatch.
SCHEMA_VERSION = 2


def build_record(timestamp, elapsed_s, bounding_boxes, class_names, confidences):
    """Serialize one frame's detections into a JSON-ready dict.

    bounding_boxes / class_names / confidences are parallel arrays (same order,
    one entry per kept detection).  Each bounding box contains four projected
    belt-plane points in clockwise order: top-left, top-right, bottom-right,
    bottom-left.  Every point is ``[X, Y, Z]`` in metres.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": float(timestamp),
        "elapsed_s": float(elapsed_s),
        "bounding_boxes": [
            [[float(v) for v in point] for point in box]
            for box in bounding_boxes
        ],
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
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_a, **_kw):
        pass

    def do_GET(self):
        server_receive_ns = time.time_ns()
        path = self.path.split("?", 1)[0]
        if path in ("/", "/stream"):
            self._serve_mjpeg()
        elif path == "/detections":
            self._serve_detection_latest()
        elif path == "/detections/stream":
            self._serve_detection_stream()
        elif path == "/latency":
            self._serve_latency(server_receive_ns)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _serve_latency(self, server_receive_ns):
        """Return timestamps used by the robot PC for an NTP-style probe."""
        payload = {
            "protocol": "gp8-latency-v1",
            "server_receive_time_ns": int(server_receive_ns),
            "server_send_time_ns": time.time_ns(),
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
