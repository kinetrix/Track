"""A tiny MJPEG-over-HTTP server (standard library only).

Used twice: camera_server.py streams the Mac camera to a Linux machine, and
headless_viewer.py streams the annotated result back so any browser can show it.
The same idea works on a robot: the camera comes in over the network, the
picture goes out over the network, nothing needs a screen.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><html><head><title>FaceTrack</title><style>
body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style></head><body><img src="/stream.mjpg"></body></html>"""


class MJPEGServer:
    def __init__(self, port: int, host: str = "0.0.0.0"):
        self._jpeg, self._seq = None, 0
        self._cond = threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the terminal quiet
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                if not self.path.startswith("/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                try:
                    while True:
                        jpeg, last = server.wait(last)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def publish(self, jpeg: bytes):
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 5.0):
        """Block until a frame newer than last_seq exists (clients always get the newest)."""
        deadline = time.time() + timeout
        with self._cond:
            while self._jpeg is None or self._seq == last_seq:
                left = deadline - time.time()
                if left <= 0:
                    raise BrokenPipeError
                self._cond.wait(left)
            return self._jpeg, self._seq

    def close(self):
        self.httpd.shutdown()
