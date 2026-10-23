#!/usr/bin/env python3
"""Desktop viewer: webcam -> Engine -> window with masks, boxes, hand skeletons
and gesture labels. This is the only platform-specific part of v2 (camera +
window), and it is deliberately thin: everything else lives in facetrack/.

    python app/mac_viewer.py                 # RealSense if plugged in, else the Mac's camera
    python app/mac_viewer.py --accel cpu     # force CPU
    python app/mac_viewer.py --video clip.mp4

Keys:  q quit   m masks on/off   r reset tracks   c next camera   f faces on/off   h hands on/off   s save screenshot

Threads:
    camera thread   keeps only the newest frame (older frames are dropped, as in v1)
    worker thread   runs the engine on the newest frame, then picks the next newest
    main thread     draws the newest processed frame (OpenCV windows must live here on macOS)
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facetrack import HAND_CONNECTIONS, Engine  # noqa: E402

COLORS = [(245, 135, 66), (120, 200, 80), (66, 170, 245), (200, 80, 200), (80, 200, 200), (200, 200, 80)]
WINDOW = "FaceTrack v2"


def list_mac_cameras():
    """Camera names in OpenCV's AVFoundation index order, from macOS's own camera list.
    OpenCV numbers cameras by sorting them on their unique ID (cap_avfoundation_mac.mm),
    so sorting system_profiler's list the same way gives the index of each name."""
    if sys.platform != "darwin":
        return []
    try:
        import json
        import subprocess
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True,
                             timeout=15).stdout
        cams = json.loads(out).get("SPCameraDataType", [])
        cams = sorted(cams, key=lambda c: c.get("spcamera_unique-id", ""))
        return [c.get("_name", "?") for c in cams]
    except Exception:
        return []


BUILTIN_NAMES = ["facetime", "built-in", "macbook", "imac"]  # e.g. "FaceTime HD Camera", "MacBook Pro Camera"


def _find(names, words, prefer=()):
    """Index of the first camera whose name contains one of `words`; among those,
    one whose name also contains a `prefer` word wins."""
    hits = [i for i, n in enumerate(names) if any(w in n.lower() for w in words)]
    for p in prefer:
        for i in hits:
            if p in names[i].lower():
                return i
    return hits[0] if hits else None


def resolve_camera(choice: str) -> int:
    """'auto' (default): an Intel RealSense camera's colour sensor if one is plugged in,
    otherwise the Mac's own camera; 'realsense'; 'builtin'; a number (OpenCV index);
    or any other text: the first camera whose name contains it (e.g. 'iPhone')."""
    if choice.isdigit():
        return int(choice)
    names = list_mac_cameras()
    if names:
        print("[camera] found:", ", ".join(f"{i}: {n}" for i, n in enumerate(names)))
    # A RealSense D4xx lists its sensors separately; only the one named RGB / Color is the colour camera.
    # (macOS may list just "... Depth", which gives no usable picture: never pick that one automatically.)
    realsense = next((i for i, n in enumerate(names) if "realsense" in n.lower()
                      and any(w in n.lower() for w in ("rgb", "color", "colour"))), None)
    if realsense is None and any("realsense" in n.lower() for n in names) and choice in ("auto", "realsense"):
        print("[camera] the RealSense shows only its depth sensor to macOS here, not its colour camera; "
              "using the Mac's camera (see README: RealSense on macOS)")
    if choice == "auto":
        idx = realsense if realsense is not None else _find(names, BUILTIN_NAMES)
    elif choice == "realsense":
        idx = realsense if realsense is not None else _find(names, BUILTIN_NAMES)
    elif choice == "builtin":
        idx = _find(names, BUILTIN_NAMES)
    else:
        idx = _find(names, [choice.lower()])
    if idx is None:
        if choice != "auto":
            print(f"[camera] no camera matching {choice!r}; using camera 0")
        return 0
    print(f"[camera] using camera {idx} ({names[idx]})")
    return idx


class Camera(threading.Thread):
    MAX_INDEX = 4  # camera indices tried when cycling (built-in, iPhone Continuity, external, virtual...)

    def __init__(self, source, width, height, fps, mirror):
        super().__init__(daemon=True)
        is_file = isinstance(source, str)
        self.api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and not is_file else cv2.CAP_ANY
        self.size, self.fps = (width, height), fps
        self.index = None if is_file else source
        self.cap = self._open(source)
        if not self.cap.isOpened() and not is_file:
            # First run on macOS: the camera permission prompt is showing; wait for the answer.
            print("Waiting for camera access (allow Terminal in the macOS prompt)...")
            for _ in range(30):
                time.sleep(2)
                self.cap = self._open(source)
                if self.cap.isOpened():
                    break
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera/video {source!r}. On macOS, allow camera access for "
                             "Terminal in System Settings > Privacy & Security > Camera, then run again.")
        self.is_file, self.mirror = is_file, mirror
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.lock = threading.Lock()
        self.frame, self.stamp, self.seq = None, 0.0, 0
        self.running = True
        self.capture_fps = 0.0
        self.switch_to = None  # camera index requested by the UI ('c' key)
        self.auto_switched = 0

    def _open(self, source):
        cap = cv2.VideoCapture(source, self.api)
        if cap.isOpened() and not isinstance(source, str):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def next_camera(self):
        self.switch_to = (self.index + 1) % self.MAX_INDEX if self.index is not None else None

    def _switch(self, idx):
        for k in range(self.MAX_INDEX):  # skip indices with no camera
            cand = (idx + k) % self.MAX_INDEX
            cap = self._open(cand)
            if cap.isOpened():
                self.cap.release()
                self.cap, self.index = cap, cand
                print(f"[camera] using camera {cand}")
                return

    def run(self):
        t_last, black = None, 0
        while self.running:
            if self.switch_to is not None:
                self._switch(self.switch_to)
                self.switch_to, black = None, 0
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    self.running = False
                    break
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            # an all-black stream for 3 s usually means a virtual/idle camera: try the next one
            if not self.is_file:
                black = black + 1 if f[::16, ::16].max() < 8 else 0
                if black > 3 * self.fps and self.auto_switched < self.MAX_INDEX:
                    print(f"[camera] camera {self.index} only sends black frames; trying the next one ('c' cycles cameras)")
                    self.auto_switched += 1
                    self.switch_to = (self.index + 1) % self.MAX_INDEX
            if self.mirror:
                f = cv2.flip(f, 1)
            with self.lock:
                self.frame, self.stamp, self.seq = f, now, self.seq + 1
            if t_last is not None:
                self.capture_fps = 0.9 * self.capture_fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now
            if self.is_file:
                time.sleep(1.0 / self.file_fps)  # play files at their own speed

    def latest(self):
        with self.lock:
            return self.frame, self.stamp, self.seq


class Worker(threading.Thread):
    def __init__(self, cam: Camera, engine: Engine, record_dir=None, record_max=300, debug_hands=None):
        super().__init__(daemon=True)
        self.cam, self.engine = cam, engine
        self.debug_hands, self.n_debug, self.prev = debug_hands, 0, None
        if debug_hands:
            os.makedirs(debug_hands, exist_ok=True)
        self.record_dir, self.record_max, self.recorded = record_dir, record_max, 0
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.result = None  # (frame, FrameResult)
        self.fps = 0.0
        self.running = True
        self.reset_requested = False
        self.t0 = time.perf_counter()

    def run(self):
        last_seq, t_last = -1, None
        while self.running:
            frame, stamp, seq = self.cam.latest()
            if frame is None or seq == last_seq:
                if not self.cam.running and frame is not None:
                    break
                time.sleep(0.002)
                continue
            last_seq = seq
            if self.reset_requested:
                self.engine.reset()
                self.reset_requested = False
            res = self.engine.process(frame, stamp - self.t0)
            if self.debug_hands and self.prev is not None and self.n_debug < 50 and \
                    any("lost (presence" in e for e in res.events):
                import json
                pf, pres = self.prev
                base = os.path.join(self.debug_hands, f"{self.n_debug:03d}")
                cv2.imwrite(base + "_a.png", pf)
                cv2.imwrite(base + "_b.png", frame)
                with open(base + ".json", "w") as fh:
                    json.dump({"prev_hands": [h.to_dict() for h in pres.hands], "events": res.events}, fh)
                self.n_debug += 1
            if self.debug_hands:
                self.prev = (frame, res)
            if self.record_dir and self.recorded < self.record_max:
                cv2.imwrite(os.path.join(self.record_dir, f"{self.recorded:05d}_{res.time:08.3f}.jpg"), frame)
                self.recorded += 1
            for e in res.events:
                print(f"[{res.time:7.2f}s] {e}")
            now = time.perf_counter()
            if t_last is not None:
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now
            with self.lock:
                self.result = (frame, res)


def _label(img, text, org, scale, color):
    """Text on a dark box (an outline drawn with a thicker stroke does not line up
    with the thin text in OpenCV 5's font renderer)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame, res, show_masks, fps, cam_fps, backends):
    img = frame.copy()
    H, W = img.shape[:2]
    for f in res.faces:
        c = COLORS[f.id % len(COLORS)]
        if not f.visible:
            continue
        if show_masks and f.mask is not None:
            m = cv2.resize(f.mask, (W, H), interpolation=cv2.INTER_LINEAR) > 0
            img[m] = (0.55 * img[m] + 0.45 * np.array(c)).astype(np.uint8)
        x1, y1, x2, y2 = f.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"face #{f.id}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    for h in res.hands:
        c = COLORS[(h.id + 3) % len(COLORS)]
        pts = h.landmarks[:, :2].astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), c, 2)
        for p in pts:
            cv2.circle(img, tuple(p), 3, (255, 255, 255), -1)
        x1, y1 = int(h.box[0]), int(h.box[1])
        label = f"{h.handedness} #{h.id}: {h.gesture or '-'} ({h.fingers})"
        _label(img, label, (max(0, x1), max(22, y1 - 8)), 0.7, c)

    t = res.timings
    lines = [f"{fps:4.1f} FPS processed  (camera {cam_fps:4.1f})   faces {len(res.faces)}  hands {len(res.hands)}",
             "ms  total {:.1f} | detect {:.1f} enc {:.1f} attn {:.1f} dec {:.1f} menc {:.1f} | palm {:.1f} landmarks {:.1f}"
             .format(t.get("total", 0), t.get("detect", 0), t.get("enc", 0), t.get("attn", 0), t.get("dec", 0),
                     t.get("menc", 0), t.get("palm", 0), t.get("landmarks", 0))]
    backend_set = sorted(set(backends.values()))
    lines.append("backend: " + ", ".join(backend_set) + "   keys: q quit  m masks  r reset  c camera  f faces  h hands")
    y = 22
    for ln in lines:
        _label(img, ln, (10, y), 0.55, (255, 255, 255))
        y += 22
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="auto",
                    help="auto (default: RealSense if plugged in, else the Mac's camera), realsense, builtin, "
                         "a camera number, or part of a camera name")
    ap.add_argument("--video", help="run on a video file instead of the camera")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--accel", default="auto", help="auto | cpu | coreml | cuda | directml")
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror the camera image")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--record", help="also save every processed camera frame (raw, as JPEG) into this folder")
    ap.add_argument("--record-max", type=int, default=300, help="stop recording after this many frames")
    ap.add_argument("--debug-hands", help="save the frame pair + landmarks whenever a tracked hand is lost (debugging)")
    args = ap.parse_args()

    print("Loading models (the first launch also measures the fastest backend per model)...")
    engine = Engine(args.models, accel=args.accel, faces=not args.no_faces, hands=not args.no_hands,
                    mirrored_input=not args.no_mirror)
    print("backends:", engine.backends)
    cam = Camera(args.video if args.video else resolve_camera(str(args.camera)), args.width, args.height, args.fps,
                 mirror=not args.no_mirror and not args.video)
    cam.start()
    worker = Worker(cam, engine, args.record, args.record_max, args.debug_hands)
    worker.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    show_masks, last_print = True, time.perf_counter()
    while True:
        with worker.lock:
            r = worker.result
        if r is not None:
            frame, res = r
            img = draw(frame, res, show_masks, worker.fps, cam.capture_fps, engine.backends)
            cv2.imshow(WINDOW, img)
            if time.perf_counter() - last_print > 2.0:
                t = res.timings
                print(f"[FaceTrack] {worker.fps:4.1f} fps | faces {len(res.faces)} hands {len(res.hands)} | " +
                      " ".join(f"{k} {v:.1f}" for k, v in t.items()) + " ms")
                last_print = time.perf_counter()
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q") or key == 27 or (not cam.running and not worker.is_alive()):
            break
        if key == ord("m"):
            show_masks = not show_masks
        elif key == ord("r"):
            worker.reset_requested = True
        elif key == ord("c"):
            cam.next_camera()
            worker.reset_requested = True
        elif key == ord("f"):
            engine.enable_faces = not engine.enable_faces and hasattr(engine, "faces")
        elif key == ord("h"):
            engine.enable_hands = not engine.enable_hands and hasattr(engine, "hands")
        elif key == ord("s") and r is not None:
            name = time.strftime("facetrack_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(ROOT, name), img)
            print("saved", name)
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cam.running = worker.running = False
    engine.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
