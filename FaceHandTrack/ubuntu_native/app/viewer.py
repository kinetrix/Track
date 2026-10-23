#!/usr/bin/env python3
"""Ubuntu desktop viewer: camera -> Engine (PyTorch) -> window with face masks,
boxes, hand skeletons and gesture labels.

    ./run.sh                          # RealSense colour stream if plugged in, else the laptop webcam
    ./run.sh --camera realsense:1     # second RealSense (or realsense:<serial number>)
    ./run.sh --camera 0               # /dev/video0
    ./run.sh --video clip.mp4         # a video file (or a folder of .jpg frames)
    ./run.sh --list-cameras

Keys:  q quit   m masks on/off   r reset tracks   f faces on/off   h hands on/off   s save screenshot

Threads:
    camera thread   keeps only the newest frame (older frames are dropped)
    worker thread   runs the engine on the newest frame, then picks the next newest
    main thread     draws the newest processed frame in the window
"""
import argparse
import glob
import os
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))
from facehand import HAND_CONNECTIONS, Engine  # noqa: E402
from facehand.engine import DEFAULT_CKPT  # noqa: E402
import realsense  # noqa: E402

COLORS = [(245, 135, 66), (120, 200, 80), (66, 170, 245), (200, 80, 200), (80, 200, 200), (200, 200, 80)]
WINDOW = "FaceHandTrack (Ubuntu native)"


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def list_cameras():
    print("V4L2 video devices:")
    for i, dev, name, fmts in realsense.v4l2_cameras():
        print(f"  {i}: {dev}  {name}  formats {fmts}")
    try:
        import pyrealsense2 as rs
        devs = list(rs.context().query_devices())
        print("RealSense cameras (pyrealsense2):")
        for k, d in enumerate(devs):
            print(f"  realsense:{k}  {d.get_info(rs.camera_info.name)}  serial {d.get_info(rs.camera_info.serial_number)}"
                  f"  USB {d.get_info(rs.camera_info.usb_type_descriptor)}")
        if not devs:
            print("  (none)")
    except ImportError:
        print("pyrealsense2 is not installed")


def realsense_present() -> bool:
    try:
        import pyrealsense2 as rs
        if len(rs.context().query_devices()) > 0:
            return True
    except Exception:
        pass
    return bool(realsense.v4l2_realsense_color())


def open_camera(choice: str, width: int, height: int, fps: int):
    """auto: a RealSense colour stream if one is plugged in, else /dev/video0."""
    if choice == "auto":
        choice = "realsense" if realsense_present() else "0"
        if choice == "0":
            print("[camera] no RealSense found; using /dev/video0 (the laptop webcam)")
    if choice.startswith("realsense"):
        return realsense.open_realsense(choice, width, height, fps)
    idx = int(choice) if choice.isdigit() else choice
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {choice!r}. See ./run.sh --list-cameras")
    return cap


class Camera(threading.Thread):
    """Camera or video file in a thread; `latest()` gives the newest frame."""

    def __init__(self, cap, mirror: bool, is_file: bool, file_fps: float = 30.0):
        super().__init__(daemon=True)
        self.cap, self.mirror, self.is_file, self.file_fps = cap, mirror, is_file, file_fps
        self.lock = threading.Lock()
        self.frame, self.stamp, self.seq = None, 0.0, 0
        self.running = True
        self.capture_fps = 0.0

    def run(self):
        t_last = None
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    break
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            if self.mirror:
                f = cv2.flip(f, 1)
            with self.lock:
                self.frame, self.stamp, self.seq = f, now, self.seq + 1
            if t_last is not None:
                self.capture_fps = 0.9 * self.capture_fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now
            if self.is_file:
                time.sleep(1.0 / self.file_fps)  # play files at their own speed
        self.running = False

    def latest(self):
        with self.lock:
            return self.frame, self.stamp, self.seq


class FolderCapture:
    """A folder of .jpg/.png frames, read like a video."""

    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
        self.i = 0

    def read(self):
        if self.i >= len(self.files):
            return False, None
        self.i += 1
        return True, cv2.imread(self.files[self.i - 1])

    def release(self):
        pass


class Worker(threading.Thread):
    def __init__(self, cam: Camera, engine: Engine):
        super().__init__(daemon=True)
        self.cam, self.engine = cam, engine
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
                if not self.cam.running:  # video file finished
                    break
                time.sleep(0.002)
                continue
            last_seq = seq
            if self.reset_requested:
                self.engine.reset()
                self.reset_requested = False
            res = self.engine.process(frame, stamp - self.t0)
            for e in res.events:
                print(f"[{res.time:7.2f}s] {e}")
            now = time.perf_counter()
            if t_last is not None:
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now
            with self.lock:
                self.result = (frame, res)


# --------------------------------------------------------------------------- #
# Drawing (same look as the ONNX version)
# --------------------------------------------------------------------------- #
def _label(img, text, org, scale, color):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame, res, show_masks, fps, cam_fps, device_name):
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
        _label(img, f"{h.handedness} #{h.id}: {h.gesture or '-'} ({h.fingers})", (max(0, x1), max(22, y1 - 8)), 0.7, c)

    t = res.timings
    lines = [f"{fps:4.1f} FPS processed  (camera {cam_fps:4.1f})   faces {len(res.faces)}  hands {len(res.hands)}",
             "ms  total {:.1f} | detect {:.1f} enc {:.1f} attn {:.1f} dec {:.1f} menc {:.1f} | palm {:.1f} landmarks {:.1f}"
             .format(t.get("total", 0), t.get("detect", 0), t.get("enc", 0), t.get("attn", 0), t.get("dec", 0),
                     t.get("menc", 0), t.get("palm", 0), t.get("landmarks", 0)),
             f"PyTorch on {device_name}   keys: q quit  m masks  r reset  f faces  h hands  s screenshot"]
    y = 22
    for ln in lines:
        _label(img, ln, (10, y), 0.55, (255, 255, 255))
        y += 22
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="auto",
                    help="auto (default: RealSense if plugged in, else /dev/video0), realsense, realsense:1, "
                         "realsense:<serial>, or a /dev/video number")
    ap.add_argument("--video", help="run on a video file or a folder of .jpg frames instead of a camera")
    ap.add_argument("--list-cameras", action="store_true", help="print the cameras this machine sees and exit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--accel", default="auto", help="auto (CUDA GPU if available, else CPU) | cuda | cpu")
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't mirror the picture (use for a camera that faces away from you, e.g. on a robot)")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="EdgeTAM checkpoint (edgetam.pt)")
    args = ap.parse_args()
    if args.list_cameras:
        list_cameras()
        return

    mirror = not args.no_mirror and not args.video
    print("Loading models...")
    engine = Engine(args.models, args.checkpoint, accel=args.accel, faces=not args.no_faces,
                    hands=not args.no_hands, mirrored_input=mirror)
    if args.video:
        if os.path.isdir(args.video):
            cam = Camera(FolderCapture(args.video), False, True)
        else:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"Could not open {args.video!r}")
            cam = Camera(cap, False, True, cap.get(cv2.CAP_PROP_FPS) or 30.0)
    else:
        cam = Camera(open_camera(args.camera, args.width, args.height, args.fps), mirror, False)
    cam.start()
    worker = Worker(cam, engine)
    worker.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    show_masks, last_print, img = True, time.perf_counter(), None
    while True:
        with worker.lock:
            r = worker.result
        if r is not None:
            frame, res = r
            img = draw(frame, res, show_masks, worker.fps, cam.capture_fps, engine.device_name)
            cv2.imshow(WINDOW, img)
            if time.perf_counter() - last_print > 2.0:
                print(f"[FaceHandTrack] {worker.fps:4.1f} fps | faces {len(res.faces)} hands {len(res.hands)} | " +
                      " ".join(f"{k} {v:.1f}" for k, v in res.timings.items()) + " ms")
                last_print = time.perf_counter()
        key = cv2.waitKey(5) & 0xFF
        if key in (ord("q"), 27) or (not cam.running and not worker.is_alive()):
            break
        if key == ord("m"):
            show_masks = not show_masks
        elif key == ord("r"):
            worker.reset_requested = True
        elif key == ord("f"):
            engine.enable_faces = not engine.enable_faces and hasattr(engine, "faces")
        elif key == ord("h"):
            engine.enable_hands = not engine.enable_hands and hasattr(engine, "hands")
        elif key == ord("s") and img is not None:
            name = time.strftime("facehand_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(ROOT, name), img)
            print("saved", name)
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cam.running = worker.running = False
    engine.close()
    cam.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
