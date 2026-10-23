"""Colour stream of an Intel RealSense camera (D455 / D455f / D435 ...) on Linux.

Two ways in, tried in this order:
  1. Intel's SDK (pip package pyrealsense2): picks a camera by serial number and
     delivers BGR frames. Needs the RealSense udev rules once (see README).
  2. Plain V4L2 (no SDK): the camera's colour sensor is an ordinary UVC webcam.
     A D4xx shows up as several /dev/video* nodes (depth, infrared, colour,
     metadata) that all have the same truncated name, so the colour node is
     found by asking each RealSense node for its pixel formats: only the colour
     sensor offers YUYV.

On macOS the colour sensor is just another camera; mac_viewer.resolve_camera
finds it by name instead.

    cap = open_realsense("realsense", 1280, 720, 30)      # first camera
    cap = open_realsense("realsense:1", ...)              # second camera
    cap = open_realsense("realsense:123456789012", ...)   # by serial number
    ok, bgr = cap.read()
"""
import fcntl
import glob
import os
import re
import struct

import cv2
import numpy as np

# VIDIOC_ENUM_FMT = _IOWR('V', 2, struct v4l2_fmtdesc), sizeof(v4l2_fmtdesc) = 64
_VIDIOC_ENUM_FMT = 0xC0405602
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1


def _formats(dev: str):
    """FourCC codes a V4L2 capture node offers, e.g. ['YUYV'] or ['Z16 ']."""
    out = []
    try:
        fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return out
    try:
        for i in range(32):
            buf = bytearray(struct.pack("II", i, _V4L2_BUF_TYPE_VIDEO_CAPTURE) + bytes(56))
            try:
                fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, buf)
            except OSError:
                break
            out.append(struct.unpack_from("4s", buf, 44)[0].decode("ascii", "replace"))
    finally:
        os.close(fd)
    return out


def v4l2_cameras():
    """[(index, '/dev/videoN', name, formats)] for every V4L2 capture node."""
    cams = []
    for d in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        try:
            name = open(os.path.join(d, "name")).read().strip()
        except OSError:
            continue
        dev = "/dev/" + os.path.basename(d)
        cams.append((int(re.sub(r"\D", "", os.path.basename(d))), dev, name, _formats(dev)))
    return cams


def v4l2_realsense_color():
    """V4L2 indices of RealSense colour sensors (one per camera), in device order."""
    return [i for i, _, name, fmts in v4l2_cameras() if "realsense" in name.lower() and "YUYV" in fmts]


class SDKColor:
    """RealSense colour stream through pyrealsense2, with a cv2.VideoCapture-like API."""

    def __init__(self, serial, width, height, fps):
        import pyrealsense2 as rs
        self.rs = rs
        ctx = rs.context()
        devices = [(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name))
                   for d in ctx.query_devices()]
        if not devices:
            raise RuntimeError("pyrealsense2 found no RealSense camera")
        print("[camera] RealSense cameras:", ", ".join(f"{n} (serial {s})" for s, n in devices))
        if serial is None:
            serial = devices[0][0]
        elif serial.isdigit() and len(serial) < 3:  # realsense:0 / realsense:1 = n-th camera
            serial = devices[min(int(serial), len(devices) - 1)][0]
        self.pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe.start(cfg)
        self.opened = True
        print(f"[camera] using RealSense {serial} colour stream {width}x{height} @ {fps} fps (pyrealsense2)")

    def isOpened(self):
        return self.opened

    def read(self):
        try:
            frames = self.pipe.wait_for_frames(2000)
        except RuntimeError:
            return False, None
        c = frames.get_color_frame()
        if not c:
            return False, None
        return True, np.asanyarray(c.get_data()).copy()

    def set(self, *_):
        return False

    def get(self, prop):
        return 0.0

    def release(self):
        if self.opened:
            self.pipe.stop()
            self.opened = False


def open_realsense(spec: str, width: int = 1280, height: int = 720, fps: int = 30):
    """spec: 'realsense' (first camera) or 'realsense:<serial>'. Returns an opened capture."""
    serial = spec.split(":", 1)[1] if ":" in spec else None
    try:
        return SDKColor(serial, width, height, fps)
    except ImportError:
        print("[camera] pyrealsense2 not installed; using the camera as a plain V4L2 webcam")
    except Exception as e:  # no permission (udev rules), busy, unsupported mode...
        print(f"[camera] pyrealsense2 could not start the camera ({e}); trying plain V4L2")
    nodes = v4l2_realsense_color()
    if not nodes:
        found = "\n  ".join(f"{dev}: {name} {fmts}" for _, dev, name, fmts in v4l2_cameras()) or "(none)"
        raise SystemExit("No RealSense colour stream found. Is the camera plugged into a USB 3 port?\n"
                         f"V4L2 cameras:\n  {found}")
    idx = nodes[0]
    if serial is not None and serial.isdigit() and len(serial) < 3:
        idx = nodes[min(int(serial), len(nodes) - 1)]  # realsense:0 / realsense:1 = n-th camera
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"Could not open /dev/video{idx}. Add yourself to the 'video' group or install the "
                         "RealSense udev rules (see README), then unplug and replug the camera.")
    print(f"[camera] using RealSense colour sensor /dev/video{idx} (V4L2)")
    return cap
