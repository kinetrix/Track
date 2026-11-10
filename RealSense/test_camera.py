"""Checks the camera is visible, reports USB speed, and reads one depth value."""
import pyrealsense2 as rs

ctx = rs.context()
devs = ctx.query_devices()
print("Devices found:", len(devs))
if len(devs) == 0:
    raise SystemExit("No RealSense found. Run with sudo, check the cable, and close realsense-viewer.")

dev = devs[0]
name = dev.get_info(rs.camera_info.name)
serial = dev.get_info(rs.camera_info.serial_number)
fw = dev.get_info(rs.camera_info.firmware_version)
usb = dev.get_info(rs.camera_info.usb_type_descriptor)
print(f"{name} | serial {serial} | firmware {fw} | USB {usb}")
if usb.startswith("2"):
    print("WARNING: USB 2 connection. Use a USB 3 data cable for full resolution/FPS.")

fps = 15 if usb.startswith("2") else 30
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device(serial)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
pipe.start(cfg)
try:
    for _ in range(30):  # let auto-exposure settle
        frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    print(f"Distance at image center: {depth.get_distance(320, 240):.3f} m")
finally:
    pipe.stop()
print("Test passed.")
