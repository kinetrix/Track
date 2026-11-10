"""Live color + depth view. Keys: q = quit, s = save snapshot (PNG + raw depth .npy)."""
import time
import numpy as np
import cv2
import pyrealsense2 as rs

ctx = rs.context()
if len(ctx.query_devices()) == 0:
    raise SystemExit("No RealSense found. Run with sudo, check the cable, and close realsense-viewer.")
dev = ctx.query_devices()[0]
usb = dev.get_info(rs.camera_info.usb_type_descriptor)
fps = 15 if usb.startswith("2") else 30
W, H = 640, 480
print(f"{dev.get_info(rs.camera_info.name)} on USB {usb}, streaming {W}x{H}@{fps}")

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, fps)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, fps)
profile = pipe.start(cfg)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)          # map depth pixels onto the color image
colorizer = rs.colorizer()

try:
    while True:
        frames = align.process(pipe.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        cx, cy = W // 2, H // 2
        dist = depth_frame.get_distance(cx, cy)
        for img in (color, depth_vis):
            cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(color, f"{dist:.3f} m", (cx + 12, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("RealSense  (left: color | right: depth)  q=quit s=save",
                   np.hstack((color, depth_vis)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"color_{ts}.png", color)
            cv2.imwrite(f"depth_vis_{ts}.png", depth_vis)
            np.save(f"depth_m_{ts}.npy", depth_raw * depth_scale)
            print(f"Saved snapshot {ts}")
finally:
    pipe.stop()
    cv2.destroyAllWindows()
