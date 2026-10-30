"""Compare run_clip (C++) and run_clip_python.py outputs: same IDs per frame, box IoU.
    python3 compare.py cpp.txt python.txt
"""
import sys
from collections import defaultdict


def load(path):
    tracks, events = defaultdict(dict), []
    for line in open(path):
        if line.startswith("["):  # log lines
            continue
        if line.startswith("#"):
            events.append(line.strip())
            continue
        f, i, x1, y1, x2, y2, v, q = line.split()
        tracks[int(f)][int(i)] = (float(x1), float(y1), float(x2), float(y2), int(v), float(q))
    return tracks, events


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 1.0


cpp, ce = load(sys.argv[1])
py, pe = load(sys.argv[2])
frames = sorted(set(cpp) | set(py))
same_ids, ious = 0, []
for f in frames:
    if set(cpp.get(f, {})) == set(py.get(f, {})):
        same_ids += 1
    for i in set(cpp.get(f, {})) & set(py.get(f, {})):
        ious.append(iou(cpp[f][i], py[f][i]))
print(f"frames with the same track IDs: {same_ids}/{len(frames)}")
if ious:
    print(f"box IoU C++ vs Python: mean {sum(ious) / len(ious):.4f}, min {min(ious):.4f} over {len(ious)} boxes")
print(f"events identical: {ce == pe} ({len(ce)} C++, {len(pe)} Python)")
if ce != pe:
    for a, b in zip(ce, pe):
        if a != b:
            print("  first difference:", a, "|", b)
            break
