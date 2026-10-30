"""Compare run_clip (C++) and run_clip_python.py outputs, frame by frame.
    python3 compare.py cpp.txt python.txt
Faces: same track IDs, box IoU. Hands: same IDs, Left/Right and gestures, landmark distance.
"""
import math
import sys
from collections import defaultdict


def load(path):
    faces, hands, events = defaultdict(dict), defaultdict(dict), []
    for line in open(path):
        if line.startswith("["):  # log lines
            continue
        if line.startswith("#"):
            events.append(line.strip())
            continue
        p = line.split()
        if p[0] == "H":
            f, i = int(p[1]), int(p[2])
            hands[f][i] = (p[3], p[4], p[5], int(p[6]), tuple(map(float, p[7:11])))
            continue
        f, i, x1, y1, x2, y2, v, q = p
        faces[int(f)][int(i)] = (float(x1), float(y1), float(x2), float(y2), int(v), float(q))
    return faces, hands, events


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 1.0


cf, ch, ce = load(sys.argv[1])
pf, ph, pe = load(sys.argv[2])

frames = sorted(set(cf) | set(pf))
if frames:
    same = sum(set(cf.get(f, {})) == set(pf.get(f, {})) for f in frames)
    ious = [iou(cf[f][i], pf[f][i]) for f in frames for i in set(cf.get(f, {})) & set(pf.get(f, {}))]
    print(f"faces: same track IDs on {same}/{len(frames)} frames", end="")
    print(f"; box IoU mean {sum(ious) / len(ious):.4f}, min {min(ious):.4f} ({len(ious)} boxes)" if ious else "")

frames = sorted(set(ch) | set(ph))
if frames:
    same = sum(set(ch.get(f, {})) == set(ph.get(f, {})) for f in frames)
    pairs = [(ch[f][i], ph[f][i]) for f in frames for i in set(ch.get(f, {})) & set(ph.get(f, {}))]
    hand_ok = sum(a[0] == b[0] for a, b in pairs)
    gest_ok = sum(a[1] == b[1] for a, b in pairs)
    raw_ok = sum(a[2] == b[2] for a, b in pairs)
    fing_ok = sum(a[3] == b[3] for a, b in pairs)
    d = [math.dist(a[4][k:k + 2], b[4][k:k + 2]) for a, b in pairs for k in (0, 2)]
    print(f"hands: same IDs on {same}/{len(frames)} frames; of {len(pairs)} hands: Left/Right {hand_ok}, "
          f"gesture {gest_ok}, raw gesture {raw_ok}, fingers {fing_ok} agree; "
          f"landmark distance mean {sum(d) / len(d):.2f} px, max {max(d):.2f} px" if pairs else "hands: none matched")
    for f in frames:
        for i in set(ch.get(f, {})) & set(ph.get(f, {})):
            if ch[f][i][:4] != ph[f][i][:4]:
                print(f"  frame {f} hand #{i}: C++ {ch[f][i][:4]} | Python {ph[f][i][:4]}")

print(f"events identical: {ce == pe} ({len(ce)} C++, {len(pe)} Python)")
if ce != pe:
    for a, b in zip(ce + [""] * len(pe), pe + [""] * len(ce)):
        if a != b:
            print("  first difference:", a, "|", b)
            break
