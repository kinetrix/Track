import sys, glob, os, numpy as np, torch, cv2
sys.argv += []
from compare_sizes import build, run
from PIL import Image
torch.set_grad_enabled(False)
clip, box, tag = sys.argv[1], [float(v) for v in sys.argv[2:6]], sys.argv[6]
files = sorted(glob.glob(os.path.join(clip, "*.jpg"))); W, H = Image.open(files[0]).size
b = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
out = {}
for S in (1024, 512):
    trk, _ = build(S); out[S] = run(trk, files, b, W, H)[0]
ious = []
for i, (a, c) in enumerate(zip(out[1024], out[512])):
    u = (a | c).sum(); iou = (a & c).sum() / u if u else 1.0; ious.append(iou)
    print(f"frame {i:2d} area1024 {a.sum():6d} area512 {c.sum():6d} IoU {iou:.3f}")
worst = np.argsort(ious)[:3]
tiles = []
for i in sorted(worst):
    img = cv2.imread(files[i])
    for S, col in ((1024, (0, 0, 255)), (512, (0, 255, 0))):
        cs, _ = cv2.findContours(out[S][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 3)
    cv2.putText(img, f"f{i} IoU {ious[i]:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    tiles.append(cv2.resize(img, (360, 640)))
cv2.imwrite(f"worst_{tag}.jpg", np.hstack(tiles))
