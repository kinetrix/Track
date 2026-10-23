import CoreGraphics
import Foundation

/// Tunables. Same values as `FacePolicy` in ../onnx/facetrack/faces.py and the iOS app.
struct TrackPolicy {
    var detScore: Float = 0.80  // detector confidence to start a track
    var nmsIoU: Float = 0.30
    var detInterval = 3  // detector period while tracks exist (every frame when none)
    var matchIoU: Float = 0.30  // detection <-> track association
    var reanchorIoU: Float = 0.50  // matched but boxes disagree -> re-prompt with the detection
    var reanchorMinGap: TimeInterval = 0.5  // seconds between re-anchors of one track
    var minTrackIoUPred: Float = 0.60  // SAM's own mask-quality estimate below this -> re-anchor
    // Time limits are in seconds, not frames, so they hold at any frame rate.
    var maxMissing: TimeInterval = 1.0  // object score <= 0 (occluded / gone) this long -> drop
    var confidentIoUPred: Float = 0.80  // SAM mask-quality estimate that counts as "confident"
    var maxUnconfirmed: TimeInterval = 5.0  // no detector match this long while NOT confident -> drop
    var maxUnconfirmedConfident: TimeInterval = 30.0  // ... and while confident (detector misses tilted/profile faces)
    var duplicateIoU: Float = 0.60  // two visible tracks overlapping this much -> drop the younger
    var maxTracks = 4  // EdgeTAM cost is per face: cap it
    var minFacePx: Float = 24  // in detector input pixels
}

final class Track {
    let id: Int
    let memory: EdgeTAM.Memory
    var box: CGRect  // frame pixels
    var mask: [Float16]  // n x n logits over the squashed frame (n = EdgeTAM mask size)
    var age = 0
    var lastSeen: TimeInterval  // last frame with object score > 0
    var lastConfirmed: TimeInterval  // last frame matched by the detector
    var lastAnchor: TimeInterval  // last box prompt (spawn or re-anchor)
    var iouPred: Float
    var visible = true

    init(id: Int, memory: EdgeTAM.Memory, box: CGRect, mask: [Float16], iouPred: Float, now: TimeInterval) {
        self.id = id
        self.memory = memory
        self.box = box
        self.mask = mask
        self.iouPred = iouPred
        lastSeen = now
        lastConfirmed = now
        lastAnchor = now
    }
}

/// Detector-guided tracking policy (DESIGN.md §5): the detector starts, confirms
/// and corrects tracks; EdgeTAM carries identity and pixel-accurate masks between
/// detections and through poses/occlusions the detector misses.
final class TrackManager {
    var policy = TrackPolicy()
    private(set) var tracks: [Track] = []
    private var nextID = 1
    private(set) var frameIndex = 0
    /// Recent lifecycle events (spawn / re-anchor / drop + reason), newest last. Shown in the HUD.
    private(set) var events: [String] = []

    private func log(_ e: String) {
        events.append("f\(frameIndex) \(e)")
        if events.count > 4 { events.removeFirst(events.count - 4) }
    }

    func shouldDetect() -> Bool { tracks.isEmpty || frameIndex % policy.detInterval == 0 }

    func reset() { tracks.removeAll() }

    /// One frame at time `now` (seconds). `frame` is nil only when there are no tracks and no detections.
    func update(detections dets: [FaceDetection], frame: EdgeTAM.Frame?, tam: EdgeTAM,
                frameSize: CGSize, now: TimeInterval) throws {
        defer { frameIndex += 1 }
        guard let frame else { return }
        let toModel = CGAffineTransform(scaleX: CGFloat(tam.cfg.image_size) / frameSize.width,
                                       y: CGFloat(tam.cfg.image_size) / frameSize.height)

        // 1) greedy IoU association between last-frame track boxes and detections
        var pairs: [(Float, Int, Int)] = []
        for (ti, t) in tracks.enumerated() {
            for (di, d) in dets.enumerated() { pairs.append((iou(t.box, d.box), ti, di)) }
        }
        pairs.sort { $0.0 > $1.0 }
        var usedT = Set<Int>(), usedD = Set<Int>(), match: [Int: Int] = [:]
        for (v, ti, di) in pairs where v >= policy.matchIoU && !usedT.contains(ti) && !usedD.contains(di) {
            usedT.insert(ti); usedD.insert(di); match[ti] = di
        }

        // 2) propagate or re-anchor every live track
        for (ti, t) in tracks.enumerated() {
            t.age += 1
            if let di = match[ti] {
                t.lastConfirmed = now
                let disagree = iou(t.box, dets[di].box) < policy.reanchorIoU || t.iouPred < policy.minTrackIoUPred
                if disagree && now - t.lastAnchor >= policy.reanchorMinGap {
                    let (_, r) = try tam.addBox(frame, box: dets[di].box.applying(toModel), memory: t.memory)
                    log(String(format: "re-anchor #%ld (IoU %.2f, q %.2f)", t.id, iou(t.box, dets[di].box), t.iouPred))
                    t.lastAnchor = now
                    apply(r, to: t, frameSize: frameSize, now: now)
                    continue
                }
            }
            let r = try tam.track(frame, memory: t.memory)
            apply(r, to: t, frameSize: frameSize, now: now)
        }

        // 3) new tracks for unmatched detections
        for (di, d) in dets.enumerated() where !usedD.contains(di) && tracks.count < policy.maxTracks {
            if tracks.contains(where: { iou($0.box, d.box) > policy.matchIoU }) { continue }
            let (mem, r) = try tam.addBox(frame, box: d.box.applying(toModel), memory: nil)
            let t = Track(id: nextID, memory: mem, box: d.box, mask: r.mask, iouPred: r.iouPred, now: now)
            nextID += 1
            apply(r, to: t, frameSize: frameSize, now: now)
            tracks.append(t)
            log("new #\(t.id)")
        }

        // 4) retire lost / never-confirmed / duplicate tracks (older tracks win)
        var keep: [Track] = []
        for t in tracks.sorted(by: { $0.age > $1.age }) {
            if now - t.lastSeen > policy.maxMissing { log("drop #\(t.id) lost"); continue }
            let limit = t.iouPred >= policy.confidentIoUPred ? policy.maxUnconfirmedConfident : policy.maxUnconfirmed
            if now - t.lastConfirmed > limit { log("drop #\(t.id) unconfirmed"); continue }
            if t.visible && keep.contains(where: { $0.visible && iou($0.box, t.box) > policy.duplicateIoU }) {
                log("drop #\(t.id) duplicate")
                continue
            }
            keep.append(t)
        }
        tracks = keep.sorted { $0.id < $1.id }
    }

    private func apply(_ r: EdgeTAM.Result, to t: Track, frameSize: CGSize, now: TimeInterval) {
        t.mask = r.mask
        t.iouPred = r.iouPred
        guard r.objScore > 0, let b = maskBox(r.mask) else {
            t.visible = false
            return
        }
        t.visible = true
        t.lastSeen = now
        let n = CGFloat(maskSide(r.mask))
        let sx = frameSize.width / n, sy = frameSize.height / n
        t.box = CGRect(x: b.minX * sx, y: b.minY * sy, width: b.width * sx, height: b.height * sy)
    }

    private func maskBox(_ m: [Float16]) -> CGRect? {
        let n = maskSide(m)
        var x0 = n, y0 = n, x1 = -1, y1 = -1
        m.withUnsafeBufferPointer { p in
            for y in 0..<n {
                let row = p.baseAddress! + y * n
                for x in 0..<n where row[x] > 0 {
                    if x < x0 { x0 = x }
                    if x > x1 { x1 = x }
                    if y < y0 { y0 = y }
                    y1 = y
                }
            }
        }
        return x1 < 0 ? nil : CGRect(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1)
    }
}

/// Side of a square low-res mask stored row-major (128 at 512 input, 256 at 1024).
func maskSide(_ m: [Float16]) -> Int { Int(Double(m.count).squareRoot().rounded()) }
