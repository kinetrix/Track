import CoreGraphics
import Foundation

/// Fast face path (default; same rules as ../jetson/src/fast_face_tracker.cpp): YuNet on every
/// frame, each detection linked to the track whose predicted box (constant velocity) it overlaps
/// most, and SFace fingerprints to give a returning person their old ID back. About 1 ms of YuNet
/// per frame plus an embedding only when a face appears or a fingerprint is refreshed, so several
/// faces cost little more than one (EdgeTAM costs ~7-8 ms per face here).
struct FastFacePolicy {
    var keepScore: Float = 0.60  // detector confidence to keep following a face
    var startScore: Float = 0.80  // ... to start (or revive) a track
    var nmsIoU: Float = 0.30
    var minFacePx: Float = 16  // in detector input pixels (YuNet finds faces down to ~10 px)
    var matchIoU: Float = 0.30  // detection <-> predicted track box
    var maxMissing: TimeInterval = 1.0  // seconds without a detection -> track is lost (ID kept in the gallery)
    var gallerySeconds: TimeInterval = 30  // how long a lost ID can come back
    var reidSimilarity: Float = 0.55  // fingerprint cosine needed to give a lost ID back (or a saved name)
    var reidMargin: Float = 0.10  // ... and this much above the next-best different candidate
    var minEmbedPx: CGFloat = 64  // smallest face side (frame pixels) worth a fingerprint
    var refreshSeconds: TimeInterval = 1.0  // re-fingerprint each visible track this often (one track per frame)
    var embedEMA: Float = 0.8  // weight of the old fingerprint when refreshing
    var maxTracks = 3
}

final class FastTrack {
    let id: Int
    var box: CGRect  // last detection, frame pixels
    var landmarks: [CGPoint]  // last detection's 5 points, frame pixels
    var velC = CGVector(dx: 0, dy: 0)  // centre velocity, pixels / s
    var velS = CGVector(dx: 0, dy: 0)  // size velocity, pixels / s
    var lastSeen: TimeInterval
    var lastEmbed: TimeInterval = -1e9
    var score: Float
    var visible = true
    var emb: [Float]?
    var name: String?  // saved name this fingerprint matches

    init(id: Int, det: FaceDetection, now: TimeInterval) {
        self.id = id
        box = det.box
        landmarks = det.landmarks
        score = det.score
        lastSeen = now
    }

    func predict(_ now: TimeInterval) -> CGRect {
        let dt = CGFloat(min(now - lastSeen, 0.5))  // don't fling a box far away
        let cx = box.midX + velC.dx * dt, cy = box.midY + velC.dy * dt
        let w = max(4, box.width + velS.dx * dt), h = max(4, box.height + velS.dy * dt)
        return CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }

    func take(_ d: FaceDetection, now: TimeInterval, keepVelocity: Bool) {
        let dt = now - lastSeen
        if keepVelocity && visible && dt > 1e-4 {
            let k: CGFloat = 0.5, inv = CGFloat(1 / dt)
            velC = CGVector(dx: velC.dx * (1 - k) + (d.box.midX - box.midX) * inv * k,
                            dy: velC.dy * (1 - k) + (d.box.midY - box.midY) * inv * k)
            velS = CGVector(dx: velS.dx * (1 - k) + (d.box.width - box.width) * inv * k,
                            dy: velS.dy * (1 - k) + (d.box.height - box.height) * inv * k)
        } else {
            velC = CGVector(dx: 0, dy: 0)
            velS = CGVector(dx: 0, dy: 0)
        }
        box = d.box
        landmarks = d.landmarks
        score = d.score
        lastSeen = now
        visible = true
    }
}

struct LostFace {
    let id: Int
    let emb: [Float]
    let lostAt: TimeInterval
    let name: String?
}

struct KnownFace: Codable {
    var name: String
    var embedding: [Float]
}

final class FastFaceTracker {
    var policy = FastFacePolicy()
    private(set) var tracks: [FastTrack] = []
    private(set) var lost: [LostFace] = []
    /// Saved names (fingerprints); a track whose fingerprint matches one shows that name.
    var known: [KnownFace] = []
    private var nextID = 1
    private var refreshNext = 0
    private(set) var frameIndex = 0
    /// Recent lifecycle events, newest last (HUD); `newEvents` holds this frame's only.
    private(set) var events: [String] = []
    private(set) var newEvents: [String] = []

    private func log(_ e: String) {
        newEvents.append(e)
        events.append("f\(frameIndex) \(e)")
        if events.count > 4 { events.removeFirst(events.count - 4) }
    }

    func reset() {
        tracks.removeAll()
        lost.removeAll()
        nextID = 1
        refreshNext = 0
    }

    private func embeddable(_ d: FaceDetection) -> Bool {
        d.score >= policy.startScore && min(d.box.width, d.box.height) >= policy.minEmbedPx && d.landmarks.count == 5
    }

    /// Best saved name for a fingerprint, if it is clearly that person.
    private func knownName(_ e: [Float]) -> String? {
        var best: Float = -1, second: Float = -1, bestName: String?
        for k in known {
            let s = similarity(e, k.embedding)
            if k.name == bestName { best = max(best, s); continue }
            if s > best {
                if bestName != nil { second = max(second, best) }
                best = s
                bestName = k.name
            } else {
                second = max(second, s)
            }
        }
        guard let bestName, best >= policy.reidSimilarity, best - second >= policy.reidMargin else { return nil }
        return bestName
    }

    private func lose(_ i: Int, now: TimeInterval) {
        let t = tracks[i]
        if let e = t.emb {
            // one gallery entry per person: a newer fingerprint of the same person replaces the older one
            lost.removeAll { similarity($0.emb, e) >= policy.reidSimilarity }
            lost.append(LostFace(id: t.id, emb: e, lostAt: now, name: t.name))
        }
        log("face drop #\(t.id) lost")
        tracks.remove(at: i)
    }

    /// `dets`: this frame's detections (>= keepScore). `embed` returns a unit fingerprint for a
    /// detection (nil: no re-identification).
    func update(_ dets: [FaceDetection], now: TimeInterval, embed: (FaceDetection) -> [Float]?) {
        frameIndex += 1
        newEvents.removeAll()
        let p = policy
        lost.removeAll { now - $0.lostAt > p.gallerySeconds }

        // 1. Link detections to tracks by overlap with each track's predicted box (best pairs first).
        var pairs: [(Float, Int, Int)] = []
        for (ti, t) in tracks.enumerated() {
            let pred = t.predict(now)
            for (di, d) in dets.enumerated() {
                let o = iou(pred, d.box)
                if o >= p.matchIoU { pairs.append((o, ti, di)) }
            }
        }
        pairs.sort { $0.0 > $1.0 }
        var detOf = [Int](repeating: -1, count: tracks.count), trackOf = [Int](repeating: -1, count: dets.count)
        for (_, ti, di) in pairs where detOf[ti] < 0 && trackOf[di] < 0 {
            detOf[ti] = di
            trackOf[di] = ti
        }
        for (ti, t) in tracks.enumerated() {
            if detOf[ti] < 0 { t.visible = false } else { t.take(dets[detOf[ti]], now: now, keepVelocity: true) }
        }

        // 2. Detections no track claimed: a returning face (fingerprint matches a missing track or a
        //    lost ID, clearly better than any other candidate) or a new face. Most confident first.
        let fresh = dets.indices.filter { trackOf[$0] < 0 && dets[$0].score >= p.startScore }
            .sorted { dets[$0].score > dets[$1].score }
        for di in fresh {
            let d = dets[di]
            let e = embeddable(d) ? embed(d) : nil
            var bestTrack = -1, bestLost = -1
            var best: Float = -1, second: Float = -1
            func consider(_ s: Float, _ ti: Int, _ li: Int) {
                if s > best {
                    second = best
                    best = s
                    bestTrack = ti
                    bestLost = li
                } else {
                    second = max(second, s)
                }
            }
            if let e {
                for (ti, t) in tracks.enumerated() where !t.visible {
                    if let te = t.emb { consider(similarity(e, te), ti, -1) }
                }
                for (li, l) in lost.enumerated() { consider(similarity(e, l.emb), -1, li) }
            }
            if best < p.reidSimilarity || best - second < p.reidMargin {
                bestTrack = -1
                bestLost = -1
            }
            if bestTrack >= 0 {  // missing for a moment, then back somewhere the prediction didn't expect
                let t = tracks[bestTrack]
                t.take(d, now: now, keepVelocity: false)
                detOf[bestTrack] = di
                log(String(format: "face back #%d (similarity %.2f)", t.id, best))
                continue
            }
            if tracks.count >= p.maxTracks { continue }
            let t: FastTrack
            if bestLost >= 0 {
                let l = lost.remove(at: bestLost)
                t = FastTrack(id: l.id, det: d, now: now)
                t.name = l.name
                log(String(format: "face back #%d after %.1fs (similarity %.2f)", t.id, now - l.lostAt, best))
            } else {
                t = FastTrack(id: nextID, det: d, now: now)
                nextID += 1
                log("face new #\(t.id)")
            }
            if let e {
                t.emb = e
                t.lastEmbed = now
                if let n = knownName(e) { t.name = n }
            }
            tracks.append(t)
            detOf.append(di)
        }

        // 3. Tracks missing too long are lost; their fingerprint stays in the gallery.
        for ti in tracks.indices.reversed() where !tracks[ti].visible && now - tracks[ti].lastSeen > p.maxMissing {
            lose(ti, now: now)
            detOf.remove(at: ti)
        }

        // 4. Refresh one visible track's fingerprint per frame (round robin), only from a clear,
        //    unobstructed view so an overlapping face doesn't blur it.
        guard !tracks.isEmpty else { return }
        for k in 0..<tracks.count {
            let ti = (refreshNext + k) % tracks.count
            let t = tracks[ti]
            guard t.visible, detOf[ti] >= 0, now - t.lastEmbed >= p.refreshSeconds else { continue }
            let d = dets[detOf[ti]]
            guard embeddable(d) else { continue }
            let overlapped = dets.indices.contains { $0 != detOf[ti] && d.box.intersects(dets[$0].box) }
            if overlapped { continue }
            guard let e = embed(d) else { break }
            if let old = t.emb {
                t.emb = normalized(zip(old, e).map { p.embedEMA * $0 + (1 - p.embedEMA) * $1 })
            } else {
                t.emb = e
            }
            t.lastEmbed = now
            if let n = knownName(t.emb!) { t.name = n }
            refreshNext = ti + 1
            break
        }
    }

    /// Fingerprint of a track (for saving a name).
    func embedding(of id: Int) -> [Float]? { tracks.first { $0.id == id }?.emb }

    /// After a name is saved: name every track (and lost ID) whose fingerprint matches it.
    func applyNames() {
        for t in tracks { if let e = t.emb, let n = knownName(e) { t.name = n } }
    }
}
