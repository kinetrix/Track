import CoreGraphics
import Foundation

/// Waving to call the robot, and where the robot should turn to face that person.
///
///   wave       an open hand (four fingers up) moving side to side: at least 3 swings, each at
///              least 0.6 x the hand's size (about 6 cm), within 1.5 s, raised to about shoulder
///              height or above
///   attention  the person who last waved, or who started addressing the robot when no one had
///              its attention; kept until they are out of sight 2 s, someone else waves, or 30 s
///              pass without a wave or a held gesture from them
///   turn       horizontal and vertical angle from the camera's centre to that person's face,
///              from the camera's field of view (robot's left / right, up / down)
///
/// No model: the hand points and face boxes the app already has.
struct AttentionPolicy {
    var window: TimeInterval = 1.5  // a wave's swings must fit in this long
    var minSwings = 3  // left-right-left (or right-left-right)
    var swingFraction: CGFloat = 0.6  // each swing at least this x the hand size (wrist..middle knuckle)
    var raisedBelowChin: CGFloat = 0.5  // hand centre at most this many face heights below the chin
    var waveCooldown: TimeInterval = 3  // one wave per person per this long
    var wavingShown: TimeInterval = 1.5  // "waving" label stays this long
    var lostAfter: TimeInterval = 2  // attention target out of sight this long -> no target
    var idleAfter: TimeInterval = 30  // no wave / held gesture from the target this long -> no target
    var horizontalFOV: Double = 70  // degrees (D435i colour 69; MacBook cameras about 70-80)
    var centredWithin: Double = 5  // degrees: |angle| below this counts as already facing
}

struct AttentionHand {
    let id: Int
    let owner: Int?
    let extended: [Bool]  // thumb, index, middle, ring, pinky
    let points: [CGPoint]  // 21 landmarks, frame pixels
}

struct AttentionFace {
    let id: Int
    let box: CGRect  // frame pixels
    let addressing: Bool
}

struct AttentionTarget {
    let faceID: Int
    let yaw: Double  // degrees, + = robot should turn to its right
    let pitch: Double  // degrees, + = up
    let reason: String  // "waved" / "addressing"
    var turnText: String {
        let h = abs(yaw) < 5 ? "facing" : String(format: "turn %@ %.0f°", yaw > 0 ? "right" : "left", abs(yaw))
        let v = abs(pitch) < 5 ? "" : String(format: ", %@ %.0f°", pitch > 0 ? "up" : "down", abs(pitch))
        return h + v
    }
}

final class AttentionTracker {
    var policy = AttentionPolicy()
    /// Frames are mirrored like a selfie view: flips the robot's left / right.
    var mirrored = true
    private var history: [Int: [(t: TimeInterval, x: CGFloat, size: CGFloat)]] = [:]  // per hand id
    private var lastWave: [String: TimeInterval] = [:]  // per person key
    private(set) var wavingUntil: [Int: TimeInterval] = [:]  // per face id
    private var targetID: Int?
    private var targetReason = ""
    private var targetLastSeen: TimeInterval = 0
    private var targetLastActive: TimeInterval = 0
    private var wasAddressing = Set<Int>()
    private(set) var target: AttentionTarget?
    private(set) var newEvents: [String] = []

    func reset() {
        history.removeAll()
        lastWave.removeAll()
        wavingUntil.removeAll()
        targetID = nil
        target = nil
        wasAddressing.removeAll()
    }

    /// Number of alternating side-to-side moves of at least `amp` in `xs`.
    static func swings(_ xs: [CGFloat], amp: CGFloat) -> Int {
        guard var lo = xs.first, amp > 0 else { return 0 }
        var hi = lo, dir = 0, n = 0, ext = lo
        for x in xs.dropFirst() {
            switch dir {
            case 0:
                lo = min(lo, x)
                hi = max(hi, x)
                if x - lo >= amp { dir = 1; n = 1; ext = x } else if hi - x >= amp { dir = -1; n = 1; ext = x }
            case 1:
                if x > ext { ext = x } else if ext - x >= amp { dir = -1; n += 1; ext = x }
            default:
                if x < ext { ext = x } else if x - ext >= amp { dir = 1; n += 1; ext = x }
            }
        }
        return n
    }

    func update(faces: [AttentionFace], hands: [AttentionHand], frameSize: CGSize, now: TimeInterval) {
        newEvents.removeAll()
        let pol = policy
        let faceByID = Dictionary(uniqueKeysWithValues: faces.map { ($0.id, $0) })
        let live = Set(hands.map(\.id))
        history = history.filter { live.contains($0.key) }

        // 1. waves, per hand
        for h in hands where h.extended.count == 5 && h.points.count == 21 {
            let open = h.extended[1] && h.extended[2] && h.extended[3] && h.extended[4]
            guard open else {
                history[h.id] = nil
                continue
            }
            let cx = h.points.map(\.x).reduce(0, +) / 21, cy = h.points.map(\.y).reduce(0, +) / 21
            var hist = (history[h.id] ?? []).filter { now - $0.t <= pol.window }
            hist.append((now, cx, PersonLinker.handSize(h.points)))
            history[h.id] = hist
            let size = hist.map(\.size).sorted()[hist.count / 2]  // median: robust to one bad frame
            guard Self.swings(hist.map(\.x), amp: pol.swingFraction * size) >= pol.minSwings else { continue }
            let face = h.owner.flatMap { faceByID[$0] }
            if let f = face, cy > f.box.maxY + pol.raisedBelowChin * f.box.height { continue }  // hand too low
            let key = h.owner.map { "f\($0)" } ?? "h\(h.id)"
            history[h.id] = []
            if let t = lastWave[key], now - t < pol.waveCooldown { continue }
            lastWave[key] = now
            if let f = face {
                wavingUntil[f.id] = now + pol.wavingShown
                newEvents.append("face #\(f.id) waved: calling the robot")
                if targetID != f.id { newEvents.append("robot attention -> face #\(f.id) (waved)") }
                targetID = f.id
                targetReason = "waved"
                targetLastActive = now
            } else {
                newEvents.append("hand #\(h.id) waved (face not seen)")
            }
        }
        wavingUntil = wavingUntil.filter { now <= $0.value }

        // 2. someone starting to address the robot gets its attention if no one has it
        let addressingNow = Set(faces.filter(\.addressing).map(\.id))
        if let id = targetID, addressingNow.contains(id) { targetLastActive = now }
        if targetID == nil, let id = addressingNow.subtracting(wasAddressing).sorted().first {
            targetID = id
            targetReason = "addressing"
            targetLastActive = now
            newEvents.append("robot attention -> face #\(id) (addressing)")
        }
        wasAddressing = addressingNow

        // 3. keep or drop the target, and where to turn
        target = nil
        guard let id = targetID else { return }
        if let f = faceByID[id] {
            targetLastSeen = now
            let w = Double(frameSize.width), hgt = Double(frameSize.height)
            let focal = (w / 2) / tan(pol.horizontalFOV / 2 * .pi / 180)  // pixels
            var yaw = atan((Double(f.box.midX) - w / 2) / focal) * 180 / .pi  // + = image right
            if mirrored { yaw = -yaw }  // a selfie view shows the robot's right on the image's left
            let pitch = -atan((Double(f.box.midY) - hgt / 2) / focal) * 180 / .pi
            target = AttentionTarget(faceID: id, yaw: yaw, pitch: pitch, reason: targetReason)
        }
        let gone = now - targetLastSeen > pol.lostAfter, idle = now - targetLastActive > pol.idleAfter
        if gone || idle {
            newEvents.append("robot attention released (face #\(id) " + (gone ? "out of sight)" : "idle)"))
            targetID = nil
            target = nil
        }
    }
}
