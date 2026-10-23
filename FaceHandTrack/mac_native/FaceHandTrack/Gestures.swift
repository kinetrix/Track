import Foundation
import simd

/// Static hand gestures from the 21 landmarks, with hand-written rules.
/// Line-by-line port of the Python rules (../onnx/facetrack/gestures.py), same thresholds.
///
/// Finger state uses the landmark model's *world* landmarks (3-D, metric, centred on
/// the hand), so it does not change with hand size, distance or in-plane rotation.
/// Pinch distance and pointing direction use the image landmarks.
enum Gestures {
    static let all = ["Open palm", "Fist", "Thumbs up", "Thumbs down", "Pointing up", "Pointing down", "Pointing left",
                      "Pointing right", "Victory", "Three", "Rock", "I love you", "Call me", "OK", "Pinch"]

    // thresholds
    static let extRatio: Float = 1.15  // fingertip at least this much farther from the wrist than the PIP joint
    static let extBend: Float = 80  // and the finger bends less than this (degrees, PIP + DIP)
    static let thumbOut: Float = 0.85  // thumb tip distance from ring-finger base, in palm lengths
    static let thumbBend: Float = 60  // thumb bends less than this (degrees, MCP + IP)
    static let pinch: Float = 0.25  // thumb tip - index tip distance, in palm lengths
    static let indexNotFolded: Float = 0.88  // pinch needs the index only partly bent
    static let indexClear: Float = 1.25  // "I love you" needs a clearly straight index

    static let fingers: [[Int]] = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]  // index..pinky

    static func angle(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let c = simd_dot(a, b) / (simd_length(a) * simd_length(b) + 1e-9)
        return acos(min(max(c, -1), 1)) * 180 / .pi
    }

    /// Extended (true) or folded for thumb, index, middle, ring, pinky.
    static func fingerStates(_ w: [SIMD3<Float>]) -> [Bool] {
        let palm = simd_length(w[9] - w[0]) + 1e-9
        let t0 = w[3] - w[2], t1 = w[4] - w[3]
        let thumb = simd_length(w[4] - w[13]) / palm > thumbOut && angle(t0, t1) + angle(w[2] - w[1], t0) < thumbBend
        var out = [thumb]
        for f in fingers {
            let s0 = w[f[1]] - w[f[0]], s1 = w[f[2]] - w[f[1]], s2 = w[f[3]] - w[f[2]]
            let bend = angle(s0, s1) + angle(s1, s2)
            let ratio = simd_length(w[f[3]] - w[0]) / (simd_length(w[f[1]] - w[0]) + 1e-9)
            out.append(ratio > extRatio && bend < extBend)
        }
        return out
    }

    /// Screen direction of an image-space vector (y grows downwards).
    static func direction(_ v: SIMD2<Float>) -> String {
        if abs(v.y) >= abs(v.x) { return v.y < 0 ? "up" : "down" }
        return v.x < 0 ? "left" : "right"
    }

    /// landmarks: 21 points, image pixels + relative z; world: 21 metric points.
    /// Returns (gesture or "", number of extended fingers).
    static func classify(landmarks lm: [SIMD3<Float>], world: [SIMD3<Float>]) -> (String, Int) {
        let st = fingerStates(world)
        let count = st.filter { $0 }.count
        let (T, I, M, R, P) = (st[0], st[1], st[2], st[3], st[4])
        let palmImg = simd_length(lm[9] - lm[0]) + 1e-9
        let indexRatio = simd_length(world[8] - world[0]) / (simd_length(world[6] - world[0]) + 1e-9)
        let isPinch = simd_length(lm[4] - lm[8]) / palmImg < pinch && indexRatio > indexNotFolded
        func xy(_ p: SIMD3<Float>) -> SIMD2<Float> { SIMD2(p.x, p.y) }

        if isPinch { return (M && R && P ? "OK" : "Pinch", count) }
        if I && M && R && P { return ("Open palm", count) }
        if !(I || M || R || P) {
            if !T { return ("Fist", count) }
            let d = direction(xy(lm[4]) - xy(lm[2]))
            if d == "up" || d == "down" { return (d == "up" ? "Thumbs up" : "Thumbs down", count) }
            return ("", count)
        }
        if I && !(M || R || P) { return ("Pointing " + direction(xy(lm[8]) - xy(lm[5])), count) }
        if I && M && !(R || P) && !T { return ("Victory", count) }
        if I && M && R && !P && !T { return ("Three", count) }
        if I && P && !(M || R) {
            if !T { return ("Rock", count) }
            return (indexRatio > indexClear ? "I love you" : "Call me", count)
        }
        if T && P && !(I || M || R) { return ("Call me", count) }
        return ("", count)
    }
}

/// Reports a gesture only after it won the majority of the last `window` frames,
/// so labels don't flicker while the hand moves between poses.
final class GestureSmoother {
    private let window: Int
    private let need: Int
    private var history: [String] = []
    private(set) var stable = ""

    init(window: Int = 5, need: Int = 3) {
        self.window = window
        self.need = need
    }

    func update(_ label: String) -> String {
        history.append(label)
        if history.count > window { history.removeFirst(history.count - window) }
        var counts: [String: Int] = [:]
        for l in history { counts[l, default: 0] += 1 }
        if let best = counts.max(by: { $0.value < $1.value }), best.value >= need { stable = best.key }
        return stable
    }
}
