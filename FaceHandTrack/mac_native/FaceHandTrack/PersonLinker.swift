import CoreGraphics
import Foundation

/// Who is gesturing at the camera (robot): each hand is linked to the person (face) it belongs
/// to, and a person who faces the camera and holds a gesture is "addressing" it.
/// No model: a few comparisons per frame on the face and hand boxes the app already has.
///
///   reach  a hand can only belong to a face within arm's reach, measured in face widths
///   size   a person's hand and face appear at a similar scale, so a hand far smaller or larger
///          than a face belongs to someone nearer or farther away (a depth cue from one camera)
///   side   a right hand is usually on the person's right of their face
///   one    each person has at most one left and one right hand
///   sticky a hand keeps its person unless another fits clearly better (no flicker)
struct LinkPolicy {
    var maxReach: CGFloat = 4.5  // hand centre to face centre, in face widths (real links seen up to ~3.7)
    var handFaceRatio: CGFloat = 0.6  // typical hand size (wrist..middle knuckle) / face box width
    var minRatio: CGFloat = 0.25, maxRatio: CGFloat = 2.5  // outside this, never the same person
    var sizeWeight: CGFloat = 2.0  // cost per unit of |ln(ratio / handFaceRatio)|
    var wrongSidePenalty: CGFloat = 1.0  // hand clearly on the other side of the face (crossed arm)
    var aboveHeadPenalty: CGFloat = 1.0  // hand far above the head
    var stickyBonus: CGFloat = 0.75  // previous owner's cost is lowered by this much
    var maxYaw: CGFloat = 0.35  // nose offset from the eye midpoint / eye distance: below = facing the camera
    var holdSeconds: TimeInterval = 0.5  // a gesture held this long while facing = addressing
    var lingerSeconds: TimeInterval = 1.0  // keep "addressing" this long after the gesture ends
}

struct LinkFace {
    let id: Int
    let box: CGRect  // frame pixels
    let landmarks: [CGPoint]  // 5 YuNet points, frame pixels (empty if unknown this frame)
    let visible: Bool
}

struct LinkHand {
    let id: Int
    let handedness: String  // the person's real hand: "Left" / "Right"
    let points: [CGPoint]  // 21 landmarks, frame pixels
    let gesture: String  // stable gesture, "" = none
}

struct PersonState {
    let faceID: Int
    var handIDs: [Int] = []
    var facing: Bool?  // nil: not known yet
    var addressing: String?  // gesture this person is showing the camera
}

final class PersonLinker {
    var policy = LinkPolicy()
    /// Frames are mirrored like a selfie view: flips which side of the face a right hand appears on.
    var mirrored = true
    private(set) var owner: [Int: Int] = [:]  // hand id -> face id
    private(set) var people: [Int: PersonState] = [:]  // face id -> state
    private var facing: [Int: Bool] = [:]  // last known, per face id
    private var held: [Int: (gesture: String, since: TimeInterval)] = [:]  // per hand id
    private var addressing: [Int: (gesture: String, until: TimeInterval)] = [:]  // per face id
    private(set) var newEvents: [String] = []

    func reset() {
        owner.removeAll()
        people.removeAll()
        facing.removeAll()
        held.removeAll()
        addressing.removeAll()
    }

    /// Is this face turned to the camera? From YuNet's points: right eye, left eye, nose, mouth corners.
    static func yaw(_ lm: [CGPoint]) -> CGFloat? {
        guard lm.count >= 3 else { return nil }
        let eyeDist = hypot(lm[1].x - lm[0].x, lm[1].y - lm[0].y)
        guard eyeDist > 1 else { return nil }
        let mid = CGPoint(x: (lm[0].x + lm[1].x) / 2, y: (lm[0].y + lm[1].y) / 2)
        // nose offset along the eye line, relative to the eye distance (0 = frontal, ~0.5+ = profile)
        let ux = (lm[1].x - lm[0].x) / eyeDist, uy = (lm[1].y - lm[0].y) / eyeDist
        return abs((lm[2].x - mid.x) * ux + (lm[2].y - mid.y) * uy) / eyeDist
    }

    /// Hand size in pixels: wrist to middle knuckle, or the knuckle span (scaled to the same
    /// length) when the palm is foreshortened.
    static func handSize(_ p: [CGPoint]) -> CGFloat {
        guard p.count == 21 else { return 0 }
        let d = { (a: Int, b: Int) in hypot(p[a].x - p[b].x, p[a].y - p[b].y) }
        return max(d(0, 9), 1.3 * d(5, 17))
    }

    /// Reach (hand centre to face centre, in face widths) and hand / face size ratio.
    func measure(_ hand: LinkHand, _ face: LinkFace) -> (reach: CGFloat, ratio: CGFloat) {
        let fw = max(face.box.width, 1)
        let n = CGFloat(max(hand.points.count, 1))
        let hx = hand.points.map(\.x).reduce(0, +) / n, hy = hand.points.map(\.y).reduce(0, +) / n
        return (hypot(hx - face.box.midX, hy - face.box.midY) / fw, Self.handSize(hand.points) / fw)
    }

    /// Cost of `hand` belonging to `face` (lower is better), nil if impossible.
    func cost(_ hand: LinkHand, _ face: LinkFace) -> CGFloat? {
        let pol = policy
        let fw = max(face.box.width, 1)
        let hc = CGPoint(x: hand.points.map(\.x).reduce(0, +) / CGFloat(max(hand.points.count, 1)),
                         y: hand.points.map(\.y).reduce(0, +) / CGFloat(max(hand.points.count, 1)))
        let dx = (hc.x - face.box.midX) / fw, dy = (hc.y - face.box.midY) / fw
        let dist = hypot(dx, dy)
        guard dist <= pol.maxReach else { return nil }
        let size = Self.handSize(hand.points)
        guard size > 0 else { return nil }
        let ratio = size / fw
        guard ratio >= pol.minRatio && ratio <= pol.maxRatio else { return nil }
        var c = dist + pol.sizeWeight * abs(log(ratio / pol.handFaceRatio))
        // Un-mirrored camera view: the person's right hand appears on the image's left (smaller x).
        let rightIsLeftInImage = !mirrored
        let onImageLeft = dx < -0.5, onImageRight = dx > 0.5
        if hand.handedness == "Right" && (rightIsLeftInImage ? onImageRight : onImageLeft) { c += pol.wrongSidePenalty }
        if hand.handedness == "Left" && (rightIsLeftInImage ? onImageLeft : onImageRight) { c += pol.wrongSidePenalty }
        if dy < -1.5 { c += pol.aboveHeadPenalty }  // more than 1.5 face widths above the face centre
        if owner[hand.id] == face.id { c -= pol.stickyBonus }
        return c
    }

    func update(faces: [LinkFace], hands: [LinkHand], now: TimeInterval) {
        newEvents.removeAll()
        let pol = policy
        let visible = faces.filter(\.visible)

        // facing, remembered per face between frames where its points are known
        for f in faces {
            if let y = Self.yaw(f.landmarks) { facing[f.id] = y <= pol.maxYaw }
        }

        // 1. hands -> people: cheapest pairs first, at most one Left and one Right per person
        var pairs: [(CGFloat, Int, Int)] = []
        for (hi, h) in hands.enumerated() {
            for (fi, f) in visible.enumerated() {
                if let c = cost(h, f) { pairs.append((c, hi, fi)) }
            }
        }
        pairs.sort { $0.0 < $1.0 }
        var newOwner: [Int: Int] = [:]
        var taken = Set<String>()  // "faceID-Left"
        for (_, hi, fi) in pairs {
            let h = hands[hi], f = visible[fi]
            if newOwner[h.id] != nil { continue }
            let slot = "\(f.id)-\(h.handedness)"
            if taken.contains(slot) { continue }
            newOwner[h.id] = f.id
            taken.insert(slot)
        }
        for h in hands where newOwner[h.id] != owner[h.id] {
            guard let fid = newOwner[h.id], let f = visible.first(where: { $0.id == fid }) else { continue }
            let m = measure(h, f)  // printed so the size rule can be tuned on real people
            newEvents.append(String(format: "hand #%d (%@) -> face #%d (reach %.1f, size %.2f)", h.id, h.handedness, fid, m.reach, m.ratio))
        }
        let prevOwner = owner
        owner = newOwner

        // 2. held gestures (per hand), then who is addressing the camera (per person)
        let live = Set(hands.map(\.id))
        held = held.filter { live.contains($0.key) }
        for h in hands {
            if h.gesture.isEmpty || newOwner[h.id] != prevOwner[h.id] {
                held[h.id] = nil  // the hold restarts when the gesture ends or the hand changes person
            }
            if !h.gesture.isEmpty && held[h.id]?.gesture != h.gesture {
                held[h.id] = (h.gesture, now)
            }
        }
        var state: [Int: PersonState] = [:]
        for f in visible { state[f.id] = PersonState(faceID: f.id, facing: facing[f.id]) }
        for h in hands {
            guard let f = owner[h.id] else { continue }
            state[f]?.handIDs.append(h.id)
        }
        for f in visible {
            // the longest-held gesture among this person's hands
            let showing = state[f.id]!.handIDs.compactMap { held[$0] }
                .filter { now - $0.since >= pol.holdSeconds }
                .min { $0.since < $1.since }
            if let g = showing, facing[f.id] == true {
                if addressing[f.id]?.gesture != g.gesture || addressing[f.id].map({ now > $0.until }) ?? true {
                    newEvents.append("face #\(f.id) addressing: \(g.gesture)")
                }
                addressing[f.id] = (g.gesture, now + pol.lingerSeconds)
            }
            if let a = addressing[f.id], now <= a.until { state[f.id]!.addressing = a.gesture }
        }
        let visibleIDs = Set(visible.map(\.id))
        addressing = addressing.filter { visibleIDs.contains($0.key) && now <= $0.value.until }
        let faceIDs = Set(faces.map(\.id))
        facing = facing.filter { faceIDs.contains($0.key) }
        people = state
    }
}
