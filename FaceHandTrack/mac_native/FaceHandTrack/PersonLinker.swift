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
///   pose   when body poses are known (hands in view), a hand whose wrist sits on the same-side
///          wrist of a person's pose (left hand, left wrist) belongs to that person; otherwise the
///          farther the person's nearest wrist, the higher the cost, and a person whose visible
///          wrists are both very far cannot own it. A pose whose nose is not on its
///          own face is someone else's body and is ignored; when two poses claim the same hand,
///          only a clearly closer one counts (same rules as ../jetson/src/person_linker.cpp)
struct LinkPolicy {
    var maxReach: CGFloat = 5.0  // hand centre to face centre, in face widths (raised hands reach ~4.5; doubtful links were 5.3+)
    var handFaceRatio: CGFloat = 0.6  // typical hand size (wrist..middle knuckle) / face box width
    var minRatio: CGFloat = 0.25, maxRatio: CGFloat = 2.5  // outside this, never the same person
    var sizeWeight: CGFloat = 2.0  // cost per unit of |ln(ratio / handFaceRatio)|
    var wrongSidePenalty: CGFloat = 1.0  // hand clearly on the other side of the face (crossed arm)
    var aboveHeadPenalty: CGFloat = 1.0  // hand far above the head
    var stickyBonus: CGFloat = 0.75  // previous owner's cost is lowered by this much
    var maxYaw: CGFloat = 0.35  // nose offset from the eye midpoint / eye distance: below = facing the camera
    var holdSeconds: TimeInterval = 0.5  // a gesture held this long while facing = addressing
    var lingerSeconds: TimeInterval = 1.0  // keep "addressing" this long after the gesture ends
    var poseVisible: Float = 0.5  // pose wrist visibility needed to count
    var poseMatch: CGFloat = 1.0  // hand wrist within this many hand sizes of a pose wrist: that person's
    var poseWeight: CGFloat = 1.5  // otherwise cost per hand size from the person's wrist (a far wrist costs more) ...
    var poseCap: CGFloat = 6  // ... up to this many; beyond it from both visible wrists: not that person's
    var handednessSlack: CGFloat = 0.5  // the other wrist counts too, this much farther (Left/Right labels flip)
    var poseBonus: CGFloat = 10  // cost lowered by this for a pose match (decides over the other rules)
    var poseNoseReach: CGFloat = 1.0  // pose nose farther than this many face heights from its face: someone else's pose
    var poseClearlyCloser: CGFloat = 0.6  // two poses claim a hand: one counts only if this much closer
}

/// A body pose for one face, as the linker needs it.
struct LinkPose {
    let nose: CGPoint  // frame pixels
    let wrists: [(point: CGPoint, visibility: Float)]  // [person's left, person's right], frame pixels
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
    private var poses: [Int: LinkPose] = [:]  // per face id, accepted poses this frame only
    private var poseIgnored: Set<Int> = []  // face ids whose pose was not on their face (logged once)

    /// Hand wrist distance to a face's pose wrists, in hand sizes: (same-side wrist or nil if not
    /// visible, nearest visible wrist or nil if none).
    func poseDistances(_ hand: LinkHand, _ faceID: Int) -> (same: CGFloat?, any: CGFloat?) {
        guard let p = poses[faceID], p.wrists.count == 2, hand.points.count == 21 else { return (nil, nil) }
        let w0 = hand.points[0], hs = max(Self.handSize(hand.points), 1)
        let d = p.wrists.map { $0.visibility >= policy.poseVisible ? hypot(w0.x - $0.point.x, w0.y - $0.point.y) / hs : nil }
        let same = hand.handedness == "Left" ? d[0] : d[1]
        return (same, d.compactMap { $0 }.min())
    }

    /// The same-side distance (what the rule uses), for the log.
    func poseDistance(_ hand: LinkHand, _ faceID: Int) -> CGFloat? { poseDistances(hand, faceID).same }

    func reset() {
        owner.removeAll()
        people.removeAll()
        facing.removeAll()
        held.removeAll()
        addressing.removeAll()
        poseIgnored.removeAll()
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
        // body pose: the strongest evidence when it is there
        // (20 Nov: a hard "both wrists elsewhere = not theirs" at 2.5 hand sizes excluded the friend,
        // whose pose wrist was 2.6-3.3 away, and handed his hand to the only one left: the nearer person)
        let pd = poseDistances(hand, face.id)
        if let any = pd.any, any > pol.poseCap,
           (poses[face.id]?.wrists ?? []).allSatisfy({ $0.visibility >= pol.poseVisible }) {
            return nil  // both wrists seen, both far away
        }
        if let same = pd.same, same <= pol.poseMatch {
            // another pose claiming the same hand: only a clearly closer claim counts
            let rival = poses.keys.filter { $0 != face.id }.compactMap { poseDistances(hand, $0).same }.filter { $0 <= pol.poseMatch }.min()
            if rival.map({ same <= pol.poseClearlyCloser * $0 }) ?? true { c -= pol.poseBonus }
        } else if let d = [pd.same, pd.any.map { $0 + pol.handednessSlack }].compactMap({ $0 }).min() {
            c += pol.poseWeight * min(d, pol.poseCap)
        }
        return c
    }

    /// `bodyPoses`: per face id, its body pose (only when computed this frame).
    func update(faces: [LinkFace], hands: [LinkHand], now: TimeInterval, bodyPoses: [Int: LinkPose] = [:]) {
        newEvents.removeAll()
        let pol = policy
        let visible = faces.filter(\.visible)

        // a pose found from a crop around a face can latch onto another person in the crop
        poses = [:]
        for f in visible {
            guard let p = bodyPoses[f.id] else { continue }
            let off = hypot(p.nose.x - f.box.midX, p.nose.y - f.box.midY) / max(f.box.height, 1)
            if off <= pol.poseNoseReach {
                poses[f.id] = p
                poseIgnored.remove(f.id)
            } else if poseIgnored.insert(f.id).inserted {
                newEvents.append(String(format: "pose for face #%d ignored: its nose is %.1f face heights from the face (someone else's body)", f.id, off))
            }
        }

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
            let m = measure(h, f)  // printed so the rules can be tuned on real people
            var line = String(format: "hand #%d (%@) -> face #%d (reach %.1f, size %.2f", h.id, h.handedness, fid, m.reach, m.ratio)
            let claims = poses.keys.sorted().compactMap { id in poseDistance(h, id).map { String(format: "#%d %.1f", id, $0) } }
            if !claims.isEmpty { line += ", pose wrist hand sizes away: " + claims.joined(separator: ", ") }
            newEvents.append(line + ")")
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
