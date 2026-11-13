import CoreGraphics
import Foundation

/// Help requests for a caregiver, from the hand points the app already has (no model).
///
///   Signal for Help   palm to the camera with the thumb tucked in (index..pinky up, thumb folded),
///                     held a moment, then the fingers fold down over the thumb (a fist), within
///                     about a second: the one-handed signal promoted by the Canadian Women's
///                     Foundation, done once.
///   Both hands up     both of one person's hands open and raised above their chin, held 2 s.
///
/// A request raises an alert naming the person (face id and saved name) that stays until it is
/// acknowledged; the same person can't raise another for a short cooldown.
struct HelpPolicy {
    var tuckedHold: TimeInterval = 0.4  // palm-with-thumb-tucked shown at least this long ...
    var foldWithin: TimeInterval = 1.0  // ... then a fist within this long after it ends
    var tuckedGap: TimeInterval = 0.3  // a break this long in the tucked palm restarts its hold
    var handsUpHold: TimeInterval = 2.0  // both hands open and raised this long
    var cooldown: TimeInterval = 10  // seconds before the same person can raise another alert
}

struct HelpAlert: Identifiable {
    let id: Int
    let faceID: Int?  // nil: the hand belonged to no visible face
    let name: String?
    let reason: String  // "Signal for Help" / "Both hands up"
    let time: TimeInterval  // seconds since start
    let clock: Date

    var who: String {
        guard let f = faceID else { return "someone (face not seen)" }
        return name.map { "\($0) (#\(f))" } ?? "face #\(f)"
    }
}

struct HelpHand {
    let id: Int
    let owner: Int?  // face id
    let extended: [Bool]  // thumb, index, middle, ring, pinky
    let gesture: String  // stable gesture
    let center: CGPoint  // frame pixels
}

struct HelpFace {
    let id: Int
    let box: CGRect  // frame pixels
    let name: String?
}

final class HelpDetector {
    var policy = HelpPolicy()
    /// Unacknowledged alerts, oldest first.
    private(set) var alerts: [HelpAlert] = []
    private(set) var newAlerts: [HelpAlert] = []
    private var nextID = 1
    private var tucked: [Int: (since: TimeInterval, last: TimeInterval)] = [:]  // per hand id
    private var handsUp: [Int: TimeInterval] = [:]  // per face id: since
    private var lastAlert: [String: TimeInterval] = [:]  // per person key: time

    func reset() {
        tucked.removeAll()
        handsUp.removeAll()
        lastAlert.removeAll()
    }

    func acknowledgeAll() { alerts.removeAll() }

    private func raise(_ reason: String, face: HelpFace?, faceID: Int?, handID: Int, now: TimeInterval) {
        let key = faceID.map { "f\($0)" } ?? "h\(handID)"
        if let t = lastAlert[key], now - t < policy.cooldown { return }
        lastAlert[key] = now
        let a = HelpAlert(id: nextID, faceID: faceID, name: face?.name, reason: reason, time: now, clock: Date())
        nextID += 1
        alerts.append(a)
        newAlerts.append(a)
    }

    func update(faces: [HelpFace], hands: [HelpHand], now: TimeInterval) {
        newAlerts.removeAll()
        let pol = policy
        let faceByID = Dictionary(uniqueKeysWithValues: faces.map { ($0.id, $0) })
        let live = Set(hands.map(\.id))
        tucked = tucked.filter { live.contains($0.key) }

        // Signal for Help, per hand
        for h in hands where h.extended.count == 5 {
            let e = h.extended
            let isTucked = !e[0] && e[1] && e[2] && e[3] && e[4]
            let isClosed = !e[1] && !e[2] && !e[3] && !e[4]
            if isTucked {
                if let t = tucked[h.id], now - t.last <= pol.tuckedGap {
                    tucked[h.id] = (t.since, now)
                } else {
                    tucked[h.id] = (now, now)
                }
            } else if let t = tucked[h.id] {
                if isClosed && t.last - t.since >= pol.tuckedHold && now - t.last <= pol.foldWithin {
                    raise("Signal for Help", face: h.owner.flatMap { faceByID[$0] }, faceID: h.owner, handID: h.id, now: now)
                    tucked[h.id] = nil
                } else if now - t.last > pol.foldWithin {
                    tucked[h.id] = nil
                }
            }
        }

        // Both hands up, per person
        var upNow = Set<Int>()
        for f in faces {
            let mine = hands.filter { $0.owner == f.id }
            let raised = mine.filter { $0.gesture == "Open palm" && $0.center.y < f.box.maxY }
            guard mine.count >= 2, raised.count >= 2 else { continue }
            upNow.insert(f.id)
            let since = handsUp[f.id] ?? now
            handsUp[f.id] = since
            if now - since >= pol.handsUpHold {
                raise("Both hands up", face: f, faceID: f.id, handID: raised[0].id, now: now)
                handsUp[f.id] = nil
                upNow.remove(f.id)
            }
        }
        handsUp = handsUp.filter { upNow.contains($0.key) }
    }
}
