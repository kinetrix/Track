import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import simd

/// Hand detection, 21-point landmarks and hand tracking on Core ML.
/// Port of the Python engine (../onnx/facetrack/hands.py), same constants and policy:
/// MediaPipe palm detector + hand landmark model (Google, Apache-2.0, OpenCV Zoo files),
/// and, like MediaPipe, a tracked hand's next crop comes from its own landmarks, so the
/// palm detector only runs to find new hands.

/// Palm-detector-style hand region: box + 7 keypoints (wrist, index/middle/ring/pinky MCP,
/// thumb CMC, thumb MCP), all in frame pixels.
struct Palm {
    var x1: Double, y1: Double, x2: Double, y2: Double
    var kps: [SIMD2<Double>]
    var score: Float = 1
}

struct LandmarkResult {
    var landmarks: [SIMD3<Float>]  // frame pixels (x, y) + relative depth
    var world: [SIMD3<Float>]  // metric, hand-centred
    var rightProb: Float  // the model's "right hand" score for an un-mirrored image
    var presence: Float
}

func iou(_ a: (Double, Double, Double, Double), _ b: (Double, Double, Double, Double)) -> Double {
    let x1 = max(a.0, b.0), y1 = max(a.1, b.1), x2 = min(a.2, b.2), y2 = min(a.3, b.3)
    let inter = max(0, x2 - x1) * max(0, y2 - y1)
    let u = (a.2 - a.0) * (a.3 - a.1) + (b.2 - b.0) * (b.3 - b.1) - inter
    return u > 0 ? inter / u : 0
}

private func loadModel(_ name: String, units: MLComputeUnits) throws -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = units
    guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
        throw NSError(domain: "FaceHandTrack", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(name).mlmodelc missing from the app"])
    }
    return try MLModel(contentsOf: url, configuration: cfg)
}

// MARK: - palm detector

final class PalmDetector {
    static let size = 192
    private let model: MLModel
    private let anchors: [SIMD2<Double>]
    var minScore: Double = 0.5
    var nmsIoU: Double = 0.3

    init(units: MLComputeUnits) throws {
        model = try loadModel("PalmDetector", units: units)
        // SSD anchors: 24x24 grid with 2 per cell (stride 8), then 12x12 with 6 per cell (stride 16 x 3 layers)
        var a: [SIMD2<Double>] = []
        for (g, n) in [(24, 2), (12, 6)] {
            for y in 0..<g {
                for x in 0..<g {
                    for _ in 0..<n { a.append(SIMD2((Double(x) + 0.5) / Double(g), (Double(y) + 0.5) / Double(g))) }
                }
            }
        }
        anchors = a  // 2016
    }

    func detect(_ img: BGRAImage, sampler: ImageSampler) throws -> [Palm] {
        let S = Self.size
        let H = img.height, W = img.width
        let ratio = min(Double(S) / Double(H), Double(S) / Double(W))
        let nh = Int(Double(H) * ratio), nw = Int(Double(W) * ratio)
        let left = (S - nw) / 2, top = (S - nh) / 2
        // letterbox: canvas pixel (u, v) -> resized pixel (u - left, v - top) -> frame (bilinear, half-pixel)
        let sx = Double(W) / Double(nw), sy = Double(H) / Double(nh)
        let toSrc = Affine(a: sx, tx: (0.5 - Double(left)) * sx - 0.5, d: sy, ty: (0.5 - Double(top)) * sy - 0.5)
        let content = Region(map: Affine(), x0: Double(left) - 0.5, y0: Double(top) - 0.5,
                             x1: Double(left + nw) - 0.5, y1: Double(top + nh) - 0.5)
        let input = sampler.render(img, side: S, toSource: toSrc, regions: [content])

        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
        let reg = out.featureValue(for: "boxes")!.multiArrayValue!.floats2D(prefix: [0], rows: anchors.count, cols: 18)
        let logit = out.featureValue(for: "scores")!.multiArrayValue!.floats2D(prefix: [0], rows: anchors.count, cols: 1)

        let padX = Double(left) / ratio, padY = Double(top) / ratio  // letterbox offset in frame pixels
        let scale = Double(max(H, W)), s = Double(S)
        var cands: [Palm] = []
        for i in 0..<anchors.count {
            let score = 1 / (1 + exp(-Double(logit[i])))
            guard score >= minScore else { continue }
            let r = i * 18, anc = anchors[i]
            let cx = Double(reg[r]) / s + anc.x, cy = Double(reg[r + 1]) / s + anc.y
            let w = Double(reg[r + 2]) / s, h = Double(reg[r + 3]) / s
            var kps: [SIMD2<Double>] = []
            for k in 0..<7 {
                kps.append(SIMD2((Double(reg[r + 4 + 2 * k]) / s + anc.x) * scale - padX,
                                 (Double(reg[r + 5 + 2 * k]) / s + anc.y) * scale - padY))
            }
            cands.append(Palm(x1: (cx - w / 2) * scale - padX, y1: (cy - h / 2) * scale - padY,
                              x2: (cx + w / 2) * scale - padX, y2: (cy + h / 2) * scale - padY, kps: kps, score: Float(score)))
        }
        // greedy NMS
        var keep: [Palm] = []
        for c in cands.sorted(by: { $0.score > $1.score })
        where !keep.contains(where: { iou(($0.x1, $0.y1, $0.x2, $0.y2), (c.x1, c.y1, c.x2, c.y2)) > nmsIoU }) {
            keep.append(c)
        }
        return keep
    }
}

// MARK: - hand landmarks

final class HandLandmarker {
    static let size = 224
    // crop geometry from MediaPipe (via OpenCV Zoo)
    static let palmBoxPreEnlarge = 4.0
    static let palmBoxShift = SIMD2<Double>(0, -0.4)
    static let palmBoxEnlarge = 3.0
    static let handBoxShift = SIMD2<Double>(0, -0.1)
    static let handBoxEnlarge = 1.65

    private let model: MLModel

    init(units: MLComputeUnits) throws {
        model = try loadModel("HandLandmarks", units: units)
    }

    /// Integer crop box as in the Python `_crop_pad`: shift + enlarge, truncate, clip to the
    /// image, then pad to a square. Returns (box x0, y0, x1, y1), padded side, left, top.
    private static func cropPad(_ b0: SIMD2<Double>, _ b1: SIMD2<Double>, shift: SIMD2<Double>, enlarge: Double,
                                squareDiag: Bool, width: Int, height: Int) -> (Int, Int, Int, Int, Int, Int, Int) {
        let wh = b1 - b0
        let s0 = b0 + shift * wh, s1 = b1 + shift * wh
        let c = (s0 + s1) / 2
        let half = (s1 - s0) * enlarge / 2
        let x0 = min(max(Int(c.x - half.x), 0), width), y0 = min(max(Int(c.y - half.y), 0), height)
        let x1 = min(max(Int(c.x + half.x), 0), width), y1 = min(max(Int(c.y + half.y), 0), height)
        let w = x1 - x0, h = y1 - y0
        let side = squareDiag ? Int(Double(w * w + h * h).squareRoot()) : max(w, h)
        return (x0, y0, x1, y1, side, (side - w) / 2, (side - h) / 2)
    }

    func run(_ img: BGRAImage, palm: Palm, sampler: ImageSampler) throws -> LandmarkResult? {
        // 1) generous crop around the palm, padded to a square so rotation never cuts corners
        let c1 = Self.cropPad(SIMD2(palm.x1, palm.y1), SIMD2(palm.x2, palm.y2), shift: .zero,
                              enlarge: Self.palmBoxPreEnlarge, squareDiag: true, width: img.width, height: img.height)
        let (bx0, by0, bx1, by1, side1, left1, top1) = c1
        let w1 = bx1 - bx0, h1 = by1 - by0
        guard side1 >= 8, w1 > 0, h1 > 0 else { return nil }  // crop (almost) entirely outside the frame
        let bias = SIMD2(Double(bx0 - left1), Double(by0 - top1))  // crop1 -> frame offset
        let kps = palm.kps.map { $0 - bias }
        // 2) rotate so the wrist -> middle-finger-base direction points up
        let p1 = kps[0], p2 = kps[2]
        var rad = Double.pi / 2 - atan2(-(p2.y - p1.y), p2.x - p1.x)
        rad -= 2 * Double.pi * floor((rad + Double.pi) / (2 * Double.pi))
        let center = SIMD2(Double(bx0 + bx1) / 2, Double(by0 + by1) / 2) - bias
        // cv2.getRotationMatrix2D(center, angle, 1): [[a, b, (1-a)cx - b cy], [-b, a, b cx + (1-a) cy]]
        let ca = cos(rad), sb = sin(rad)
        let rot = Affine(a: ca, b: sb, tx: (1 - ca) * center.x - sb * center.y,
                         c: -sb, d: ca, ty: sb * center.x + (1 - ca) * center.y)
        let rk = kps.map { k -> SIMD2<Double> in let r = rot.apply(k.x, k.y); return SIMD2(r.0, r.1) }
        var mn = rk[0], mx = rk[0]
        for k in rk { mn = simd_min(mn, k); mx = simd_max(mx, k) }
        // 3) final crop from the rotated keypoints' box, resized to 224
        let c2 = Self.cropPad(mn, mx, shift: Self.palmBoxShift, enlarge: Self.palmBoxEnlarge, squareDiag: false,
                              width: side1, height: side1)
        let (rx0, ry0, rx1, ry1, side2, left2, top2) = c2
        guard side2 >= 8 else { return nil }

        // output pixel -> padded hand crop (resize) -> rotated image -> crop1 (inverse rotation) -> frame
        let S = Double(Self.size), s2 = Double(side2) / S
        let toHandCrop = Affine(a: s2, tx: 0.5 * s2 - 0.5, d: s2, ty: 0.5 * s2 - 0.5)
        let toRotated = Affine.translate(Double(rx0 - left2), Double(ry0 - top2)).after(toHandCrop)
        let det = rot.a * rot.d - rot.b * rot.c
        let invRot = Affine(a: rot.d / det, b: -rot.b / det, tx: (rot.b * rot.ty - rot.d * rot.tx) / det,
                            c: -rot.c / det, d: rot.a / det, ty: (rot.c * rot.tx - rot.a * rot.ty) / det)
        let toCrop1 = invRot.after(toRotated)
        let toFrame = Affine.translate(bias.x, bias.y).after(toCrop1)
        let regions = [
            Region(map: toRotated, x0: Double(rx0) - 0.5, y0: Double(ry0) - 0.5, x1: Double(rx1) - 0.5, y1: Double(ry1) - 0.5),
            Region(map: toCrop1, x0: Double(left1) - 0.5, y0: Double(top1) - 0.5,
                   x1: Double(left1 + w1) - 0.5, y1: Double(top1 + h1) - 0.5),
        ]
        let input = sampler.render(img, side: Self.size, toSource: toFrame, regions: regions,
                                   supersample: min(3, max(1, Int(s2.rounded(.up)))))

        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
        let lmRaw = out.featureValue(for: "landmarks")!.multiArrayValue!.floats2D(prefix: [], rows: 1, cols: 63)
        let wRaw = out.featureValue(for: "world")!.multiArrayValue!.floats2D(prefix: [], rows: 1, cols: 63)
        let presence = out.featureValue(for: "presence")!.multiArrayValue!.value([0, 0])
        let right = out.featureValue(for: "handedness")!.multiArrayValue!.value([0, 0])

        // 4) map landmarks back: crop -> rotated image -> frame
        let scale = Double(max(rx1 - rx0, ry1 - ry0)) / S
        let cRot = SIMD2(Double(rx0 + rx1) / 2, Double(ry0 + ry1) / 2)
        let back = invRot.apply(cRot.x, cRot.y)  // centre of the hand crop in crop1 coordinates
        var lms: [SIMD3<Float>] = [], world: [SIMD3<Float>] = []
        for i in 0..<21 {
            let x = (Double(lmRaw[3 * i]) - S / 2) * scale, y = (Double(lmRaw[3 * i + 1]) - S / 2) * scale
            let z = Double(lmRaw[3 * i + 2]) * scale
            // rotate back by the crop angle (row vector times cv2.getRotationMatrix2D((0,0), angle, 1)[:, :2])
            let xr = x * ca - y * sb, yr = x * sb + y * ca
            lms.append(SIMD3(Float(xr + back.0 + bias.x), Float(yr + back.1 + bias.y), Float(z)))
            let wx = Double(wRaw[3 * i]), wy = Double(wRaw[3 * i + 1])
            world.append(SIMD3(Float(wx * ca - wy * sb), Float(wx * sb + wy * ca), wRaw[3 * i + 2]))
        }
        return LandmarkResult(landmarks: lms, world: world, rightProb: right, presence: presence)
    }

    /// Box around the landmarks, shifted/enlarged like MediaPipe's hand rect (frame pixels).
    static func handBox(_ lm: [SIMD3<Float>]) -> (Double, Double, Double, Double) {
        var mn = SIMD2<Double>(Double(lm[0].x), Double(lm[0].y)), mx = mn
        for p in lm { let q = SIMD2(Double(p.x), Double(p.y)); mn = simd_min(mn, q); mx = simd_max(mx, q) }
        let wh = mx - mn
        let b0 = mn + handBoxShift * wh, b1 = mx + handBoxShift * wh
        let c = (b0 + b1) / 2, half = (b1 - b0) * handBoxEnlarge / 2
        return (c.x - half.x, c.y - half.y, c.x + half.x, c.y + half.y)
    }

    /// Palm-detector-style input (square box + 7 keypoints) from a hand's landmarks,
    /// so a tracked hand needs no palm detection.
    static func palmFromLandmarks(_ lm: [SIMD3<Float>]) -> Palm {
        let k = [0, 5, 9, 13, 17, 1, 2].map { SIMD2(Double(lm[$0].x), Double(lm[$0].y)) }
        var mn = k[0], mx = k[0]
        for p in k { mn = simd_min(mn, p); mx = simd_max(mx, p) }
        let c = (mn + mx) / 2, half = max(mx.x - mn.x, mx.y - mn.y) / 2
        return Palm(x1: c.x - half, y1: c.y - half, x2: c.x + half, y2: c.y + half, kps: k)
    }
}

// MARK: - tracker

struct HandPolicy {
    var maxHands = 2
    var palmInterval = 5  // palm detector every N frames while hands are tracked (every frame when none)
    var minPresence: Float = 0.6  // landmark model's hand-presence score to keep a hand
    var minPresenceNew: Float = 0.8  // ... to start a new hand
    var overlapIoU = 0.3  // a detected palm overlapping a tracked hand's palm this much is that hand
    var duplicateIoU = 0.5  // two tracked hands overlapping this much -> drop the younger
}

final class HandTrack {
    let id: Int
    var result: LandmarkResult
    var age = 0

    init(id: Int, result: LandmarkResult) {
        self.id = id
        self.result = result
    }
}

final class HandTracker {
    var policy = HandPolicy()
    /// Frames are mirrored like a selfie view (affects Left/Right only).
    var mirrored = true
    private(set) var tracks: [HandTrack] = []
    private var nextID = 1
    private let palm: PalmDetector
    private let landmarker: HandLandmarker
    private let sampler = ImageSampler()
    /// Milliseconds of this frame per stage ("palm", "landmarks").
    private(set) var timings: [String: Double] = [:]

    init(palmUnits: MLComputeUnits = .cpuAndNeuralEngine, landmarkUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        palm = try PalmDetector(units: palmUnits)
        landmarker = try HandLandmarker(units: landmarkUnits)
    }

    func reset() {
        tracks.removeAll()
        nextID = 1
    }

    private func timed<T>(_ key: String, _ body: () throws -> T) rethrows -> T {
        let s = CFAbsoluteTimeGetCurrent()
        let r = try body()
        timings[key, default: 0] += (CFAbsoluteTimeGetCurrent() - s) * 1000
        return r
    }

    // single-image access for the self-test
    func detectPalms(_ pb: CVPixelBuffer) throws -> [Palm] {
        try withBGRA(pb) { try palm.detect($0, sampler: sampler) }
    }

    func landmarks(_ pb: CVPixelBuffer, palm p: Palm) throws -> LandmarkResult? {
        try withBGRA(pb) { try landmarker.run($0, palm: p, sampler: sampler) }
    }

    func update(_ pb: CVPixelBuffer, frameIndex: Int, events: inout [String]) throws {
        timings = [:]
        let pol = policy
        try withBGRA(pb) { img in
            // 1) follow every tracked hand from its previous landmarks
            var alive: [HandTrack] = []
            for t in tracks {
                let r = try timed("landmarks") {
                    try landmarker.run(img, palm: HandLandmarker.palmFromLandmarks(t.result.landmarks), sampler: sampler)
                }
                guard let r, r.presence >= pol.minPresence else {
                    events.append("hand drop #\(t.id) lost (" + (r == nil ? "crop outside frame" : String(format: "presence %.2f", r!.presence)) + ")")
                    continue
                }
                t.result = r
                t.age += 1
                alive.append(t)
            }
            tracks = alive

            // 2) look for new hands
            if tracks.count < pol.maxHands && (tracks.isEmpty || frameIndex % pol.palmInterval == 0) {
                let palms = try timed("palm") { try palm.detect(img, sampler: sampler) }
                for p in palms {
                    if tracks.count >= pol.maxHands { break }
                    if tracks.contains(where: { sameHand(p, $0) }) { continue }
                    let r = try timed("landmarks") { try landmarker.run(img, palm: p, sampler: sampler) }
                    guard let r, r.presence >= pol.minPresenceNew else { continue }
                    let t = HandTrack(id: nextID, result: r)
                    nextID += 1
                    tracks.append(t)
                    events.append("hand new #\(t.id)")
                }
            }
        }

        // 3) two tracks on the same hand -> keep the older one
        var keep: [HandTrack] = []
        for t in tracks.sorted(by: { $0.age > $1.age }) {
            let b = HandLandmarker.handBox(t.result.landmarks)
            if keep.contains(where: { iou(b, HandLandmarker.handBox($0.result.landmarks)) > pol.duplicateIoU }) {
                events.append("hand drop #\(t.id) duplicate")
                continue
            }
            keep.append(t)
        }
        tracks = keep.sorted { $0.id < $1.id }
    }

    /// A detected palm belongs to a tracked hand if its centre lies inside that hand's box,
    /// or the two palm boxes overlap.
    private func sameHand(_ p: Palm, _ t: HandTrack) -> Bool {
        let hb = HandLandmarker.handBox(t.result.landmarks)
        let cx = (p.x1 + p.x2) / 2, cy = (p.y1 + p.y2) / 2
        if hb.0 <= cx && cx <= hb.2 && hb.1 <= cy && cy <= hb.3 { return true }
        let q = HandLandmarker.palmFromLandmarks(t.result.landmarks)
        return iou((p.x1, p.y1, p.x2, p.y2), (q.x1, q.y1, q.x2, q.y2)) > policy.overlapIoU
    }

    /// The person's real hand. The model's score is "right hand" for an ordinary,
    /// un-mirrored image; a mirrored (selfie) image swaps left and right, so flip it back.
    func handedness(_ t: HandTrack) -> String {
        var right = t.result.rightProb > 0.5
        if mirrored { right.toggle() }
        return right ? "Right" : "Left"
    }
}
