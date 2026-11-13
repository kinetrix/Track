import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import os

struct FaceTrackOutput: Identifiable {
    let id: Int
    let box: CGRect  // normalised [0,1] frame coordinates
    let mask: CGImage?  // low-res (n x n) alpha mask covering the whole frame (EdgeTAM only)
    let visible: Bool
    let quality: Float  // detector score (fast tracker) or EdgeTAM's mask-quality estimate
    var name: String? = nil  // saved name this face matches (fast tracker)
    var facing: Bool? = nil  // turned to the camera (nil: not known yet)
    var addressing: String? = nil  // gesture this person is showing the camera
    var waving = false  // just waved to call the robot
}

struct HandTrackOutput: Identifiable {
    let id: Int
    let handedness: String  // the person's real hand
    let gesture: String  // stable (smoothed) label, "" = none
    let rawGesture: String  // this frame's label
    let fingers: Int
    let score: Float
    let box: CGRect  // normalised
    let points: [CGPoint]  // 21 landmarks, normalised
    let landmarks: [SIMD3<Float>]  // 21 landmarks, frame pixels (for the self-test)
    var owner: Int? = nil  // face id of the person this hand belongs to
    var extended: [Bool] = []  // thumb, index, middle, ring, pinky (for the help signal)
}

enum FaceMode: String, CaseIterable, Identifiable {
    case fast = "Fast (YuNet + SFace)"
    case edgetam = "EdgeTAM (masks)"
    var id: String { rawValue }
}

struct PipelineStats {
    var fps: Double = 0
    var detectMs: Double = 0
    var encodeMs: Double = 0
    var perFaceMs: Double = 0
    var embedMs: Double = 0  // SFace fingerprints this frame (fast tracker)
    var palmMs: Double = 0
    var landmarksMs: Double = 0
    var totalMs: Double = 0  // moving average
    var frameMs: Double = 0  // this frame
    var faces = 0
    var hands = 0
    var events: [String] = []
}

struct FrameOutput {
    var faces: [FaceTrackOutput]
    var hands: [HandTrackOutput]
    var stats: PipelineStats
    var size: CGSize
    var alerts: [HelpAlert] = []  // unacknowledged help requests
    var attention: AttentionTarget? = nil  // who the robot should turn to, and how far
}

/// Owns the models and runs one frame at a time on a private queue. Frames that arrive
/// while a frame is in flight are dropped (latest-frame-wins), so latency never builds up.
///
///     face path (pipeline queue)                 hand path (hand queue, in parallel)
///     fast (default): YuNet every frame ->       follow tracked hands from their landmarks
///       face tracks, SFace fingerprints for      palm detector for new hands (every 5th
///       returning faces and saved names          frame, every frame if none); gestures
///     edgetam: YuNet (GPU) || EdgeTAM encoder,
///       per face: memory attention, decoder, memory encoder (ANE)
///
/// Then the person linker joins each hand to its face and finds who is addressing the camera.
final class Pipeline {
    private let detector: FaceDetector
    private let tam: EdgeTAM
    private let faceManager = TrackManager()
    private let fastFaces = FastFaceTracker()
    private let embedder: FaceEmbedder
    private let linker = PersonLinker()
    private let help = HelpDetector()
    private let attention = AttentionTracker()
    private let handTracker: HandTracker
    private var smoothers: [Int: GestureSmoother] = [:]
    private var handEventLog: [String] = []  // recent hand events, newest last (HUD)
    private let scaler = PixelBufferScaler()
    private let queue = DispatchQueue(label: "facehandtrack.pipeline", qos: .userInitiated)
    private let detectQueue = DispatchQueue(label: "facehandtrack.detect", qos: .userInitiated)
    private let handQueue = DispatchQueue(label: "facehandtrack.hands", qos: .userInitiated)
    private let busy = OSAllocatedUnfairLock(initialState: false)
    private let resetRequested = OSAllocatedUnfairLock(initialState: false)
    private let modeLock = OSAllocatedUnfairLock(initialState: FaceMode.fast)
    private var lastEvents: [String] = []  // recent face / person events, newest last (HUD)
    private var lastTAMEvent: String?
    private var lastFrameTime: CFAbsoluteTime = 0
    private var stats = PipelineStats()
    private var frameIndex = 0
    private let start = CFAbsoluteTimeGetCurrent()

    var onResult: ((FrameOutput) -> Void)?
    var renderMasks = true
    var facesEnabled = true
    var handsEnabled = true
    /// Face tracker; switching forgets all tracks.
    var faceMode: FaceMode {
        get { modeLock.withLock { $0 } }
        set {
            modeLock.withLock { $0 = newValue }
            reset()
        }
    }
    /// Frames are mirrored like a selfie view (affects Left/Right and which side of a face a hand is on).
    var mirrored: Bool {
        get { handTracker.mirrored }
        set {
            handTracker.mirrored = newValue
            linker.mirrored = newValue
            attention.mirrored = newValue
        }
    }

    /// Compute units per model from SelfTest --benchmark on an M5 (logs/benchmark.json): every model is
    /// fastest on the Neural Engine except YuNet, which goes to the GPU so it runs alongside the EdgeTAM
    /// encoder instead of queueing behind it.
    init() throws {
        detector = try FaceDetector(computeUnits: .cpuAndGPU)
        tam = try EdgeTAM(computeUnits: .cpuAndNeuralEngine)
        handTracker = try HandTracker(palmUnits: .cpuAndNeuralEngine, landmarkUnits: .cpuAndNeuralEngine)
        embedder = try FaceEmbedder(computeUnits: .cpuAndNeuralEngine)
        handTracker.policy.maxHands = 2 * fastFaces.policy.maxTracks  // up to 2 hands per person
        fastFaces.known = KnownFaces.load()
    }

    // MARK: - saved names

    /// Saves the fingerprint of the person addressing the camera (else the largest visible face)
    /// under `name`. `done` gets a status line, on the main queue.
    func saveFace(name: String, done: @escaping (String) -> Void) {
        queue.async { [self] in
            let msg: String
            let n = name.trimmingCharacters(in: .whitespaces)
            let visible = fastFaces.tracks.filter { $0.visible && $0.emb != nil }
            let speaker = visible.first { linker.people[$0.id]?.addressing != nil }
            if faceMode != .fast {
                msg = "Names need the Fast face tracker."
            } else if n.isEmpty {
                msg = "Type a name first."
            } else if let t = speaker ?? visible.max(by: { $0.box.width < $1.box.width }), let e = t.emb {
                fastFaces.known.append(KnownFace(name: n, embedding: e))
                KnownFaces.save(fastFaces.known)
                t.name = n
                fastFaces.applyNames()
                msg = "Saved face #\(t.id) as \(n) (\(fastFaces.known.filter { $0.name == n }.count) view(s))."
            } else {
                msg = "No face close enough to save (needs a face at least \(Int(fastFaces.policy.minEmbedPx)) px tall)."
            }
            print("[names] " + msg)
            DispatchQueue.main.async { done(msg) }
        }
    }

    func forgetNames(done: @escaping (String) -> Void) {
        queue.async { [self] in
            fastFaces.known.removeAll()
            KnownFaces.save([])
            for t in fastFaces.tracks { t.name = nil }
            DispatchQueue.main.async { done("Forgot all saved names.") }
        }
    }

    func reset() { resetRequested.withLock { $0 = true } }

    /// The caregiver has seen the help requests: clear them.
    func acknowledgeAlerts() { queue.async { [self] in help.acknowledgeAll() } }

    /// Called from the camera queue. Returns immediately.
    func submit(_ pixelBuffer: CVPixelBuffer) {
        let accepted = busy.withLock { b -> Bool in
            if b { return false }
            b = true
            return true
        }
        guard accepted else { return }
        queue.async { [weak self] in
            guard let self else { return }
            defer { self.busy.withLock { $0 = false } }
            do {
                let out = try self.process(pixelBuffer, now: CFAbsoluteTimeGetCurrent() - self.start)
                DispatchQueue.main.async { self.onResult?(out) }
            } catch {
                print("pipeline error:", error)
            }
        }
    }

    /// One frame, synchronously (self-test). `now` in seconds: track timeouts are in seconds.
    func process(_ pb: CVPixelBuffer, now: TimeInterval) throws -> FrameOutput {
        if resetRequested.withLock({ let r = $0; $0 = false; return r }) {
            faceManager.reset()
            fastFaces.reset()
            linker.reset()
            help.reset()
            attention.reset()
            handTracker.reset()
            lastEvents.removeAll()
            smoothers.removeAll()
            handEventLog.removeAll()
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))

        // hand path, in parallel with the face path
        var hands: [HandTrackOutput] = []
        var handEvents: [String] = []
        var handError: Error?
        let handGroup = DispatchGroup()
        if handsEnabled {
            handQueue.async(group: handGroup) {
                do { hands = try self.handPath(pb, size: size, events: &handEvents) } catch { handError = error }
            }
        }

        var faces: [FaceTrackOutput] = []
        var faceLandmarks: [Int: [CGPoint]] = [:]  // per face id, frame pixels (when known this frame)
        var faceEvents: [String] = []
        var detMs = 0.0, encMs = 0.0, faceMs = 0.0, embMs = 0.0
        var ranDetector = false, encoded = false
        let mode = faceMode
        if facesEnabled && mode == .fast {
            (faces, faceLandmarks, detMs, embMs) = try fastFacePath(pb, size: size, now: now)
            faceEvents = fastFaces.newEvents
            ranDetector = true
        } else if facesEnabled {
            (faces, faceLandmarks, detMs, encMs, faceMs, ranDetector, encoded) = try facePath(pb, size: size, now: now)
            if let e = faceManager.events.last, e != lastTAMEvent {
                faceEvents = [e]
                lastTAMEvent = e
            }
        }
        handGroup.wait()
        if let handError { throw handError }
        frameIndex += 1

        // who each hand belongs to, and who is addressing the camera
        let W = size.width, H = size.height
        var linkFaces: [LinkFace] = []
        for f in faces {
            let box = CGRect(x: f.box.minX * W, y: f.box.minY * H, width: f.box.width * W, height: f.box.height * H)
            linkFaces.append(LinkFace(id: f.id, box: box, landmarks: faceLandmarks[f.id] ?? [], visible: f.visible))
        }
        var linkHands: [LinkHand] = []
        for h in hands {
            let pts = h.landmarks.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            linkHands.append(LinkHand(id: h.id, handedness: h.handedness, points: pts, gesture: h.gesture))
        }
        linker.update(faces: linkFaces, hands: linkHands, now: now)
        for i in hands.indices { hands[i].owner = linker.owner[hands[i].id] }
        for i in faces.indices {
            let st = linker.people[faces[i].id]
            faces[i].facing = st?.facing
            faces[i].addressing = st?.addressing
        }
        // help requests (Signal for Help, both hands up)
        var helpHands: [HelpHand] = []
        for h in hands {
            let n = CGFloat(max(h.landmarks.count, 1))
            let c = CGPoint(x: CGFloat(h.landmarks.reduce(0) { $0 + $1.x }) / n, y: CGFloat(h.landmarks.reduce(0) { $0 + $1.y }) / n)
            let pts = h.landmarks.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            helpHands.append(HelpHand(id: h.id, owner: h.owner, extended: h.extended, gesture: h.gesture, center: c,
                                      wrist: pts.first ?? c, size: PersonLinker.handSize(pts)))
        }
        var helpFaces: [HelpFace] = []
        for (f, lf) in zip(faces, linkFaces) where f.visible { helpFaces.append(HelpFace(id: f.id, box: lf.box, name: f.name)) }
        help.update(faces: helpFaces, hands: helpHands, now: now)
        var helpEvents: [String] = []
        for a in help.newAlerts { helpEvents.append("HELP REQUESTED: \(a.reason) from \(a.who)") }

        // waving to call the robot, and where it should turn
        var attFaces: [AttentionFace] = []
        for (f, lf) in zip(faces, linkFaces) where f.visible {
            attFaces.append(AttentionFace(id: f.id, box: lf.box, addressing: f.addressing != nil))
        }
        var attHands: [AttentionHand] = []
        for (h, lh) in zip(hands, linkHands) { attHands.append(AttentionHand(id: h.id, owner: h.owner, handedness: h.handedness, extended: h.extended, points: lh.points)) }
        attention.update(faces: attFaces, hands: attHands, frameSize: size, now: now)
        for i in faces.indices { faces[i].waving = attention.wavingUntil[faces[i].id] != nil }

        for e in faceEvents + linker.newEvents + helpEvents + attention.newEvents {
            print(String(format: "[%7.2fs] ", now) + e)
            lastEvents.append(e)
        }
        if lastEvents.count > 4 { lastEvents.removeFirst(lastEvents.count - 4) }

        let tEnd = CFAbsoluteTimeGetCurrent()
        let ema = { (old: Double, new: Double) in old == 0 ? new : 0.9 * old + 0.1 * new }
        if lastFrameTime > 0 { stats.fps = ema(stats.fps, 1 / (tEnd - lastFrameTime)) }
        lastFrameTime = tEnd
        if ranDetector { stats.detectMs = ema(stats.detectMs, detMs) }
        if encoded { stats.encodeMs = ema(stats.encodeMs, encMs) }
        if !faces.isEmpty && mode == .edgetam { stats.perFaceMs = ema(stats.perFaceMs, faceMs / Double(faces.count)) }
        if mode == .fast { stats.embedMs = ema(stats.embedMs, embMs) }
        let ht = handTracker.timings
        if let p = ht["palm"] { stats.palmMs = ema(stats.palmMs, p) }
        if let l = ht["landmarks"] { stats.landmarksMs = ema(stats.landmarksMs, l) }
        stats.frameMs = (tEnd - t0) * 1000
        stats.totalMs = ema(stats.totalMs, stats.frameMs)
        stats.faces = faces.count
        stats.hands = hands.count
        for e in handEvents {
            print(String(format: "[%7.2fs] ", now) + e)
            handEventLog.append("f\(frameIndex - 1) " + e)
        }
        if handEventLog.count > 3 { handEventLog.removeFirst(handEventLog.count - 3) }
        stats.events = Array(lastEvents.suffix(3)) + handEventLog
        if frameIndex % 30 == 0 {  // periodic line for remote profiling
            let head = String(format: "[FaceHandTrack] %.1f fps | faces %ld hands %ld | detect %.1f embed %.1f encode %.1f per-face %.1f palm %.1f landmarks %.1f total %.1f ms",
                              stats.fps, stats.faces, stats.hands, stats.detectMs, stats.embedMs, stats.encodeMs, stats.perFaceMs,
                              stats.palmMs, stats.landmarksMs, stats.totalMs)
            var handText: [String] = []
            for h in hands {
                let who = h.owner.map { "#\($0) " } ?? "? "
                handText.append(who + h.handedness + " " + (h.gesture.isEmpty ? "-" : h.gesture))
            }
            var talk = ""
            for f in faces { if let g = f.addressing { talk += " | #\(f.id) addressing: " + g } }
            let att = attention.target.map { " | robot -> #\($0.faceID) (\($0.reason)): \($0.turnText)" } ?? ""
            print(head + " | " + handText.joined(separator: ", ") + talk + att + (mode == .fast ? fastFaces.summary() : ""))
        }
        return FrameOutput(faces: faces, hands: hands, stats: stats, size: size, alerts: help.alerts, attention: attention.target)
    }

    // MARK: - fast face path (as in ../jetson)

    private func fastFacePath(_ pb: CVPixelBuffer, size: CGSize, now: TimeInterval) throws
        -> ([FaceTrackOutput], [Int: [CGPoint]], Double, Double) {
        let fp = fastFaces.policy
        var detPolicy = TrackPolicy()  // the detector's thresholds
        detPolicy.detScore = fp.keepScore
        detPolicy.nmsIoU = fp.nmsIoU
        detPolicy.minFacePx = fp.minFacePx
        let det = scaler.letterbox(pb, width: detector.inputWidth, height: detector.inputHeight)
        let s = CFAbsoluteTimeGetCurrent()
        let dets = try detector.detect(det.0, scale: det.1, policy: detPolicy)
        let detMs = (CFAbsoluteTimeGetCurrent() - s) * 1000
        var embMs = 0.0
        withBGRA(pb) { img in
            fastFaces.update(dets, now: now) { d in
                let t = CFAbsoluteTimeGetCurrent()
                defer { embMs += (CFAbsoluteTimeGetCurrent() - t) * 1000 }
                return try? embedder.embed(img, landmarks: d.landmarks)
            }
        }
        var landmarks: [Int: [CGPoint]] = [:]
        let outputs = fastFaces.tracks.map { t -> FaceTrackOutput in
            if t.visible { landmarks[t.id] = t.landmarks }
            return FaceTrackOutput(id: t.id,
                                   box: CGRect(x: t.box.minX / size.width, y: t.box.minY / size.height,
                                               width: t.box.width / size.width, height: t.box.height / size.height),
                                   mask: nil, visible: t.visible, quality: t.score, name: t.name)
        }
        return (outputs, landmarks, detMs, embMs)
    }

    // MARK: - EdgeTAM face path (as in the iOS app)

    private func facePath(_ pb: CVPixelBuffer, size: CGSize, now: TimeInterval) throws
        -> ([FaceTrackOutput], [Int: [CGPoint]], Double, Double, Double, Bool, Bool) {
        let runDet = faceManager.shouldDetect()
        let hasTracks = !faceManager.tracks.isEmpty
        let policy = faceManager.policy

        // Pre-scale both inputs serially (cheap, vImage) ...
        let det = runDet ? scaler.letterbox(pb, width: detector.inputWidth, height: detector.inputHeight) : nil
        let imgTAM = hasTracks ? scaler.squash(pb, size: tam.cfg.image_size) : nil

        // ... then run the detector (GPU) and the EdgeTAM encoder (Neural Engine) concurrently.
        var dets: [FaceDetection] = []
        var frame: EdgeTAM.Frame?
        var detMs = 0.0, encMs = 0.0
        let group = DispatchGroup()
        if let det {
            detectQueue.async(group: group) { [detector] in
                let s = CFAbsoluteTimeGetCurrent()
                dets = (try? detector.detect(det.0, scale: det.1, policy: policy)) ?? []
                detMs = (CFAbsoluteTimeGetCurrent() - s) * 1000
            }
        }
        if let imgTAM {
            let s = CFAbsoluteTimeGetCurrent()
            frame = try tam.encode(imgTAM)
            encMs = (CFAbsoluteTimeGetCurrent() - s) * 1000
        }
        group.wait()

        // No tracks yet: only pay for the EdgeTAM encoder when the detector found a face.
        if frame == nil && !dets.isEmpty {
            let s = CFAbsoluteTimeGetCurrent()
            frame = try tam.encode(scaler.squash(pb, size: tam.cfg.image_size))
            encMs = (CFAbsoluteTimeGetCurrent() - s) * 1000
        }

        let s = CFAbsoluteTimeGetCurrent()
        try faceManager.update(detections: dets, frame: frame, tam: tam, frameSize: size, now: now)
        let faceMs = (CFAbsoluteTimeGetCurrent() - s) * 1000

        let outputs = faceManager.tracks.map { t in
            FaceTrackOutput(id: t.id,
                            box: CGRect(x: t.box.minX / size.width, y: t.box.minY / size.height,
                                        width: t.box.width / size.width, height: t.box.height / size.height),
                            mask: renderMasks && t.visible ? MaskRenderer.image(t.mask, colorIndex: t.id) : nil,
                            visible: t.visible, quality: t.iouPred)
        }
        // face points for the "facing the camera" check: from the detection that matches each track
        var landmarks: [Int: [CGPoint]] = [:]
        for t in faceManager.tracks where t.visible {
            if let d = dets.max(by: { iou($0.box, t.box) < iou($1.box, t.box) }), iou(d.box, t.box) > 0.3 {
                landmarks[t.id] = d.landmarks
            }
        }
        return (outputs, landmarks, detMs, encMs, faceMs, runDet, frame != nil)
    }

    // MARK: - hand path

    private func handPath(_ pb: CVPixelBuffer, size: CGSize, events: inout [String]) throws -> [HandTrackOutput] {
        try handTracker.update(pb, frameIndex: frameIndex, events: &events)
        var out: [HandTrackOutput] = []
        for t in handTracker.tracks {
            let (raw, fingers) = Gestures.classify(landmarks: t.result.landmarks, world: t.result.world)
            let sm = smoothers[t.id] ?? GestureSmoother()
            smoothers[t.id] = sm
            let stable = sm.update(raw)
            let b = HandLandmarker.handBox(t.result.landmarks)
            let w = Double(size.width), h = Double(size.height)
            out.append(HandTrackOutput(
                id: t.id, handedness: handTracker.handedness(t), gesture: stable, rawGesture: raw, fingers: fingers,
                score: t.result.presence,
                box: CGRect(x: b.0 / w, y: b.1 / h, width: (b.2 - b.0) / w, height: (b.3 - b.1) / h),
                points: t.result.landmarks.map { CGPoint(x: Double($0.x) / w, y: Double($0.y) / h) },
                landmarks: t.result.landmarks, extended: Gestures.fingerStates(t.result.world)))
        }
        let live = Set(handTracker.tracks.map(\.id))
        smoothers = smoothers.filter { live.contains($0.key) }
        return out
    }
}

/// n x n fp16 logits -> tinted RGBA CGImage (premultiplied), drawn stretched over the frame.
enum MaskRenderer {
    static let palette: [(UInt8, UInt8, UInt8)] = [(66, 135, 245), (80, 200, 120), (245, 170, 66), (200, 80, 200), (240, 90, 90)]

    static func image(_ logits: [Float16], colorIndex: Int) -> CGImage? {
        let (r, g, b) = palette[colorIndex % palette.count]
        let a: UInt8 = 110
        let pr = UInt8(Int(r) * Int(a) / 255), pg = UInt8(Int(g) * Int(a) / 255), pbv = UInt8(Int(b) * Int(a) / 255)
        let n = maskSide(logits)
        var px = [UInt8](repeating: 0, count: n * n * 4)
        for i in 0..<(n * n) where logits[i] > 0 {
            px[4 * i] = pr; px[4 * i + 1] = pg; px[4 * i + 2] = pbv; px[4 * i + 3] = a
        }
        guard let provider = CGDataProvider(data: Data(px) as CFData) else { return nil }
        return CGImage(width: n, height: n, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: n * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

/// Saved names (face fingerprints), in ~/Library/Application Support/FaceHandTrack/known_faces.json.
enum KnownFaces {
    static var url: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FaceHandTrack", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("known_faces.json")
    }

    static func load() -> [KnownFace] {
        guard let data = try? Data(contentsOf: url), let k = try? JSONDecoder().decode([KnownFace].self, from: data) else { return [] }
        print("[names] loaded \(k.count) saved face(s): " + Set(k.map(\.name)).sorted().joined(separator: ", "))
        return k
    }

    static func save(_ k: [KnownFace]) {
        if let data = try? JSONEncoder().encode(k) { try? data.write(to: url) }
    }
}
