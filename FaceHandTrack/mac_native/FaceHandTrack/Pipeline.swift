import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import os

struct FaceTrackOutput: Identifiable {
    let id: Int
    let box: CGRect  // normalised [0,1] frame coordinates
    let mask: CGImage?  // low-res (n x n) alpha mask covering the whole frame
    let visible: Bool
    let quality: Float
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
}

struct PipelineStats {
    var fps: Double = 0
    var detectMs: Double = 0
    var encodeMs: Double = 0
    var perFaceMs: Double = 0
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
}

/// Owns the models and runs one frame at a time on a private queue. Frames that arrive
/// while a frame is in flight are dropped (latest-frame-wins), so latency never builds up.
///
///     face path (pipeline queue)                 hand path (hand queue, in parallel)
///     YuNet (GPU) || EdgeTAM encoder (ANE)       follow tracked hands from their landmarks
///     per face: memory attention, decoder,       palm detector for new hands (every 5th
///     memory encoder (ANE)                       frame, every frame if none); gestures
final class Pipeline {
    private let detector: FaceDetector
    private let tam: EdgeTAM
    private let faceManager = TrackManager()
    private let handTracker: HandTracker
    private var smoothers: [Int: GestureSmoother] = [:]
    private var handEventLog: [String] = []  // recent hand events, newest last (HUD)
    private let scaler = PixelBufferScaler()
    private let queue = DispatchQueue(label: "facehandtrack.pipeline", qos: .userInitiated)
    private let detectQueue = DispatchQueue(label: "facehandtrack.detect", qos: .userInitiated)
    private let handQueue = DispatchQueue(label: "facehandtrack.hands", qos: .userInitiated)
    private let busy = OSAllocatedUnfairLock(initialState: false)
    private let resetRequested = OSAllocatedUnfairLock(initialState: false)
    private var lastFrameTime: CFAbsoluteTime = 0
    private var stats = PipelineStats()
    private var frameIndex = 0
    private let start = CFAbsoluteTimeGetCurrent()

    var onResult: ((FrameOutput) -> Void)?
    var renderMasks = true
    var facesEnabled = true
    var handsEnabled = true
    /// Frames are mirrored like a selfie view (affects Left/Right only).
    var mirrored: Bool {
        get { handTracker.mirrored }
        set { handTracker.mirrored = newValue }
    }

    /// Compute units per model from SelfTest --benchmark on an M5 (logs/benchmark.json): every model is
    /// fastest on the Neural Engine except YuNet, which goes to the GPU so it runs alongside the EdgeTAM
    /// encoder instead of queueing behind it.
    init() throws {
        detector = try FaceDetector(computeUnits: .cpuAndGPU)
        tam = try EdgeTAM(computeUnits: .cpuAndNeuralEngine)
        handTracker = try HandTracker(palmUnits: .cpuAndNeuralEngine, landmarkUnits: .cpuAndNeuralEngine)
    }

    func reset() { resetRequested.withLock { $0 = true } }

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
            handTracker.reset()
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
        var detMs = 0.0, encMs = 0.0, faceMs = 0.0
        var ranDetector = false, encoded = false
        if facesEnabled {
            (faces, detMs, encMs, faceMs, ranDetector, encoded) = try facePath(pb, size: size, now: now)
        }
        handGroup.wait()
        if let handError { throw handError }
        frameIndex += 1

        let tEnd = CFAbsoluteTimeGetCurrent()
        let ema = { (old: Double, new: Double) in old == 0 ? new : 0.9 * old + 0.1 * new }
        if lastFrameTime > 0 { stats.fps = ema(stats.fps, 1 / (tEnd - lastFrameTime)) }
        lastFrameTime = tEnd
        if ranDetector { stats.detectMs = ema(stats.detectMs, detMs) }
        if encoded { stats.encodeMs = ema(stats.encodeMs, encMs) }
        if !faces.isEmpty { stats.perFaceMs = ema(stats.perFaceMs, faceMs / Double(faces.count)) }
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
        stats.events = (facesEnabled ? Array(faceManager.events.suffix(3)) : []) + handEventLog
        if frameIndex % 30 == 0 {  // periodic line for remote profiling
            print(String(format: "[FaceHandTrack] %.1f fps | faces %ld hands %ld | detect %.1f encode %.1f per-face %.1f palm %.1f landmarks %.1f total %.1f ms",
                         stats.fps, stats.faces, stats.hands, stats.detectMs, stats.encodeMs, stats.perFaceMs,
                         stats.palmMs, stats.landmarksMs, stats.totalMs)
                  + " | " + hands.map { "\($0.handedness) \($0.gesture.isEmpty ? "-" : $0.gesture)" }.joined(separator: ", "))
        }
        return FrameOutput(faces: faces, hands: hands, stats: stats, size: size)
    }

    // MARK: - face path (as in the iOS app)

    private func facePath(_ pb: CVPixelBuffer, size: CGSize, now: TimeInterval) throws
        -> ([FaceTrackOutput], Double, Double, Double, Bool, Bool) {
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
        return (outputs, detMs, encMs, faceMs, runDet, frame != nil)
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
                landmarks: t.result.landmarks))
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
