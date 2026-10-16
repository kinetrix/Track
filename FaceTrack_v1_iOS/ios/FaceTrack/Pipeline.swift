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

struct PipelineStats {
    var fps: Double = 0
    var detectMs: Double = 0
    var encodeMs: Double = 0
    var perFaceMs: Double = 0
    var totalMs: Double = 0
    var faces = 0
    var events: [String] = []
}

/// Owns the models and runs one frame at a time on a private queue. Frames that
/// arrive while a frame is in flight are dropped (latest-frame-wins), so latency
/// never accumulates.
final class Pipeline {
    private let detector: FaceDetector
    private let tam: EdgeTAM
    private let manager = TrackManager()
    private let scaler = PixelBufferScaler()
    private let queue = DispatchQueue(label: "facetrack.pipeline", qos: .userInitiated)
    private let detectQueue = DispatchQueue(label: "facetrack.detect", qos: .userInitiated)
    private let busy = OSAllocatedUnfairLock(initialState: false)
    private let resetRequested = OSAllocatedUnfairLock(initialState: false)
    private var lastFrameTime: CFAbsoluteTime = 0
    private var stats = PipelineStats()
    private var statsFrame = 0

    var onResult: (([FaceTrackOutput], PipelineStats, CGSize) -> Void)?
    var renderMasks = true

    init() throws {
        detector = try FaceDetector()
        tam = try EdgeTAM()
        ComputePlanReport.printAll()  // one-off: which chip each model's ops are placed on
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
            do { try self.process(pixelBuffer) } catch { print("pipeline error:", error) }
        }
    }

    private func process(_ pb: CVPixelBuffer) throws {
        if resetRequested.withLock({ let r = $0; $0 = false; return r }) { manager.reset() }
        let t0 = CFAbsoluteTimeGetCurrent()
        let size = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        let runDet = manager.shouldDetect()
        let hasTracks = !manager.tracks.isEmpty
        let policy = manager.policy

        // Pre-scale both inputs serially (cheap, NEON) ...
        let det = runDet ? scaler.letterbox(pb, width: detector.inputWidth, height: detector.inputHeight) : nil
        let imgTAM = hasTracks ? scaler.squash(pb, size: tam.cfg.image_size) : nil

        // ... then run the detector (GPU) and the EdgeTAM encoder (ANE) concurrently.
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
        try manager.update(detections: dets, frame: frame, tam: tam, frameSize: size, now: t0)
        let faceMs = (CFAbsoluteTimeGetCurrent() - s) * 1000

        let outputs = manager.tracks.map { t in
            FaceTrackOutput(id: t.id,
                            box: CGRect(x: t.box.minX / size.width, y: t.box.minY / size.height,
                                        width: t.box.width / size.width, height: t.box.height / size.height),
                            mask: renderMasks && t.visible ? MaskRenderer.image(t.mask, colorIndex: t.id) : nil,
                            visible: t.visible, quality: t.iouPred)
        }

        let now = CFAbsoluteTimeGetCurrent()
        let ema = { (old: Double, new: Double) in old == 0 ? new : 0.9 * old + 0.1 * new }
        if lastFrameTime > 0 { stats.fps = ema(stats.fps, 1 / (now - lastFrameTime)) }
        lastFrameTime = now
        if runDet { stats.detectMs = ema(stats.detectMs, detMs) }
        if frame != nil { stats.encodeMs = ema(stats.encodeMs, encMs) }
        if !manager.tracks.isEmpty { stats.perFaceMs = ema(stats.perFaceMs, faceMs / Double(manager.tracks.count)) }
        stats.totalMs = ema(stats.totalMs, (now - t0) * 1000)
        stats.faces = manager.tracks.count
        stats.events = manager.events
        let snapshot = stats
        statsFrame += 1
        if statsFrame % 30 == 0 {  // periodic line in Xcode's console for remote profiling
            print(String(format: "[FaceTrack] %.1f fps | faces %ld | detect %.1f ms | encode %.1f ms | per face %.1f ms | total %.1f ms",
                         stats.fps, stats.faces, stats.detectMs, stats.encodeMs, stats.perFaceMs, stats.totalMs)
                  + (stats.events.isEmpty ? "" : " | " + stats.events.joined(separator: "; ")))
            let t = tam.timings
            print("[FaceTrack] stages (ms): " + ["enc", "attn", "dec", "menc", "assemble", "cpu", "decPrompt"]
                .compactMap { k in t[k].map { String(format: "%@ %.1f", k, $0) } }.joined(separator: " | "))
            if statsFrame % 150 == 0 { ComputePlanReport.lines.forEach { print($0) } }
        }
        DispatchQueue.main.async { self.onResult?(outputs, snapshot, size) }
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
