import AVFoundation
import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO

/// Command-line checks, run without a window or camera:
///
///     FaceHandTrack.app/Contents/MacOS/FaceHandTrack --selftest-hands <image folder> <out.json>
///     FaceHandTrack.app/Contents/MacOS/FaceHandTrack --selftest-sequence <frame folder> <out.json> [--mirrored]
///     FaceHandTrack.app/Contents/MacOS/FaceHandTrack --benchmark <out.json>
///     FaceHandTrack.app/Contents/MacOS/FaceHandTrack --list-cameras
///
/// The JSON is compared with the Python (ONNX) engine on the same images by
/// tests/compare_selftest.py, so the Swift port can be checked number by number.
enum SelfTest {
    /// Runs a test if the command line asks for one; returns its exit code, or nil for the normal app.
    static func runIfRequested() -> Int32? {
        let args = CommandLine.arguments
        func arg(_ flag: String, _ k: Int) -> String? {
            guard let i = args.firstIndex(of: flag), i + k < args.count else { return nil }
            return args[i + k]
        }
        do {
            if let dir = arg("--selftest-hands", 1), let out = arg("--selftest-hands", 2) {
                try hands(dir: dir, out: out)
                return 0
            }
            if let dir = arg("--selftest-sequence", 1), let out = arg("--selftest-sequence", 2) {
                try sequence(dir: dir, out: out, mirrored: args.contains("--mirrored"))
                return 0
            }
            if let out = arg("--benchmark", 1) {
                try benchmark(out: out)
                return 0
            }
            if args.contains("--list-cameras") {
                listCameras()
                return 0
            }
        } catch {
            print("selftest failed:", error)
            return 1
        }
        return nil
    }

    /// Every camera macOS offers, with its video formats (a RealSense lists one entry per sensor it exposes).
    static func listCameras() {
        for d in CameraController.cameras() {
            print("\(d.localizedName) | type \(d.deviceType.rawValue) | model \(d.modelID) | maker \(d.manufacturer)")
            var seen = Set<String>()
            for f in d.formats {
                let dims = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let sub = CMFormatDescriptionGetMediaSubType(f.formatDescription)
                let fourcc = String(bytes: [24, 16, 8, 0].map { UInt8((sub >> $0) & 0xff) }, encoding: .ascii) ?? "?"
                let fps = f.videoSupportedFrameRateRanges.map { String(format: "%.0f", $0.maxFrameRate) }.joined(separator: "/")
                let line = "    \(fourcc) \(dims.width)x\(dims.height) @ \(fps) fps"
                if seen.insert(line).inserted { print(line) }
            }
        }
    }

    // MARK: - inputs

    static func images(in dir: String) -> [URL] {
        let url = URL(fileURLWithPath: dir)
        let files = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        return files.filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Image file -> BGRA pixel buffer of the same size.
    static func pixelBuffer(_ url: URL) throws -> CVPixelBuffer {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "SelfTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot read \(url.path)"])
        }
        let w = img.width, h = img.height
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:]]
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        let out = pb!
        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(out), width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(out), space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return out
    }

    static func write(_ obj: Any, to path: String) throws {
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        print("wrote", path)
    }

    static func flat(_ v: [SIMD3<Float>]) -> [Double] { v.flatMap { [Double($0.x), Double($0.y), Double($0.z)] } }

    // MARK: - tests

    /// Palm detector + landmarks + gesture rules on single images (and the tracking path:
    /// landmarks again from the image's own landmarks).
    static func hands(dir: String, out: String) throws {
        let tracker = try HandTracker()
        var rows: [[String: Any]] = []
        for url in images(in: dir) {
            let pb = try pixelBuffer(url)
            var row: [String: Any] = ["name": url.deletingPathExtension().lastPathComponent]
            let palms = try tracker.detectPalms(pb)
            row["palms"] = palms.count
            if let p = palms.first, let r = try tracker.landmarks(pb, palm: p) {
                let (g, n) = Gestures.classify(landmarks: r.landmarks, world: r.world)
                row["palm"] = [p.x1, p.y1, p.x2, p.y2] + p.kps.flatMap { [$0.x, $0.y] }
                row["palm_score"] = Double(p.score)
                row["landmarks"] = flat(r.landmarks)
                row["world"] = flat(r.world)
                row["right"] = Double(r.rightProb)
                row["presence"] = Double(r.presence)
                row["gesture"] = g
                row["fingers"] = n
                if let r2 = try tracker.landmarks(pb, palm: HandLandmarker.palmFromLandmarks(r.landmarks)) {
                    row["gesture_tracking"] = Gestures.classify(landmarks: r2.landmarks, world: r2.world).0
                }
            }
            print(row["name"]!, row["gesture"] ?? "no hand")
            rows.append(row)
        }
        try write(rows, to: out)
    }

    /// The whole pipeline on a frame sequence (time = index / 30 s).
    static func sequence(dir: String, out: String, mirrored: Bool) throws {
        let p = try Pipeline()
        p.mirrored = mirrored
        p.renderMasks = false
        var rows: [[String: Any]] = []
        for (i, url) in images(in: dir).enumerated() {
            let pb = try pixelBuffer(url)
            let o = try p.process(pb, now: Double(i) / 30)
            let W = Double(o.size.width), H = Double(o.size.height)
            rows.append([
                "index": i,
                "faces": o.faces.map { f -> [String: Any] in
                    ["id": f.id, "visible": f.visible, "quality": Double(f.quality),
                     "box": [f.box.minX * W, f.box.minY * H, f.box.maxX * W, f.box.maxY * H]]
                },
                "hands": o.hands.map { h -> [String: Any] in
                    ["id": h.id, "handedness": h.handedness, "gesture": h.gesture, "raw": h.rawGesture,
                     "landmarks": flat(h.landmarks)]
                },
                "total_ms": o.stats.frameMs,
            ])
        }
        try write(rows, to: out)
    }

    /// Milliseconds per prediction for every model on every compute-unit choice.
    static func benchmark(out: String) throws {
        let models: [(String, String?)] = [("YuNet", nil), ("EdgeTAMImageEncoder", nil), ("EdgeTAMMemoryAttention", nil),
                                           ("EdgeTAMMaskDecoder", "track"), ("EdgeTAMMemoryEncoder", nil),
                                           ("PalmDetector", nil), ("HandLandmarks", nil)]
        let units: [(String, MLComputeUnits)] = [("cpu", .cpuOnly), ("gpu", .cpuAndGPU), ("ane", .cpuAndNeuralEngine), ("all", .all)]
        var result: [String: [String: Double]] = [:]
        for (name, function) in models {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { continue }
            for (uname, u) in units {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = u
                if let function { cfg.functionName = function }
                let m = try MLModel(contentsOf: url, configuration: cfg)
                let input = try exampleInput(m)
                for _ in 0..<3 { _ = try m.prediction(from: input) }  // warm-up
                let n = 20
                let s = CFAbsoluteTimeGetCurrent()
                for _ in 0..<n { _ = try m.prediction(from: input) }
                let ms = (CFAbsoluteTimeGetCurrent() - s) * 1000 / Double(n)
                result[name, default: [:]][uname] = ms
                print(String(format: "%-24@ %-4@ %7.2f ms", name, uname, ms))
            }
        }
        try write(result, to: out)
    }

    /// Zero inputs of the right shapes and types (images: mid-grey).
    static func exampleInput(_ m: MLModel) throws -> MLFeatureProvider {
        var d: [String: MLFeatureValue] = [:]
        for (name, desc) in m.modelDescription.inputDescriptionsByName {
            if let c = desc.imageConstraint {
                var pb: CVPixelBuffer?
                let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:]]
                CVPixelBufferCreate(nil, c.pixelsWide, c.pixelsHigh, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
                CVPixelBufferLockBaseAddress(pb!, [])
                memset(CVPixelBufferGetBaseAddress(pb!), 128, CVPixelBufferGetDataSize(pb!))
                CVPixelBufferUnlockBaseAddress(pb!, [])
                d[name] = MLFeatureValue(pixelBuffer: pb!)
            } else if let c = desc.multiArrayConstraint {
                let shape = c.shape.isEmpty ? [1] : c.shape
                let a = try MLMultiArray(shape: shape, dataType: c.dataType)
                a.withUnsafeMutableBytes { p, _ in memset(p.baseAddress!, 0, p.count) }
                d[name] = MLFeatureValue(multiArray: a)
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: d)
    }
}
