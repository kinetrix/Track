import CoreGraphics
import CoreML
import CoreVideo

struct FaceDetection {
    var box: CGRect  // frame pixels
    var score: Float
    var landmarks: [CGPoint]  // 5 points, frame pixels (eyes, nose, mouth corners)
}

/// YuNet (OpenCV Zoo, MIT). Box decoding is inside the Core ML graph; the app
/// only thresholds and runs NMS over the candidates.
final class FaceDetector {
    let inputWidth: Int
    let inputHeight: Int
    private let model: MLModel

    init(computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        // GPU by default: YuNet is tiny, and keeping it off the Neural Engine
        // lets it run in parallel with the EdgeTAM encoder (see DESIGN.md §6).
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let url = Bundle.main.url(forResource: "YuNet", withExtension: "mlmodelc")!
        model = try MLModel(contentsOf: url, configuration: cfg)
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        inputWidth = Int(meta["input_width"] ?? "384") ?? 384
        inputHeight = Int(meta["input_height"] ?? "640") ?? 640
    }

    /// `input` must be the letterboxed buffer, `scale` the letterbox scale (input = frame * scale).
    func detect(_ input: CVPixelBuffer, scale: Float, policy: TrackPolicy) throws -> [FaceDetection] {
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
        let faces = out.featureValue(for: "faces")!.multiArrayValue!
        let n = faces.shape[1].intValue, k = faces.shape[2].intValue
        // One read of the whole [n, 15] output instead of a locked access per value
        // (5040 candidates x up to 15 values per frame).
        let f = faces.floats2D(prefix: [0], rows: n, cols: k)
        var cands: [FaceDetection] = []
        let inv = CGFloat(1 / scale)
        for i in 0..<n {
            let r = i * k
            let score = f[r]
            guard score >= policy.detScore else { continue }
            let w = f[r + 3], h = f[r + 4]
            guard min(w, h) >= policy.minFacePx else { continue }
            let x = f[r + 1], y = f[r + 2]
            var lm: [CGPoint] = []
            for j in 0..<5 {
                lm.append(CGPoint(x: CGFloat(f[r + 5 + 2 * j]) * inv, y: CGFloat(f[r + 6 + 2 * j]) * inv))
            }
            cands.append(FaceDetection(box: CGRect(x: CGFloat(x) * inv, y: CGFloat(y) * inv,
                                                   width: CGFloat(w) * inv, height: CGFloat(h) * inv),
                                       score: score, landmarks: lm))
        }
        return nms(cands, threshold: policy.nmsIoU)
    }

    private func nms(_ d: [FaceDetection], threshold: Float) -> [FaceDetection] {
        var keep: [FaceDetection] = []
        for c in d.sorted(by: { $0.score > $1.score }) where !keep.contains(where: { iou($0.box, c.box) > threshold }) {
            keep.append(c)
        }
        return keep
    }
}

func iou(_ a: CGRect, _ b: CGRect) -> Float {
    let i = a.intersection(b)
    guard !i.isNull, !i.isEmpty else { return 0 }
    let inter = i.width * i.height
    return Float(inter / (a.width * a.height + b.width * b.height - inter))
}
