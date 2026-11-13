import CoreGraphics
import CoreML
import CoreVideo
import Foundation

/// SFace (OpenCV Zoo face_recognition_sface_2021dec, Apache-2.0): a 128-number "fingerprint"
/// per face, used to give a returning person their old ID back and to recognise saved names.
/// The face is aligned from YuNet's 5 landmarks to the standard 112 x 112 template first
/// (same template and transform as OpenCV's FaceRecognizerSF and the Jetson version).
final class FaceEmbedder {
    static let side = 112
    /// eyes, nose tip, mouth corners in the 112 x 112 crop
    static let template: [SIMD2<Double>] = [SIMD2(38.2946, 51.6963), SIMD2(73.5318, 51.5014), SIMD2(56.0252, 71.7366),
                                            SIMD2(41.5493, 92.3655), SIMD2(70.7299, 92.2041)]
    private let model: MLModel
    private let sampler = ImageSampler()

    init(computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        guard let url = Bundle.main.url(forResource: "SFace", withExtension: "mlmodelc") else {
            throw NSError(domain: "FaceHandTrack", code: 1, userInfo: [NSLocalizedDescriptionKey: "SFace.mlmodelc missing from the app"])
        }
        model = try MLModel(contentsOf: url, configuration: cfg)
    }

    /// Unit-length embedding of the face whose 5 landmarks (frame pixels) are given.
    func embed(_ img: BGRAImage, landmarks: [CGPoint]) throws -> [Float] {
        let pts = landmarks.map { SIMD2(Double($0.x), Double($0.y)) }
        // frame -> crop fitted like OpenCV's FaceRecognizerSF.alignCrop, then inverted (crop pixel ->
        // frame pixel); outside the frame is black, as with its warpAffine
        let toFrame = similarityTransform(from: pts, to: Self.template).inverted()
        let frame = Region(map: toFrame, x0: -0.5, y0: -0.5, x1: Double(img.width) - 0.5, y1: Double(img.height) - 0.5)
        let input = sampler.render(img, side: Self.side, toSource: toFrame, regions: [frame])
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
        let v = out.featureValue(for: "embedding")!.multiArrayValue!.floats2D(prefix: [], rows: 1, cols: 128)  // output (1, 128)
        return normalized(v)
    }
}

/// Rotation + uniform scale + shift mapping `src` onto `dst`, least squares (Umeyama without reflection).
func similarityTransform(from src: [SIMD2<Double>], to dst: [SIMD2<Double>]) -> Affine {
    let n = Double(min(src.count, dst.count))
    guard n > 0 else { return Affine() }
    var ms = SIMD2<Double>(0, 0), md = SIMD2<Double>(0, 0)
    for i in 0..<Int(n) {
        ms += src[i]
        md += dst[i]
    }
    ms /= n
    md /= n
    var a = 0.0, b = 0.0, v = 0.0
    for i in 0..<Int(n) {
        let s = src[i] - ms, d = dst[i] - md
        a += s.x * d.x + s.y * d.y
        b += s.x * d.y - s.y * d.x
        v += s.x * s.x + s.y * s.y
    }
    let c = v > 0 ? a / v : 1, sn = v > 0 ? b / v : 0  // scale * cos, scale * sin
    return Affine(a: c, b: -sn, tx: md.x - (c * ms.x - sn * ms.y), c: sn, d: c, ty: md.y - (sn * ms.x + c * ms.y))
}

extension Affine {
    func inverted() -> Affine {
        let det = a * d - b * c
        guard abs(det) > 1e-12 else { return Affine() }
        let ia = d / det, ib = -b / det, ic = -c / det, id = a / det
        return Affine(a: ia, b: ib, tx: -(ia * tx + ib * ty), c: ic, d: id, ty: -(ic * tx + id * ty))
    }
}

func normalized(_ v: [Float]) -> [Float] {
    let n = v.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
    return n > 0 ? v.map { Float(Double($0) / n) } : v
}

/// Cosine similarity of two unit-length embeddings (1 = same face, about 0 = unrelated).
func similarity(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0
    for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }
    return s
}
