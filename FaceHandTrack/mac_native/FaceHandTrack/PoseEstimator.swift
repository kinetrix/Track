import CoreGraphics
import CoreML
import CoreVideo
import Foundation

/// Body pose for one person (MediaPipe pose landmarker, OpenCV Zoo pose_estimation_mediapipe_2023mar,
/// Apache-2.0): 33 body points from a crop built around that person's face. No person detector:
/// the face is already known (YuNet), so each pose belongs to a known face from the start.
/// Used only when 2+ people are in view, to join each hand to its own body (one camera has no
/// depth, and hand size alone could not tell a farther person's hand from a nearer one's).
struct BodyPose {
    let faceID: Int
    let points: [CGPoint]  // 33 + 6 helper points, frame pixels (MediaPipe order: 15/16 wrists, 11/12 shoulders, 13/14 elbows)
    let visibility: [Float]  // 0..1 per point
    let presence: Float  // the model's confidence that a body is there

    var wrists: [(point: CGPoint, visibility: Float)] { [(points[15], visibility[15]), (points[16], visibility[16])] }
}

final class PoseEstimator {
    static let side = 256
    /// The body crop around a face: hip assumed this many face heights below the face centre,
    /// crop half-size this many face heights (checked on a 2-person photo: wrists on the hands).
    var hipBelow: CGFloat = 3.0
    var halfSize: CGFloat = 4.0
    private let model: MLModel
    private let sampler = ImageSampler()

    init(computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        guard let url = Bundle.main.url(forResource: "Pose", withExtension: "mlmodelc") else {
            throw NSError(domain: "FaceHandTrack", code: 1, userInfo: [NSLocalizedDescriptionKey: "Pose.mlmodelc missing from the app"])
        }
        model = try MLModel(contentsOf: url, configuration: cfg)
    }

    /// Pose of the person whose face box (frame pixels) is given.
    func estimate(_ img: BGRAImage, face: CGRect, faceID: Int) throws -> BodyPose {
        let h = Double(face.height)
        let hipX = Double(face.midX), hipY = Double(face.midY) + Double(hipBelow) * h
        let r = Double(halfSize) * h
        let k = 2 * r / Double(Self.side)  // frame pixels per crop pixel
        // crop pixel -> frame pixel (upright square of side 2r centred on the hip); outside the frame is black
        let toFrame = Affine(a: k, b: 0, tx: hipX - Double(Self.side) / 2 * k, c: 0, d: k, ty: hipY - Double(Self.side) / 2 * k)
        let frame = Region(map: toFrame, x0: -0.5, y0: -0.5, x1: Double(img.width) - 0.5, y1: Double(img.height) - 0.5)
        let input = sampler.render(img, side: Self.side, toSource: toFrame, regions: [frame])
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)]))
        let lm = out.featureValue(for: "landmarks")!.multiArrayValue!.floats2D(prefix: [], rows: 1, cols: 195)
        let presence = out.featureValue(for: "presence")!.multiArrayValue!.floats2D(prefix: [], rows: 1, cols: 1)[0]
        var points: [CGPoint] = [], vis: [Float] = []
        for j in 0..<39 {
            let u = Double(lm[j * 5]), v = Double(lm[j * 5 + 1])
            points.append(CGPoint(x: toFrame.a * u + toFrame.tx, y: toFrame.d * v + toFrame.ty))
            vis.append(1 / (1 + exp(-lm[j * 5 + 3])))
        }
        return BodyPose(faceID: faceID, points: points, visibility: vis, presence: presence)
    }
}
