import CoreGraphics
import CoreML
import CoreVideo
import Foundation

/// Streaming EdgeTAM (Meta, Apache-2.0) on Core ML.
/// Mirrors the Python reference tracker line by line (models from FaceTrack_v1_iOS, 512 x 512).
final class EdgeTAM {
    struct Config: Decodable {
        let num_cond: Int
        let num_recent: Int
        let num_ptr: Int
        let tokens_per_mem: Int
        let tokens_per_ptr: Int
        let mem_dim: Int
        let image_size: Int  // EdgeTAM input side (1024, 768 or 512)
        let mask_size: Int?  // low-res mask side (image_size / 4); absent in older configs
        let stability_delta: Float
        let stability_thresh: Float

        var maskSize: Int { mask_size ?? image_size / 4 }
    }

    /// Per-frame encoder outputs, shared by every tracked face.
    struct Frame {
        let feat: MLMultiArray  // raw stride-16 features (memory encoder input)
        let featInit: MLMultiArray  // feat + no_mem_embed (first frame of a track)
        let tgt0: MLMultiArray  // after the shared layer-0 self-attention
        let hr0: MLMultiArray
        let hr1: MLMultiArray
    }

    /// Per-face memory bank (fixed slot layout, see edgetam_config.json).
    final class Memory {
        var condMem: [MLMultiArray] = []  // oldest first; slot 0 = first anchor
        var condPtr: [[Float16]] = []
        var recentMem: [MLMultiArray] = []  // newest first
        var recentPtr: [[Float16]] = []  // newest first
    }

    struct Result {
        let mask: [Float16]  // maskSize x maskSize logits over the squashed frame
        let iouPred: Float
        let objScore: Float
    }

    let cfg: Config
    private let encoder: MLModel
    private let attention: MLModel
    private let decTrack: MLModel
    private let decPrompt: MLModel
    private let memEncoder: MLModel

    // reusable inputs (all calls happen on the pipeline queue, one at a time)
    private let memoryIn: MLMultiArray
    private let validIn: MLMultiArray
    private let maskIn: MLMultiArray
    private let binarizeOn: MLMultiArray
    private let binarizeOff = MLMultiArray.zerosF16([1])
    private let trackCoords = MLMultiArray.f32([0, 0], shape: [1, 1, 2])
    private let trackLabels = MLMultiArray.f32([-1], shape: [1, 1])

    private var slots: Int { cfg.num_cond + cfg.num_recent }

    /// Moving-average wall time (ms) of each stage, for profiling. Keys: enc, attn, dec, menc, cpu.
    private(set) var timings: [String: Double] = [:]

    private func timed<T>(_ key: String, _ body: () throws -> T) rethrows -> T {
        let s = CFAbsoluteTimeGetCurrent()
        let r = try body()
        let ms = (CFAbsoluteTimeGetCurrent() - s) * 1000
        timings[key] = timings[key].map { 0.9 * $0 + 0.1 * ms } ?? ms
        return r
    }

    /// attentionUnits: memory attention alone can run elsewhere. On an M5 the GPU is faster for it
    /// (3.5 vs 5.5 ms, SelfTest --benchmark) but its masks drift further from the reference
    /// (group clip: median box IoU 0.94 vs 1.00), so the default keeps it on the Neural Engine.
    init(computeUnits: MLComputeUnits = .cpuAndNeuralEngine, attentionUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        let cfgURL = Bundle.main.url(forResource: "edgetam_config", withExtension: "json")!
        cfg = try JSONDecoder().decode(Config.self, from: Data(contentsOf: cfgURL))
        maskIn = .zerosF16([1, 1, cfg.maskSize, cfg.maskSize])

        func load(_ name: String, function: String? = nil, units: MLComputeUnits? = nil) throws -> MLModel {
            let c = MLModelConfiguration()
            c.computeUnits = units ?? computeUnits
            if let function { c.functionName = function }
            return try MLModel(contentsOf: Bundle.main.url(forResource: name, withExtension: "mlmodelc")!, configuration: c)
        }
        encoder = try load("EdgeTAMImageEncoder")
        attention = try load("EdgeTAMMemoryAttention", units: attentionUnits)
        decTrack = try load("EdgeTAMMaskDecoder", function: "track")
        decPrompt = try load("EdgeTAMMaskDecoder", function: "prompt")
        memEncoder = try load("EdgeTAMMemoryEncoder")

        let lk = (cfg.num_cond + cfg.num_recent) * cfg.tokens_per_mem + cfg.num_ptr * cfg.tokens_per_ptr
        memoryIn = .zerosF16([1, lk, cfg.mem_dim])
        validIn = .zerosF16([1, cfg.num_cond + cfg.num_recent + cfg.num_ptr])
        binarizeOn = .zerosF16([1])
        binarizeOn.withUnsafeMutableBufferPointer(ofType: Float16.self) { b, _ in b[0] = 1 }
    }

    // MARK: - per frame

    func encode(_ image: CVPixelBuffer) throws -> Frame {
        let o = try timed("enc") { try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: image)])) }
        func a(_ n: String) -> MLMultiArray { o.featureValue(for: n)!.multiArrayValue! }
        return Frame(feat: a("feat"), featInit: a("feat_init"), tgt0: a("tgt0"), hr0: a("hr0"), hr1: a("hr1"))
    }

    // MARK: - per face

    /// Start a track (memory == nil) or re-anchor one from a detector box given
    /// in model-input coordinates (0...image_size; x1, y1, x2, y2).
    func addBox(_ f: Frame, box: CGRect, memory: Memory?) throws -> (Memory, Result) {
        let mem = memory ?? Memory()
        let pix: MLMultiArray
        if memory == nil {
            pix = f.featInit  // first frame of the track: no memory, no_mem_embed added
        } else {
            pix = try attend(f, mem)  // re-anchor: condition on the track's memory, like a SAM 2 correction click
        }
        let coords = MLMultiArray.f32([Float(box.minX), Float(box.minY), Float(box.maxX), Float(box.maxY)], shape: [1, 2, 2])
        let labels = MLMultiArray.f32([2, 3], shape: [1, 2])
        let d = try decode(decPrompt, pix: pix, f: f, coords: coords, labels: labels)
        // single-mask output with SAM 2's dynamic fallback to the best multimask output
        let k = stability(d.masks, index: 0) >= cfg.stability_thresh ? 0 : 1 + argmax(d.iou, from: 1)
        let mask = plane(d.masks, k)
        let ptr = pointer(d.ptr, 0)  // pointer always from the single-mask token for prompts
        let tokens = try encodeMemory(f, mask: mask, binarize: true)
        mem.condMem.append(tokens)
        mem.condPtr.append(ptr)
        if mem.condMem.count > cfg.num_cond {  // keep first anchor + newest re-anchors
            mem.condMem.remove(at: 1)
            mem.condPtr.remove(at: 1)
        }
        return (mem, Result(mask: mask, iouPred: d.iou.value([0, k]), objScore: d.score))
    }

    /// Propagate a track by one frame using its memory bank.
    func track(_ f: Frame, memory mem: Memory) throws -> Result {
        let pix = try attend(f, mem)
        let d = try decode(decTrack, pix: pix, f: f, coords: trackCoords, labels: trackLabels)
        let (k, mask, ptr) = timed("cpu") { () -> (Int, [Float16], [Float16]) in
            let k = 1 + argmax(d.iou, from: 1)  // multimask output while tracking
            return (k, plane(d.masks, k), pointer(d.ptr, k))
        }
        let tokens = try encodeMemory(f, mask: mask, binarize: false)
        mem.recentMem.insert(tokens, at: 0)
        mem.recentPtr.insert(ptr, at: 0)
        if mem.recentMem.count > cfg.num_recent { mem.recentMem.removeLast(mem.recentMem.count - cfg.num_recent) }
        if mem.recentPtr.count > cfg.num_ptr - 1 { mem.recentPtr.removeLast(mem.recentPtr.count - (cfg.num_ptr - 1)) }
        return Result(mask: mask, iouPred: d.iou.value([0, k]), objScore: d.score)
    }

    // MARK: - graph calls

    private func attend(_ f: Frame, _ mem: Memory) throws -> MLMultiArray {
        timed("assemble") { assemble(mem) }
        let o = try timed("attn") {
            try attention.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "tgt0": MLFeatureValue(multiArray: f.tgt0),
                "memory": MLFeatureValue(multiArray: memoryIn),
                "slot_valid": MLFeatureValue(multiArray: validIn),
            ]))
        }
        return o.featureValue(for: "pix_feat")!.multiArrayValue!
    }

    private struct Decoded {
        let masks: MLMultiArray
        let iou: MLMultiArray
        let ptr: MLMultiArray
        let score: Float
    }

    private func decode(_ m: MLModel, pix: MLMultiArray, f: Frame, coords: MLMultiArray, labels: MLMultiArray) throws -> Decoded {
        let o = try timed(m === decTrack ? "dec" : "decPrompt") {
            try m.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "pix_feat": MLFeatureValue(multiArray: pix),
                "hr0": MLFeatureValue(multiArray: f.hr0),
                "hr1": MLFeatureValue(multiArray: f.hr1),
                "point_coords": MLFeatureValue(multiArray: coords),
                "point_labels": MLFeatureValue(multiArray: labels),
            ]))
        }
        func a(_ n: String) -> MLMultiArray { o.featureValue(for: n)!.multiArrayValue! }
        return Decoded(masks: a("masks"), iou: a("iou"), ptr: a("obj_ptr"), score: a("obj_score").value([0, 0]))
    }

    private func encodeMemory(_ f: Frame, mask: [Float16], binarize: Bool) throws -> MLMultiArray {
        maskIn.withUnsafeMutableBufferPointer(ofType: Float16.self) { dst, strides in
            let row = strides[2], n = cfg.maskSize
            mask.withUnsafeBufferPointer { src in
                for r in 0..<n { (dst.baseAddress! + r * row).update(from: src.baseAddress! + r * n, count: n) }
            }
        }
        let o = try timed("menc") {
            try memEncoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "feat": MLFeatureValue(multiArray: f.feat),
                "mask_logits": MLFeatureValue(multiArray: maskIn),
                "binarize": MLFeatureValue(multiArray: binarize ? binarizeOn : binarizeOff),
            ]))
        }
        return o.featureValue(for: "memory_tokens")!.multiArrayValue!
    }

    // MARK: - memory bank assembly

    /// Layout: [cond 0..C-1 | recent 0 (newest)..R-1 | P pointer slots]; empty slots zeroed + masked.
    private func assemble(_ mem: Memory) {
        let tpm = cfg.tokens_per_mem, d = cfg.mem_dim
        let slotElems = tpm * d
        let ptrs = mem.condPtr + mem.recentPtr.prefix(max(0, cfg.num_ptr - mem.condPtr.count))
        memoryIn.withUnsafeMutableBufferPointer(ofType: Float16.self) { dst, strides in
            precondition(strides[1] == d && strides[2] == 1, "memory input must be dense")
            let base = dst.baseAddress!
            func put(_ src: MLMultiArray?, slot: Int) {
                let p = base + slot * slotElems
                if let src { src.copyPlaneF16(prefix: [0], rows: tpm, cols: d, into: p) } else { p.update(repeating: 0, count: slotElems) }
            }
            for i in 0..<cfg.num_cond { put(i < mem.condMem.count ? mem.condMem[i] : nil, slot: i) }
            for i in 0..<cfg.num_recent { put(i < mem.recentMem.count ? mem.recentMem[i] : nil, slot: cfg.num_cond + i) }
            let ptrBase = base + slots * slotElems
            for i in 0..<cfg.num_ptr {
                let p = ptrBase + i * 256  // 4 tokens x 64 = one 256-d pointer
                if i < ptrs.count { ptrs[i].withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: 256) } } else { p.update(repeating: 0, count: 256) }
            }
        }
        validIn.withUnsafeMutableBufferPointer(ofType: Float16.self) { v, _ in
            for i in 0..<cfg.num_cond { v[i] = i < mem.condMem.count ? 1 : 0 }
            for i in 0..<cfg.num_recent { v[cfg.num_cond + i] = i < mem.recentMem.count ? 1 : 0 }
            for i in 0..<cfg.num_ptr { v[slots + i] = i < ptrs.count ? 1 : 0 }
        }
    }

    // MARK: - small helpers

    private func plane(_ masks: MLMultiArray, _ k: Int) -> [Float16] {
        let n = cfg.maskSize
        var out = [Float16](repeating: 0, count: n * n)
        out.withUnsafeMutableBufferPointer { masks.copyPlaneF16(prefix: [0, k], rows: n, cols: n, into: $0.baseAddress!) }
        return out
    }

    private func pointer(_ ptr: MLMultiArray, _ k: Int) -> [Float16] {
        var out = [Float16](repeating: 0, count: 256)
        out.withUnsafeMutableBufferPointer { ptr.copyF16(from: [0, k, 0], count: 256, into: $0.baseAddress!) }
        return out
    }

    private func argmax(_ iou: MLMultiArray, from start: Int) -> Int {
        var best = start
        for i in start..<iou.shape[1].intValue where iou.value([0, i]) > iou.value([0, best]) { best = i }
        return best - start
    }

    private func stability(_ masks: MLMultiArray, index k: Int) -> Float {
        let m = plane(masks, k)
        var inter = 0, union = 0
        let d = Float16(cfg.stability_delta)
        for v in m {
            if v > d { inter += 1 }
            if v > -d { union += 1 }
        }
        return union > 0 ? Float(inter) / Float(union) : 1
    }
}
