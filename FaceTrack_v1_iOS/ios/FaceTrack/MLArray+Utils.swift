import CoreML
import Foundation

/// Small helpers for fp16 MLMultiArrays. Graph-to-graph tensors are fp16 end to
/// end (see export_edgetam.py), so the app never converts large tensors to fp32.
extension MLMultiArray {
    static func zerosF16(_ shape: [Int]) -> MLMultiArray {
        let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        a.withUnsafeMutableBytes { ptr, _ in memset(ptr.baseAddress!, 0, ptr.count) }
        return a
    }

    static func f32(_ values: [Float], shape: [Int]) -> MLMultiArray {
        let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        a.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for (i, v) in values.enumerated() { buf[i] = v }
        }
        return a
    }

    /// Scalar read that works for fp16 / fp32 outputs and any strides.
    func value(_ idx: [Int]) -> Float {
        var off = 0
        for (i, s) in strides.enumerated() { off += idx[i] * s.intValue }
        switch dataType {
        case .float16:
            return withUnsafeBufferPointer(ofType: Float16.self) { Float($0[off]) }
        case .float32:
            return withUnsafeBufferPointer(ofType: Float.self) { $0[off] }
        case .double:
            return withUnsafeBufferPointer(ofType: Double.self) { Float($0[off]) }
        default:
            return Float(truncating: self[off])
        }
    }

    /// Copies a contiguous run of `count` fp16 values that starts at `idx`
    /// (inner dimension must be dense, which Core ML guarantees for the last axis).
    func copyF16(from idx: [Int], count: Int, into dst: UnsafeMutablePointer<Float16>) {
        precondition(dataType == .float16)
        var off = 0
        for (i, s) in strides.enumerated() { off += idx[i] * s.intValue }
        withUnsafeBufferPointer(ofType: Float16.self) { src in
            dst.update(from: src.baseAddress! + off, count: count)
        }
    }

    /// Integer strides. (`strides` builds a new [NSNumber] on every access, so read it once.)
    var intStrides: [Int] { strides.map { $0.intValue } }

    /// Copies a whole [..., rows, cols] fp16 plane, honouring any strides.
    /// One buffer access per plane: locking an output array once per row (512 times
    /// per memory slot) cost ~5 ms per frame on iPhone with Neural Engine outputs.
    func copyPlaneF16(prefix: [Int], rows: Int, cols: Int, into dst: UnsafeMutablePointer<Float16>) {
        precondition(dataType == .float16)
        let st = intStrides
        var base = 0
        for (i, p) in prefix.enumerated() { base += p * st[i] }
        let rs = st[prefix.count], cs = st[prefix.count + 1]
        withUnsafeBufferPointer(ofType: Float16.self) { buf in
            let src = buf.baseAddress! + base
            if cs == 1 && rs == cols {
                dst.update(from: src, count: rows * cols)
            } else if cs == 1 {
                for r in 0..<rows { (dst + r * cols).update(from: src + r * rs, count: cols) }
            } else {
                for r in 0..<rows { for c in 0..<cols { dst[r * cols + c] = src[r * rs + c * cs] } }
            }
        }
    }

    /// Reads a [rows, cols] slab starting at `prefix` into a dense row-major [Float]
    /// in one buffer access (fp16 / fp32 / fp64, any strides).
    func floats2D(prefix: [Int], rows: Int, cols: Int) -> [Float] {
        let st = intStrides
        var base = 0
        for (i, p) in prefix.enumerated() { base += p * st[i] }
        let rs = st[prefix.count], cs = st[prefix.count + 1]
        var out = [Float](repeating: 0, count: rows * cols)
        func read<T: BinaryFloatingPoint>(_ buf: UnsafeBufferPointer<T>) {
            let src = buf.baseAddress! + base
            for r in 0..<rows {
                for c in 0..<cols { out[r * cols + c] = Float(src[r * rs + c * cs]) }
            }
        }
        switch dataType {
        case .float16: withUnsafeBufferPointer(ofType: Float16.self) { read($0) }
        case .float32: withUnsafeBufferPointer(ofType: Float.self) { read($0) }
        case .double: withUnsafeBufferPointer(ofType: Double.self) { read($0) }
        default:
            for r in 0..<rows {
                for c in 0..<cols { out[r * cols + c] = value(prefix + [r, c]) }
            }
        }
        return out
    }
}
