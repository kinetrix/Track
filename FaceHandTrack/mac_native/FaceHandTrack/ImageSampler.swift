import CoreVideo
import Foundation

/// Read access to a locked 32-bit BGRA pixel buffer.
struct BGRAImage {
    let base: UnsafePointer<UInt8>
    let width: Int
    let height: Int
    let rowBytes: Int

    /// Bilinear sample at (x, y), pixel centres on integer coordinates, edges clamped.
    @inline(__always)
    func bilinear(_ x: Double, _ y: Double) -> (Double, Double, Double) {
        let xf = floor(x), yf = floor(y)
        let fx = x - xf, fy = y - yf
        let x0 = min(max(Int(xf), 0), width - 1), x1 = min(max(Int(xf) + 1, 0), width - 1)
        let y0 = min(max(Int(yf), 0), height - 1), y1 = min(max(Int(yf) + 1, 0), height - 1)
        let r0 = base + y0 * rowBytes, r1 = base + y1 * rowBytes
        let p00 = r0 + 4 * x0, p01 = r0 + 4 * x1, p10 = r1 + 4 * x0, p11 = r1 + 4 * x1
        let w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy
        func ch(_ c: Int) -> Double {
            w00 * Double(p00[c]) + w01 * Double(p01[c]) + w10 * Double(p10[c]) + w11 * Double(p11[c])
        }
        return (ch(0), ch(1), ch(2))
    }
}

func withBGRA<T>(_ pb: CVPixelBuffer, _ body: (BGRAImage) throws -> T) rethrows -> T {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let img = BGRAImage(base: CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self),
                        width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                        rowBytes: CVPixelBufferGetBytesPerRow(pb))
    return try body(img)
}

/// 2-D affine map: (x, y) -> (a x + b y + tx, c x + d y + ty).
struct Affine {
    var a = 1.0, b = 0.0, tx = 0.0
    var c = 0.0, d = 1.0, ty = 0.0

    @inline(__always)
    func apply(_ x: Double, _ y: Double) -> (Double, Double) { (a * x + b * y + tx, c * x + d * y + ty) }

    /// self after `first`: p -> self(first(p)).
    func after(_ f: Affine) -> Affine {
        Affine(a: a * f.a + b * f.c, b: a * f.b + b * f.d, tx: a * f.tx + b * f.ty + tx,
               c: c * f.a + d * f.c, d: c * f.b + d * f.d, ty: c * f.tx + d * f.ty + ty)
    }

    static func translate(_ x: Double, _ y: Double) -> Affine { Affine(tx: x, ty: y) }
    static func scale(_ sx: Double, _ sy: Double) -> Affine { Affine(a: sx, d: sy) }
}

/// A rectangle [x0, x1) x [y0, y1) in some intermediate image, reached from output
/// coordinates through `map`. Samples outside it are black (the zero padding of the
/// Python crop / pad / rotate steps).
struct Region {
    var map: Affine
    var x0: Double, y0: Double, x1: Double, y1: Double

    @inline(__always)
    func contains(_ u: Double, _ v: Double) -> Bool {
        let (x, y) = map.apply(u, v)
        return x >= x0 && x < x1 && y >= y0 && y < y1
    }
}

/// Small BGRA model inputs (192 x 192 palm detector, 224 x 224 landmarks) sampled straight
/// from the camera frame through one affine map, instead of resize + pad + rotate + crop +
/// resize as separate image copies. `supersample` x `supersample` samples per output pixel
/// approximate OpenCV's INTER_AREA when shrinking.
final class ImageSampler {
    private var pools: [Int: CVPixelBufferPool] = [:]

    private func buffer(side: Int) -> CVPixelBuffer {
        if pools[side] == nil {
            let attrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: side,
                kCVPixelBufferHeightKey: side,
                kCVPixelBufferIOSurfacePropertiesKey: [:],
            ]
            var p: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
            pools[side] = p!
        }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pools[side]!, &out)
        return out!
    }

    /// `toSource` maps output pixel coordinates (pixel centres on integers) to frame coordinates.
    func render(_ src: BGRAImage, side: Int, toSource: Affine, regions: [Region], supersample: Int = 1) -> CVPixelBuffer {
        let out = buffer(side: side)
        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        let dst = CVPixelBufferGetBaseAddress(out)!.assumingMemoryBound(to: UInt8.self)
        let row = CVPixelBufferGetBytesPerRow(out)
        let n = max(1, supersample)
        let inv = 1.0 / Double(n * n)
        for j in 0..<side {
            let line = dst + j * row
            for i in 0..<side {
                var b = 0.0, g = 0.0, r = 0.0
                for sy in 0..<n {
                    let v = Double(j) - 0.5 + (Double(sy) + 0.5) / Double(n)
                    for sx in 0..<n {
                        let u = Double(i) - 0.5 + (Double(sx) + 0.5) / Double(n)
                        var inside = true
                        for reg in regions where !reg.contains(u, v) {
                            inside = false
                            break
                        }
                        if !inside { continue }
                        let (x, y) = toSource.apply(u, v)
                        let p = src.bilinear(x, y)
                        b += p.0
                        g += p.1
                        r += p.2
                    }
                }
                let px = line + 4 * i
                px[0] = UInt8(min(255, (b * inv).rounded()))
                px[1] = UInt8(min(255, (g * inv).rounded()))
                px[2] = UInt8(min(255, (r * inv).rounded()))
                px[3] = 255
            }
        }
        return out
    }
}
