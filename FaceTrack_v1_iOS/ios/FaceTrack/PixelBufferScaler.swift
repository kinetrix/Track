import Accelerate
import CoreVideo

/// CPU (vImage/NEON) scaling of camera BGRA frames into the two model inputs.
/// Both outputs come from CVPixelBufferPools, so nothing is allocated per frame.
final class PixelBufferScaler {
    private var pools: [String: CVPixelBufferPool] = [:]

    private func pool(width: Int, height: Int) -> CVPixelBufferPool {
        let key = "\(width)x\(height)"
        if let p = pools[key] { return p }
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:],  // IOSurface-backed: Core ML/ANE can read it without a copy
        ]
        var p: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
        pools[key] = p!
        return p!
    }

    /// Aspect-preserving resize into the top-left of a `width x height` canvas
    /// (rest is black). Returns the buffer and the scale applied (dst = src * scale).
    func letterbox(_ src: CVPixelBuffer, width: Int, height: Int) -> (CVPixelBuffer, Float) {
        let sw = CVPixelBufferGetWidth(src), sh = CVPixelBufferGetHeight(src)
        let s = min(Float(width) / Float(sw), Float(height) / Float(sh))
        let tw = min(width, Int((Float(sw) * s).rounded())), th = min(height, Int((Float(sh) * s).rounded()))
        var dst: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool(width: width, height: height), &dst)
        let out = dst!
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(out, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        let dstBase = CVPixelBufferGetBaseAddress(out)!
        let dstRow = CVPixelBufferGetBytesPerRow(out)
        memset(dstBase, 0, dstRow * height)
        var inBuf = vImage_Buffer(data: CVPixelBufferGetBaseAddress(src), height: vImagePixelCount(sh),
                                  width: vImagePixelCount(sw), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var outBuf = vImage_Buffer(data: dstBase, height: vImagePixelCount(th), width: vImagePixelCount(tw), rowBytes: dstRow)
        vImageScale_ARGB8888(&inBuf, &outBuf, nil, vImage_Flags(kvImageNoFlags))
        return (out, s)
    }

    /// Non-uniform resize to `size x size` (SAM 2 / EdgeTAM squash the frame to 1024x1024).
    func squash(_ src: CVPixelBuffer, size: Int) -> CVPixelBuffer {
        var dst: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool(width: size, height: size), &dst)
        let out = dst!
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(out, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        var inBuf = vImage_Buffer(data: CVPixelBufferGetBaseAddress(src), height: vImagePixelCount(CVPixelBufferGetHeight(src)),
                                  width: vImagePixelCount(CVPixelBufferGetWidth(src)), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var outBuf = vImage_Buffer(data: CVPixelBufferGetBaseAddress(out), height: vImagePixelCount(size),
                                   width: vImagePixelCount(size), rowBytes: CVPixelBufferGetBytesPerRow(out))
        vImageScale_ARGB8888(&inBuf, &outBuf, nil, vImage_Flags(kvImageNoFlags))
        return out
    }
}
