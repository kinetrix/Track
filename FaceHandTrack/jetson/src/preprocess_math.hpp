// The per-pixel arithmetic of resize_to_planar, shared by the CUDA kernel (preprocess.cu)
// and the CPU version used by the desktop check, so both compute exactly the same values.
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define FHT_HD __host__ __device__ __forceinline__
#else
#define FHT_HD inline
#endif

namespace fht {

// Output pixel (x, y) of the planar float image. Bilinear sampling with OpenCV's
// INTER_LINEAR convention (pixel centres, edge clamping) and rounding to 8 bits, as
// cv::resize on an 8-bit image followed by convertTo(float) would give.
FHT_HD void resize_pixel(const uint8_t* src, int sw, int sh, float* dst, int dw, int dh, int rw, int rh, bool rgb,
                         int x, int y) {
    const int plane = dw * dh;
    float* o = dst + y * dw + x;
    if (x >= rw || y >= rh) {
        o[0] = o[plane] = o[2 * plane] = 0.f;
        return;
    }
    const float scale_x = static_cast<float>(sw) / rw, scale_y = static_cast<float>(sh) / rh;
    float fx = (x + 0.5f) * scale_x - 0.5f, fy = (y + 0.5f) * scale_y - 0.5f;
    int sx = static_cast<int>(fx >= 0.f ? fx : fx - 1.f);  // floor
    int sy = static_cast<int>(fy >= 0.f ? fy : fy - 1.f);
    fx -= sx;
    fy -= sy;
    if (sx < 0) {
        sx = 0;
        fx = 0.f;
    }
    if (sx >= sw - 1) {
        sx = sw - 1;
        fx = 0.f;
    }
    if (sy < 0) {
        sy = 0;
        fy = 0.f;
    }
    if (sy >= sh - 1) {
        sy = sh - 1;
        fy = 0.f;
    }
    const int sx1 = sx + 1 < sw ? sx + 1 : sx, sy1 = sy + 1 < sh ? sy + 1 : sy;
    const uint8_t* p00 = src + (sy * sw + sx) * 3;
    const uint8_t* p01 = src + (sy * sw + sx1) * 3;
    const uint8_t* p10 = src + (sy1 * sw + sx) * 3;
    const uint8_t* p11 = src + (sy1 * sw + sx1) * 3;
    for (int c = 0; c < 3; ++c) {
        const float v = (1.f - fx) * (1.f - fy) * p00[c] + fx * (1.f - fy) * p01[c] + (1.f - fx) * fy * p10[c] +
                        fx * fy * p11[c];
        float r = static_cast<float>(static_cast<int>(v + 0.5f));  // round to 8 bits
        r = r > 255.f ? 255.f : r;
        const int out_c = rgb ? 2 - c : c;  // source is BGR
        o[out_c * plane] = r;
    }
}

}  // namespace fht
