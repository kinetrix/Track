// Model-input preparation on the GPU: the camera frame is uploaded once, then each model's
// input (resize, letterbox, BGR -> RGB, uint8 -> float CHW) is written by one CUDA kernel
// straight into the model's input buffer.
#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstdint>

namespace fht {

// The current camera frame (BGR, 8-bit) on the GPU.
class GpuFrame {
public:
    GpuFrame() = default;
    ~GpuFrame();
    GpuFrame(const GpuFrame&) = delete;
    GpuFrame& operator=(const GpuFrame&) = delete;

    // Copies the frame to the GPU on `stream` and waits for the copy to finish.
    void upload(const cv::Mat& bgr, cudaStream_t stream);
    const uint8_t* data() const { return dev_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    uint8_t* dev_ = nullptr;
    uint8_t* pinned_ = nullptr;
    size_t bytes_ = 0;
    int w_ = 0, h_ = 0;
};

// Writes dst (3 x dh x dw floats, 0..255) from the frame: the frame is resized (bilinear, like
// cv::resize INTER_LINEAR) to rw x rh and placed in the top-left corner; the rest is 0.
// rgb: channel order R, G, B (else B, G, R). Runs asynchronously on `stream`.
void resize_to_planar(const GpuFrame& frame, float* dst, int dw, int dh, int rw, int rh, bool rgb,
                      cudaStream_t stream);

}  // namespace fht
