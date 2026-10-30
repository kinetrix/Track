// Desktop check only: preprocess.hpp on the CPU ("device" memory is ordinary memory in the
// desktop build), using the same per-pixel function as the CUDA kernel.
#include <cstdlib>
#include <cstring>

#include "preprocess.hpp"
#include "preprocess_math.hpp"

namespace fht {

GpuFrame::~GpuFrame() { std::free(dev_); }

void GpuFrame::upload(const cv::Mat& bgr, cudaStream_t) {
    const cv::Mat src = bgr.isContinuous() ? bgr : bgr.clone();
    const size_t bytes = src.total() * src.elemSize();
    if (bytes != bytes_) {
        std::free(dev_);
        dev_ = static_cast<uint8_t*>(std::malloc(bytes));
        bytes_ = bytes;
    }
    w_ = src.cols;
    h_ = src.rows;
    std::memcpy(dev_, src.data, bytes);
}

void resize_to_planar(const GpuFrame& frame, float* dst, int dw, int dh, int rw, int rh, bool rgb, cudaStream_t) {
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            resize_pixel(frame.data(), frame.width(), frame.height(), dst, dw, dh, rw, rh, rgb, x, y);
}

}  // namespace fht
