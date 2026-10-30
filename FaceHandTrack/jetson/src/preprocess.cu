// CUDA side of preprocess.hpp (compiled by nvcc in build.sh).
#include <cstring>

#include "preprocess.hpp"
#include "preprocess_math.hpp"
#include "trt_model.hpp"

namespace fht {

namespace {
__global__ void resize_kernel(const uint8_t* src, int sw, int sh, float* dst, int dw, int dh, int rw, int rh, bool rgb) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dw && y < dh) resize_pixel(src, sw, sh, dst, dw, dh, rw, rh, rgb, x, y);
}
}  // namespace

GpuFrame::~GpuFrame() {
    if (dev_) cudaFree(dev_);
    if (pinned_) cudaFreeHost(pinned_);
}

void GpuFrame::upload(const cv::Mat& bgr, cudaStream_t stream) {
    const cv::Mat src = bgr.isContinuous() ? bgr : bgr.clone();
    const size_t bytes = src.total() * src.elemSize();
    if (bytes != bytes_) {
        if (dev_) cudaFree(dev_);
        if (pinned_) cudaFreeHost(pinned_);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_), bytes));
        bytes_ = bytes;
    }
    w_ = src.cols;
    h_ = src.rows;
    std::memcpy(pinned_, src.data, bytes);
    CUDA_CHECK(cudaMemcpyAsync(dev_, pinned_, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void resize_to_planar(const GpuFrame& frame, float* dst, int dw, int dh, int rw, int rh, bool rgb,
                      cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((dw + block.x - 1) / block.x, (dh + block.y - 1) / block.y);
    resize_kernel<<<grid, block, 0, stream>>>(frame.data(), frame.width(), frame.height(), dst, dw, dh, rw, rh, rgb);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fht
