#include "trt_model.hpp"

namespace fht {

GpuTimer::GpuTimer(int n_stages) : open_(static_cast<size_t>(n_stages), nullptr) {}

GpuTimer::~GpuTimer() {
    for (cudaEvent_t e : pool_) cudaEventDestroy(e);
}

cudaEvent_t GpuTimer::take() {
    if (used_ == pool_.size()) {
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreate(&e));
        pool_.push_back(e);
    }
    return pool_[used_++];
}

void GpuTimer::begin(int stage, cudaStream_t s) {
    cudaEvent_t e = take();
    CUDA_CHECK(cudaEventRecord(e, s));
    open_[stage] = e;
}

void GpuTimer::end(int stage, cudaStream_t s) {
    cudaEvent_t e = take();
    CUDA_CHECK(cudaEventRecord(e, s));
    spans_.push_back({stage, open_[stage], e});
}

void GpuTimer::collect(double* sums) {
    for (const Span& sp : spans_) {
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, sp.a, sp.b) == cudaSuccess) sums[sp.stage] += ms;
    }
    spans_.clear();
    used_ = 0;
}

}  // namespace fht
