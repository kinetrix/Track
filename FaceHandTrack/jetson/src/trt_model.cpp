#include "trt_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

namespace fht {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[cuda] %s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", msg);
    }
};

Logger& logger() {
    static Logger l;
    return l;
}

size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

}  // namespace

TrtModel::TrtModel(const std::string& engine_path) : path_(engine_path) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[trt] cannot open engine %s\n", engine_path.c_str());
        std::exit(1);
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    runtime_ = nvinfer1::createInferRuntime(logger());
    engine_ = runtime_ ? runtime_->deserializeCudaEngine(blob.data(), blob.size()) : nullptr;
    context_ = engine_ ? engine_->createExecutionContext() : nullptr;
    if (!context_) {
        std::fprintf(stderr, "[trt] cannot load engine %s (built with another TensorRT version?)\n",
                     engine_path.c_str());
        std::exit(1);
    }
    CUDA_CHECK(cudaStreamCreate(&stream_));

    const int n = engine_->getNbBindings();
    tensors_.resize(n);
    bindings_.resize(n);
    for (int i = 0; i < n; ++i) {
        Tensor& t = tensors_[i];
        t.name = engine_->getBindingName(i);
        t.is_input = engine_->bindingIsInput(i);
        t.dims = engine_->getBindingDimensions(i);
        if (engine_->getBindingDataType(i) != nvinfer1::DataType::kFLOAT) {
            std::fprintf(stderr, "[trt] %s: tensor %s is not float32\n", engine_path.c_str(), t.name.c_str());
            std::exit(1);
        }
        for (int k = 0; k < t.dims.nbDims; ++k) {
            if (t.dims.d[k] < 0) {
                std::fprintf(stderr, "[trt] %s: tensor %s has a dynamic shape\n", engine_path.c_str(), t.name.c_str());
                std::exit(1);
            }
        }
        t.count = volume(t.dims);
        CUDA_CHECK(cudaMalloc(&t.device, t.count * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&t.host), t.count * sizeof(float)));
        bindings_[i] = t.device;
    }
}

TrtModel::~TrtModel() {
    for (Tensor& t : tensors_) {
        cudaFree(t.device);
        cudaFreeHost(t.host);
    }
    if (stream_) cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
}

Tensor& TrtModel::tensor(const std::string& name) {
    for (Tensor& t : tensors_)
        if (t.name == name) return t;
    std::fprintf(stderr, "[trt] %s has no tensor %s\n", path_.c_str(), name.c_str());
    std::exit(1);
}

void TrtModel::infer() {
    for (Tensor& t : tensors_)
        if (t.is_input)
            CUDA_CHECK(cudaMemcpyAsync(t.device, t.host, t.count * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
        std::fprintf(stderr, "[trt] %s: inference failed\n", path_.c_str());
        std::exit(1);
    }
    for (Tensor& t : tensors_)
        if (!t.is_input)
            CUDA_CHECK(cudaMemcpyAsync(t.host, t.device, t.count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace fht
