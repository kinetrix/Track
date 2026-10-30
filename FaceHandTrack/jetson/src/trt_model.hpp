// A TensorRT engine with its input / output buffers (TensorRT 8.5, JetPack 5).
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace fht {

// Aborts with a message if a CUDA call fails.
void cuda_check(cudaError_t e, const char* what);
#define CUDA_CHECK(x) ::fht::cuda_check((x), #x)

struct Tensor {
    std::string name;
    bool is_input = false;
    nvinfer1::Dims dims{};
    size_t count = 0;          // number of float elements
    void* device = nullptr;    // GPU buffer
    float* host = nullptr;     // pinned CPU buffer of the same size
};

class TrtModel {
public:
    // Loads a serialized engine (built by trtexec). All inputs and outputs must be float32.
    explicit TrtModel(const std::string& engine_path);
    ~TrtModel();
    TrtModel(const TrtModel&) = delete;
    TrtModel& operator=(const TrtModel&) = delete;

    Tensor& tensor(const std::string& name);   // aborts if missing
    const std::vector<Tensor>& tensors() const { return tensors_; }

    // Copies every input host -> device, runs, copies every output device -> host, waits.
    void infer();

    const std::string& path() const { return path_; }

private:
    std::string path_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<Tensor> tensors_;
    std::vector<void*> bindings_;
};

}  // namespace fht
