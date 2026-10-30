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

// A float buffer on the GPU (freed automatically).
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count);
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&& o) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* get() const { return p_; }
    size_t count() const { return n_; }

private:
    float* p_ = nullptr;
    size_t n_ = 0;
};

// CUDA graphs: each model records its GPU work once per set of buffers and replays it
// afterwards, which removes most kernel-launch cost on the Jetson's CPU. On by default.
void set_cuda_graphs(bool on);

// GPU time of named stages, measured with CUDA events (no extra waiting).
class GpuTimer {
public:
    explicit GpuTimer(int n_stages);
    ~GpuTimer();
    void begin(int stage, cudaStream_t s);
    void end(int stage, cudaStream_t s);
    // Call after the stream's work has finished: adds the elapsed times to sums[stage].
    void collect(double* sums);

private:
    struct Span {
        int stage;
        cudaEvent_t a, b;
    };
    cudaEvent_t take();
    std::vector<cudaEvent_t> pool_;
    size_t used_ = 0;
    std::vector<Span> spans_;
    std::vector<cudaEvent_t> open_;
};

struct Tensor {
    std::string name;
    bool is_input = false;
    nvinfer1::Dims dims{};
    size_t count = 0;          // number of float elements
    float* device = nullptr;   // the engine's own GPU buffer
    float* host = nullptr;     // pinned CPU buffer of the same size
    int binding = -1;
};

class TrtModel {
public:
    // Loads a serialized engine (built by trtexec). All inputs and outputs must be float32.
    // stream: CUDA stream to run on (nullptr: the model makes its own).
    explicit TrtModel(const std::string& engine_path, cudaStream_t stream = nullptr);
    ~TrtModel();
    TrtModel(const TrtModel&) = delete;
    TrtModel& operator=(const TrtModel&) = delete;

    Tensor& tensor(const std::string& name);   // aborts if missing
    const std::vector<Tensor>& tensors() const { return tensors_; }

    // Points a tensor at another GPU buffer (e.g. another model's output) instead of its own;
    // nullptr goes back to the model's own buffer.
    void bind(const std::string& name, const float* device);

    void upload(const std::string& name);    // host -> device (async on the stream)
    void download(const std::string& name);  // device -> host (async on the stream)
    void enqueue();                          // run (async on the stream)
    void sync();                             // wait for everything queued on the stream

    // Copies every input host -> device, runs, copies every output device -> host, waits.
    void infer();

    cudaStream_t stream() const { return stream_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool own_stream_ = false;
    std::vector<Tensor> tensors_;
    std::vector<void*> bindings_;
    struct Graph {
        std::vector<void*> bindings;
        cudaGraphExec_t exec;
    };
    std::vector<Graph> graphs_;               // recorded runs, one per set of buffers
    std::vector<std::vector<void*>> seen_;    // sets of buffers run once (recorded on the second run)
    bool graphs_failed_ = false;
};

}  // namespace fht
