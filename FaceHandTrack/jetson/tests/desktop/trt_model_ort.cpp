// Desktop check only: the TrtModel interface backed by ONNX Runtime on the CPU, so the C++
// tracker can be compared with the Python engine on a PC without a GPU or TensorRT.
// An engine path ".../<model>.fp16.engine" loads "$FHT_ONNX_DIR/<model>.onnx".
#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <utility>

#include "trt_model.hpp"

namespace fht {

void set_cuda_graphs(bool) {}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[cuda stub] %s failed\n", what);
        std::exit(1);
    }
}

DeviceBuffer::DeviceBuffer(size_t count) : n_(count) {
    p_ = static_cast<float*>(std::calloc(count, sizeof(float)));
}
DeviceBuffer::~DeviceBuffer() { std::free(p_); }
DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) {
    o.p_ = nullptr;
    o.n_ = 0;
}
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
        std::free(p_);
        p_ = std::exchange(o.p_, nullptr);
        n_ = std::exchange(o.n_, 0);
    }
    return *this;
}

namespace {
struct OrtState {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> in_names, out_names;
    std::vector<std::vector<int64_t>> shapes;  // per tensor, same order as tensors_
};
Ort::Env& env() {
    static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "fht");
    return e;
}
std::map<const TrtModel*, std::unique_ptr<OrtState>>& states() {
    static std::map<const TrtModel*, std::unique_ptr<OrtState>> s;
    return s;
}
}  // namespace

TrtModel::TrtModel(const std::string& engine_path, cudaStream_t stream) : path_(engine_path), stream_(stream) {
    std::string name = engine_path.substr(engine_path.find_last_of('/') + 1);
    name = name.substr(0, name.find('.'));
    const char* dir = std::getenv("FHT_ONNX_DIR");
    const std::string onnx = std::string(dir ? dir : ".") + "/" + name + ".onnx";
    auto st = std::make_unique<OrtState>();
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(4);
    st->session = std::make_unique<Ort::Session>(env(), onnx.c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto add = [&](bool input, size_t i) {
        Tensor t;
        t.is_input = input;
        t.name = input ? st->session->GetInputNameAllocated(i, alloc).get()
                       : st->session->GetOutputNameAllocated(i, alloc).get();
        auto info = input ? st->session->GetInputTypeInfo(i) : st->session->GetOutputTypeInfo(i);
        std::vector<int64_t> shape = info.GetTensorTypeAndShapeInfo().GetShape();
        t.dims.nbDims = static_cast<int>(shape.size());
        t.count = 1;
        for (size_t k = 0; k < shape.size(); ++k) {
            t.dims.d[k] = static_cast<int32_t>(shape[k]);
            t.count *= static_cast<size_t>(shape[k]);
        }
        t.device = static_cast<float*>(std::calloc(t.count, sizeof(float)));
        t.host = static_cast<float*>(std::calloc(t.count, sizeof(float)));
        t.binding = static_cast<int>(tensors_.size());
        (input ? st->in_names : st->out_names).push_back(t.name);
        st->shapes.push_back(shape);
        tensors_.push_back(t);
        bindings_.push_back(t.device);
    };
    for (size_t i = 0; i < st->session->GetInputCount(); ++i) add(true, i);
    for (size_t i = 0; i < st->session->GetOutputCount(); ++i) add(false, i);
    states()[this] = std::move(st);
}

TrtModel::~TrtModel() {
    for (Tensor& t : tensors_) {
        std::free(t.device);
        std::free(t.host);
    }
    states().erase(this);
}

Tensor& TrtModel::tensor(const std::string& name) {
    for (Tensor& t : tensors_)
        if (t.name == name) return t;
    std::fprintf(stderr, "[ort] %s has no tensor %s\n", path_.c_str(), name.c_str());
    std::exit(1);
}

void TrtModel::bind(const std::string& name, const float* device) {
    Tensor& t = tensor(name);
    bindings_[t.binding] = device ? const_cast<float*>(device) : t.device;
}

void TrtModel::upload(const std::string& name) {
    Tensor& t = tensor(name);
    std::memcpy(bindings_[t.binding], t.host, t.count * sizeof(float));
}

void TrtModel::download(const std::string& name) {
    Tensor& t = tensor(name);
    std::memcpy(t.host, bindings_[t.binding], t.count * sizeof(float));
}

void TrtModel::enqueue() {
    OrtState& st = *states()[this];
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> ins, outs;
    std::vector<const char*> in_n, out_n;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        Tensor& t = tensors_[i];
        Ort::Value v = Ort::Value::CreateTensor<float>(mi, static_cast<float*>(bindings_[i]), t.count,
                                                       st.shapes[i].data(), st.shapes[i].size());
        if (t.is_input) {
            ins.push_back(std::move(v));
            in_n.push_back(t.name.c_str());
        } else {
            outs.push_back(std::move(v));
            out_n.push_back(t.name.c_str());
        }
    }
    st.session->Run(Ort::RunOptions{nullptr}, in_n.data(), ins.data(), ins.size(), out_n.data(), outs.data(),
                    outs.size());
}

void TrtModel::sync() {}

void TrtModel::infer() {
    for (Tensor& t : tensors_)
        if (t.is_input) upload(t.name);
    enqueue();
    for (Tensor& t : tensors_)
        if (!t.is_input) download(t.name);
}

}  // namespace fht

// CUDA runtime stand-ins: "device" memory is ordinary memory here.
extern "C" {
cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::calloc(1, n);
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) {
    std::free(p);
    return cudaSuccess;
}
cudaError_t cudaMemset(void* p, int v, size_t n) {
    std::memset(p, v, n);
    return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) {
    std::memset(p, v, n);
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, enum cudaMemcpyKind, cudaStream_t) {
    std::memmove(d, s, n);
    return cudaSuccess;
}
cudaError_t cudaMallocHost(void** p, size_t n) {
    *p = std::calloc(1, n);
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void* p) {
    std::free(p);
    return cudaSuccess;
}
cudaError_t cudaStreamCreate(cudaStream_t* s) {
    *s = reinterpret_cast<cudaStream_t>(0x1);
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "stub"; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = reinterpret_cast<cudaEvent_t>(0x1);
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    *ms = 0.f;
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
}
