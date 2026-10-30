// EdgeTAM streaming face tracker on TensorRT: one image encoding per frame, then per face
// memory attention -> mask decoder -> memory encoder. Every tensor that feeds another model
// stays on the GPU; only the 128x128 masks and a few scores come back to the CPU.
// Port of onnx/facetrack/faces.py (EdgeTAM class), same memory layout and rules.
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "preprocess.hpp"
#include "trt_model.hpp"

namespace fht {

struct EdgeTamConfig {
    int S = 512;   // model input size (whole frame squashed to S x S)
    int M = 128;   // mask size
    int C = 2;     // anchor ("conditioning") memory slots
    int R = 6;     // recent memory slots
    int P = 16;    // object-pointer slots
    int TM = 512;  // tokens per memory slot
    int TP = 4;    // tokens per pointer (256 = 4 x 64)
    int D = 64;    // memory token width
    float stability_delta = 0.05f;
    float stability_thresh = 0.98f;
};

// Per-face memory bank, kept on the GPU in fixed, preallocated slots.
struct TrackMemory {
    std::vector<DeviceBuffer> cond_mem, cond_ptr;  // anchors, oldest first (n_cond used)
    std::vector<DeviceBuffer> recent_mem;          // newest first (n_recent used)
    std::vector<DeviceBuffer> recent_ptr;          // newest first (n_recent_ptr used)
    int n_cond = 0, n_recent = 0, n_recent_ptr = 0;
};

struct TamStep {
    cv::Mat mask;        // M x M float logits (> 0 = face)
    float iou_pred = 0;  // EdgeTAM's own quality estimate
    float score = 0;     // object score (> 0 = present)
};

struct AttnCompare {     // fp16 vs fp32 memory attention, measured on tracking frames
    long n = 0;
    double iou_sum = 0;
    double iou_min = 1;
};

class EdgeTAM {
public:
    // engines: folder with <model>.fp16.engine files; attn_precision: "fp16" or "fp32";
    // compare_attn: also run the fp32 attention and measure how far fp16 masks differ.
    EdgeTAM(const std::string& engines, const std::string& attn_precision, bool compare_attn);
    ~EdgeTAM();

    std::unique_ptr<TrackMemory> new_memory() const;
    void encode(const GpuFrame& frame);  // once per frame, before start / track (asynchronous)

    // Start a track (is_new) or re-anchor one from a detector box (x1, y1, x2, y2 in 0..S).
    TamStep start(const cv::Vec4f& box_S, TrackMemory& mem, bool is_new);
    TamStep track(TrackMemory& mem);
    // End of the frame: waits for the remaining GPU work and adds this frame's stage times.
    void finish();

    const EdgeTamConfig& cfg() const { return cfg_; }
    cudaStream_t stream() const { return stream_; }
    // GPU time per stage, summed since reset_timings()
    double ms_enc = 0, ms_attn = 0, ms_dec = 0, ms_menc = 0;
    void reset_timings() { ms_enc = ms_attn = ms_dec = ms_menc = 0; }
    AttnCompare compare;

private:
    const float* attend(TrtModel& att, const TrackMemory& mem);
    TrtModel& decode(const float* pix_feat, const float* coords, const float* labels, int n_points);
    void memorize(TrtModel& dec, int k, float binarize, float* dst_tokens);
    float stability(const float* mask) const;

    EdgeTamConfig cfg_;
    cudaStream_t stream_ = nullptr;
    GpuTimer timer_{4};
    std::unique_ptr<TrtModel> enc_, att_, att32_, dec1_, dec2_, men_;
};

}  // namespace fht
