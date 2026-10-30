#include "edgetam.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fht {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int best_multimask(const float* iou) {  // index of the best of masks 1..3
    int k = 1;
    for (int i = 2; i < 4; ++i)
        if (iou[i] > iou[k]) k = i;
    return k;
}

double mask_iou(const cv::Mat& a, const cv::Mat& b) {
    long inter = 0, uni = 0;
    const float* pa = a.ptr<float>();
    const float* pb = b.ptr<float>();
    for (size_t i = 0; i < a.total(); ++i) {
        bool x = pa[i] > 0, y = pb[i] > 0;
        inter += x && y;
        uni += x || y;
    }
    return uni ? static_cast<double>(inter) / uni : 1.0;
}

}  // namespace

EdgeTAM::EdgeTAM(const std::string& engines, const std::string& attn_precision, bool compare_attn) {
    if (attn_precision != "fp16" && attn_precision != "fp32") {
        std::fprintf(stderr, "[edgetam] --attn must be fp16 or fp32\n");
        std::exit(1);
    }
    CUDA_CHECK(cudaStreamCreate(&stream_));
    auto load = [&](const std::string& name) { return std::make_unique<TrtModel>(engines + "/" + name, stream_); };
    enc_ = load("edgetam_image_encoder.fp16.engine");
    att_ = load("edgetam_memory_attention." + attn_precision + ".engine");
    if (compare_attn && attn_precision == "fp16") att32_ = load("edgetam_memory_attention.fp32.engine");
    dec1_ = load("edgetam_mask_decoder_n1.fp16.engine");
    dec2_ = load("edgetam_mask_decoder_n2.fp16.engine");
    men_ = load("edgetam_memory_encoder.fp16.engine");

    const Tensor& img = enc_->tensor("image");
    cfg_.S = img.dims.d[2];
    const Tensor& masks = dec1_->tensor("masks");
    cfg_.M = masks.dims.d[2];
    const Tensor& mem = att_->tensor("memory");
    const Tensor& valid = att_->tensor("slot_valid");
    const int K = (cfg_.C + cfg_.R) * cfg_.TM + cfg_.P * cfg_.TP;
    if (mem.dims.d[1] != K || mem.dims.d[2] != cfg_.D || valid.dims.d[1] != cfg_.C + cfg_.R + cfg_.P) {
        std::fprintf(stderr, "[edgetam] memory attention engine does not match C=%d R=%d P=%d\n", cfg_.C, cfg_.R, cfg_.P);
        std::exit(1);
    }
    std::printf("[edgetam] input %d, mask %d, memory attention %s%s\n", cfg_.S, cfg_.M, attn_precision.c_str(),
                att32_ ? " (comparing with fp32)" : "");
}

EdgeTAM::~EdgeTAM() {
    enc_.reset();
    att_.reset();
    att32_.reset();
    dec1_.reset();
    dec2_.reset();
    men_.reset();
    if (stream_) cudaStreamDestroy(stream_);
}

std::unique_ptr<TrackMemory> EdgeTAM::new_memory() const {
    auto m = std::make_unique<TrackMemory>();
    const size_t tok = static_cast<size_t>(cfg_.TM) * cfg_.D, ptr = static_cast<size_t>(cfg_.TP) * cfg_.D;
    for (int i = 0; i < cfg_.C; ++i) {
        m->cond_mem.emplace_back(tok);
        m->cond_ptr.emplace_back(ptr);
    }
    for (int i = 0; i < cfg_.R; ++i) m->recent_mem.emplace_back(tok);
    for (int i = 0; i < cfg_.P - 1; ++i) m->recent_ptr.emplace_back(ptr);
    return m;
}

void EdgeTAM::encode(const cv::Mat& bgr) {
    auto t = Clock::now();
    const int S = cfg_.S;
    cv::Mat small, rgb, f;
    cv::resize(bgr, small, cv::Size(S, S), 0, 0, cv::INTER_LINEAR);  // whole frame squashed to S x S
    cv::cvtColor(small, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f, CV_32F);  // 0..255; normalisation is inside the graph
    Tensor& img = enc_->tensor("image");
    std::vector<cv::Mat> planes;
    for (int c = 0; c < 3; ++c) planes.emplace_back(S, S, CV_32F, img.host + static_cast<size_t>(c) * S * S);
    cv::split(f, planes);
    enc_->upload("image");
    enc_->enqueue();
    enc_->sync();
    ms_enc += ms_since(t);
}

const float* EdgeTAM::attend(TrtModel& att, const TrackMemory& mem) {
    auto t = Clock::now();
    const int C = cfg_.C, R = cfg_.R, P = cfg_.P;
    const size_t tok = static_cast<size_t>(cfg_.TM) * cfg_.D, ptr = static_cast<size_t>(cfg_.TP) * cfg_.D;
    Tensor& m = att.tensor("memory");
    Tensor& v = att.tensor("slot_valid");
    CUDA_CHECK(cudaMemsetAsync(m.device, 0, m.count * sizeof(float), stream_));
    std::fill(v.host, v.host + v.count, 0.f);
    auto put = [&](const DeviceBuffer& src, size_t offset, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(m.device + offset, src.get(), n * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    };
    for (int i = 0; i < std::min(mem.n_cond, C); ++i) {
        put(mem.cond_mem[i], i * tok, tok);
        v.host[i] = 1.f;
    }
    for (int i = 0; i < std::min(mem.n_recent, R); ++i) {
        put(mem.recent_mem[i], (C + i) * tok, tok);
        v.host[C + i] = 1.f;
    }
    // pointers: anchors first, then the newest recent ones
    const size_t base = (C + R) * tok;
    int np = 0;
    for (int i = 0; i < mem.n_cond && np < P; ++i, ++np) {
        put(mem.cond_ptr[i], base + np * ptr, ptr);
        v.host[C + R + np] = 1.f;
    }
    for (int i = 0; i < mem.n_recent_ptr && np < P; ++i, ++np) {
        put(mem.recent_ptr[i], base + np * ptr, ptr);
        v.host[C + R + np] = 1.f;
    }
    att.upload("slot_valid");
    att.bind("tgt0", enc_->tensor("tgt0").device);
    att.enqueue();
    att.sync();
    ms_attn += ms_since(t);
    return att.tensor("pix_feat").device;
}

TrtModel& EdgeTAM::decode(const float* pix_feat, const float* coords, const float* labels, int n_points) {
    auto t = Clock::now();
    TrtModel& dec = n_points == 1 ? *dec1_ : *dec2_;
    dec.bind("pix_feat", pix_feat);
    dec.bind("hr0", enc_->tensor("hr0").device);
    dec.bind("hr1", enc_->tensor("hr1").device);
    std::memcpy(dec.tensor("point_coords").host, coords, 2 * n_points * sizeof(float));
    std::memcpy(dec.tensor("point_labels").host, labels, n_points * sizeof(float));
    dec.upload("point_coords");
    dec.upload("point_labels");
    dec.enqueue();
    dec.download("masks");
    dec.download("iou");
    dec.download("obj_score");
    dec.sync();
    ms_dec += ms_since(t);
    return dec;
}

void EdgeTAM::memorize(TrtModel& dec, int k, float binarize, float* dst_tokens) {
    auto t = Clock::now();
    const size_t mm = static_cast<size_t>(cfg_.M) * cfg_.M;
    men_->bind("feat", enc_->tensor("feat").device);
    men_->bind("mask_logits", dec.tensor("masks").device + k * mm);
    men_->tensor("binarize").host[0] = binarize;
    men_->upload("binarize");
    men_->bind("tokens", dst_tokens);
    men_->enqueue();
    men_->sync();
    ms_menc += ms_since(t);
}

float EdgeTAM::stability(const float* mask) const {
    const float d = cfg_.stability_delta;
    long inside = 0, uni = 0;
    for (int i = 0; i < cfg_.M * cfg_.M; ++i) {
        inside += mask[i] > d;
        uni += mask[i] > -d;
    }
    return uni > 0 ? static_cast<float>(inside) / uni : 1.f;
}

TamStep EdgeTAM::start(const cv::Vec4f& box_S, TrackMemory& mem, bool is_new) {
    const float coords[4] = {box_S[0], box_S[1], box_S[2], box_S[3]};
    const float labels[2] = {2.f, 3.f};  // box corners
    const float* pix = is_new ? enc_->tensor("feat_init").device : attend(*att_, mem);
    TrtModel& dec = decode(pix, coords, labels, 2);
    const size_t mm = static_cast<size_t>(cfg_.M) * cfg_.M;
    const float* masks = dec.tensor("masks").host;
    const float* iou = dec.tensor("iou").host;
    // single-mask output, falling back to the best multimask output when unstable
    const int k = stability(masks) >= cfg_.stability_thresh ? 0 : best_multimask(iou);

    int slot;
    if (mem.n_cond < cfg_.C) {
        slot = mem.n_cond++;
    } else {  // keep the first anchor + the newest ones
        std::rotate(mem.cond_mem.begin() + 1, mem.cond_mem.begin() + 2, mem.cond_mem.end());
        std::rotate(mem.cond_ptr.begin() + 1, mem.cond_ptr.begin() + 2, mem.cond_ptr.end());
        slot = cfg_.C - 1;
    }
    TamStep out;
    out.mask = cv::Mat(cfg_.M, cfg_.M, CV_32F, const_cast<float*>(masks + k * mm)).clone();
    out.iou_pred = iou[k];
    out.score = dec.tensor("obj_score").host[0];
    memorize(dec, k, 1.f, mem.cond_mem[slot].get());
    // the single-mask token always gives the pointer
    const size_t ptr = static_cast<size_t>(cfg_.TP) * cfg_.D;
    CUDA_CHECK(cudaMemcpyAsync(mem.cond_ptr[slot].get(), dec.tensor("ptr").device, ptr * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));
    return out;
}

TamStep EdgeTAM::track(TrackMemory& mem) {
    const float coords[2] = {0.f, 0.f};
    const float labels[1] = {-1.f};  // no prompt
    const size_t mm = static_cast<size_t>(cfg_.M) * cfg_.M;

    cv::Mat mask32;
    if (att32_) {  // same step with the fp32 attention, for the accuracy check only
        const float* pix32 = attend(*att32_, mem);
        TrtModel& d = decode(pix32, coords, labels, 1);
        const int k32 = best_multimask(d.tensor("iou").host);
        mask32 = cv::Mat(cfg_.M, cfg_.M, CV_32F, d.tensor("masks").host + k32 * mm).clone();
    }

    const float* pix = attend(*att_, mem);
    TrtModel& dec = decode(pix, coords, labels, 1);
    const float* masks = dec.tensor("masks").host;
    const float* iou = dec.tensor("iou").host;
    const int k = best_multimask(iou);  // multimask for tracking

    TamStep out;
    out.mask = cv::Mat(cfg_.M, cfg_.M, CV_32F, const_cast<float*>(masks + k * mm)).clone();
    out.iou_pred = iou[k];
    out.score = dec.tensor("obj_score").host[0];
    if (att32_) {
        const double v = mask_iou(out.mask, mask32);
        ++compare.n;
        compare.iou_sum += v;
        compare.iou_min = std::min(compare.iou_min, v);
    }

    // newest first: the oldest slot's buffer moves to the front and is overwritten
    std::rotate(mem.recent_mem.begin(), mem.recent_mem.end() - 1, mem.recent_mem.end());
    memorize(dec, k, 0.f, mem.recent_mem[0].get());
    mem.n_recent = std::min(mem.n_recent + 1, cfg_.R);
    std::rotate(mem.recent_ptr.begin(), mem.recent_ptr.end() - 1, mem.recent_ptr.end());
    const size_t ptr = static_cast<size_t>(cfg_.TP) * cfg_.D;
    CUDA_CHECK(cudaMemcpyAsync(mem.recent_ptr[0].get(), dec.tensor("ptr").device + k * ptr, ptr * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));
    mem.n_recent_ptr = std::min(mem.n_recent_ptr + 1, cfg_.P - 1);
    return out;
}

}  // namespace fht
