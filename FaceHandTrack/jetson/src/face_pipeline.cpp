#include "face_pipeline.hpp"

#include <cstdio>

namespace fht {

FacePipeline::FacePipeline(const std::string& engines, const std::string& attn_precision, bool compare_attn,
                           const FacePolicy& policy)
    : detector_(engines + "/yunet_640x384.fp16.engine"),
      tam_(engines, attn_precision, compare_attn),
      tracker_(tam_, policy) {
    params_.min_score = policy.det_score;
    params_.nms_iou = policy.nms_iou;
    params_.min_px = policy.min_face_px;
}

void FacePipeline::process(const cv::Mat& bgr, double now, std::vector<std::string>& events) {
    dets_.clear();
    frame_.upload(bgr, tam_.stream());
    const bool encoded = tracker_.has_tracks();
    if (encoded) tam_.encode(frame_);  // queued; runs while YuNet runs
    if (tracker_.wants_detection(index_)) {
        dets_ = detector_.detect(frame_, params_);
        ms_detect += detector_.ms;
    }
    if (!encoded && !dets_.empty()) tam_.encode(frame_);  // faces just appeared
    if (encoded || !dets_.empty()) {
        tracker_.update(dets_, bgr.cols, bgr.rows, now, events);
        tam_.finish();
    }
    ++index_;
}

std::vector<FaceOut> FacePipeline::faces() const {
    std::vector<FaceOut> out;
    for (const auto& t : tracker_.tracks()) out.push_back({t->id, t->box, t->visible, t->iou_pred});
    return out;
}

std::string FacePipeline::timing(int frames) {
    char s[192];
    int n = std::snprintf(s, sizeof(s), "detect %.1f enc %.1f attn %.1f dec %.1f menc %.1f", ms_detect / frames,
                          tam_.ms_enc / frames, tam_.ms_attn / frames, tam_.ms_dec / frames, tam_.ms_menc / frames);
    if (tam_.compare.n > 0)
        std::snprintf(s + n, sizeof(s) - n, " | fp16 vs fp32 mask IoU mean %.3f min %.3f (%ld)",
                      tam_.compare.iou_sum / tam_.compare.n, tam_.compare.iou_min, tam_.compare.n);
    reset_timings();
    return s;
}

}  // namespace fht
