#include "face_pipeline.hpp"

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
    if (tracker_.wants_detection(index_)) {
        dets_ = detector_.detect(bgr, params_);
        ms_detect += detector_.ms_pre + detector_.ms_infer + detector_.ms_post;
    }
    if (tracker_.has_tracks() || !dets_.empty()) {
        tam_.encode(bgr);
        tracker_.update(dets_, bgr.cols, bgr.rows, now, events);
    }
    ++index_;
}

}  // namespace fht
