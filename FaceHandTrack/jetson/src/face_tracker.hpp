// Face track manager: detector boxes start / re-anchor EdgeTAM tracks, EdgeTAM keeps the IDs.
// Port of onnx/facetrack/faces.py (FacePolicy + FaceTracker), same rules and thresholds.
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "edgetam.hpp"
#include "face_detector.hpp"

namespace fht {

struct FacePolicy {
    float det_score = 0.80f;          // detector confidence to start a track
    float nms_iou = 0.30f;
    int det_interval = 3;             // detector every N frames while faces are tracked (every frame when none)
    float match_iou = 0.30f;          // detection <-> track association
    float reanchor_iou = 0.50f;       // matched but overlap below this -> re-prompt with the detection box
    double reanchor_min_gap = 0.5;    // seconds between re-anchors of one track
    float min_track_iou_pred = 0.60f; // EdgeTAM's quality estimate below this -> re-anchor on next detection
    double max_missing = 1.0;         // seconds with no mask -> drop
    float confident_iou_pred = 0.80f;
    double max_unconfirmed = 5.0;     // seconds without a detector match (not confident) -> drop
    double max_unconfirmed_confident = 30.0;  // ... while confident (detector misses tilted / profile faces)
    float duplicate_iou = 0.60f;      // two tracks overlapping this much -> drop the younger
    int max_tracks = 3;
    float min_face_px = 24.f;         // in detector input pixels
};

struct FaceTrack {
    int id = 0;
    std::unique_ptr<TrackMemory> mem;
    cv::Rect2f box;    // frame pixels (from the mask)
    cv::Mat mask;      // M x M logits
    long age = 0;
    double last_seen = 0, last_confirmed = 0, last_anchor = 0;
    float iou_pred = 1.f;
    bool visible = true;
};

class FaceTracker {
public:
    FaceTracker(EdgeTAM& tam, const FacePolicy& pol) : tam_(tam), pol_(pol) {}

    bool wants_detection(long frame_index) const {
        return tracks_.empty() || frame_index % pol_.det_interval == 0;
    }
    bool has_tracks() const { return !tracks_.empty(); }
    // dets: this frame's detections (may be empty on frames without detection). W, H: frame size.
    void update(const std::vector<FaceDet>& dets, int W, int H, double now, std::vector<std::string>& events);
    const std::vector<std::unique_ptr<FaceTrack>>& tracks() const { return tracks_; }
    const FacePolicy& policy() const { return pol_; }

private:
    void update_box(FaceTrack& t, int W, int H, float score, double now);

    EdgeTAM& tam_;
    FacePolicy pol_;
    std::vector<std::unique_ptr<FaceTrack>> tracks_;
    int next_id_ = 1;
};

}  // namespace fht
