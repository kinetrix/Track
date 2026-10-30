// Face path for one frame: YuNet (every det_interval-th frame while faces are tracked, every
// frame when none) -> EdgeTAM image encoder -> track manager. Same rules as the Python engine.
// The frame goes to the GPU once; YuNet (its own CUDA stream) runs alongside the encoder.
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "edgetam.hpp"
#include "face_detector.hpp"
#include "face_tracker.hpp"

namespace fht {

class FacePipeline {
public:
    FacePipeline(const std::string& engines, const std::string& attn_precision, bool compare_attn,
                 const FacePolicy& policy);

    // now: seconds (track timeouts are in seconds, so they behave the same at any frame rate).
    void process(const cv::Mat& bgr, double now, std::vector<std::string>& events);

    const std::vector<std::unique_ptr<FaceTrack>>& tracks() const { return tracker_.tracks(); }
    const std::vector<FaceDet>& detections() const { return dets_; }  // this frame's (may be empty)
    EdgeTAM& tam() { return tam_; }
    double ms_detect = 0;  // summed since reset_timings()
    void reset_timings() {
        ms_detect = 0;
        tam_.reset_timings();
    }

private:
    FaceDetector detector_;
    EdgeTAM tam_;
    FaceTracker tracker_;
    GpuFrame frame_;
    DetectorParams params_;
    std::vector<FaceDet> dets_;
    long index_ = 0;
};

}  // namespace fht
