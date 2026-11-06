// What the rest of the app needs from a face path, whichever tracker is behind it.
#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "face_detector.hpp"

namespace fht {

struct FaceOut {
    int id = 0;
    cv::Rect2f box;        // frame pixels
    bool visible = true;   // false while a track is briefly missing
    float quality = 0;     // detector score (fast tracker) or EdgeTAM's IoU estimate
};

class FaceSource {
public:
    virtual ~FaceSource() = default;
    // now: seconds since start (track timeouts are in seconds).
    virtual void process(const cv::Mat& bgr, double now, std::vector<std::string>& events) = 0;
    virtual std::vector<FaceOut> faces() const = 0;
    virtual const std::vector<FaceDet>& detections() const = 0;  // this frame's (may be empty)
    // Per-stage times averaged over `frames` frames (for the status line), then resets the sums.
    virtual std::string timing(int frames) = 0;
};

}  // namespace fht
