// Hand path for one frame: hand tracking + gesture rules + smoothing (as the Python engine).
#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

#include "gestures.hpp"
#include "hands.hpp"

namespace fht {

struct HandResult {
    int id = 0;
    std::string handedness;      // "Left" / "Right": the person's real hand
    float presence = 0;
    std::array<float, 4> box{};  // x1, y1, x2, y2
    Points21 landmarks;          // x, y in frame pixels
    std::string gesture;         // after smoothing ("" = none)
    std::string raw_gesture;     // this frame's label
    int fingers = 0;
};

class HandPipeline {
public:
    HandPipeline(const std::string& engines, const HandPolicy& pol, bool mirrored_input)
        : tracker_(engines, pol, mirrored_input) {}

    void process(const cv::Mat& bgr, std::vector<std::string>& events);
    const std::vector<HandResult>& hands() const { return hands_; }
    double ms_palm() const { return tracker_.ms_palm; }
    double ms_landmarks() const { return tracker_.ms_landmarks; }
    double ms_gesture = 0;
    void reset_timings() {
        tracker_.ms_palm = tracker_.ms_landmarks = 0;
        ms_gesture = 0;
    }

private:
    HandTracker tracker_;
    std::map<int, GestureSmoother> smoothers_;
    std::vector<HandResult> hands_;
    long index_ = 0;
};

}  // namespace fht
