// Hand detection, 21-point hand landmarks and hand tracking (MediaPipe palm detector + hand
// landmark model, Apache-2.0, as published in OpenCV Zoo). Port of onnx/facetrack/hands.py:
// same crops, rotations, thresholds and tracking rules. Crops are made on the CPU (they are
// small); both models run as TensorRT engines on their own CUDA streams.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "trt_model.hpp"

namespace fht {

using Palm = std::array<float, 18>;  // box x1, y1, x2, y2, then 7 keypoints (x, y), frame pixels
using Points21 = std::array<cv::Point3f, 21>;

struct PalmDet {
    Palm palm;
    float score = 0;
};

struct LandmarkResult {
    Points21 landmarks;  // x, y in frame pixels, z relative depth (scaled like x, y)
    Points21 world;      // metric, centred on the hand
    float right_prob = 0, presence = 0;
};

class PalmDetector {
public:
    explicit PalmDetector(const std::string& engine_path, float score = 0.5f, float nms_iou = 0.3f);
    std::vector<PalmDet> detect(const cv::Mat& bgr);

private:
    TrtModel model_;
    std::vector<cv::Point2f> anchors_;
    float score_, nms_iou_;
};

class HandLandmarker {
public:
    explicit HandLandmarker(const std::string& engine_path);
    // palm: from the palm detector or from the previous frame's landmarks.
    std::optional<LandmarkResult> run(const cv::Mat& bgr, const Palm& palm);

private:
    TrtModel model_;
};

std::array<float, 4> hand_box(const Points21& lm);  // MediaPipe-style box around the landmarks
Palm palm_from_landmarks(const Points21& lm);       // palm-detector-style input from landmarks
float iou4(const float* a, const float* b);         // boxes as x1, y1, x2, y2

struct HandPolicy {
    int max_hands = 2;
    int palm_interval = 5;           // palm detector every N frames while hands are tracked (every frame when none)
    float min_presence = 0.6f;       // landmark model's hand-presence score to keep a hand
    float min_presence_new = 0.8f;   // ... to start a new hand
    float overlap_iou = 0.3f;        // a detected palm overlapping a tracked hand's palm this much is that hand
    float duplicate_iou = 0.5f;      // two tracked hands overlapping this much -> drop the younger
};

struct HandTrack {
    int id = 0;
    LandmarkResult r;
    long age = 0;
};

class HandTracker {
public:
    // mirrored_input: frames are mirrored like a selfie view (only swaps Left / Right).
    HandTracker(const std::string& engines, const HandPolicy& pol, bool mirrored_input);
    void update(const cv::Mat& bgr, long frame_index, std::vector<std::string>& events);
    const std::vector<HandTrack>& tracks() const { return tracks_; }
    std::string handedness(const HandTrack& t) const;  // the person's real hand: "Left" / "Right"

    double ms_palm = 0, ms_landmarks = 0;  // summed since the caller resets them

private:
    bool same_hand(const Palm& p, const HandTrack& t) const;

    PalmDetector palm_;
    HandLandmarker lmk_;
    HandPolicy pol_;
    bool mirrored_;
    std::vector<HandTrack> tracks_;
    int next_id_ = 1;
};

}  // namespace fht
