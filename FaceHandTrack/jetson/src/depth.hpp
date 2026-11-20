// Depth from the head RealSense (D435i), read at colour-image points without aligning whole
// frames: each colour pixel is mapped to the depth image with the camera's calibration
// (librealsense rs2_project_color_pixel_to_depth_pixel), so only the few points the rules need
// cost anything. Built without librealsense, every depth is unknown (-1).
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace fht {

struct DepthFrame {
    struct Impl;
    std::shared_ptr<const Impl> impl;  // empty: no depth for this frame
    bool valid() const { return static_cast<bool>(impl); }
    // Metres at these colour-image points: the q-quantile (0 = nearest) of the valid depth samples
    // in a (2r+1)^2 window around each mapped point; -1 if there are none.
    float at(const std::vector<cv::Point2f>& colour_points, float q = 0.5f, int r = 2) const;
};

// Distance to a face: the median depth over its inner box.
float face_depth(const DepthFrame& d, const cv::Rect2f& box);
// Distance to a hand: palm points (wrist, knuckles), biased to the near side, since samples that
// miss the hand land on whatever is behind it.
float hand_depth(const DepthFrame& d, const std::vector<cv::Point2f>& landmarks);
// The arm check: along the line from a hand's wrist to a person's shoulder (estimated from the face
// box, on the hand's side), the share of depth readings that lie on a body at the depth expected
// between the hand and that face (within `tolerance` metres). An arm reaching to its own shoulder is
// there; the line to someone else's shoulder crosses empty background or a nearer body.
// -1 when there are too few readings.
float arm_support(const DepthFrame& d, const cv::Point2f& wrist, float hand_m, const cv::Rect2f& face, float face_m,
                  float tolerance = 0.3f);

}  // namespace fht
