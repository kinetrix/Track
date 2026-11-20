// Inside of DepthFrame (only with librealsense).
#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>

#include "depth.hpp"

namespace fht {

struct DepthFrame::Impl {
    cv::Mat z16;  // depth image, CV_16U, units of `scale` metres (0 = no reading)
    float scale = 0.001f;
    rs2_intrinsics depth_intrin{}, color_intrin{};
    rs2_extrinsics color_to_depth{}, depth_to_color{};
};

}  // namespace fht
