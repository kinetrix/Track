#include "depth.hpp"

#include <algorithm>
#include <cmath>

#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "depth_impl.hpp"
#endif

namespace fht {

#ifdef HAVE_REALSENSE
float DepthFrame::at(const std::vector<cv::Point2f>& colour_points, float q, int r) const {
    if (!impl || colour_points.empty()) return -1;
    const Impl& d = *impl;
    const uint16_t* data = reinterpret_cast<const uint16_t*>(d.z16.data);
    std::vector<float> m;
    for (const cv::Point2f& p : colour_points) {
        float from[2] = {p.x, p.y}, to[2] = {-1, -1};
        rs2_project_color_pixel_to_depth_pixel(to, data, d.scale, 0.1f, 6.0f, &d.depth_intrin, &d.color_intrin,
                                               &d.color_to_depth, &d.depth_to_color, from);
        const int cx = static_cast<int>(to[0] + 0.5f), cy = static_cast<int>(to[1] + 0.5f);
        if (to[0] < 0 || to[1] < 0) continue;
        for (int y = std::max(cy - r, 0); y <= std::min(cy + r, d.z16.rows - 1); ++y)
            for (int x = std::max(cx - r, 0); x <= std::min(cx + r, d.z16.cols - 1); ++x) {
                const uint16_t z = d.z16.at<uint16_t>(y, x);
                if (z) m.push_back(z * d.scale);
            }
    }
    if (m.empty()) return -1;
    const size_t k = std::min(m.size() - 1, static_cast<size_t>(q * (m.size() - 1) + 0.5f));
    std::nth_element(m.begin(), m.begin() + k, m.end());
    return m[k];
}
#else
float DepthFrame::at(const std::vector<cv::Point2f>&, float, int) const { return -1; }
#endif

float face_depth(const DepthFrame& d, const cv::Rect2f& box) {
    if (!d.valid()) return -1;
    std::vector<cv::Point2f> pts;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            pts.emplace_back(box.x + box.width * (0.3f + 0.2f * j), box.y + box.height * (0.3f + 0.2f * i));
    return d.at(pts, 0.5f, 2);
}

float hand_depth(const DepthFrame& d, const std::vector<cv::Point2f>& lm) {
    if (!d.valid() || lm.size() != 21) return -1;
    std::vector<cv::Point2f> pts;
    for (int j : {0, 1, 5, 9, 13, 17}) pts.push_back(lm[j]);
    pts.push_back((lm[0] + lm[9]) * 0.5f);  // palm centre
    return d.at(pts, 0.3f, 1);
}

float arm_support(const DepthFrame& d, const cv::Point2f& wrist, float hand_m, const cv::Rect2f& face, float face_m,
                  float tolerance) {
    if (!d.valid() || hand_m <= 0 || face_m <= 0) return -1;
    const cv::Point2f fc(face.x + face.width / 2, face.y + face.height / 2);
    const float side = wrist.x < fc.x ? -1.f : 1.f;
    const cv::Point2f shoulder(fc.x + side * 0.8f * face.width, face.y + face.height * 1.8f);
    int valid = 0, on_body = 0;
    for (int i = 1; i <= 8; ++i) {
        const float t = 0.1f * i + 0.05f;  // 0.15 .. 0.85: skip the hand and the shoulder themselves
        const float m = d.at({wrist + t * (shoulder - wrist)}, 0.5f, 1);
        if (m <= 0) continue;
        ++valid;
        on_body += std::fabs(m - (hand_m + t * (face_m - hand_m))) <= tolerance;
    }
    return valid < 4 ? -1.f : static_cast<float>(on_body) / valid;
}

}  // namespace fht
