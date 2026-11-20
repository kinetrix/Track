#include "depth.hpp"

#include <algorithm>

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

}  // namespace fht
