#include "face_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace fht {

namespace {
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

FaceDetector::FaceDetector(const std::string& engine_path) : model_(engine_path) {
    in_ = &model_.tensor("image");    // (1, 3, 384, 640), BGR 0..255
    out_ = &model_.tensor("faces");   // (1, N, 15): score, x, y, w, h, 5 x (lx, ly)
    if (in_->dims.nbDims != 4 || in_->dims.d[1] != 3 || out_->dims.nbDims != 3 || out_->dims.d[2] != 15) {
        std::fprintf(stderr, "[yunet] unexpected tensor shapes in %s\n", engine_path.c_str());
        std::exit(1);
    }
    H_ = in_->dims.d[2];
    W_ = in_->dims.d[3];
}

std::vector<FaceDet> FaceDetector::detect(const GpuFrame& frame, const DetectorParams& p) {
    auto t = std::chrono::steady_clock::now();
    // Letterbox into the top-left corner of the input, keeping the aspect ratio (BGR, 0..255).
    const float s = std::min(static_cast<float>(W_) / frame.width(), static_cast<float>(H_) / frame.height());
    const int rw = static_cast<int>(std::lround(frame.width() * s)), rh = static_cast<int>(std::lround(frame.height() * s));
    resize_to_planar(frame, in_->device, W_, H_, rw, rh, false, model_.stream());
    model_.enqueue();
    model_.download("faces");
    model_.sync();

    const int n = out_->dims.d[1];
    const float* o = out_->host;
    std::vector<FaceDet> cand;
    for (int i = 0; i < n; ++i, o += 15) {
        if (o[0] < p.min_score || std::min(o[3], o[4]) < p.min_px) continue;
        FaceDet d;
        d.score = o[0];
        d.box = cv::Rect2f(o[1] / s, o[2] / s, o[3] / s, o[4] / s);
        for (int k = 0; k < 5; ++k) d.landmarks[k] = cv::Point2f(o[5 + 2 * k] / s, o[6 + 2 * k] / s);
        cand.push_back(d);
    }
    std::sort(cand.begin(), cand.end(), [](const FaceDet& a, const FaceDet& b) { return a.score > b.score; });
    std::vector<FaceDet> keep;
    for (const FaceDet& d : cand) {
        bool overlaps = false;
        for (const FaceDet& k : keep)
            if (iou(d.box, k.box) > p.nms_iou) {
                overlaps = true;
                break;
            }
        if (!overlaps) keep.push_back(d);
    }
    ms = ms_since(t);
    return keep;
}

}  // namespace fht
