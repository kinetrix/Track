#include "face_embedder.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace fht {

namespace {
// ArcFace / SFace template for a 112x112 crop (eyes, nose, mouth corners), as OpenCV's FaceRecognizerSF.
const cv::Point2f kTemplate[5] = {{38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
                                  {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
}  // namespace

float similarity(const Embedding& a, const Embedding& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

cv::Mat similarity_transform(const cv::Point2f* src, const cv::Point2f* dst, int n) {
    double msx = 0, msy = 0, mdx = 0, mdy = 0;
    for (int i = 0; i < n; ++i) {
        msx += src[i].x;
        msy += src[i].y;
        mdx += dst[i].x;
        mdy += dst[i].y;
    }
    msx /= n;
    msy /= n;
    mdx /= n;
    mdy /= n;
    double a = 0, b = 0, var = 0;
    for (int i = 0; i < n; ++i) {
        const double xs = src[i].x - msx, ys = src[i].y - msy, xd = dst[i].x - mdx, yd = dst[i].y - mdy;
        a += xs * xd + ys * yd;
        b += xs * yd - ys * xd;
        var += xs * xs + ys * ys;
    }
    const double c = var > 0 ? a / var : 1.0, s = var > 0 ? b / var : 0.0;  // scale * cos, scale * sin
    cv::Mat M = (cv::Mat_<double>(2, 3) << c, -s, mdx - (c * msx - s * msy), s, c, mdy - (s * msx + c * msy));
    return M;
}

FaceEmbedder::FaceEmbedder(const std::string& engine_path) : model_(engine_path) {
    const Tensor& in = model_.tensor("data");
    const Tensor& out = model_.tensor("fc1");
    if (in.count != 3 * 112 * 112 || out.count != 128) {
        std::fprintf(stderr, "[sface] unexpected tensor shapes in %s\n", engine_path.c_str());
        std::exit(1);
    }
}

cv::Mat FaceEmbedder::align(const cv::Mat& bgr, const FaceDet& face) const {
    const cv::Mat M = similarity_transform(face.landmarks.data(), kTemplate, 5);
    cv::Mat crop;
    cv::warpAffine(bgr, crop, M, cv::Size(112, 112), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return crop;
}

Embedding FaceEmbedder::embed(const cv::Mat& bgr, const FaceDet& face) {
    const auto t = std::chrono::steady_clock::now();
    cv::Mat crop = align(bgr, face), rgb, f;
    cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f, CV_32F);  // 0..255; normalisation is inside the model
    Tensor& in = model_.tensor("data");
    std::vector<cv::Mat> planes;
    for (int c = 0; c < 3; ++c) planes.emplace_back(112, 112, CV_32F, in.host + c * 112 * 112);
    cv::split(f, planes);
    model_.infer();
    const float* o = model_.tensor("fc1").host;
    Embedding e;
    double n = 0;
    for (int i = 0; i < 128; ++i) n += static_cast<double>(o[i]) * o[i];
    const float inv = n > 0 ? static_cast<float>(1.0 / std::sqrt(n)) : 0.f;
    for (int i = 0; i < 128; ++i) e[i] = o[i] * inv;
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    ++calls;
    return e;
}

}  // namespace fht
