// SFace face embeddings (OpenCV Zoo, Apache-2.0): a 128-number "fingerprint" per face, used to
// give a person their old ID back after they were hidden or left the picture.
// The face is aligned from YuNet's 5 landmarks to the standard 112x112 template first.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>

#include "face_detector.hpp"
#include "trt_model.hpp"

namespace fht {

using Embedding = std::array<float, 128>;  // unit length

// Cosine similarity of two unit-length embeddings (1 = same, about 0 = unrelated).
float similarity(const Embedding& a, const Embedding& b);

// Similarity transform (rotation + uniform scale + shift) mapping src to dst, least squares.
cv::Mat similarity_transform(const cv::Point2f* src, const cv::Point2f* dst, int n);

class FaceEmbedder {
public:
    explicit FaceEmbedder(const std::string& engine_path);
    Embedding embed(const cv::Mat& bgr, const FaceDet& face);
    cv::Mat align(const cv::Mat& bgr, const FaceDet& face) const;  // 112x112 BGR crop

    double ms = 0;  // summed since the caller resets it
    int calls = 0;

private:
    TrtModel model_;
};

}  // namespace fht
