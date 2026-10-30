// YuNet face detector (TensorRT). The engine already decodes boxes; this class
// letterboxes the frame into the 640x384 input and does thresholding + NMS.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

#include "trt_model.hpp"

namespace fht {

struct FaceDet {
    cv::Rect2f box;                       // frame pixels
    float score = 0.f;
    std::array<cv::Point2f, 5> landmarks; // eyes, nose, mouth corners (frame pixels)
};

struct DetectorParams {
    float min_score = 0.80f;  // detector confidence to report a face
    float nms_iou = 0.30f;
    float min_px = 24.f;      // smallest face side, in detector input pixels
};

class FaceDetector {
public:
    explicit FaceDetector(const std::string& engine_path);
    std::vector<FaceDet> detect(const cv::Mat& bgr, const DetectorParams& p);

    double ms_pre = 0, ms_infer = 0, ms_post = 0;  // last call's timings

private:
    TrtModel model_;
    Tensor* in_ = nullptr;
    Tensor* out_ = nullptr;
    int W_ = 0, H_ = 0;  // input width / height
    cv::Mat canvas_;
};

float iou(const cv::Rect2f& a, const cv::Rect2f& b);

}  // namespace fht
