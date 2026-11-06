// YuNet face detector (TensorRT). The engine already decodes boxes; this class
// letterboxes the frame into the 640x384 input and does thresholding + NMS.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

#include "preprocess.hpp"
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
    // Runs on the detector's own CUDA stream and waits for its result.
    std::vector<FaceDet> detect(const GpuFrame& frame, const DetectorParams& p);

    double ms = 0;  // last call's time
    cudaStream_t stream() const { return model_.stream(); }

private:
    TrtModel model_;
    Tensor* in_ = nullptr;
    Tensor* out_ = nullptr;
    int W_ = 0, H_ = 0;  // input width / height
};

float iou(const cv::Rect2f& a, const cv::Rect2f& b);

}  // namespace fht
