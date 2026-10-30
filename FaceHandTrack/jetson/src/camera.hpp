// Camera reader on its own thread: the newest frame wins, older ones are dropped.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace fht {

class Camera {
public:
    // device: e.g. "/dev/video4" (the G1 head RealSense colour stream).
    Camera(const std::string& device, int width, int height, int fps);
    ~Camera();

    // Waits for a frame newer than `last_id`; returns its id (0 if the camera stopped).
    uint64_t wait_newer(uint64_t last_id, cv::Mat& out);
    double capture_fps() const { return capture_fps_.load(); }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    void loop();

    cv::VideoCapture cap_;
    int width_ = 0, height_ = 0;
    std::thread thread_;
    std::mutex m_;
    std::condition_variable cv_;
    cv::Mat latest_;
    uint64_t latest_id_ = 0;
    std::atomic<bool> running_{true};
    std::atomic<double> capture_fps_{0.0};
};

}  // namespace fht
