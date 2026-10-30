#include "camera.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace fht {

Camera::Camera(const std::string& device, int width, int height, int fps) {
    // OpenCV 4.2 opens V4L2 devices most reliably by index: "/dev/video4" -> 4.
    bool ok = false;
    if (device.rfind("/dev/video", 0) == 0 && device.size() > 10) ok = cap_.open(std::stoi(device.substr(10)), cv::CAP_V4L2);
    if (!ok) ok = cap_.open(device, cv::CAP_V4L2);
    if (!ok) {
        std::fprintf(stderr,
                     "[camera] cannot open %s. Another program may be using it: run\n"
                     "  sudo fuser -v /dev/video*\n",
                     device.c_str());
        std::exit(1);
    }
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap_.set(cv::CAP_PROP_FPS, fps);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cv::Mat probe;
    if (!cap_.read(probe) || probe.empty()) {
        std::fprintf(stderr, "[camera] %s opened but gives no frames (in use by another program?)\n", device.c_str());
        std::exit(1);
    }
    width_ = probe.cols;
    height_ = probe.rows;
    std::printf("[camera] %s: %dx%d, asked for %d fps (driver reports %.0f)\n", device.c_str(), width_, height_, fps,
                cap_.get(cv::CAP_PROP_FPS));
    thread_ = std::thread(&Camera::loop, this);
}

Camera::~Camera() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    cv_.notify_all();
}

void Camera::loop() {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int frames = 0;
    cv::Mat f;
    while (running_) {
        if (!cap_.read(f) || f.empty()) {
            std::fprintf(stderr, "[camera] frame read failed\n");
            break;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            f.copyTo(latest_);
            ++latest_id_;
        }
        cv_.notify_all();
        ++frames;
        double dt = std::chrono::duration<double>(clock::now() - t0).count();
        if (dt >= 1.0) {
            capture_fps_ = frames / dt;
            frames = 0;
            t0 = clock::now();
        }
    }
    running_ = false;
    cv_.notify_all();
}

uint64_t Camera::wait_newer(uint64_t last_id, cv::Mat& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return latest_id_ > last_id || !running_; });
    if (latest_id_ <= last_id) return 0;
    latest_.copyTo(out);
    return latest_id_;
}

}  // namespace fht
