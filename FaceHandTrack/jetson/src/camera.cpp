#include "camera.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace fht {

// Unitree's videohub_pc4 owns the head camera (1920x1080 @ 15 fps) and multicasts it as
// 1280x720 H.264 over RTP on eth0. Receiving that stream keeps Unitree's own video working;
// the Jetson's hardware decoder (nvv4l2decoder) unpacks it.
static const char* kVideohubPipeline =
    "udpsrc multicast-group=230.1.1.1 port=1720 multicast-iface=eth0 auto-multicast=true "
    "caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false";

Camera::Camera(const std::string& device, int width, int height, int fps) {
    bool ok = false;
    std::string what = device;
    if (device.rfind("/dev/video", 0) == 0) {
        // Direct V4L2 camera. OpenCV 4.2 opens these most reliably by index: "/dev/video4" -> 4.
        if (device.size() > 10) ok = cap_.open(std::stoi(device.substr(10)), cv::CAP_V4L2);
        if (!ok) ok = cap_.open(device, cv::CAP_V4L2);
        if (ok) {
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            cap_.set(cv::CAP_PROP_FPS, fps);
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
    } else {
        // "videohub" (Unitree's head-camera stream) or any GStreamer pipeline ending in appsink.
        const std::string pipeline = device == "videohub" ? kVideohubPipeline : device;
        what = device == "videohub" ? "videohub stream (rtp://230.1.1.1:1720)" : "GStreamer pipeline";
        ok = cap_.open(pipeline, cv::CAP_GSTREAMER);
    }
    if (!ok) {
        std::fprintf(stderr,
                     "[camera] cannot open %s.\n"
                     "  /dev/videoN: another program may be using it (sudo fuser -v /dev/video*)\n"
                     "  videohub: is videohub_pc4 running, and does OpenCV have GStreamer support?\n",
                     what.c_str());
        std::exit(1);
    }
    cv::Mat probe;
    if (!cap_.read(probe) || probe.empty()) {
        std::fprintf(stderr, "[camera] %s opened but gives no frames\n", what.c_str());
        std::exit(1);
    }
    width_ = probe.cols;
    height_ = probe.rows;
    std::printf("[camera] %s: %dx%d\n", what.c_str(), width_, height_);
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
