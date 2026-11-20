#include "camera.hpp"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#include "depth_impl.hpp"
#endif

namespace fht {

#ifdef HAVE_REALSENSE
struct Camera::RealSense {
    rs2::pipeline pipe;
    float scale = 0.001f;
    rs2_intrinsics depth_intrin{}, color_intrin{};
    rs2_extrinsics color_to_depth{}, depth_to_color{};
};
#else
struct Camera::RealSense {};
#endif

// The RealSense colour camera's auto exposure may lengthen each frame in dim light, which drops
// the frame rate (15-30 fps on the robot) and blurs moving hands. Turning off "exposure priority"
// keeps the frame rate fixed; auto exposure then brightens by gain instead.
static void keep_frame_rate(const std::string& device) {
    const int fd = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) return;
    v4l2_control c{};
    c.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
    c.value = 0;
    if (::ioctl(fd, VIDIOC_S_CTRL, &c) == 0)
        std::printf("[camera] exposure priority off: frame rate stays fixed in dim light\n");
    else
        std::printf("[camera] could not turn exposure priority off (frame rate may drop in dim light)\n");
    ::close(fd);
}

// Unitree's videohub_pc4 owns the head camera (1920x1080 @ 15 fps) and multicasts it as
// 1280x720 H.264 over RTP on eth0. Receiving that stream keeps Unitree's own video working;
// the Jetson's hardware decoder (nvv4l2decoder) unpacks it.
static const char* kVideohubPipeline =
    "udpsrc multicast-group=230.1.1.1 port=1720 multicast-iface=eth0 auto-multicast=true "
    "caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false";

Camera::Camera(const std::string& device, int width, int height, int fps) {
    if (device == "realsense") {
#ifdef HAVE_REALSENSE
        rs_ = std::make_unique<RealSense>();
        try {
            rs2::config cfg;
            cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
            cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, fps);  // the D435i's sharpest depth mode
            rs2::pipeline_profile prof = rs_->pipe.start(cfg);
            for (rs2::sensor s : prof.get_device().query_sensors()) {
                try {  // keep the frame rate fixed in dim light (as for /dev/videoN)
                    if (s.supports(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)) s.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0.f);
                } catch (const rs2::error&) {
                }
                if (auto ds = s.as<rs2::depth_sensor>()) rs_->scale = ds.get_depth_scale();
            }
            auto dp = prof.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            auto cp = prof.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            rs_->depth_intrin = dp.get_intrinsics();
            rs_->color_intrin = cp.get_intrinsics();
            rs_->color_to_depth = cp.get_extrinsics_to(dp);
            rs_->depth_to_color = dp.get_extrinsics_to(cp);
            width_ = rs_->color_intrin.width;
            height_ = rs_->color_intrin.height;
        } catch (const rs2::error& e) {
            std::fprintf(stderr,
                         "[camera] realsense: %s\n"
                         "  Unitree's videohub holds the head camera: sudo pkill -f videohub_pc4\n",
                         e.what());
            std::exit(1);
        }
        std::printf("[camera] realsense: colour %dx%d + depth 848x480 @ %d fps (depth unit %.4f m)\n", width_,
                    height_, fps, rs_->scale);
        thread_ = std::thread(&Camera::loop_realsense, this);
        return;
#else
        std::fprintf(stderr, "[camera] built without librealsense: --camera realsense is not available\n");
        std::exit(1);
#endif
    }
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
            keep_frame_rate(device);
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
                     "  /dev/videoN: another program may be using it (sudo fuser -v /dev/video*);\n"
                     "               Unitree's videohub holds the head camera: sudo pkill -f videohub_pc4\n"
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
    if (device.rfind("/dev/video", 0) == 0)
        std::printf("[camera] %s: %dx%d @ %.0f fps (asked %dx%d @ %d)\n", what.c_str(), width_, height_,
                    cap_.get(cv::CAP_PROP_FPS), width, height, fps);
    else
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

void Camera::loop_realsense() {
#ifdef HAVE_REALSENSE
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int frames = 0;
    while (running_) {
        rs2::frameset fs;
        try {
            if (!rs_->pipe.try_wait_for_frames(&fs, 2000)) {
                std::fprintf(stderr, "[camera] realsense: no frames for 2 s\n");
                break;
            }
        } catch (const rs2::error& e) {
            std::fprintf(stderr, "[camera] realsense: %s\n", e.what());
            break;
        }
        rs2::video_frame c = fs.get_color_frame();
        rs2::depth_frame z = fs.get_depth_frame();
        if (!c) continue;
        auto impl = std::make_shared<DepthFrame::Impl>();
        if (z) {
            impl->z16 = cv::Mat(z.get_height(), z.get_width(), CV_16U, const_cast<void*>(z.get_data())).clone();
            impl->scale = rs_->scale;
            impl->depth_intrin = rs_->depth_intrin;
            impl->color_intrin = rs_->color_intrin;
            impl->color_to_depth = rs_->color_to_depth;
            impl->depth_to_color = rs_->depth_to_color;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            cv::Mat(c.get_height(), c.get_width(), CV_8UC3, const_cast<void*>(c.get_data())).copyTo(latest_);
            latest_depth_.impl = z ? std::shared_ptr<const DepthFrame::Impl>(impl) : nullptr;
            ++latest_id_;
        }
        cv_.notify_all();
        ++frames;
        const double dt = std::chrono::duration<double>(clock::now() - t0).count();
        if (dt >= 1.0) {
            capture_fps_ = frames / dt;
            frames = 0;
            t0 = clock::now();
        }
    }
    rs_->pipe.stop();
#endif
    running_ = false;
    cv_.notify_all();
}

uint64_t Camera::wait_newer(uint64_t last_id, cv::Mat& out, DepthFrame* depth) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return latest_id_ > last_id || !running_; });
    if (latest_id_ <= last_id) return 0;
    latest_.copyTo(out);
    if (depth) *depth = latest_depth_;
    return latest_id_;
}

}  // namespace fht
