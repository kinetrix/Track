// Minimal MJPEG-over-HTTP server: open http://<robot>:<port>/ in a browser.
// Frames are JPEG-encoded on the server's own thread (so the tracking loop never waits
// for it), and only while at least one viewer is connected.
#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fht {

class MjpegServer {
public:
    MjpegServer(int port, int jpeg_quality);
    ~MjpegServer();

    bool has_viewers() const { return viewers_.load() > 0; }
    // Hands over an annotated frame; the newest one replaces any not yet encoded.
    void submit(cv::Mat bgr);
    int port() const { return port_; }

private:
    void accept_loop();
    void encode_loop();
    void serve(int fd);

    int port_ = 0;
    int listen_fd_ = -1;
    std::atomic<bool> running_{true};
    std::atomic<int> viewers_{0};
    int jpeg_quality_ = 80;
    std::thread accept_thread_, encode_thread_;
    std::mutex frame_m_;
    std::condition_variable frame_cv_;
    cv::Mat pending_;
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<unsigned char> jpeg_;
    uint64_t jpeg_id_ = 0;
};

}  // namespace fht
