// Minimal HTTP server for the robot's browser page: open http://<robot>:<port>/ in a browser.
//   /         the page: live video, help alerts with an Acknowledge button, who the robot attends
//             to, and saving a person's name (the robot has no screen of its own)
//   /stream   the annotated video (MJPEG)
//   /status   the latest status JSON (alerts, people, attention), polled by the page
//   /cmd?...  save=<name> | forget | ack: queued for the tracking loop (take_commands)
// Frames are JPEG-encoded on the server's own thread (so the tracking loop never waits for it),
// and only while at least one viewer is connected.
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
    void set_status(std::string json);          // from the tracking loop
    std::vector<std::string> take_commands();   // "save=<name>", "forget", "ack"

private:
    void accept_loop();
    void encode_loop();
    void serve(int fd);
    void stream(int fd);

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
    std::mutex status_m_;
    std::string status_ = "{}";
    std::vector<std::string> commands_;
};

}  // namespace fht
