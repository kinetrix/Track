// Minimal MJPEG-over-HTTP server: open http://<robot>:<port>/ in a browser.
// JPEGs are encoded only while at least one viewer is connected.
#pragma once

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
    explicit MjpegServer(int port);
    ~MjpegServer();

    bool has_viewers() const { return viewers_.load() > 0; }
    void publish(std::vector<unsigned char> jpeg);  // newest frame replaces the old one
    int port() const { return port_; }

private:
    void accept_loop();
    void serve(int fd);

    int port_ = 0;
    int listen_fd_ = -1;
    std::atomic<bool> running_{true};
    std::atomic<int> viewers_{0};
    std::thread accept_thread_;
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<unsigned char> jpeg_;
    uint64_t jpeg_id_ = 0;
};

}  // namespace fht
