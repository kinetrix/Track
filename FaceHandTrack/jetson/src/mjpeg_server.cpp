#include "mjpeg_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fht {

namespace {
bool send_all(int fd, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    while (n > 0) {
        ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k;
        n -= static_cast<size_t>(k);
    }
    return true;
}
}  // namespace

MjpegServer::MjpegServer(int port) : port_(port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(listen_fd_, 4) != 0) {
        std::fprintf(stderr, "[stream] cannot listen on port %d: %s\n", port, std::strerror(errno));
        std::exit(1);
    }
    accept_thread_ = std::thread(&MjpegServer::accept_loop, this);
}

MjpegServer::~MjpegServer() {
    running_ = false;
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    cv_.notify_all();
    if (accept_thread_.joinable()) accept_thread_.join();
}

void MjpegServer::publish(std::vector<unsigned char> jpeg) {
    {
        std::lock_guard<std::mutex> lk(m_);
        jpeg_ = std::move(jpeg);
        ++jpeg_id_;
    }
    cv_.notify_all();
}

void MjpegServer::accept_loop() {
    while (running_) {
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        std::thread(&MjpegServer::serve, this, fd).detach();
    }
}

void MjpegServer::serve(int fd) {
    char req[1024];
    ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);  // the request itself does not matter
    if (n <= 0) {
        ::close(fd);
        return;
    }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    const char* header =
        "HTTP/1.0 200 OK\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!send_all(fd, header, std::strlen(header))) {
        ::close(fd);
        return;
    }
    ++viewers_;
    uint64_t sent_id = 0;
    std::vector<unsigned char> frame;
    while (running_) {
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, std::chrono::seconds(1), [&] { return jpeg_id_ != sent_id || !running_; });
            if (!running_) break;
            if (jpeg_id_ == sent_id) continue;
            frame = jpeg_;
            sent_id = jpeg_id_;
        }
        char part[128];
        int len = std::snprintf(part, sizeof(part), "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                                frame.size());
        if (!send_all(fd, part, static_cast<size_t>(len)) || !send_all(fd, frame.data(), frame.size()) ||
            !send_all(fd, "\r\n", 2))
            break;
    }
    --viewers_;
    ::close(fd);
}

}  // namespace fht
