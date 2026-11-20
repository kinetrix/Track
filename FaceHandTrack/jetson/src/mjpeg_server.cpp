#include "mjpeg_server.hpp"

#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fht {

namespace {
std::string url_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') o += ' ';
        else if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
                 std::isxdigit(static_cast<unsigned char>(s[i + 2]))) {
            o += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else o += s[i];
    }
    return o;
}

const char* kPage = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FaceHandTrack on the G1</title>
<style>
body{margin:0;background:#101418;color:#e8ecf1;font-family:system-ui,sans-serif}
main{max-width:1320px;margin:0 auto;padding:12px}
img{width:100%;border-radius:8px;background:#000;display:block}
#alert{display:none;background:#c62828;color:#fff;border-radius:10px;padding:14px 16px;margin-bottom:10px}
#alert b{font-size:22px} #pending{display:none;background:#b26a00;color:#fff;border-radius:10px;padding:10px 14px;margin-bottom:10px}
#att{margin:10px 0;font-size:17px;color:#7fdbe8}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button{font-size:16px;padding:10px 16px;border-radius:8px;border:0;background:#2b3542;color:#fff;cursor:pointer;min-height:44px}
#ack{background:#fff;color:#c62828;font-weight:700}
input{font-size:16px;padding:10px;border-radius:8px;border:1px solid #445;background:#1b222a;color:#fff;min-height:24px}
#msg,#people{color:#a9b4c0;font-size:14px}
</style></head><body><main>
<div id="alert" role="alert"><b>HELP REQUESTED</b><div id="alerts"></div><div class="row"><button id="ack">Acknowledge</button></div></div>
<div id="pending"></div>
<img src="/stream" alt="Robot head camera with tracking overlay">
<div id="att">Robot attention: no one</div>
<div id="people"></div>
<div class="row"><label for="name">Save the face of the person talking to the robot (else the largest face) as</label>
<input id="name" placeholder="Name"><button id="save">Save face</button><button id="forget">Forget names</button></div>
<div id="msg"></div>
</main><script>
const $=id=>document.getElementById(id);
function cmd(q){fetch('/cmd?'+q).catch(()=>{});}
$('ack').onclick=()=>cmd('ack');
$('save').onclick=()=>{const n=$('name').value.trim(); if(n){cmd('save='+encodeURIComponent(n)); $('name').value='';}};
$('name').onkeydown=e=>{if(e.key==='Enter')$('save').click();};
$('forget').onclick=()=>cmd('forget');
async function poll(){
 try{const s=await (await fetch('/status',{cache:'no-store'})).json();
  const a=s.alerts||[]; $('alert').style.display=a.length?'block':'none';
  $('alerts').textContent=a.map(x=>x.who+' · '+x.reason+' · '+x.clock).join('  |  ');
  const p=s.pending||[]; $('pending').style.display=p.length?'block':'none';
  $('pending').textContent=p.map(x=>'Help signal from '+x+': waiting for the second signal to confirm').join(' | ');
  $('att').textContent='Robot attention: '+(s.attention?s.attention:'no one');
  $('people').textContent=(s.people||[]).join('   ');
  $('msg').textContent=s.message||'';
 }catch(e){}
 setTimeout(poll,500);}
poll();
</script></body></html>)HTML";

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

MjpegServer::MjpegServer(int port, int jpeg_quality) : port_(port), jpeg_quality_(jpeg_quality) {
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
    encode_thread_ = std::thread(&MjpegServer::encode_loop, this);
}

MjpegServer::~MjpegServer() {
    running_ = false;
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    cv_.notify_all();
    frame_cv_.notify_all();
    if (accept_thread_.joinable()) accept_thread_.join();
    if (encode_thread_.joinable()) encode_thread_.join();
}

void MjpegServer::submit(cv::Mat bgr) {
    {
        std::lock_guard<std::mutex> lk(frame_m_);
        pending_ = std::move(bgr);
    }
    frame_cv_.notify_one();
}

void MjpegServer::encode_loop() {
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    while (running_) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(frame_m_);
            frame_cv_.wait_for(lk, std::chrono::milliseconds(500), [&] { return !pending_.empty() || !running_; });
            if (pending_.empty()) continue;
            frame = std::move(pending_);
            pending_ = cv::Mat();
        }
        std::vector<unsigned char> jpg;
        cv::imencode(".jpg", frame, jpg, params);
        {
            std::lock_guard<std::mutex> lk(m_);
            jpeg_ = std::move(jpg);
            ++jpeg_id_;
        }
        cv_.notify_all();
    }
}

void MjpegServer::accept_loop() {
    while (running_) {
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        std::thread(&MjpegServer::serve, this, fd).detach();
    }
}

void MjpegServer::set_status(std::string json) {
    std::lock_guard<std::mutex> lk(status_m_);
    status_ = std::move(json);
}

std::vector<std::string> MjpegServer::take_commands() {
    std::lock_guard<std::mutex> lk(status_m_);
    std::vector<std::string> c;
    c.swap(commands_);
    return c;
}

void MjpegServer::serve(int fd) {
    char req[2048];
    ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
    if (n <= 0) {
        ::close(fd);
        return;
    }
    req[n] = 0;
    // request line: "GET /path?query HTTP/1.1"
    std::string line(req, strcspn(req, "\r\n"));
    std::string path = "/";
    const size_t a = line.find(' '), b = line.find(' ', a + 1);
    if (a != std::string::npos && b != std::string::npos) path = line.substr(a + 1, b - a - 1);
    auto reply = [&](const char* type, const std::string& body) {
        char head[256];
        const int len = std::snprintf(head, sizeof(head),
                                      "HTTP/1.0 200 OK\r\nCache-Control: no-store\r\nConnection: close\r\n"
                                      "Content-Type: %s\r\nContent-Length: %zu\r\n\r\n", type, body.size());
        if (send_all(fd, head, static_cast<size_t>(len))) send_all(fd, body.data(), body.size());
        ::close(fd);
    };
    if (path == "/" || path == "/index.html") return reply("text/html; charset=utf-8", kPage);
    if (path == "/status") {
        std::string s;
        {
            std::lock_guard<std::mutex> lk(status_m_);
            s = status_;
        }
        return reply("application/json", s);
    }
    if (path.rfind("/cmd?", 0) == 0) {
        const std::string q = path.substr(5);
        if (q == "ack" || q == "forget" || q.rfind("save=", 0) == 0) {
            std::lock_guard<std::mutex> lk(status_m_);
            if (commands_.size() < 32) commands_.push_back(q.rfind("save=", 0) == 0 ? "save=" + url_decode(q.substr(5)) : q);
        }
        return reply("text/plain", "ok");
    }
    if (path != "/stream") return reply("text/plain", "not found");
    stream(fd);
}

void MjpegServer::stream(int fd) {
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
