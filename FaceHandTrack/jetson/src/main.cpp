// FaceHandTrack on the Unitree G1 (Jetson Orin NX, TensorRT 8.5).
// Step 1: head camera -> YuNet face boxes (TensorRT fp16) -> browser video stream.

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "camera.hpp"
#include "face_detector.hpp"
#include "mjpeg_server.hpp"

using namespace fht;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct Args {
    std::string camera = "videohub";
    int width = 640, height = 480, fps = 60;
    std::string engines = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/facehandtrack/engines";
    int port = 8090;
    int stream_fps = 15;
    int jpeg_quality = 80;
};

void usage() {
    std::printf(
        "usage: facehandtrack [--camera videohub|/dev/videoN|<gstreamer pipeline>]\n"
        "                     [--width 640] [--height 480] [--fps 60]  (only for /dev/videoN)\n"
        "                     [--engines ~/facehandtrack/engines] [--port 8090] [--stream-fps 15]\n"
        "                     [--jpeg-quality 80]\n");
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(1);
            }
            return argv[++i];
        };
        if (k == "--camera") a.camera = val();
        else if (k == "--width") a.width = std::stoi(val());
        else if (k == "--height") a.height = std::stoi(val());
        else if (k == "--fps") a.fps = std::stoi(val());
        else if (k == "--engines") a.engines = val();
        else if (k == "--port") a.port = std::stoi(val());
        else if (k == "--stream-fps") a.stream_fps = std::stoi(val());
        else if (k == "--jpeg-quality") a.jpeg_quality = std::stoi(val());
        else {
            usage();
            std::exit(k == "--help" || k == "-h" ? 0 : 1);
        }
    }
    return a;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

void draw(cv::Mat& img, const std::vector<FaceDet>& faces, double fps, double cam_fps) {
    for (const FaceDet& f : faces) {
        cv::rectangle(img, f.box, cv::Scalar(80, 220, 80), 2);
        char s[32];
        std::snprintf(s, sizeof(s), "%.2f", f.score);
        cv::putText(img, s, f.box.tl() + cv::Point2f(0, -6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 220, 80), 1,
                    cv::LINE_AA);
        for (const cv::Point2f& p : f.landmarks) cv::circle(img, p, 2, cv::Scalar(0, 200, 255), -1, cv::LINE_AA);
    }
    char hud[96];
    std::snprintf(hud, sizeof(hud), "%.1f fps (camera %.0f)  faces %zu", fps, cam_fps, faces.size());
    cv::putText(img, hud, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, hud, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    FaceDetector detector(a.engines + "/yunet_640x384.fp16.engine");
    Camera cam(a.camera, a.width, a.height, a.fps);
    MjpegServer stream(a.port);
    std::printf("[stream] open http://<robot address>:%d/ in a browser (e.g. http://192.168.123.164:%d/)\n", a.port,
                a.port);
    std::fflush(stdout);

    DetectorParams params;
    cv::Mat frame;
    uint64_t id = 0;
    auto t_report = std::chrono::steady_clock::now();
    auto t_stream = t_report;
    int frames = 0;
    double sum_pre = 0, sum_inf = 0, sum_post = 0, sum_total = 0, fps = 0;
    size_t last_faces = 0;
    const std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, a.jpeg_quality};

    while (!g_stop) {
        id = cam.wait_newer(id, frame);
        if (id == 0) break;
        auto t0 = std::chrono::steady_clock::now();
        std::vector<FaceDet> faces = detector.detect(frame, params);
        sum_total += ms_since(t0);
        sum_pre += detector.ms_pre;
        sum_inf += detector.ms_infer;
        sum_post += detector.ms_post;
        last_faces = faces.size();
        ++frames;

        // The stream gets an annotated frame at most stream_fps times a second, and only while someone watches.
        if (stream.has_viewers() && ms_since(t_stream) >= 1000.0 / std::max(1, a.stream_fps)) {
            t_stream = std::chrono::steady_clock::now();
            draw(frame, faces, fps, cam.capture_fps());
            std::vector<unsigned char> jpg;
            cv::imencode(".jpg", frame, jpg, jpeg_params);
            stream.publish(std::move(jpg));
        }

        double dt = ms_since(t_report);
        if (dt >= 2000.0) {
            fps = frames * 1000.0 / dt;
            std::printf("[facehandtrack] %5.1f fps (camera %4.1f) | faces %zu | pre %.2f infer %.2f post %.2f total %.2f ms\n",
                        fps, cam.capture_fps(), last_faces, sum_pre / frames, sum_inf / frames, sum_post / frames,
                        sum_total / frames);
            std::fflush(stdout);
            frames = 0;
            sum_pre = sum_inf = sum_post = sum_total = 0;
            t_report = std::chrono::steady_clock::now();
        }
    }
    std::printf("stopped\n");
    return 0;
}
