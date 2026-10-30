// FaceHandTrack on the Unitree G1 (Jetson Orin NX, TensorRT 8.5).
// Step 2: head camera -> YuNet face boxes -> EdgeTAM face tracks with stable IDs (TensorRT fp16)
//         -> browser video stream.

#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "camera.hpp"
#include "face_pipeline.hpp"
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
    std::string attn = "fp16";
    bool compare_attn = false;
    int max_faces = 3;
};

void usage() {
    std::printf(
        "usage: facehandtrack [--camera videohub|/dev/videoN|<gstreamer pipeline>]\n"
        "                     [--width 640] [--height 480] [--fps 60]  (only for /dev/videoN)\n"
        "                     [--engines ~/facehandtrack/engines] [--port 8090] [--stream-fps 15]\n"
        "                     [--jpeg-quality 80] [--max-faces 3] [--attn fp16|fp32] [--compare-attn]\n");
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
        else if (k == "--max-faces") a.max_faces = std::stoi(val());
        else if (k == "--attn") a.attn = val();
        else if (k == "--compare-attn") a.compare_attn = true;
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

cv::Scalar track_color(int id) {
    static const cv::Scalar palette[] = {{80, 220, 80}, {255, 160, 40}, {60, 120, 255}, {220, 80, 220}, {40, 220, 220}};
    return palette[(id - 1) % 5];
}

void draw(cv::Mat& img, const std::vector<std::unique_ptr<FaceTrack>>& tracks, const std::vector<FaceDet>& dets,
          double fps, double cam_fps) {
    for (const FaceDet& d : dets) cv::rectangle(img, d.box, cv::Scalar(200, 200, 200), 1);
    for (const auto& t : tracks) {
        if (!t->visible) continue;
        const cv::Scalar c = track_color(t->id);
        cv::rectangle(img, t->box, c, 2);
        char s[48];
        std::snprintf(s, sizeof(s), "#%d  q%.2f", t->id, t->iou_pred);
        const cv::Point org(static_cast<int>(t->box.x), static_cast<int>(t->box.y) - 6);
        cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv::LINE_AA);
    }
    int shown = 0;
    for (const auto& t : tracks) shown += t->visible;
    char hud[96];
    std::snprintf(hud, sizeof(hud), "%.1f fps (camera %.0f)  faces %d", fps, cam_fps, shown);
    cv::putText(img, hud, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, hud, cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    FacePolicy policy;
    policy.max_tracks = a.max_faces;
    FacePipeline faces(a.engines, a.attn, a.compare_attn, policy);
    Camera cam(a.camera, a.width, a.height, a.fps);
    MjpegServer stream(a.port, a.jpeg_quality);
    std::printf("[stream] open http://<robot address>:%d/ in a browser (e.g. http://192.168.123.164:%d/)\n", a.port,
                a.port);
    std::fflush(stdout);
    EdgeTAM& tam = faces.tam();

    const auto t_start = std::chrono::steady_clock::now();
    cv::Mat frame;
    uint64_t id = 0;
    auto t_report = t_start;
    auto t_stream = t_start;
    int frames = 0;
    double sum_total = 0, fps = 0;
    std::vector<std::string> events;

    while (!g_stop) {
        id = cam.wait_newer(id, frame);
        if (id == 0) break;
        const auto t0 = std::chrono::steady_clock::now();
        const double now = std::chrono::duration<double>(t0 - t_start).count();

        events.clear();
        faces.process(frame, now, events);
        for (const std::string& e : events) std::printf("[%8.2fs] %s\n", now, e.c_str());
        sum_total += ms_since(t0);
        ++frames;

        // The stream gets an annotated frame at most stream_fps times a second, and only while someone watches.
        if (stream.has_viewers() && ms_since(t_stream) >= 1000.0 / std::max(1, a.stream_fps)) {
            t_stream = std::chrono::steady_clock::now();
            draw(frame, faces.tracks(), faces.detections(), fps, cam.capture_fps());
            stream.submit(std::move(frame));  // `frame` is empty now; the camera fills a fresh one
        }

        const double dt = ms_since(t_report);
        if (dt >= 2000.0) {
            fps = frames * 1000.0 / dt;
            int shown = 0;
            for (const auto& t : faces.tracks()) shown += t->visible;
            std::printf("[facehandtrack] %5.1f fps (camera %4.1f) | faces %d | detect %.1f enc %.1f attn %.1f dec %.1f "
                        "menc %.1f total %.1f ms",
                        fps, cam.capture_fps(), shown, faces.ms_detect / frames, tam.ms_enc / frames,
                        tam.ms_attn / frames, tam.ms_dec / frames, tam.ms_menc / frames, sum_total / frames);
            if (tam.compare.n > 0)
                std::printf(" | fp16 vs fp32 mask IoU mean %.3f min %.3f (%ld)", tam.compare.iou_sum / tam.compare.n,
                            tam.compare.iou_min, tam.compare.n);
            std::printf("\n");
            std::fflush(stdout);
            frames = 0;
            sum_total = 0;
            faces.reset_timings();
            t_report = std::chrono::steady_clock::now();
        }
    }
    std::printf("stopped\n");
    return 0;
}
