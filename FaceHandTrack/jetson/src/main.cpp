// FaceHandTrack on the Unitree G1 (Jetson Orin NX, TensorRT 8.5).
// Per frame, two paths run in parallel:
//   face path: YuNet -> EdgeTAM face tracks with stable IDs (up to 3 faces)
//   hand path: palm detector (new hands) / hand landmarks (tracked hands) -> gestures
// Results are drawn on a browser video stream.

#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "camera.hpp"
#include "face_pipeline.hpp"
#include "hand_pipeline.hpp"
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
    int max_faces = 3, max_hands = 2;
    bool faces = true, hands = true, graphs = true, mirrored = false;
};

void usage() {
    std::printf(
        "usage: facehandtrack [--camera videohub|/dev/videoN|<gstreamer pipeline>]\n"
        "                     [--width 640] [--height 480] [--fps 60]  (only for /dev/videoN)\n"
        "                     [--engines ~/facehandtrack/engines] [--port 8090] [--stream-fps 15]\n"
        "                     [--jpeg-quality 80] [--max-faces 3] [--max-hands 2] [--no-faces] [--no-hands]\n"
        "                     [--attn fp16|fp32] [--compare-attn] [--no-graphs] [--mirrored]\n");
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
        else if (k == "--max-hands") a.max_hands = std::stoi(val());
        else if (k == "--attn") a.attn = val();
        else if (k == "--compare-attn") a.compare_attn = true;
        else if (k == "--no-faces") a.faces = false;
        else if (k == "--no-hands") a.hands = false;
        else if (k == "--no-graphs") a.graphs = false;
        else if (k == "--mirrored") a.mirrored = true;
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

void label(cv::Mat& img, const std::string& s, cv::Point org, const cv::Scalar& c, double size = 0.6) {
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, size, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, size, c, 1, cv::LINE_AA);
}

const int kBones[21][2] = {{0, 1},   {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
                           {7, 8},   {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14},
                           {14, 15}, {15, 16}, {13, 17}, {17, 18}, {18, 19}, {19, 20}, {0, 17}};

void draw(cv::Mat& img, const FacePipeline* faces, const HandPipeline* hands, double fps, double cam_fps) {
    int n_faces = 0;
    if (faces) {
        for (const FaceDet& d : faces->detections()) cv::rectangle(img, d.box, cv::Scalar(200, 200, 200), 1);
        for (const auto& t : faces->tracks()) {
            if (!t->visible) continue;
            ++n_faces;
            const cv::Scalar c = track_color(t->id);
            cv::rectangle(img, t->box, c, 2);
            char s[48];
            std::snprintf(s, sizeof(s), "face #%d  q%.2f", t->id, t->iou_pred);
            label(img, s, cv::Point(static_cast<int>(t->box.x), static_cast<int>(t->box.y) - 6), c);
        }
    }
    int n_hands = 0;
    if (hands) {
        for (const HandResult& h : hands->hands()) {
            ++n_hands;
            const cv::Scalar c = h.handedness == "Right" ? cv::Scalar(255, 200, 0) : cv::Scalar(0, 200, 255);
            for (const auto& b : kBones) {
                const cv::Point3f& p = h.landmarks[b[0]];
                const cv::Point3f& q = h.landmarks[b[1]];
                cv::line(img, cv::Point2f(p.x, p.y), cv::Point2f(q.x, q.y), c, 2, cv::LINE_AA);
            }
            for (const auto& p : h.landmarks) cv::circle(img, cv::Point2f(p.x, p.y), 3, cv::Scalar(255, 255, 255), -1);
            char s[96];
            std::snprintf(s, sizeof(s), "%s #%d: %s (%d)", h.handedness.c_str(), h.id,
                          h.gesture.empty() ? "-" : h.gesture.c_str(), h.fingers);
            label(img, s, cv::Point(static_cast<int>(h.box[0]), static_cast<int>(h.box[1]) - 6), c);
        }
    }
    char hud[96];
    std::snprintf(hud, sizeof(hud), "%.1f fps (camera %.0f)  faces %d  hands %d", fps, cam_fps, n_faces, n_hands);
    label(img, hud, cv::Point(10, 26), cv::Scalar(255, 255, 255), 0.7);
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    set_cuda_graphs(a.graphs);

    std::unique_ptr<FacePipeline> faces;
    std::unique_ptr<HandPipeline> hands;
    if (a.faces) {
        FacePolicy fp;
        fp.max_tracks = a.max_faces;
        faces = std::make_unique<FacePipeline>(a.engines, a.attn, a.compare_attn, fp);
    }
    if (a.hands) {
        HandPolicy hp;
        hp.max_hands = a.max_hands;
        hands = std::make_unique<HandPipeline>(a.engines, hp, a.mirrored);
    }
    Camera cam(a.camera, a.width, a.height, a.fps);
    MjpegServer stream(a.port, a.jpeg_quality);
    std::printf("[stream] open http://<robot address>:%d/ in a browser (e.g. http://192.168.123.164:%d/)\n", a.port,
                a.port);
    std::fflush(stdout);

    const auto t_start = std::chrono::steady_clock::now();
    cv::Mat frame;
    uint64_t id = 0;
    auto t_report = t_start;
    auto t_stream = t_start;
    int frames = 0;
    double sum_total = 0, sum_face = 0, sum_hand = 0, fps = 0;
    std::vector<std::string> face_events, hand_events;

    while (!g_stop) {
        id = cam.wait_newer(id, frame);
        if (id == 0) break;
        const auto t0 = std::chrono::steady_clock::now();
        const double now = std::chrono::duration<double>(t0 - t_start).count();

        face_events.clear();
        hand_events.clear();
        double hand_ms = 0;
        std::future<void> hand_job;
        if (hands)
            hand_job = std::async(std::launch::async, [&] {
                const auto th = std::chrono::steady_clock::now();
                hands->process(frame, hand_events);
                hand_ms = ms_since(th);
            });
        if (faces) {
            const auto tf = std::chrono::steady_clock::now();
            faces->process(frame, now, face_events);
            sum_face += ms_since(tf);
        }
        if (hand_job.valid()) hand_job.get();
        sum_hand += hand_ms;
        for (const std::string& e : face_events) std::printf("[%8.2fs] %s\n", now, e.c_str());
        for (const std::string& e : hand_events) std::printf("[%8.2fs] %s\n", now, e.c_str());
        sum_total += ms_since(t0);
        ++frames;

        // The stream gets an annotated frame at most stream_fps times a second, and only while someone watches.
        if (stream.has_viewers() && ms_since(t_stream) >= 1000.0 / std::max(1, a.stream_fps)) {
            t_stream = std::chrono::steady_clock::now();
            draw(frame, faces.get(), hands.get(), fps, cam.capture_fps());
            stream.submit(std::move(frame));  // `frame` is empty now; the camera fills a fresh one
        }

        const double dt = ms_since(t_report);
        if (dt >= 2000.0) {
            fps = frames * 1000.0 / dt;
            int n_faces = 0;
            if (faces)
                for (const auto& t : faces->tracks()) n_faces += t->visible;
            const int n_hands = hands ? static_cast<int>(hands->hands().size()) : 0;
            std::printf("[facehandtrack] %5.1f fps (camera %4.1f) | faces %d hands %d | total %.1f ms", fps,
                        cam.capture_fps(), n_faces, n_hands, sum_total / frames);
            if (faces) {
                EdgeTAM& tam = faces->tam();
                std::printf(" | face %.1f: detect %.1f enc %.1f attn %.1f dec %.1f menc %.1f", sum_face / frames,
                            faces->ms_detect / frames, tam.ms_enc / frames, tam.ms_attn / frames, tam.ms_dec / frames,
                            tam.ms_menc / frames);
                if (tam.compare.n > 0)
                    std::printf(" | fp16 vs fp32 mask IoU mean %.3f min %.3f (%ld)",
                                tam.compare.iou_sum / tam.compare.n, tam.compare.iou_min, tam.compare.n);
                faces->reset_timings();
            }
            if (hands) {
                std::printf(" | hand %.1f: palm %.1f landmarks %.1f", sum_hand / frames, hands->ms_palm() / frames,
                            hands->ms_landmarks() / frames);
                for (const HandResult& h : hands->hands())
                    std::printf(" | %s #%d %s", h.handedness.c_str(), h.id, h.gesture.empty() ? "-" : h.gesture.c_str());
                hands->reset_timings();
            }
            std::printf("\n");
            std::fflush(stdout);
            frames = 0;
            sum_total = sum_face = sum_hand = 0;
            t_report = std::chrono::steady_clock::now();
        }
    }
    std::printf("stopped\n");
    return 0;
}
