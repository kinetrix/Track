// FaceHandTrack on the Unitree G1 (Jetson Orin NX, TensorRT 8.5).
// Per frame, two paths run in parallel:
//   face path: YuNet every frame -> face tracks with stable IDs, SFace re-identification and
//              saved names (up to 3 faces; --tracker edgetam for the EdgeTAM tracker instead)
//   hand path: palm detector (new hands) / hand landmarks (tracked hands) -> gestures
// then the caregiver rules (same as the Mac app): whose hand is whose, who is addressing the
// robot, help requests, waving to call the robot and where it should turn.
// Results are drawn on a browser page with the video, alerts and controls.

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "camera.hpp"
#include "depth.hpp"
#include "face_pipeline.hpp"
#include "fast_face_tracker.hpp"
#include "attention_tracker.hpp"
#include "hand_pipeline.hpp"
#include "help_detector.hpp"
#include "mjpeg_server.hpp"
#include "person_linker.hpp"

using namespace fht;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct Args {
#ifdef HAVE_REALSENSE
    std::string camera = "realsense";  // head camera colour + depth (stop Unitree's videohub first)
#else
    std::string camera = "/dev/video4";  // head camera, direct (stop Unitree's videohub first)
#endif
    int width = 1280, height = 720, fps = 30;
    std::string engines = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/facehandtrack/engines";
    int port = 8090;
    int stream_fps = 15;
    int jpeg_quality = 80;
    std::string tracker = "fast";
    bool reid = true;
    std::string attn = "fp16";
    bool compare_attn = false;
    int max_faces = 3, max_hands = -1;  // -1: 2 per face
    bool faces = true, hands = true, graphs = true, mirrored = false;
    std::string names = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/facehandtrack/known_faces.txt";
};

void usage() {
    std::printf(
        "usage: facehandtrack [--camera realsense|/dev/video4|videohub|<gstreamer pipeline>]\n"
        "                     [--width 1280] [--height 720] [--fps 30]  (realsense, /dev/videoN)\n"
        "                     [--engines ~/facehandtrack/engines] [--port 8090] [--stream-fps 15]\n"
        "                     [--jpeg-quality 80] [--max-faces 3] [--max-hands 2 x faces] [--no-faces] [--no-hands]\n"
        "                     [--tracker fast|edgetam] [--no-reid] [--attn fp16|fp32] [--compare-attn]\n"
        "                     [--no-graphs] [--mirrored] [--names ~/facehandtrack/known_faces.txt]\n");
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
        else if (k == "--tracker") a.tracker = val();
        else if (k == "--no-reid") a.reid = false;
        else if (k == "--attn") a.attn = val();
        else if (k == "--compare-attn") a.compare_attn = true;
        else if (k == "--no-faces") a.faces = false;
        else if (k == "--no-hands") a.hands = false;
        else if (k == "--no-graphs") a.graphs = false;
        else if (k == "--mirrored") a.mirrored = true;
        else if (k == "--names") a.names = val();
        else {
            usage();
            std::exit(k == "--help" || k == "-h" ? 0 : 1);
        }
    }
    if (a.tracker != "fast" && a.tracker != "edgetam") {
        usage();
        std::exit(1);
    }
    if (a.max_hands < 0) a.max_hands = 2 * a.max_faces;
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

// What the caregiver rules decided this frame (for drawing and the page).
struct Scene {
    std::map<int, int> owner;               // hand id -> face id
    std::map<int, PersonState> people;      // face id -> state
    std::vector<HelpAlert> alerts;
    std::map<int, double> pending;          // help signals waiting for the second
    AttentionTarget attention;
    std::set<int> waving;
};

void dashed(cv::Mat& img, cv::Point2f a, cv::Point2f b, const cv::Scalar& c) {
    const float len = std::hypot(b.x - a.x, b.y - a.y);
    for (float t = 0; t < len; t += 14) {
        const float u = t / len, v = std::min(len, t + 7) / len;
        cv::line(img, a + (b - a) * u, a + (b - a) * v, c, 2, cv::LINE_AA);
    }
}

void banner(cv::Mat& img, const std::string& s, cv::Point org, const cv::Scalar& bg, const cv::Scalar& fg, double size) {
    int base = 0;
    const cv::Size ts = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, size, 2, &base);
    cv::rectangle(img, cv::Rect(org.x - 8, org.y - ts.height - 10, ts.width + 16, ts.height + base + 18), bg, cv::FILLED);
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, size, fg, 2, cv::LINE_AA);
}

void draw(cv::Mat& img, const FaceSource* faces, const HandPipeline* hands, const Scene& sc, double fps, double cam_fps) {
    int n_faces = 0;
    std::map<int, cv::Rect2f> face_box;
    std::map<int, std::string> face_name;
    if (faces) {
        for (const FaceDet& d : faces->detections()) cv::rectangle(img, d.box, cv::Scalar(200, 200, 200), 1);
        for (const FaceOut& t : faces->faces()) {
            if (!t.visible) continue;
            ++n_faces;
            face_box[t.id] = t.box;
            face_name[t.id] = t.name;
            const cv::Scalar c = track_color(t.id);
            auto st = sc.people.find(t.id);
            const bool talking = st != sc.people.end() && !st->second.addressing.empty();
            if (talking) cv::rectangle(img, t.box + cv::Point2f(-6, -6) + cv::Size2f(12, 12), cv::Scalar(0, 230, 255), 5);
            cv::rectangle(img, t.box, c, 2);
            std::string s = t.name.empty() ? "face #" + std::to_string(t.id) : "#" + std::to_string(t.id) + " " + t.name;
            if (st != sc.people.end() && st->second.facing == 0) s += " - looking away";
            if (sc.waving.count(t.id)) s += " - waving";
            label(img, s, cv::Point(static_cast<int>(t.box.x), static_cast<int>(t.box.y) - 8), c, 0.7);
            if (talking)
                banner(img, "talking: " + st->second.addressing, cv::Point(static_cast<int>(t.box.x), static_cast<int>(t.box.br().y) + 34),
                       cv::Scalar(0, 230, 255), cv::Scalar(0, 0, 0), 0.8);
        }
    }
    int n_hands = 0;
    if (hands) {
        for (const HandResult& h : hands->hands()) {
            ++n_hands;
            auto o = sc.owner.find(h.id);
            const int owner = o == sc.owner.end() ? -1 : o->second;
            const cv::Scalar c = owner >= 0 ? track_color(owner) : cv::Scalar(150, 150, 150);  // the person's colour
            if (owner >= 0 && face_box.count(owner)) {
                const cv::Rect2f& fb = face_box[owner];
                dashed(img, cv::Point2f(h.landmarks[0].x, h.landmarks[0].y), cv::Point2f(fb.x + fb.width / 2, fb.br().y), c);
            }
            for (const auto& b : kBones) {
                const cv::Point3f& p = h.landmarks[b[0]];
                const cv::Point3f& q = h.landmarks[b[1]];
                cv::line(img, cv::Point2f(p.x, p.y), cv::Point2f(q.x, q.y), c, 3, cv::LINE_AA);
            }
            for (const auto& p : h.landmarks) cv::circle(img, cv::Point2f(p.x, p.y), 3, cv::Scalar(255, 255, 255), -1);
            char s[128];
            std::snprintf(s, sizeof(s), "%s - %s: %s (%d)", owner >= 0 ? ("#" + std::to_string(owner)).c_str() : "?",
                          h.handedness.c_str(), h.gesture.empty() ? "-" : h.gesture.c_str(), h.fingers);
            label(img, s, cv::Point(static_cast<int>(h.box[0]), static_cast<int>(h.box[1]) - 8), c, 0.7);
        }
    }
    char hud[96];
    std::snprintf(hud, sizeof(hud), "%.1f fps (camera %.0f)  faces %d  hands %d", fps, cam_fps, n_faces, n_hands);
    label(img, hud, cv::Point(10, 26), cv::Scalar(255, 255, 255), 0.7);
    if (sc.attention.face_id >= 0) {
        const std::string n = face_name.count(sc.attention.face_id) ? face_name[sc.attention.face_id] : "";
        const std::string who = n.empty() ? "face #" + std::to_string(sc.attention.face_id) : "#" + std::to_string(sc.attention.face_id) + " " + n;
        banner(img, "Robot -> " + who + " (" + sc.attention.reason + "): " + sc.attention.turn_text(),
               cv::Point(20, img.rows - 24), cv::Scalar(230, 210, 60), cv::Scalar(0, 0, 0), 0.8);
    }
    if (!sc.alerts.empty()) {
        const HelpAlert& a = sc.alerts.back();
        banner(img, "HELP REQUESTED: " + a.who() + " - " + a.reason + " - " + a.clock, cv::Point(20, 70),
               cv::Scalar(40, 40, 200), cv::Scalar(255, 255, 255), 1.0);
    } else if (!sc.pending.empty()) {
        banner(img, "Help signal seen - give it once more to confirm", cv::Point(20, 70), cv::Scalar(0, 120, 200),
               cv::Scalar(255, 255, 255), 0.8);
    }
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if (static_cast<unsigned char>(ch) < 0x20) o += ' ';
        else o += ch;
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    set_cuda_graphs(a.graphs);

    std::unique_ptr<FaceSource> faces;
    std::unique_ptr<HandPipeline> hands;
    if (a.faces && a.tracker == "edgetam") {
        FacePolicy fp;
        fp.max_tracks = a.max_faces;
        faces = std::make_unique<FacePipeline>(a.engines, a.attn, a.compare_attn, fp);
    } else if (a.faces) {
        FastFacePolicy fp;
        fp.max_tracks = a.max_faces;
        faces = std::make_unique<FastFacePipeline>(a.engines, fp, a.reid, a.names);
    }
    FastFacePipeline* fast = dynamic_cast<FastFacePipeline*>(faces.get());
    PersonLinker linker;
    HelpDetector help;
    AttentionTracker attention;
    linker.mirrored = attention.mirrored = a.mirrored;
    Scene scene;
    std::string message;  // last answer to a page command
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
    DepthFrame depth;
    std::vector<LinkFace> last_lfaces;  // for the status line: distances
    std::vector<LinkHand> last_lhands;
    uint64_t id = 0;
    auto t_report = t_start;
    auto t_stream = t_start;
    int frames = 0;
    double sum_total = 0, sum_face = 0, sum_hand = 0, fps = 0;
    std::vector<std::string> face_events, hand_events;

    while (!g_stop) {
        id = cam.wait_newer(id, frame, &depth);
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
            if (fast) fast->set_depth(&depth);  // far or turned faces keep their number by place and distance
            faces->process(frame, now, face_events);
            sum_face += ms_since(tf);
        }
        if (hand_job.valid()) hand_job.get();
        sum_hand += hand_ms;
        for (const std::string& e : face_events) std::printf("[%8.2fs] %s\n", now, e.c_str());
        for (const std::string& e : hand_events) std::printf("[%8.2fs] %s\n", now, e.c_str());

        // caregiver rules: whose hand, who is addressing the robot, help, waves, where to turn
        std::vector<FaceOut> fouts = faces ? faces->faces() : std::vector<FaceOut>();
        const std::vector<HandResult> houts = hands ? hands->hands() : std::vector<HandResult>();
        std::vector<LinkFace> lfaces;
        for (const FaceOut& f : fouts)
            lfaces.push_back({f.id, f.box, f.landmarks, f.visible, f.visible ? face_depth(depth, f.box) : -1.f});
        std::vector<LinkHand> lhands;
        for (const HandResult& h : houts) {
            LinkHand lh{h.id, h.handedness, {}, h.gesture};
            for (const auto& q : h.landmarks) lh.points.emplace_back(q.x, q.y);
            lh.depth = hand_depth(depth, lh.points);
            if (lh.depth > 0 && lh.points.size() == 21)
                for (const LinkFace& f : lfaces)
                    if (f.visible && f.depth > 0) lh.arm[f.id] = arm_support(depth, lh.points[0], lh.depth, f.box, f.depth);
            lhands.push_back(lh);
        }
        std::vector<std::string> rule_events;
        linker.update(lfaces, lhands, now, rule_events);
        last_lfaces = lfaces;
        last_lhands = lhands;
        std::vector<HelpFace> hfaces;
        std::vector<AttentionFace> afaces;
        for (const FaceOut& f : fouts) {
            if (!f.visible) continue;
            auto st = linker.people().find(f.id);
            hfaces.push_back({f.id, f.box, f.name, st != linker.people().end() ? st->second.facing : -1});
            afaces.push_back({f.id, f.box, st != linker.people().end() && !st->second.addressing.empty()});
        }
        std::vector<HelpHand> hhands;
        std::vector<AttentionHand> ahands;
        for (size_t i = 0; i < houts.size(); ++i) {
            auto o = linker.owner().find(houts[i].id);
            const int owner = o == linker.owner().end() ? -1 : o->second;
            const auto& pts = lhands[i].points;
            hhands.push_back({houts[i].id, owner, houts[i].extended, houts[i].gesture, mean_point(pts), pts[0], hand_size(pts)});
            ahands.push_back({houts[i].id, owner, houts[i].handedness, houts[i].extended, pts});
        }
        help.update(hfaces, hhands, now, rule_events);
        for (const HelpAlert& al : help.new_alerts())
            rule_events.push_back("HELP REQUESTED: " + al.reason + " from " + al.who());
        attention.update(afaces, ahands, frame.size(), now, rule_events);
        for (const std::string& e : rule_events) std::printf("[%8.2fs] %s\n", now, e.c_str());
        scene.owner = linker.owner();
        scene.people = linker.people();
        scene.alerts = help.alerts();
        scene.pending = help.pending();
        scene.attention = attention.target();
        scene.waving.clear();
        for (const FaceOut& f : fouts)
            if (attention.waving(f.id)) scene.waving.insert(f.id);

        // commands from the browser page
        for (const std::string& c : stream.take_commands()) {
            if (c == "ack") {
                help.acknowledge_all();
                message = "Help alerts acknowledged.";
            } else if (c == "forget") {
                message = fast ? fast->forget_names() : "Names need the fast face tracker.";
            } else if (c.rfind("save=", 0) == 0) {
                int prefer = attention.target().face_id;
                for (const auto& [fid, st] : linker.people())
                    if (!st.addressing.empty()) prefer = fid;
                message = fast ? fast->save_face(c.substr(5), prefer) : "Names need the fast face tracker.";
            }
            std::printf("[%8.2fs] [page] %s\n", now, message.c_str());
            scene.alerts = help.alerts();
        }
        // status for the page (a few times a second is plenty)
        if (frames % 5 == 0) {
            std::string js = "{\"alerts\":[";
            for (size_t i = 0; i < scene.alerts.size(); ++i)
                js += (i ? "," : "") + std::string("{\"id\":") + std::to_string(scene.alerts[i].id) + ",\"who\":\"" +
                      json_escape(scene.alerts[i].who()) + "\",\"reason\":\"" + json_escape(scene.alerts[i].reason) +
                      "\",\"clock\":\"" + scene.alerts[i].clock + "\"}";
            js += "],\"pending\":[";
            bool first = true;
            for (const auto& [k, t] : scene.pending) {
                js += std::string(first ? "" : ",") + "\"" + (k >= 0 ? "face #" + std::to_string(k) : std::string("someone")) + "\"";
                first = false;
            }
            js += "],\"people\":[";
            first = true;
            for (const FaceOut& f : fouts) {
                if (!f.visible) continue;
                auto st = scene.people.find(f.id);
                std::string d = "#" + std::to_string(f.id) + (f.name.empty() ? "" : " " + f.name);
                if (st != scene.people.end() && !st->second.addressing.empty()) d += " (talking: " + st->second.addressing + ")";
                js += std::string(first ? "" : ",") + "\"" + json_escape(d) + "\"";
                first = false;
            }
            js += "],\"attention\":";
            if (scene.attention.face_id >= 0)
                js += "\"#" + std::to_string(scene.attention.face_id) + " (" + scene.attention.reason + "): " +
                      json_escape(scene.attention.turn_text()) + "\"";
            else
                js += "null";
            js += ",\"message\":\"" + json_escape(message) + "\"}";
            stream.set_status(js);
        }
        sum_total += ms_since(t0);
        ++frames;

        // The stream gets an annotated frame at most stream_fps times a second, and only while someone watches.
        if (stream.has_viewers() && ms_since(t_stream) >= 1000.0 / std::max(1, a.stream_fps)) {
            t_stream = std::chrono::steady_clock::now();
            draw(frame, faces.get(), hands.get(), scene, fps, cam.capture_fps());
            stream.submit(std::move(frame));  // `frame` is empty now; the camera fills a fresh one
        }

        const double dt = ms_since(t_report);
        if (dt >= 2000.0) {
            fps = frames * 1000.0 / dt;
            int n_faces = 0;
            if (faces)
                for (const FaceOut& t : faces->faces()) n_faces += t.visible;
            const int n_hands = hands ? static_cast<int>(hands->hands().size()) : 0;
            std::printf("[facehandtrack] %5.1f fps (camera %4.1f) | faces %d hands %d | total %.1f ms", fps,
                        cam.capture_fps(), n_faces, n_hands, sum_total / frames);
            if (faces) std::printf(" | face %.1f: %s", sum_face / frames, faces->timing(frames).c_str());
            if (hands) {
                std::printf(" | hand %.1f: palm %.1f landmarks %.1f", sum_hand / frames, hands->ms_palm() / frames,
                            hands->ms_landmarks() / frames);
                for (const HandResult& h : hands->hands()) {
                    auto o = scene.owner.find(h.id);
                    std::printf(" | %s %s %s", o == scene.owner.end() ? "?" : ("#" + std::to_string(o->second)).c_str(),
                                h.handedness.c_str(), h.gesture.empty() ? "-" : h.gesture.c_str());
                }
                hands->reset_timings();
            }
            if (cam.has_depth()) {  // distances from the camera, latest frame
                std::printf(" | depth");
                for (const LinkFace& f : last_lfaces)
                    if (f.visible) std::printf(f.depth > 0 ? " face #%d %.2f m" : " face #%d ?", f.id, f.depth);
                for (const LinkHand& h : last_lhands)
                    std::printf(h.depth > 0 ? " hand #%d %.2f m" : " hand #%d ?", h.id, h.depth);
            }
            for (const auto& [fid, st] : scene.people)
                if (!st.addressing.empty()) std::printf(" | #%d addressing: %s", fid, st.addressing.c_str());
            if (scene.attention.face_id >= 0)
                std::printf(" | robot -> #%d (%s): %s", scene.attention.face_id, scene.attention.reason.c_str(),
                            scene.attention.turn_text().c_str());
            if (fast) std::printf("%s", fast->tracker().summary().c_str());
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
