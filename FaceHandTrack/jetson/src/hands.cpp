#include "hands.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace fht {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int PALM_SIZE = 192, HAND_SIZE = 224;
constexpr double PALM_BOX_PRE_ENLARGE = 4.0, PALM_BOX_SHIFT_Y = -0.4, PALM_BOX_ENLARGE = 3.0;
constexpr double HAND_BOX_SHIFT_Y = -0.1, HAND_BOX_ENLARGE = 1.65;
constexpr int PALM_FROM_HAND[7] = {0, 5, 9, 13, 17, 1, 2};  // wrist, index/middle/ring/pinky MCP, thumb CMC, MCP

// Shift + enlarge a box, crop it from the image and pad it to a square (as OpenCV Zoo's
// mp_handpose). box: x0, y0, x1, y1. Returns the padded crop; ibox: the integer crop box;
// offset: frame position of the padded crop's top-left corner.
cv::Mat crop_pad(const cv::Mat& image, double x0, double y0, double x1, double y1, double shift_y, double enlarge,
                 bool square_diag, int ibox[4], cv::Point& offset) {
    const double h = y1 - y0;
    y0 += shift_y * h;
    y1 += shift_y * h;  // (the x shift is always 0)
    const double cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const double hw = (x1 - x0) * enlarge / 2, hh = (y1 - y0) * enlarge / 2;
    int b[4] = {static_cast<int>(cx - hw), static_cast<int>(cy - hh), static_cast<int>(cx + hw),
                static_cast<int>(cy + hh)};  // truncation toward zero, like numpy astype(int32)
    b[0] = std::clamp(b[0], 0, image.cols);
    b[2] = std::clamp(b[2], 0, image.cols);
    b[1] = std::clamp(b[1], 0, image.rows);
    b[3] = std::clamp(b[3], 0, image.rows);
    std::copy(b, b + 4, ibox);
    const int cw = std::max(0, b[2] - b[0]), ch = std::max(0, b[3] - b[1]);
    if (cw == 0 || ch == 0) {
        offset = cv::Point(b[0], b[1]);
        return cv::Mat();
    }
    const cv::Mat crop = image(cv::Rect(b[0], b[1], cw, ch));
    const int side = square_diag ? static_cast<int>(std::sqrt(static_cast<double>(cw) * cw + static_cast<double>(ch) * ch))
                                 : std::max(cw, ch);
    const int ph = side - ch, pw = side - cw, left = pw / 2, top = ph / 2;
    cv::Mat out;
    // BORDER_ISOLATED: pad with black, not with the image around the crop (OpenCV would otherwise
    // fill the border from the parent image, unlike a numpy slice in the Python version)
    cv::copyMakeBorder(crop, out, top, ph - top, left, pw - left, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED,
                       cv::Scalar(0, 0, 0));
    offset = cv::Point(b[0] - left, b[1] - top);
    return out;
}

void nms(const std::vector<std::array<float, 4>>& boxes, const std::vector<double>& scores, float thr,
         std::vector<int>& keep) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<bool> removed(order.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(order[i]);
        for (size_t j = i + 1; j < order.size(); ++j)
            if (!removed[j] && iou4(boxes[order[i]].data(), boxes[order[j]].data()) > thr) removed[j] = true;
    }
}

}  // namespace

float iou4(const float* a, const float* b) {
    const float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0.f;
}

// --------------------------------------------------------------------------- palm detector
PalmDetector::PalmDetector(const std::string& engine_path, float score, float nms_iou)
    : model_(engine_path), score_(score), nms_iou_(nms_iou) {
    // SSD anchors: 24x24 grid with 2 per cell, then 12x12 with 6 per cell (2016 in all)
    const int grids[2][2] = {{24, 2}, {12, 6}};
    for (const auto& gn : grids)
        for (int y = 0; y < gn[0]; ++y)
            for (int x = 0; x < gn[0]; ++x)
                for (int k = 0; k < gn[1]; ++k)
                    anchors_.emplace_back((x + 0.5f) / gn[0], (y + 0.5f) / gn[0]);
    if (model_.tensor("Identity").count != anchors_.size() * 18) {
        std::fprintf(stderr, "[hands] unexpected palm detector outputs in %s\n", engine_path.c_str());
        std::exit(1);
    }
}

std::vector<PalmDet> PalmDetector::detect(const cv::Mat& bgr) {
    const int H = bgr.rows, W = bgr.cols, S = PALM_SIZE;
    const double ratio = std::min(static_cast<double>(S) / H, static_cast<double>(S) / W);
    const int nh = static_cast<int>(H * ratio), nw = static_cast<int>(W * ratio);
    cv::Mat img, padded, rgb;
    cv::resize(bgr, img, cv::Size(nw, nh));
    const int left = (S - nw) / 2, top = (S - nh) / 2;
    cv::copyMakeBorder(img, padded, top, S - nh - top, left, S - nw - left, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    Tensor& in = model_.tensor("input_1");  // (1, 192, 192, 3) NHWC, 0..1
    cv::Mat dst(S, S, CV_32FC3, in.host);
    rgb.convertTo(dst, CV_32F, 1.0 / 255.0);
    model_.infer();

    const float* reg = model_.tensor("Identity").host;     // (1, 2016, 18)
    const float* logit = model_.tensor("Identity_1").host; // (1, 2016, 1)
    const float pad_x = static_cast<float>(left / ratio), pad_y = static_cast<float>(top / ratio);
    const float scale = static_cast<float>(std::max(H, W));
    std::vector<std::array<float, 4>> boxes;
    std::vector<double> scores;
    std::vector<Palm> palms;
    for (size_t i = 0; i < anchors_.size(); ++i) {
        const double sc = 1.0 / (1.0 + std::exp(-static_cast<double>(logit[i])));
        if (sc < score_) continue;
        const float* r = reg + i * 18;
        const cv::Point2f a = anchors_[i];
        const float cx = r[0] / S + a.x, cy = r[1] / S + a.y, w = r[2] / S, h = r[3] / S;
        Palm p;
        p[0] = (cx - w / 2) * scale - pad_x;
        p[1] = (cy - h / 2) * scale - pad_y;
        p[2] = (cx + w / 2) * scale - pad_x;
        p[3] = (cy + h / 2) * scale - pad_y;
        for (int k = 0; k < 7; ++k) {
            p[4 + 2 * k] = (r[4 + 2 * k] / S + a.x) * scale - pad_x;
            p[5 + 2 * k] = (r[5 + 2 * k] / S + a.y) * scale - pad_y;
        }
        boxes.push_back({p[0], p[1], p[2], p[3]});
        scores.push_back(sc);
        palms.push_back(p);
    }
    std::vector<int> keep;
    nms(boxes, scores, nms_iou_, keep);
    std::vector<PalmDet> out;
    for (int i : keep) out.push_back({palms[i], static_cast<float>(scores[i])});
    return out;
}

// --------------------------------------------------------------------------- landmarks
HandLandmarker::HandLandmarker(const std::string& engine_path) : model_(engine_path) {
    if (model_.tensor("Identity").count != 63 || model_.tensor("Identity_3").count != 63) {
        std::fprintf(stderr, "[hands] unexpected hand landmark outputs in %s\n", engine_path.c_str());
        std::exit(1);
    }
}

std::optional<LandmarkResult> HandLandmarker::run(const cv::Mat& bgr, const Palm& palm) {
    // 1) generous crop around the palm, padded so rotation never cuts corners
    int cbox[4];
    cv::Point bias;
    cv::Mat crop = crop_pad(bgr, palm[0], palm[1], palm[2], palm[3], 0.0, PALM_BOX_PRE_ENLARGE, true, cbox, bias);
    if (crop.empty() || std::min(crop.rows, crop.cols) < 8) return std::nullopt;
    cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    double kx[7], ky[7];
    for (int k = 0; k < 7; ++k) {
        kx[k] = palm[4 + 2 * k] - bias.x;
        ky[k] = palm[5 + 2 * k] - bias.y;
    }
    // 2) rotate so the wrist -> middle-finger-base direction points up
    double rad = M_PI / 2 - std::atan2(-(ky[2] - ky[0]), kx[2] - kx[0]);
    rad = rad - 2 * M_PI * std::floor((rad + M_PI) / (2 * M_PI));
    const double angle = rad * 180.0 / M_PI;
    const double ccx = ((cbox[0] - bias.x) + (cbox[2] - bias.x)) / 2.0;
    const double ccy = ((cbox[1] - bias.y) + (cbox[3] - bias.y)) / 2.0;
    const cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(ccx), static_cast<float>(ccy)), angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(crop, rotated, rot, crop.size());
    const double* R = rot.ptr<double>();  // 2 x 3
    double mnx = 1e30, mny = 1e30, mxx = -1e30, mxy = -1e30;
    for (int k = 0; k < 7; ++k) {
        const double x = R[0] * kx[k] + R[1] * ky[k] + R[2], y = R[3] * kx[k] + R[4] * ky[k] + R[5];
        mnx = std::min(mnx, x);
        mny = std::min(mny, y);
        mxx = std::max(mxx, x);
        mxy = std::max(mxy, y);
    }
    // 3) final crop from the rotated keypoints' box, resized to 224
    int rbox[4];
    cv::Point unused;
    cv::Mat hand = crop_pad(rotated, mnx, mny, mxx, mxy, PALM_BOX_SHIFT_Y, PALM_BOX_ENLARGE, false, rbox, unused);
    if (hand.empty() || std::min(hand.rows, hand.cols) < 8) return std::nullopt;
    cv::Mat resized;
    cv::resize(hand, resized, cv::Size(HAND_SIZE, HAND_SIZE), 0, 0, cv::INTER_AREA);
    Tensor& in = model_.tensor("input_1");  // (1, 224, 224, 3) NHWC RGB, 0..1
    cv::Mat dst(HAND_SIZE, HAND_SIZE, CV_32FC3, in.host);
    resized.convertTo(dst, CV_32F, 1.0 / 255.0);
    model_.infer();
    const float* lm = model_.tensor("Identity").host;
    const float* world = model_.tensor("Identity_3").host;

    // 4) map landmarks back: crop -> rotated image -> frame
    LandmarkResult out;
    out.presence = model_.tensor("Identity_1").host[0];
    out.right_prob = model_.tensor("Identity_2").host[0];
    const double scale = std::max((rbox[2] - rbox[0]) / static_cast<double>(HAND_SIZE),
                                  (rbox[3] - rbox[1]) / static_cast<double>(HAND_SIZE));
    const double ca = std::cos(rad), sa = std::sin(rad);  // getRotationMatrix2D((0,0), angle): [[ca, sa], [-sa, ca]]
    // inverse of the rotation about the crop centre, applied to the final crop's centre
    const double rcx = (rbox[0] + rbox[2]) / 2.0, rcy = (rbox[1] + rbox[3]) / 2.0;
    const double tx = R[2], ty = R[5];
    const double ix = R[0] * rcx + R[3] * rcy - (R[0] * tx + R[3] * ty);
    const double iy = R[1] * rcx + R[4] * rcy - (R[1] * tx + R[4] * ty);
    for (int i = 0; i < 21; ++i) {
        const double x = (lm[3 * i] - HAND_SIZE / 2.0) * scale, y = (lm[3 * i + 1] - HAND_SIZE / 2.0) * scale;
        const double z = lm[3 * i + 2] * scale;
        out.landmarks[i] = cv::Point3f(static_cast<float>(x * ca - y * sa + ix + bias.x),
                                       static_cast<float>(x * sa + y * ca + iy + bias.y), static_cast<float>(z));
        const double wx = world[3 * i], wy = world[3 * i + 1];
        out.world[i] = cv::Point3f(static_cast<float>(wx * ca - wy * sa), static_cast<float>(wx * sa + wy * ca),
                                   world[3 * i + 2]);
    }
    return out;
}

std::array<float, 4> hand_box(const Points21& lm) {
    double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
    for (const auto& p : lm) {
        x0 = std::min(x0, static_cast<double>(p.x));
        y0 = std::min(y0, static_cast<double>(p.y));
        x1 = std::max(x1, static_cast<double>(p.x));
        y1 = std::max(y1, static_cast<double>(p.y));
    }
    const double h = y1 - y0;
    y0 += HAND_BOX_SHIFT_Y * h;
    y1 += HAND_BOX_SHIFT_Y * h;
    const double cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const double hw = (x1 - x0) * HAND_BOX_ENLARGE / 2, hh = (y1 - y0) * HAND_BOX_ENLARGE / 2;
    return {static_cast<float>(cx - hw), static_cast<float>(cy - hh), static_cast<float>(cx + hw),
            static_cast<float>(cy + hh)};
}

Palm palm_from_landmarks(const Points21& lm) {
    float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
    Palm p;
    for (int k = 0; k < 7; ++k) {
        const cv::Point3f& q = lm[PALM_FROM_HAND[k]];
        x0 = std::min(x0, q.x);
        y0 = std::min(y0, q.y);
        x1 = std::max(x1, q.x);
        y1 = std::max(y1, q.y);
        p[4 + 2 * k] = q.x;
        p[5 + 2 * k] = q.y;
    }
    const float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, half = std::max(x1 - x0, y1 - y0) / 2;
    p[0] = cx - half;
    p[1] = cy - half;
    p[2] = cx + half;
    p[3] = cy + half;
    return p;
}

// --------------------------------------------------------------------------- tracker
HandTracker::HandTracker(const std::string& engines, const HandPolicy& pol, bool mirrored_input)
    : palm_(engines + "/palm_detection_mediapipe_2023feb.fp16.engine"),
      lmk_(engines + "/handpose_estimation_mediapipe_2023feb.fp16.engine"),
      pol_(pol),
      mirrored_(mirrored_input) {}

void HandTracker::update(const cv::Mat& bgr, long frame_index, std::vector<std::string>& events) {
    // 1) follow every tracked hand from its previous landmarks
    std::vector<HandTrack> alive;
    for (HandTrack& t : tracks_) {
        auto t0 = Clock::now();
        auto r = lmk_.run(bgr, palm_from_landmarks(t.r.landmarks));
        ms_landmarks += ms_since(t0);
        if (!r || r->presence < pol_.min_presence) {
            char why[48];
            if (r)
                std::snprintf(why, sizeof(why), "presence %.2f", r->presence);
            else
                std::snprintf(why, sizeof(why), "crop outside frame");
            events.push_back("hand drop #" + std::to_string(t.id) + " lost (" + why + ")");
            continue;
        }
        t.r = *r;
        ++t.age;
        alive.push_back(t);
    }
    tracks_ = alive;

    // 2) look for new hands
    if (static_cast<int>(tracks_.size()) < pol_.max_hands && (tracks_.empty() || frame_index % pol_.palm_interval == 0)) {
        auto t0 = Clock::now();
        std::vector<PalmDet> palms = palm_.detect(bgr);
        ms_palm += ms_since(t0);
        for (const PalmDet& p : palms) {
            if (static_cast<int>(tracks_.size()) >= pol_.max_hands) break;
            bool known = false;
            for (const HandTrack& t : tracks_) known = known || same_hand(p.palm, t);
            if (known) continue;
            t0 = Clock::now();
            auto r = lmk_.run(bgr, p.palm);
            ms_landmarks += ms_since(t0);
            if (!r || r->presence < pol_.min_presence_new) continue;
            HandTrack t;
            t.id = next_id_++;
            t.r = *r;
            tracks_.push_back(t);
            events.push_back("hand new #" + std::to_string(t.id));
        }
    }

    // 3) two tracks on the same hand -> keep the older one
    std::vector<HandTrack> sorted = tracks_;
    std::stable_sort(sorted.begin(), sorted.end(), [](const HandTrack& a, const HandTrack& b) { return a.age > b.age; });
    std::vector<HandTrack> keep;
    for (const HandTrack& t : sorted) {
        const auto bt = hand_box(t.r.landmarks);
        bool dup = false;
        for (const HandTrack& k : keep) {
            const auto bk = hand_box(k.r.landmarks);
            dup = dup || iou4(bt.data(), bk.data()) > pol_.duplicate_iou;
        }
        if (dup) {
            events.push_back("hand drop #" + std::to_string(t.id) + " duplicate");
            continue;
        }
        keep.push_back(t);
    }
    std::sort(keep.begin(), keep.end(), [](const HandTrack& a, const HandTrack& b) { return a.id < b.id; });
    tracks_ = keep;
}

bool HandTracker::same_hand(const Palm& p, const HandTrack& t) const {
    // A detected palm belongs to a tracked hand if its centre lies inside that hand's box, or
    // the two palm boxes overlap (a palm box is much smaller than a hand box).
    const auto hb = hand_box(t.r.landmarks);
    const float cx = (p[0] + p[2]) / 2, cy = (p[1] + p[3]) / 2;
    if (hb[0] <= cx && cx <= hb[2] && hb[1] <= cy && cy <= hb[3]) return true;
    const Palm tp = palm_from_landmarks(t.r.landmarks);
    return iou4(p.data(), tp.data()) > pol_.overlap_iou;
}

std::string HandTracker::handedness(const HandTrack& t) const {
    // The model's score is "right hand" for an ordinary, un-mirrored camera image; a mirrored
    // (selfie) image swaps left and right.
    bool right = t.r.right_prob > 0.5f;
    if (mirrored_) right = !right;
    return right ? "Right" : "Left";
}

}  // namespace fht
