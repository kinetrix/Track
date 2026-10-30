#include "gestures.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fht {

namespace {

constexpr double EXT_RATIO = 1.15;       // fingertip at least this much farther from the wrist than the PIP joint
constexpr double EXT_BEND = 80.0;        // and the finger bends less than this (degrees, PIP + DIP)
constexpr double THUMB_OUT = 0.85;       // thumb tip distance from ring-finger base, in palm lengths
constexpr double THUMB_BEND = 60.0;      // thumb bends less than this (degrees, MCP + IP)
constexpr double PINCH = 0.25;           // thumb tip - index tip distance, in palm lengths
constexpr double INDEX_NOT_FOLDED = 0.88;
constexpr double INDEX_CLEAR = 1.25;
constexpr int FINGERS[4][4] = {{5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}};

struct V3 {
    double x, y, z;
};
V3 v(const cv::Point3f& p) { return {p.x, p.y, p.z}; }
V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double norm(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
double angle(V3 a, V3 b) {
    double c = (a.x * b.x + a.y * b.y + a.z * b.z) / (norm(a) * norm(b) + 1e-9);
    c = std::clamp(c, -1.0, 1.0);
    return std::acos(c) * 180.0 / M_PI;
}
double dist(const cv::Point3f& a, const cv::Point3f& b) { return norm(sub(v(a), v(b))); }

// screen direction of an image-space vector (y grows downwards)
std::string direction(double dx, double dy) {
    if (std::abs(dy) >= std::abs(dx)) return dy < 0 ? "up" : "down";
    return dx < 0 ? "left" : "right";
}

}  // namespace

Gesture classify(const Points21& lm, const Points21& w) {
    // finger states from the world landmarks
    const double palm = dist(w[9], w[0]) + 1e-9;
    const V3 t0 = sub(v(w[3]), v(w[2])), t1 = sub(v(w[4]), v(w[3]));
    const bool T = dist(w[4], w[13]) / palm > THUMB_OUT && angle(t0, t1) + angle(sub(v(w[2]), v(w[1])), t0) < THUMB_BEND;
    bool ext[4];
    for (int f = 0; f < 4; ++f) {
        const int mcp = FINGERS[f][0], pip = FINGERS[f][1], dip = FINGERS[f][2], tip = FINGERS[f][3];
        const V3 s0 = sub(v(w[pip]), v(w[mcp])), s1 = sub(v(w[dip]), v(w[pip])), s2 = sub(v(w[tip]), v(w[dip]));
        const double bend = angle(s0, s1) + angle(s1, s2);
        const double ratio = dist(w[tip], w[0]) / (dist(w[pip], w[0]) + 1e-9);
        ext[f] = ratio > EXT_RATIO && bend < EXT_BEND;
    }
    const bool I = ext[0], M = ext[1], R = ext[2], P = ext[3];
    Gesture g;
    g.fingers = T + I + M + R + P;

    const double palm_img = dist(lm[9], lm[0]) + 1e-9;
    const double index_ratio = dist(w[8], w[0]) / (dist(w[6], w[0]) + 1e-9);
    const bool pinch = dist(lm[4], lm[8]) / palm_img < PINCH && index_ratio > INDEX_NOT_FOLDED;

    if (pinch) {
        g.label = M && R && P ? "OK" : "Pinch";
    } else if (I && M && R && P) {
        g.label = "Open palm";
    } else if (!(I || M || R || P)) {
        if (!T) {
            g.label = "Fist";
        } else {
            const std::string d = direction(lm[4].x - lm[2].x, lm[4].y - lm[2].y);
            if (d == "up") g.label = "Thumbs up";
            if (d == "down") g.label = "Thumbs down";
        }
    } else if (I && !(M || R || P)) {
        g.label = "Pointing " + direction(lm[8].x - lm[5].x, lm[8].y - lm[5].y);
    } else if (I && M && !(R || P) && !T) {
        g.label = "Victory";
    } else if (I && M && R && !P && !T) {
        g.label = "Three";
    } else if (I && P && !(M || R)) {
        g.label = !T ? "Rock" : (index_ratio > INDEX_CLEAR ? "I love you" : "Call me");
    } else if (T && P && !(I || M || R)) {
        g.label = "Call me";
    }
    return g;
}

const std::string& GestureSmoother::operator()(const std::string& label) {
    hist_.push_back(label);
    if (static_cast<int>(hist_.size()) > window_) hist_.pop_front();
    // most frequent label; ties go to the one seen first (as Python's Counter.most_common)
    std::vector<std::pair<std::string, int>> counts;
    for (const std::string& h : hist_) {
        auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& c) { return c.first == h; });
        if (it == counts.end())
            counts.emplace_back(h, 1);
        else
            ++it->second;
    }
    auto best = counts.begin();
    for (auto it = counts.begin(); it != counts.end(); ++it)
        if (it->second > best->second) best = it;
    if (best->second >= need_) stable_ = best->first;
    return stable_;
}

}  // namespace fht
