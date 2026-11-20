#include "attention_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "person_linker.hpp"

namespace fht {

std::string AttentionTarget::turn_text() const {
    char s[64];
    std::string h = std::fabs(yaw) < 5 ? "facing" : "";
    if (h.empty()) {
        std::snprintf(s, sizeof(s), "turn %s %.0f deg", yaw > 0 ? "right" : "left", std::fabs(yaw));
        h = s;
    }
    if (std::fabs(pitch) >= 5) {
        std::snprintf(s, sizeof(s), ", %s %.0f deg", pitch > 0 ? "up" : "down", std::fabs(pitch));
        h += s;
    }
    return h;
}

int count_swings(const std::vector<float>& xs, float amp) {
    if (xs.empty() || amp <= 0) return 0;
    float lo = xs[0], hi = xs[0], ext = xs[0];
    int dir = 0, n = 0;
    for (size_t i = 1; i < xs.size(); ++i) {
        const float x = xs[i];
        if (dir == 0) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
            if (x - lo >= amp) { dir = 1; n = 1; ext = x; }
            else if (hi - x >= amp) { dir = -1; n = 1; ext = x; }
        } else if (dir == 1) {
            if (x > ext) ext = x;
            else if (ext - x >= amp) { dir = -1; ++n; ext = x; }
        } else {
            if (x < ext) ext = x;
            else if (x - ext >= amp) { dir = 1; ++n; ext = x; }
        }
    }
    return n;
}

void AttentionTracker::reset() {
    history_.clear();
    last_wave_.clear();
    waving_until_.clear();
    target_id_ = -1;
    target_ = AttentionTarget();
    was_addressing_.clear();
}

void AttentionTracker::update(const std::vector<AttentionFace>& faces, const std::vector<AttentionHand>& hands,
                              cv::Size frame, double now, std::vector<std::string>& events) {
    const AttentionPolicy& p = policy;
    std::map<int, const AttentionFace*> face_by_id;
    for (const auto& f : faces) face_by_id[f.id] = &f;
    for (auto it = history_.begin(); it != history_.end();) {
        auto& v = it->second;
        v.erase(std::remove_if(v.begin(), v.end(), [&](const Sample& s) { return now - s.t > p.window; }), v.end());
        it = v.empty() ? history_.erase(it) : std::next(it);
    }

    // 1. waves, per person and hand side (open-hand frames only; blurred frames are skipped)
    for (const auto& h : hands) {
        if (h.points.size() != 21) continue;
        if (!(h.extended[1] && h.extended[2] && h.extended[3] && h.extended[4])) continue;
        const std::string series = (h.owner >= 0 ? "f" + std::to_string(h.owner) : "h" + std::to_string(h.id)) + "-" + h.handedness;
        const cv::Point2f c = mean_point(h.points);
        auto f = face_by_id.find(h.owner);
        const AttentionFace* face = f == face_by_id.end() ? nullptr : f->second;
        // relative to the person's face: the robot turning moves both together
        const float x = face ? c.x - (face->box.x + face->box.width / 2) : c.x;
        auto& hist = history_[series];
        hist.push_back({now, x, hand_size(h.points)});
        std::vector<float> sizes, xs;
        for (const auto& s : hist) { sizes.push_back(s.size); xs.push_back(s.x); }
        std::nth_element(sizes.begin(), sizes.begin() + sizes.size() / 2, sizes.end());
        if (count_swings(xs, p.swing_fraction * sizes[sizes.size() / 2]) < p.min_swings) continue;
        if (face && c.y > face->box.y + face->box.height * (1 + p.raised_below_chin)) continue;  // too low
        hist.clear();
        const std::string key = h.owner >= 0 ? "f" + std::to_string(h.owner) : "h" + std::to_string(h.id);
        auto lw = last_wave_.find(key);
        if (lw != last_wave_.end() && now - lw->second < p.wave_cooldown) continue;
        last_wave_[key] = now;
        if (face) {
            waving_until_[face->id] = now + p.waving_shown;
            events.push_back("face #" + std::to_string(face->id) + " waved: calling the robot");
            if (target_id_ != face->id) events.push_back("robot attention -> face #" + std::to_string(face->id) + " (waved)");
            target_id_ = face->id;
            target_reason_ = "waved";
            target_last_active_ = now;
        } else {
            events.push_back("hand #" + std::to_string(h.id) + " waved (face not seen)");
        }
    }
    for (auto it = waving_until_.begin(); it != waving_until_.end();) it = now <= it->second ? std::next(it) : waving_until_.erase(it);

    // 2. someone starting to address the robot gets its attention if no one has it
    std::set<int> addressing_now;
    for (const auto& f : faces)
        if (f.addressing) addressing_now.insert(f.id);
    if (target_id_ >= 0 && addressing_now.count(target_id_)) target_last_active_ = now;
    if (target_id_ < 0)
        for (int id : addressing_now)
            if (!was_addressing_.count(id)) {
                target_id_ = id;
                target_reason_ = "addressing";
                target_last_active_ = now;
                events.push_back("robot attention -> face #" + std::to_string(id) + " (addressing)");
                break;
            }
    was_addressing_ = addressing_now;

    // 3. keep or drop the target, and where to turn
    target_ = AttentionTarget();
    if (target_id_ < 0) return;
    auto f = face_by_id.find(target_id_);
    if (f != face_by_id.end()) {
        target_last_seen_ = now;
        const double w = frame.width, hgt = frame.height;
        const double focal = (w / 2) / std::tan(p.horizontal_fov / 2 * M_PI / 180);
        const cv::Rect2f& b = f->second->box;
        double yaw = std::atan((b.x + b.width / 2 - w / 2) / focal) * 180 / M_PI;  // + = image right = robot's right
        if (mirrored) yaw = -yaw;
        const double pitch = -std::atan((b.y + b.height / 2 - hgt / 2) / focal) * 180 / M_PI;
        target_.face_id = target_id_;
        target_.yaw = yaw;
        target_.pitch = pitch;
        target_.reason = target_reason_;
    }
    const bool gone = now - target_last_seen_ > p.lost_after, idle = now - target_last_active_ > p.idle_after;
    if (gone || idle) {
        events.push_back("robot attention released (face #" + std::to_string(target_id_) + (gone ? " out of sight)" : " idle)"));
        target_id_ = -1;
        target_ = AttentionTarget();
    }
}

}  // namespace fht
