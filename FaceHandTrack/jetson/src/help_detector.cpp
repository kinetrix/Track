#include "help_detector.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <set>

namespace fht {

std::string HelpAlert::who() const {
    if (face_id < 0) return "someone (face not seen)";
    return name.empty() ? "face #" + std::to_string(face_id) : name + " (#" + std::to_string(face_id) + ")";
}

void HelpDetector::reset() {
    tucked_.clear();
    hands_up_.clear();
    last_alert_.clear();
    signals_.clear();
}

std::map<int, double> HelpDetector::pending() const {
    std::map<int, double> out;
    for (const auto& [k, v] : signals_)
        if (!v.empty()) out[k] = v.back();
    return out;
}

void HelpDetector::raise(const std::string& reason, const HelpFace* face, int face_id, int hand_id, double now) {
    const std::string key = face_id >= 0 ? "f" + std::to_string(face_id) : "h" + std::to_string(hand_id);
    auto it = last_alert_.find(key);
    if (it != last_alert_.end() && now - it->second < policy.cooldown) return;
    last_alert_[key] = now;
    HelpAlert a;
    a.id = next_id_++;
    a.face_id = face_id;
    a.name = face ? face->name : "";
    a.reason = reason;
    a.time = now;
    char buf[16];
    const std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    a.clock = buf;
    alerts_.push_back(a);
    new_alerts_.push_back(a);
}

void HelpDetector::signal(int owner, int hand_id, const HelpFace* face, double now, std::vector<std::string>& events) {
    const int key = owner >= 0 ? owner : -1 - hand_id;
    auto& times = signals_[key];
    std::vector<double> kept;
    for (double t : times)
        if (now - t <= policy.repeat_within) kept.push_back(t);
    kept.push_back(now);
    times = kept;
    const std::string who = owner < 0 ? "someone (face not seen)"
                                      : (face && !face->name.empty() ? face->name + " (#" + std::to_string(owner) + ")"
                                                                     : "face #" + std::to_string(owner));
    if (static_cast<int>(times.size()) >= policy.repeats) {
        times.clear();
        raise("Signal for Help", face, owner, hand_id, now);
    } else {
        events.push_back("help signal " + std::to_string(times.size()) + " of " + std::to_string(policy.repeats) +
                         " from " + who + " (repeat within " + std::to_string(static_cast<int>(policy.repeat_within)) +
                         " s to confirm)");
    }
}

void HelpDetector::update(const std::vector<HelpFace>& faces, const std::vector<HelpHand>& hands, double now,
                          std::vector<std::string>& events) {
    new_alerts_.clear();
    const HelpPolicy& p = policy;
    std::map<int, const HelpFace*> face_by_id;
    for (const auto& f : faces) face_by_id[f.id] = &f;
    std::set<int> live;
    for (const auto& h : hands) live.insert(h.id);
    for (auto it = tucked_.begin(); it != tucked_.end();) it = live.count(it->first) ? std::next(it) : tucked_.erase(it);
    for (auto it = signals_.begin(); it != signals_.end();) {  // forget unconfirmed signals
        while (!it->second.empty() && now - it->second.front() > p.repeat_within) it->second.erase(it->second.begin());
        it = it->second.empty() ? signals_.erase(it) : std::next(it);
    }

    // Signal for Help, per hand
    for (const auto& h : hands) {
        const auto& e = h.extended;
        const bool is_tucked = !e[0] && e[1] && e[2] && e[3] && e[4];
        const bool is_closed = !e[1] && !e[2] && !e[3] && !e[4];
        auto still = [&](const cv::Point2f& q) {
            return std::hypot(h.wrist.x - q.x, h.wrist.y - q.y) <= p.still_fraction * std::max(h.size, 1.f);
        };
        auto it = tucked_.find(h.id);
        if (is_tucked) {
            if (it != tucked_.end() && now - it->second.last <= p.tucked_gap && still(it->second.start))
                it->second = {it->second.since, now, it->second.start, h.wrist};
            else
                tucked_[h.id] = {now, now, h.wrist, h.wrist};  // (re)start: new, broken off, or moving
        } else if (it != tucked_.end()) {
            const Tucked t = it->second;
            if (is_closed && t.last - t.since >= p.tucked_hold && now - t.last <= p.fold_within && still(t.at)) {
                tucked_.erase(it);
                auto f = face_by_id.find(h.owner);
                signal(h.owner, h.id, f == face_by_id.end() ? nullptr : f->second, now, events);
            } else if (now - t.last > p.fold_within) {
                tucked_.erase(it);
            }
        }
    }

    // Both hands up, per person
    std::set<int> up_now;
    for (const auto& f : faces) {
        int mine = 0, raised = 0, first = -1;
        for (const auto& h : hands) {
            if (h.owner != f.id) continue;
            ++mine;
            if (h.gesture == "Open palm" && h.center.y < f.box.y + f.box.height) {
                ++raised;
                if (first < 0) first = h.id;
            }
        }
        if (mine < 2 || raised < 2) continue;
        up_now.insert(f.id);
        if (!hands_up_.count(f.id)) hands_up_[f.id] = now;
        if (now - hands_up_[f.id] >= p.hands_up_hold) {
            raise("Both hands up", &f, f.id, first, now);
            hands_up_.erase(f.id);
            up_now.erase(f.id);
        }
    }
    for (auto it = hands_up_.begin(); it != hands_up_.end();) it = up_now.count(it->first) ? std::next(it) : hands_up_.erase(it);
}

}  // namespace fht
