#include "person_linker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <tuple>

namespace fht {

float hand_size(const std::vector<cv::Point2f>& p) {
    if (p.size() != 21) return 0.f;
    auto d = [&](int a, int b) { return std::hypot(p[a].x - p[b].x, p[a].y - p[b].y); };
    return std::max(d(0, 9), 1.3f * d(5, 17));
}

cv::Point2f mean_point(const std::vector<cv::Point2f>& p) {
    cv::Point2f s(0, 0);
    for (const auto& q : p) s += q;
    return p.empty() ? s : s * (1.f / static_cast<float>(p.size()));
}

bool PersonLinker::yaw(const std::vector<cv::Point2f>& lm, float& out) {
    if (lm.size() < 3) return false;
    const float ed = std::hypot(lm[1].x - lm[0].x, lm[1].y - lm[0].y);
    if (ed <= 1.f) return false;
    const cv::Point2f mid = (lm[0] + lm[1]) * 0.5f;
    const float ux = (lm[1].x - lm[0].x) / ed, uy = (lm[1].y - lm[0].y) / ed;
    out = std::fabs((lm[2].x - mid.x) * ux + (lm[2].y - mid.y) * uy) / ed;
    return true;
}

float PersonLinker::pose_distance(const LinkHand& h, int face_id) const {
    auto it = poses_.find(face_id);
    if (it == poses_.end() || h.points.size() != 21) return -1;
    float best = -1;
    for (const PoseWrist& w : it->second)
        if (w.visibility >= policy.pose_visible) {
            const float d = std::hypot(h.points[0].x - w.point.x, h.points[0].y - w.point.y);
            if (best < 0 || d < best) best = d;
        }
    return best < 0 ? -1 : best / std::max(hand_size(h.points), 1.f);
}

bool PersonLinker::cost(const LinkHand& h, const LinkFace& f, float& out) const {
    const LinkPolicy& p = policy;
    const float fw = std::max(f.box.width, 1.f);
    const cv::Point2f hc = mean_point(h.points);
    const float dx = (hc.x - (f.box.x + f.box.width / 2)) / fw, dy = (hc.y - (f.box.y + f.box.height / 2)) / fw;
    const float dist = std::hypot(dx, dy);
    if (dist > p.max_reach) return false;
    const float size = hand_size(h.points);
    if (size <= 0) return false;
    const float ratio = size / fw;
    if (ratio < p.min_ratio || ratio > p.max_ratio) return false;
    float c = dist + p.size_weight * std::fabs(std::log(ratio / p.hand_face_ratio));
    // un-mirrored camera: the person's right hand appears on the image's left (smaller x)
    const bool right_is_left_in_image = !mirrored;
    const bool on_left = dx < -0.5f, on_right = dx > 0.5f;
    if (h.handedness == "Right" && (right_is_left_in_image ? on_right : on_left)) c += p.wrong_side_penalty;
    if (h.handedness == "Left" && (right_is_left_in_image ? on_left : on_right)) c += p.wrong_side_penalty;
    if (dy < -1.5f) c += p.above_head_penalty;
    auto it = owner_.find(h.id);
    if (it != owner_.end() && it->second == f.id) c -= p.sticky_bonus;
    // body pose: the strongest evidence when it is there
    const float pd = pose_distance(h, f.id);
    if (pd >= 0) {
        int visible = 0;
        for (const PoseWrist& w : poses_.at(f.id)) visible += w.visibility >= p.pose_visible;
        if (pd <= p.pose_match) c -= p.pose_bonus;
        else if (pd > p.pose_elsewhere && visible == 2) return false;  // both wrists seen, both elsewhere
        else if (pd > p.pose_elsewhere) c += p.pose_penalty;
    }
    out = c;
    return true;
}

void PersonLinker::reset() {
    owner_.clear();
    people_.clear();
    facing_.clear();
    held_.clear();
    addressing_.clear();
}

void PersonLinker::update(const std::vector<LinkFace>& faces, const std::vector<LinkHand>& hands, double now,
                          std::vector<std::string>& events, const std::map<int, std::vector<PoseWrist>>& poses) {
    const LinkPolicy& p = policy;
    poses_ = poses;
    std::vector<const LinkFace*> visible;
    for (const auto& f : faces)
        if (f.visible) visible.push_back(&f);
    for (const auto& f : faces) {
        float y;
        if (yaw(f.landmarks, y)) facing_[f.id] = y <= p.max_yaw;
    }

    // 1. hands -> people: cheapest pairs first, at most one Left and one Right per person
    std::vector<std::tuple<float, size_t, size_t>> pairs;
    for (size_t hi = 0; hi < hands.size(); ++hi)
        for (size_t fi = 0; fi < visible.size(); ++fi) {
            float c;
            if (cost(hands[hi], *visible[fi], c)) pairs.emplace_back(c, hi, fi);
        }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    std::map<int, int> new_owner;
    std::set<std::string> taken;
    for (const auto& [c, hi, fi] : pairs) {
        const LinkHand& h = hands[hi];
        const LinkFace& f = *visible[fi];
        if (new_owner.count(h.id)) continue;
        const std::string slot = std::to_string(f.id) + "-" + h.handedness;
        if (taken.count(slot)) continue;
        new_owner[h.id] = f.id;
        taken.insert(slot);
    }
    for (const auto& h : hands) {
        auto n = new_owner.find(h.id);
        auto o = owner_.find(h.id);
        if (n == new_owner.end() || (o != owner_.end() && o->second == n->second)) continue;
        for (const LinkFace* f : visible)
            if (f->id == n->second) {  // printed so the rules can be tuned on real people
                const float fw = std::max(f->box.width, 1.f);
                const cv::Point2f hc = mean_point(h.points);
                char s[200];
                int n = std::snprintf(s, sizeof(s), "hand #%d (%s) -> face #%d (reach %.1f, size %.2f", h.id, h.handedness.c_str(),
                              f->id, std::hypot(hc.x - (f->box.x + f->box.width / 2), hc.y - (f->box.y + f->box.height / 2)) / fw,
                              hand_size(h.points) / fw);
                const float pd = pose_distance(h, f->id);
                if (pd >= 0) n += std::snprintf(s + n, sizeof(s) - n, ", pose wrist %.1f hand sizes away", pd);
                std::snprintf(s + n, sizeof(s) - n, ")");
                events.push_back(s);
            }
    }
    const std::map<int, int> prev_owner = owner_;
    owner_ = new_owner;

    // 2. held gestures (per hand; restart when the gesture ends or the hand changes person)
    std::set<int> live;
    for (const auto& h : hands) live.insert(h.id);
    for (auto it = held_.begin(); it != held_.end();) it = live.count(it->first) ? std::next(it) : held_.erase(it);
    for (const auto& h : hands) {
        const auto po = prev_owner.find(h.id);
        const auto no = owner_.find(h.id);
        const int was = po == prev_owner.end() ? -1 : po->second, is = no == owner_.end() ? -1 : no->second;
        if (h.gesture.empty() || was != is) held_.erase(h.id);
        if (!h.gesture.empty() && (!held_.count(h.id) || held_[h.id].gesture != h.gesture)) held_[h.id] = {h.gesture, now};
    }

    // 3. who is addressing the robot
    std::map<int, PersonState> state;
    for (const LinkFace* f : visible) {
        PersonState s;
        s.face_id = f->id;
        auto it = facing_.find(f->id);
        s.facing = it == facing_.end() ? -1 : (it->second ? 1 : 0);
        state[f->id] = s;
    }
    for (const auto& h : hands) {
        auto o = owner_.find(h.id);
        if (o != owner_.end() && state.count(o->second)) state[o->second].hand_ids.push_back(h.id);
    }
    std::set<int> visible_ids;
    for (const LinkFace* f : visible) {
        visible_ids.insert(f->id);
        PersonState& s = state[f->id];
        const Held* best = nullptr;  // longest-held gesture among this person's hands
        for (int hid : s.hand_ids) {
            auto it = held_.find(hid);
            if (it != held_.end() && now - it->second.since >= p.hold_seconds && (!best || it->second.since < best->since))
                best = &it->second;
        }
        if (best && s.facing == 1) {
            auto a = addressing_.find(f->id);
            if (a == addressing_.end() || a->second.gesture != best->gesture || now > a->second.until)
                events.push_back("face #" + std::to_string(f->id) + " addressing: " + best->gesture);
            addressing_[f->id] = {best->gesture, now + p.linger_seconds};
        }
        auto a = addressing_.find(f->id);
        if (a != addressing_.end() && now <= a->second.until) s.addressing = a->second.gesture;
    }
    for (auto it = addressing_.begin(); it != addressing_.end();)
        it = visible_ids.count(it->first) && now <= it->second.until ? std::next(it) : addressing_.erase(it);
    std::set<int> face_ids;
    for (const auto& f : faces) face_ids.insert(f.id);
    for (auto it = facing_.begin(); it != facing_.end();) it = face_ids.count(it->first) ? std::next(it) : facing_.erase(it);
    people_ = state;
}

}  // namespace fht
