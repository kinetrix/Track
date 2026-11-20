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
    if (it == poses_.end() || it->second.wrists.size() != 2 || h.points.size() != 21) return -1;
    const PoseWrist& w = it->second.wrists[h.handedness == "Left" ? 0 : 1];
    if (w.visibility < policy.pose_visible) return -1;
    return std::hypot(h.points[0].x - w.point.x, h.points[0].y - w.point.y) / std::max(hand_size(h.points), 1.f);
}

float PersonLinker::pose_distance_any(const LinkHand& h, int face_id) const {
    auto it = poses_.find(face_id);
    if (it == poses_.end() || h.points.size() != 21) return -1;
    float best = -1;
    for (const PoseWrist& w : it->second.wrists)
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
    // depth: one camera can't tell a hand a metre behind someone from theirs, the depth stream can
    if (h.depth > 0 && f.depth > 0) {
        const float dz = h.depth - f.depth;  // < 0: hand nearer the camera than the face
        if (dz < -p.depth_front || dz > p.depth_behind) return false;
        c += p.depth_weight * std::fabs(dz);
    }
    auto arm = h.arm.find(f.id);
    if (arm != h.arm.end() && arm->second >= 0) c += p.arm_weight * (1 - arm->second);
    // body pose: the strongest evidence when it is there
    // (20 Nov: a hard "both wrists elsewhere = not theirs" at 2.5 hand sizes excluded the friend,
    // whose pose wrist was 2.6-3.3 away, and handed his hand to the only one left: the nearer person)
    const float any = pose_distance_any(h, f.id);
    if (any > p.pose_cap) {
        int visible = 0;
        for (const PoseWrist& w : poses_.at(f.id).wrists) visible += w.visibility >= p.pose_visible;
        if (visible == 2) return false;  // both wrists seen, both far away
    }
    const float pd = pose_distance(h, f.id);
    if (pd >= 0 && pd <= p.pose_match) {
        float rival = -1;  // another pose claiming the same hand: only a clearly closer claim counts
        for (const auto& kv : poses_) {
            const float d = kv.first == f.id ? -1 : pose_distance(h, kv.first);
            if (d >= 0 && d <= p.pose_match && (rival < 0 || d < rival)) rival = d;
        }
        if (rival < 0 || pd <= p.pose_clearly_closer * rival) c -= p.pose_bonus;
    } else {
        float d = pd;
        if (any >= 0 && (d < 0 || any + p.handedness_slack < d)) d = any + p.handedness_slack;
        if (d >= 0) c += p.pose_weight * std::min(d, p.pose_cap);
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
    pose_ignored_.clear();
    pending_.clear();
    last_seen_.clear();
}

void PersonLinker::update(const std::vector<LinkFace>& faces, const std::vector<LinkHand>& hands, double now,
                          std::vector<std::string>& events, const std::map<int, LinkPose>& poses) {
    const LinkPolicy& p = policy;
    std::vector<const LinkFace*> visible;
    for (const auto& f : faces)
        if (f.visible) visible.push_back(&f);
    // a pose found from a crop around a face can latch onto another person in the crop
    poses_.clear();
    for (const LinkFace* f : visible) {
        auto it = poses.find(f->id);
        if (it == poses.end()) continue;
        const float off = std::hypot(it->second.nose.x - (f->box.x + f->box.width / 2),
                                     it->second.nose.y - (f->box.y + f->box.height / 2)) / std::max(f->box.height, 1.f);
        if (off <= p.pose_nose_reach) {
            poses_[f->id] = it->second;
            pose_ignored_.erase(f->id);
        } else if (pose_ignored_.insert(f->id).second) {
            char s[160];
            std::snprintf(s, sizeof(s), "pose for face #%d ignored: its nose is %.1f face heights from the face (someone else's body)", f->id, off);
            events.push_back(s);
        }
    }
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
    // a hand keeps its person until the new one has won switch_frames frames in a row, unless its
    // person is gone, can no longer own it, or already has another hand on that side
    std::set<int> live_hands;
    for (const auto& h : hands) live_hands.insert(h.id);
    for (auto it = pending_.begin(); it != pending_.end();) it = live_hands.count(it->first) ? std::next(it) : pending_.erase(it);
    for (const auto& h : hands) {
        auto o = owner_.find(h.id);
        auto n = new_owner.find(h.id);
        if (o == owner_.end() || n == new_owner.end() || n->second == o->second) {
            pending_.erase(h.id);
            continue;
        }
        const LinkFace* fo = nullptr;
        for (const LinkFace* f : visible)
            if (f->id == o->second) fo = f;
        float c;
        if (!fo || !cost(h, *fo, c)) {
            pending_.erase(h.id);
            continue;
        }
        auto pd = pending_.find(h.id);
        if (pd == pending_.end() || pd->second.face != n->second) pd = pending_.insert_or_assign(h.id, Pending{n->second, 0}).first;
        if (++pd->second.frames >= p.switch_frames) {
            pending_.erase(pd);
            continue;
        }
        bool side_free = true;
        for (const auto& h2 : hands) {
            auto o2 = new_owner.find(h2.id);
            if (h2.id != h.id && o2 != new_owner.end() && o2->second == o->second && h2.handedness == h.handedness) side_free = false;
        }
        if (side_free) new_owner[h.id] = o->second;
    }
    for (const auto& h : hands) {
        auto n = new_owner.find(h.id);
        auto o = owner_.find(h.id);
        if (n == new_owner.end() || (o != owner_.end() && o->second == n->second)) continue;
        for (const LinkFace* f : visible)
            if (f->id == n->second) {  // printed so the rules can be tuned on real people
                const float fw = std::max(f->box.width, 1.f);
                const cv::Point2f hc = mean_point(h.points);
                char b[96];
                std::snprintf(b, sizeof(b), "hand #%d (%s) -> face #%d (reach %.1f, size %.2f", h.id, h.handedness.c_str(),
                              f->id, std::hypot(hc.x - (f->box.x + f->box.width / 2), hc.y - (f->box.y + f->box.height / 2)) / fw,
                              hand_size(h.points) / fw);
                std::string line = b;
                if (h.depth > 0 && f->depth > 0) {
                    std::snprintf(b, sizeof(b), ", depth hand %.2f m face %.2f m", h.depth, f->depth);
                    line += b;
                }
                for (const auto& [fid, v] : h.arm)
                    if (v >= 0) {
                        std::snprintf(b, sizeof(b), ", arm to #%d %.2f", fid, v);
                        line += b;
                    }
                const char* sep = ", pose wrist hand sizes away: ";
                for (const auto& kv : poses_) {
                    const float pd = pose_distance(h, kv.first);
                    if (pd < 0) continue;
                    std::snprintf(b, sizeof(b), "%s#%d %.1f", sep, kv.first, pd);
                    line += b;
                    sep = ", ";
                }
                events.push_back(line + ")");
            }
    }
    const std::map<int, int> prev_owner = owner_;
    // hands missing for a moment keep their person (and so their stickiness) if they come back
    for (const auto& h : hands) last_seen_[h.id] = now;
    for (auto it = last_seen_.begin(); it != last_seen_.end();) {
        if (now - it->second > p.remember_seconds) {
            it = last_seen_.erase(it);
            continue;
        }
        auto o = owner_.find(it->first);
        if (!live_hands.count(it->first) && o != owner_.end()) new_owner[it->first] = o->second;
        ++it;
    }
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
