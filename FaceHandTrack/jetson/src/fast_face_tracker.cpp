#include "fast_face_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <tuple>

namespace fht {

namespace {

std::string fmt(const char* f, int id, double a, double b = 0) {
    char s[96];
    std::snprintf(s, sizeof(s), f, id, a, b);
    return s;
}

cv::Point2f centre(const cv::Rect2f& r) { return {r.x + r.width / 2, r.y + r.height / 2}; }

}  // namespace

cv::Rect2f FastTrack::predict(double now) const {
    const float dt = static_cast<float>(std::min(now - last_seen, 0.5));  // don't fling a box far away
    const cv::Point2f c = centre(box) + vel_c * dt;
    const float w = std::max(4.f, box.width + vel_s.x * dt), h = std::max(4.f, box.height + vel_s.y * dt);
    return {c.x - w / 2, c.y - h / 2, w, h};
}

bool FastFaceTracker::embeddable(const FaceDet& d) const {
    return emb_ && d.score >= pol_.start_score && std::min(d.box.width, d.box.height) >= pol_.min_embed_px;
}

void FastFaceTracker::lose(size_t i, double now, std::vector<std::string>& events, const char* why) {
    const FastTrack& t = tracks_[i];
    if (t.has_emb) lost_.push_back({t.id, t.emb, now});
    events.push_back("face drop #" + std::to_string(t.id) + " " + why);
    tracks_.erase(tracks_.begin() + static_cast<long>(i));
}

void FastFaceTracker::update(const std::vector<FaceDet>& dets, const cv::Mat& bgr, double now,
                             std::vector<std::string>& events) {
    // Forget lost IDs that have been gone too long.
    lost_.erase(std::remove_if(lost_.begin(), lost_.end(),
                               [&](const LostFace& l) { return now - l.lost_at > pol_.gallery_seconds; }),
                lost_.end());

    // 1. Link detections to tracks by overlap with each track's predicted box (best pairs first).
    std::vector<std::tuple<float, int, int>> pairs;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        const cv::Rect2f p = tracks_[ti].predict(now);
        for (size_t di = 0; di < dets.size(); ++di) {
            const float o = iou(p, dets[di].box);
            if (o >= pol_.match_iou) pairs.emplace_back(o, static_cast<int>(ti), static_cast<int>(di));
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<int> det_of(tracks_.size(), -1), track_of(dets.size(), -1);
    for (const auto& [o, ti, di] : pairs) {
        if (det_of[ti] >= 0 || track_of[di] >= 0) continue;
        det_of[ti] = di;
        track_of[di] = ti;
    }
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        FastTrack& t = tracks_[ti];
        if (det_of[ti] < 0) {
            t.visible = false;
            continue;
        }
        const FaceDet& d = dets[det_of[ti]];
        const double dt = now - t.last_seen;
        if (t.visible && dt > 1e-4) {
            const float k = 0.5f, inv = static_cast<float>(1.0 / dt);
            t.vel_c = t.vel_c * (1 - k) + (centre(d.box) - centre(t.box)) * (inv * k);
            t.vel_s = t.vel_s * (1 - k) + cv::Point2f(d.box.width - t.box.width, d.box.height - t.box.height) * (inv * k);
        } else {
            t.vel_c = t.vel_s = cv::Point2f(0, 0);
        }
        t.box = d.box;
        t.score = d.score;
        t.last_seen = now;
        t.visible = true;
    }

    // 2. Detections no track claimed: a returning face (fingerprint matches a missing track or a
    //    lost ID) or a new face. Most confident first.
    std::vector<int> fresh;
    for (size_t di = 0; di < dets.size(); ++di)
        if (track_of[di] < 0 && dets[di].score >= pol_.start_score) fresh.push_back(static_cast<int>(di));
    std::sort(fresh.begin(), fresh.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });
    for (int di : fresh) {
        const FaceDet& d = dets[di];
        Embedding e{};
        const bool has_e = embeddable(d);
        if (has_e) e = emb_->embed(bgr, d);
        int best_track = -1, best_lost = -1;
        float best = -1.f, second = -1.f;
        auto consider = [&](float s, int ti, int li) {
            if (s > best) {
                second = best;
                best = s, best_track = ti, best_lost = li;
            } else {
                second = std::max(second, s);
            }
        };
        if (has_e) {
            for (size_t ti = 0; ti < tracks_.size(); ++ti) {
                const FastTrack& t = tracks_[ti];
                if (t.visible || !t.has_emb) continue;  // only tracks that went missing
                consider(similarity(e, t.emb), static_cast<int>(ti), -1);
            }
            for (size_t li = 0; li < lost_.size(); ++li) consider(similarity(e, lost_[li].emb), -1, static_cast<int>(li));
        }
        if (best < pol_.reid_similarity || best - second < pol_.reid_margin) best_track = best_lost = -1;
        if (best_track >= 0) {  // missing for a moment, then back somewhere the prediction didn't expect
            FastTrack& t = tracks_[best_track];
            t.box = d.box;
            t.score = d.score;
            t.vel_c = t.vel_s = cv::Point2f(0, 0);
            t.last_seen = now;
            t.visible = true;
            events.push_back(fmt("face back #%d (similarity %.2f)", t.id, best));
            continue;
        }
        if (static_cast<int>(tracks_.size()) >= pol_.max_tracks) continue;
        FastTrack t;
        t.box = d.box;
        t.score = d.score;
        t.last_seen = now;
        t.has_emb = has_e;
        t.emb = e;
        t.last_embed = has_e ? now : -1e9;
        if (best_lost >= 0) {
            t.id = lost_[best_lost].id;
            events.push_back(fmt("face back #%d after %.1fs (similarity %.2f)", t.id, now - lost_[best_lost].lost_at, best));
            lost_.erase(lost_.begin() + best_lost);
        } else {
            t.id = next_id_++;
            events.push_back("face new #" + std::to_string(t.id));
        }
        tracks_.push_back(t);
        det_of.push_back(di);
    }

    // 3. Tracks missing too long are lost; their fingerprint stays in the gallery.
    for (size_t ti = tracks_.size(); ti-- > 0;)
        if (!tracks_[ti].visible && now - tracks_[ti].last_seen > pol_.max_missing) {
            lose(ti, now, events, "lost");
            det_of.erase(det_of.begin() + static_cast<long>(ti));
        }

    // 4. Refresh one visible track's fingerprint per frame (round robin), only from a clear,
    //    unobstructed view so an overlapping face doesn't blur it.
    if (!emb_ || tracks_.empty()) return;
    for (size_t k = 0; k < tracks_.size(); ++k) {
        const size_t ti = (refresh_next_ + k) % tracks_.size();
        FastTrack& t = tracks_[ti];
        if (!t.visible || det_of[ti] < 0 || now - t.last_embed < pol_.refresh_seconds) continue;
        const FaceDet& d = dets[det_of[ti]];
        if (!embeddable(d)) continue;
        bool overlapped = false;
        for (size_t j = 0; j < dets.size(); ++j)
            overlapped = overlapped || (static_cast<int>(j) != det_of[ti] && (d.box & dets[j].box).area() > 0);
        if (overlapped) continue;
        const Embedding e = emb_->embed(bgr, d);
        if (t.has_emb) {
            double n = 0;
            for (int i = 0; i < 128; ++i) {
                t.emb[i] = pol_.embed_ema * t.emb[i] + (1 - pol_.embed_ema) * e[i];
                n += static_cast<double>(t.emb[i]) * t.emb[i];
            }
            const float inv = static_cast<float>(1.0 / std::sqrt(std::max(n, 1e-12)));
            for (float& v : t.emb) v *= inv;
        } else {
            t.emb = e;
            t.has_emb = true;
        }
        t.last_embed = now;
        refresh_next_ = ti + 1;
        break;
    }
}

FastFacePipeline::FastFacePipeline(const std::string& engines, const FastFacePolicy& pol, bool reid)
    : detector_(engines + "/yunet_640x384.fp16.engine"),
      embedder_(reid ? std::make_unique<FaceEmbedder>(engines + "/face_recognition_sface_2021dec.fp16.engine") : nullptr),
      tracker_(embedder_.get(), pol) {
    params_.min_score = pol.keep_score;
    params_.nms_iou = pol.nms_iou;
    params_.min_px = pol.min_face_px;
}

void FastFacePipeline::process(const cv::Mat& bgr, double now, std::vector<std::string>& events) {
    frame_.upload(bgr, detector_.stream());
    dets_ = detector_.detect(frame_, params_);
    ms_detect_ += detector_.ms;
    tracker_.update(dets_, bgr, now, events);
}

std::vector<FaceOut> FastFacePipeline::faces() const {
    std::vector<FaceOut> out;
    for (const FastTrack& t : tracker_.tracks()) out.push_back({t.id, t.box, t.visible, t.score});
    return out;
}

std::string FastFacePipeline::timing(int frames) {
    char s[128];
    const double emb_ms = embedder_ ? embedder_->ms : 0;
    const int calls = embedder_ ? embedder_->calls : 0;
    std::snprintf(s, sizeof(s), "detect %.1f embed %.1f (%d, %.1f each)", ms_detect_ / frames, emb_ms / frames, calls,
                  calls ? emb_ms / calls : 0.0);
    ms_detect_ = 0;
    if (embedder_) embedder_->ms = 0, embedder_->calls = 0;
    return s;
}

}  // namespace fht
