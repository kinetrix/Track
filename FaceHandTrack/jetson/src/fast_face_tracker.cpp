#include "fast_face_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <tuple>

namespace fht {

namespace {

cv::Point2f centre(const cv::Rect2f& r) { return {r.x + r.width / 2, r.y + r.height / 2}; }

std::string fmt(const char* f, ...) {
    char s[192];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(s, sizeof(s), f, ap);
    va_end(ap);
    return s;
}

Embedding normalized(const Embedding& v) {
    double n = 0;
    for (float x : v) n += static_cast<double>(x) * x;
    Embedding o = v;
    const float inv = static_cast<float>(1.0 / std::sqrt(std::max(n, 1e-12)));
    for (float& x : o) x *= inv;
    return o;
}

void take(FastTrack& t, const FaceDet& d, double now, bool keep_velocity) {
    const double dt = now - t.last_seen;
    if (keep_velocity && t.visible && dt > 1e-4) {
        const float k = 0.5f, inv = static_cast<float>(1.0 / dt);
        t.vel_c = t.vel_c * (1 - k) + (centre(d.box) - centre(t.box)) * (inv * k);
        t.vel_s = t.vel_s * (1 - k) + cv::Point2f(d.box.width - t.box.width, d.box.height - t.box.height) * (inv * k);
    } else {
        t.vel_c = t.vel_s = cv::Point2f(0, 0);
    }
    t.box = d.box;
    t.landmarks = d.landmarks;
    t.score = d.score;
    t.last_seen = now;
    t.visible = true;
}

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

bool FastFaceTracker::known_name(const Embedding& e, std::string& out) const {
    float best = -2, second = -2;
    std::string best_name;
    for (const KnownFace& k : known) {
        const float s = similarity(e, k.emb);
        if (k.name == best_name) { best = std::max(best, s); continue; }
        if (s > best) {
            if (!best_name.empty()) second = std::max(second, best);
            best = s;
            best_name = k.name;
        } else {
            second = std::max(second, s);
        }
    }
    if (best_name.empty() || best < pol_.reid_similarity || best - second < pol_.reid_margin) return false;
    out = best_name;
    return true;
}

int FastFaceTracker::lost_match(const Embedding& e, float& best, int& best_id) const {
    float second = -2;
    int bi = -1;
    best = -2;
    for (size_t li = 0; li < lost_.size(); ++li) {
        if (!lost_[li].has_emb) continue;
        const float s = similarity(e, lost_[li].emb);
        if (s > best) {
            second = best;
            best = s;
            bi = static_cast<int>(li);
        } else {
            second = std::max(second, s);
        }
    }
    best_id = bi >= 0 ? lost_[bi].id : -1;
    return bi >= 0 && best >= pol_.reid_similarity && best - second >= pol_.reid_margin ? bi : -1;
}

void FastFaceTracker::lose(size_t i, double now, std::vector<std::string>& events) {
    const FastTrack& t = tracks_[i];
    if (t.has_emb)  // one gallery entry per person: a newer fingerprint of the same person replaces the older one
        lost_.erase(std::remove_if(lost_.begin(), lost_.end(),
                                   [&](const LostFace& l) { return l.has_emb && similarity(l.emb, t.emb) >= pol_.reid_similarity; }),
                    lost_.end());
    // kept with or without a fingerprint: without one it can still come back by place and distance
    lost_.push_back({t.id, t.emb, now, t.name, t.has_emb, t.box, t.depth});
    events.push_back("face drop #" + std::to_string(t.id) + " lost");
    tracks_.erase(tracks_.begin() + static_cast<long>(i));
}

void FastFaceTracker::apply_names() {
    for (FastTrack& t : tracks_) {
        std::string n;
        if (t.has_emb && known_name(t.emb, n)) t.name = n;
    }
}

void FastFaceTracker::update(const std::vector<FaceDet>& dets, const cv::Mat& bgr, double now,
                             std::vector<std::string>& events, const std::vector<float>& det_depth) {
    auto depth_of = [&](size_t di) { return di < det_depth.size() ? det_depth[di] : -1.f; };
    const FastFacePolicy& p = pol_;
    lost_.erase(std::remove_if(lost_.begin(), lost_.end(),
                               [&](const LostFace& l) {
                                   return now - l.lost_at > (l.has_emb ? p.gallery_seconds : p.relink_seconds);
                               }),
                lost_.end());

    // 1. Link detections to tracks by overlap with each track's predicted box (best pairs first).
    std::vector<std::tuple<float, int, int>> pairs;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        const cv::Rect2f pr = tracks_[ti].predict(now);
        for (size_t di = 0; di < dets.size(); ++di) {
            const float o = iou(pr, dets[di].box);
            if (o >= p.match_iou) pairs.emplace_back(o, static_cast<int>(ti), static_cast<int>(di));
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
        if (det_of[ti] < 0) {
            tracks_[ti].visible = false;
        } else {
            take(tracks_[ti], dets[det_of[ti]], now, true);
            if (depth_of(det_of[ti]) > 0) tracks_[ti].depth = depth_of(det_of[ti]);
        }
    }

    // 2. Detections no track claimed: a returning face (fingerprint matches a missing track or a
    //    lost ID, clearly better than any other candidate) or a new face. Most confident first.
    std::vector<int> fresh;
    for (size_t di = 0; di < dets.size(); ++di)
        if (track_of[di] < 0 && dets[di].score >= p.start_score) fresh.push_back(static_cast<int>(di));
    std::sort(fresh.begin(), fresh.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });
    for (int di : fresh) {
        const FaceDet& d = dets[di];
        Embedding e{};
        const bool has_e = embeddable(d);
        if (has_e) e = emb_->embed(bgr, d);
        int best_track = -1, best_lost = -1;
        float best = -2, second = -2;
        auto consider = [&](float s, int ti, int li) {
            if (s > best) {
                second = best;
                best = s;
                best_track = ti;
                best_lost = li;
            } else {
                second = std::max(second, s);
            }
        };
        if (has_e) {
            for (size_t ti = 0; ti < tracks_.size(); ++ti)
                if (!tracks_[ti].visible && tracks_[ti].has_emb) consider(similarity(e, tracks_[ti].emb), static_cast<int>(ti), -1);
            for (size_t li = 0; li < lost_.size(); ++li)
                if (lost_[li].has_emb) consider(similarity(e, lost_[li].emb), -1, static_cast<int>(li));
        }
        if (best < p.reid_similarity || best - second < p.reid_margin) best_track = best_lost = -1;
        // no fingerprint match: the one face lost just there, at the same distance, if there is exactly one
        bool by_depth = false;
        const float dz = depth_of(di);
        if (best_track < 0 && best_lost < 0 && dz > 0) {
            int n = 0, rt = -1, rl = -1;
            auto near = [&](const cv::Rect2f& b, float bz) {
                const float w = std::max(b.width, d.box.width);
                const float ratio = d.box.width / std::max(b.width, 1.f);
                return bz > 0 && std::fabs(bz - dz) <= p.relink_depth && ratio > 0.6f && ratio < 1.6f &&
                       cv::norm(centre(d.box) - centre(b)) <= p.relink_reach * w;
            };
            for (size_t ti = 0; ti < tracks_.size(); ++ti)
                if (!tracks_[ti].visible && near(tracks_[ti].box, tracks_[ti].depth)) ++n, rt = static_cast<int>(ti);
            for (size_t li = 0; li < lost_.size(); ++li)
                if (now - lost_[li].lost_at <= p.relink_seconds && near(lost_[li].box, lost_[li].depth))
                    ++n, rl = static_cast<int>(li), rt = -1;
            if (n == 1) {
                best_track = rt;
                best_lost = rt >= 0 ? -1 : rl;
                by_depth = true;
            }
        }
        if (best_track >= 0) {  // missing for a moment, then back somewhere the prediction didn't expect
            FastTrack& t = tracks_[best_track];
            take(t, d, now, false);
            if (dz > 0) t.depth = dz;
            det_of[best_track] = di;
            events.push_back(by_depth ? fmt("face back #%d (same place and distance, %.2f m)", t.id, dz)
                                      : fmt("face back #%d (similarity %.2f)", t.id, best));
            continue;
        }
        if (static_cast<int>(tracks_.size()) >= p.max_tracks) continue;
        FastTrack t;
        take(t, d, now, false);
        t.born = now;
        t.depth = dz;
        if (best_lost >= 0) {
            const LostFace l = lost_[best_lost];
            lost_.erase(lost_.begin() + best_lost);
            t.id = l.id;
            t.name = l.name;
            t.revived = true;
            events.push_back(by_depth ? fmt("face back #%d after %.1fs (same place and distance, %.2f m)", t.id, now - l.lost_at, dz)
                                      : fmt("face back #%d after %.1fs (similarity %.2f)", t.id, now - l.lost_at, best));
        } else {
            t.id = next_id_++;
            std::string why = " (no fingerprint yet: face small, turned or unclear)";
            if (has_e) {
                float lb;
                int lid;
                lost_match(e, lb, lid);
                why = lid >= 0 ? fmt(" (closest lost face #%d, similarity %.2f)", lid, lb) : "";
            }
            events.push_back("face new #" + std::to_string(t.id) + why);
        }
        if (has_e) {
            t.emb = e;
            t.has_emb = true;
            t.last_embed = now;
            std::string n;
            if (known_name(e, n)) t.name = n;
        }
        tracks_.push_back(t);
        det_of.push_back(di);
    }

    // 3. Tracks missing too long are lost; their fingerprint stays in the gallery.
    for (size_t ti = tracks_.size(); ti-- > 0;)
        if (!tracks_[ti].visible && now - tracks_[ti].last_seen > p.max_missing) {
            lose(ti, now, events);
            det_of.erase(det_of.begin() + static_cast<long>(ti));
        }

    // Say once why a new face kept its new ID after its checking window.
    for (FastTrack& t : tracks_)
        if (!t.revived && !t.late_logged && now - t.born > p.late_reid_seconds) {
            t.late_logged = true;
            if (t.late_best_id >= 0)
                events.push_back(fmt("face #%d stays new (closest lost face in its first %.0fs: #%d, similarity %.2f)", t.id,
                                     p.late_reid_seconds, t.late_best_id, t.late_best));
        }

    // 4. Refresh one visible track's fingerprint per frame (round robin), only from a clear,
    //    unobstructed view. A track that just started is refreshed more often and each fresh
    //    fingerprint is checked against lost faces.
    if (!emb_ || tracks_.empty()) return;
    for (size_t k = 0; k < tracks_.size(); ++k) {
        const size_t ti = (refresh_next_ + k) % tracks_.size();
        FastTrack& t = tracks_[ti];
        const bool young = !t.revived && now - t.born <= p.late_reid_seconds;
        if (!t.visible || det_of[ti] < 0 || now - t.last_embed < (young ? p.late_reid_interval : p.refresh_seconds)) continue;
        const FaceDet& d = dets[det_of[ti]];
        if (!embeddable(d)) continue;
        bool overlapped = false;
        for (size_t j = 0; j < dets.size(); ++j)
            overlapped = overlapped || (static_cast<int>(j) != det_of[ti] && (d.box & dets[j].box).area() > 0);
        if (overlapped) continue;
        const Embedding e = emb_->embed(bgr, d);
        if (young) {
            float lb;
            int lid;
            const int li = lost_match(e, lb, lid);
            if (lid >= 0 && lb > t.late_best) {
                t.late_best = lb;
                t.late_best_id = lid;
            }
            if (li >= 0) {
                const LostFace l = lost_[li];
                lost_.erase(lost_.begin() + li);
                events.push_back(fmt("face #%d is #%d again (similarity %.2f)", t.id, l.id, lb));
                t.id = l.id;
                t.revived = true;
                if (t.name.empty()) t.name = l.name;
            }
        }
        if (t.has_emb) t.self_sim = similarity(e, t.emb);
        t.name_sim = -2;
        for (const KnownFace& kf : known) {
            const float s = similarity(e, kf.emb);
            if (s > t.name_sim) {
                t.name_sim = s;
                t.closest_name = kf.name;
            }
        }
        if (t.has_emb) {
            Embedding m;
            for (int i = 0; i < 128; ++i) m[i] = p.embed_ema * t.emb[i] + (1 - p.embed_ema) * e[i];
            t.emb = normalized(m);
        } else {
            t.emb = e;
            t.has_emb = true;
        }
        t.last_embed = now;
        std::string n;
        if (known_name(t.emb, n)) t.name = n;
        refresh_next_ = ti + 1;
        break;
    }
}

std::string FastFaceTracker::summary() const {
    std::string s;
    for (const FastTrack& t : tracks_) {
        if (!t.visible) continue;
        s += " | #" + std::to_string(t.id) + (t.name.empty() ? "" : " " + t.name);
        if (t.name_sim > -2) s += fmt(" (closest name %s %.2f, self %.2f)", t.closest_name.c_str(), t.name_sim, t.self_sim);
        else if (t.has_emb) s += fmt(" (self %.2f)", t.self_sim);
    }
    return s;
}

FastFacePipeline::FastFacePipeline(const std::string& engines, const FastFacePolicy& pol, bool reid,
                                   const std::string& names_file)
    : detector_(engines + "/yunet_640x384.fp16.engine"),
      embedder_(reid ? std::make_unique<FaceEmbedder>(engines + "/face_recognition_sface_2021dec.fp16.engine") : nullptr),
      tracker_(embedder_.get(), pol),
      names_file_(names_file) {
    params_.min_score = pol.keep_score;
    params_.nms_iou = pol.nms_iou;
    params_.min_px = pol.min_face_px;
    load_names();
}

// Saved names: one line per saved view, "name<TAB>128 numbers".
void FastFacePipeline::load_names() {
    if (names_file_.empty()) return;
    std::ifstream in(names_file_);
    std::string line;
    while (std::getline(in, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        KnownFace k;
        k.name = line.substr(0, tab);
        std::istringstream ss(line.substr(tab + 1));
        int i = 0;
        while (i < 128 && ss >> k.emb[i]) ++i;
        if (i == 128) tracker_.known.push_back(k);
    }
    if (!tracker_.known.empty())
        std::printf("[names] loaded %zu saved face view(s) from %s\n", tracker_.known.size(), names_file_.c_str());
}

void FastFacePipeline::store_names() const {
    if (names_file_.empty()) return;
    std::ofstream out(names_file_);
    for (const KnownFace& k : tracker_.known) {
        out << k.name << '\t';
        for (int i = 0; i < 128; ++i) out << k.emb[i] << (i == 127 ? '\n' : ' ');
    }
}

std::string FastFacePipeline::save_face(const std::string& name, int prefer_id) {
    if (name.empty()) return "Type a name first.";
    if (name.find('\t') != std::string::npos || name.find('\n') != std::string::npos) return "Names can't contain tabs or line breaks.";
    FastTrack* pick = nullptr;  // the preferred face (the person addressing the robot), else the largest
    for (FastTrack& t : tracker_.tracks())
        if (t.visible && t.has_emb && t.id == prefer_id) pick = &t;
    if (!pick)
        for (FastTrack& t : tracker_.tracks())
            if (t.visible && t.has_emb && (!pick || t.box.width > pick->box.width)) pick = &t;
    if (!pick)
        return fmt("No face close enough to save (needs a face at least %.0f px tall).", 64.0);
    tracker_.known.push_back({name, pick->emb});
    store_names();
    pick->name = name;
    tracker_.apply_names();
    int views = 0;
    for (const KnownFace& k : tracker_.known) views += k.name == name;
    return fmt("Saved face #%d as %s (%d view(s)).", pick->id, name.c_str(), views);
}

std::string FastFacePipeline::forget_names() {
    tracker_.known.clear();
    store_names();
    for (FastTrack& t : tracker_.tracks()) t.name.clear();
    return "Forgot all saved names.";
}

void FastFacePipeline::process(const cv::Mat& bgr, double now, std::vector<std::string>& events) {
    frame_.upload(bgr, detector_.stream());
    dets_ = detector_.detect(frame_, params_);
    ms_detect_ += detector_.ms;
    std::vector<float> dz;
    if (depth_ && depth_->valid())
        for (const FaceDet& d : dets_) dz.push_back(face_depth(*depth_, d.box));
    tracker_.update(dets_, bgr, now, events, dz);
}

std::vector<FaceOut> FastFacePipeline::faces() const {
    std::vector<FaceOut> out;
    for (const FastTrack& t : tracker_.tracks()) {
        FaceOut o;
        o.id = t.id;
        o.box = t.box;
        o.visible = t.visible;
        o.quality = t.score;
        o.name = t.name;
        if (t.visible) o.landmarks.assign(t.landmarks.begin(), t.landmarks.end());
        out.push_back(o);
    }
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
