#include "face_tracker.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace fht {

namespace {

cv::Vec4f to_model(const cv::Rect2f& b, int W, int H, int S) {  // frame box -> x1, y1, x2, y2 in 0..S
    const float sx = static_cast<float>(S) / W, sy = static_cast<float>(S) / H;
    return {b.x * sx, b.y * sy, (b.x + b.width) * sx, (b.y + b.height) * sy};
}

bool mask_box(const cv::Mat& mask, cv::Rect2f& out) {  // in mask pixels
    int x1 = mask.cols, y1 = mask.rows, x2 = -1, y2 = -1;
    for (int y = 0; y < mask.rows; ++y) {
        const float* r = mask.ptr<float>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (r[x] > 0) {
                x1 = std::min(x1, x);
                x2 = std::max(x2, x);
                y1 = std::min(y1, y);
                y2 = std::max(y2, y);
            }
        }
    }
    if (x2 < 0) return false;
    out = cv::Rect2f(static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2 + 1 - x1),
                     static_cast<float>(y2 + 1 - y1));
    return true;
}

}  // namespace

void FaceTracker::update_box(FaceTrack& t, int W, int H, float score, double now) {
    cv::Rect2f b;
    t.visible = score > 0 && mask_box(t.mask, b);
    if (!t.visible) return;
    t.last_seen = now;
    const float sx = static_cast<float>(W) / t.mask.cols, sy = static_cast<float>(H) / t.mask.rows;
    t.box = cv::Rect2f(b.x * sx, b.y * sy, b.width * sx, b.height * sy);
}

void FaceTracker::update(const std::vector<FaceDet>& dets, int W, int H, double now, std::vector<std::string>& events) {
    const int S = tam_.cfg().S;

    // 1) associate detections with last-frame track boxes (greedy by IoU)
    std::vector<std::tuple<float, int, int>> pairs;
    for (size_t ti = 0; ti < tracks_.size(); ++ti)
        for (size_t di = 0; di < dets.size(); ++di)
            pairs.emplace_back(iou(tracks_[ti]->box, dets[di].box), static_cast<int>(ti), static_cast<int>(di));
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a > b; });
    std::set<int> t_used, d_used;
    std::map<int, int> match;
    for (const auto& [v, ti, di] : pairs) {
        if (v < pol_.match_iou || t_used.count(ti) || d_used.count(di)) continue;
        t_used.insert(ti);
        d_used.insert(di);
        match[ti] = di;
    }

    // 2) propagate or re-anchor every live track
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        FaceTrack& t = *tracks_[ti];
        ++t.age;
        auto m = match.find(static_cast<int>(ti));
        if (m != match.end()) {
            const FaceDet& d = dets[m->second];
            t.last_confirmed = now;
            const bool want = iou(t.box, d.box) < pol_.reanchor_iou || t.iou_pred < pol_.min_track_iou_pred;
            if (want && now - t.last_anchor >= pol_.reanchor_min_gap) {
                TamStep s = tam_.start(to_model(d.box, W, H, S), *t.mem, false);
                t.mask = s.mask;
                t.iou_pred = s.iou_pred;
                t.last_anchor = now;
                events.push_back("face re-anchor #" + std::to_string(t.id));
                update_box(t, W, H, s.score, now);
                continue;
            }
        }
        TamStep s = tam_.track(*t.mem);
        t.mask = s.mask;
        t.iou_pred = s.iou_pred;
        update_box(t, W, H, s.score, now);
    }

    // 3) start tracks for unmatched detections
    for (size_t di = 0; di < dets.size(); ++di) {
        if (d_used.count(static_cast<int>(di)) || static_cast<int>(tracks_.size()) >= pol_.max_tracks) continue;
        const FaceDet& d = dets[di];
        bool overlaps = false;
        for (const auto& t : tracks_)
            if (iou(d.box, t->box) > pol_.match_iou) overlaps = true;
        if (overlaps) continue;
        auto t = std::make_unique<FaceTrack>();
        t->id = next_id_++;
        t->mem = tam_.new_memory();
        t->box = d.box;
        TamStep s = tam_.start(to_model(d.box, W, H, S), *t->mem, true);
        t->mask = s.mask;
        t->iou_pred = s.iou_pred;
        t->last_seen = t->last_confirmed = t->last_anchor = now;
        update_box(*t, W, H, s.score, now);
        events.push_back("face new #" + std::to_string(t->id));
        tracks_.push_back(std::move(t));
    }

    // 4) retire lost / unconfirmed / duplicate tracks (older tracks win duplicates)
    std::stable_sort(tracks_.begin(), tracks_.end(), [](const auto& a, const auto& b) { return a->age > b->age; });
    std::vector<std::unique_ptr<FaceTrack>> keep;
    for (auto& tp : tracks_) {
        FaceTrack& t = *tp;
        const double limit =
            t.iou_pred >= pol_.confident_iou_pred ? pol_.max_unconfirmed_confident : pol_.max_unconfirmed;
        if (now - t.last_seen > pol_.max_missing) {
            events.push_back("face drop #" + std::to_string(t.id) + " lost");
            continue;
        }
        if (now - t.last_confirmed > limit) {
            events.push_back("face drop #" + std::to_string(t.id) + " unconfirmed");
            continue;
        }
        bool dup = false;
        for (const auto& k : keep)
            if (t.visible && k->visible && iou(t.box, k->box) > pol_.duplicate_iou) dup = true;
        if (dup) {
            events.push_back("face drop #" + std::to_string(t.id) + " duplicate");
            continue;
        }
        keep.push_back(std::move(tp));
    }
    std::sort(keep.begin(), keep.end(), [](const auto& a, const auto& b) { return a->id < b->id; });
    tracks_ = std::move(keep);
}

}  // namespace fht
