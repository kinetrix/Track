#include "hand_pipeline.hpp"

#include <chrono>
#include <set>

namespace fht {

void HandPipeline::process(const cv::Mat& bgr, std::vector<std::string>& events) {
    tracker_.update(bgr, index_, events);
    const auto t0 = std::chrono::steady_clock::now();
    hands_.clear();
    std::set<int> live;
    for (const HandTrack& t : tracker_.tracks()) {
        HandResult h;
        h.id = t.id;
        h.handedness = tracker_.handedness(t);
        h.presence = t.r.presence;
        h.box = hand_box(t.r.landmarks);
        h.landmarks = t.r.landmarks;
        const Gesture g = classify(t.r.landmarks, t.r.world);
        h.raw_gesture = g.label;
        h.fingers = g.fingers;
        h.gesture = smoothers_[t.id](g.label);
        live.insert(t.id);
        hands_.push_back(h);
    }
    for (auto it = smoothers_.begin(); it != smoothers_.end();)
        it = live.count(it->first) ? std::next(it) : smoothers_.erase(it);
    ms_gesture += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++index_;
}

}  // namespace fht
