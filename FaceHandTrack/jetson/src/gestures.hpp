// Static hand gestures from the 21 landmarks, with hand-written rules (no model).
// Port of onnx/facetrack/gestures.py: same thresholds, same rule order.
#pragma once

#include <deque>
#include <string>

#include "hands.hpp"

namespace fht {

struct Gesture {
    std::string label;  // "" = none
    int fingers = 0;    // extended fingers
};

// landmarks: image pixels (+ relative z); world: metric 3-D.
Gesture classify(const Points21& landmarks, const Points21& world);

// Reports a gesture only after it won the majority (`need`) of the last `window` frames.
class GestureSmoother {
public:
    explicit GestureSmoother(int window = 5, int need = 3) : window_(window), need_(need) {}
    const std::string& operator()(const std::string& label);

private:
    int window_, need_;
    std::deque<std::string> hist_;
    std::string stable_;
};

}  // namespace fht
