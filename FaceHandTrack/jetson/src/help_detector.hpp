// Help requests for a caregiver, from the hand points (no model). Same rules as
// ../mac_native/FaceHandTrack/HelpDetector.swift, plus one change from the 2-person test (12
// alerts in 7 minutes, many false): the Signal for Help must be given twice within 6 s.
//
//   Signal for Help   palm to the camera with the thumb tucked in (index..pinky up, thumb folded),
//                     held a moment, then the fingers fold over the thumb (a fist) within about a
//                     second, wrist held still, by someone facing the camera (or whose face isn't
//                     seen). Done twice within 6 s: relaxing or lowering an open hand can look like
//                     it once, rarely twice.
//   Both hands up     both of one person's hands open and raised above their chin, held 2 s.
//
// An alert names the person and stays until acknowledged (the stream page's button); the same
// person can't raise another for a short cooldown.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace fht {

struct HelpPolicy {
    double tucked_hold = 0.4;      // palm-with-thumb-tucked shown at least this long ...
    double fold_within = 1.0;      // ... then a fist within this long after it ends
    double tucked_gap = 0.3;       // a break this long in the tucked palm restarts its hold
    float still_fraction = 0.5f;   // the wrist may drift at most this x the hand size during the signal
    int repeats = 2;               // signals needed ...
    double repeat_within = 6.0;    // ... within this many seconds
    double hands_up_hold = 2.0;    // both hands open and raised this long
    double cooldown = 10.0;        // seconds before the same person can raise another alert
};

struct HelpAlert {
    int id = 0;
    int face_id = -1;              // -1: the hand belonged to no visible face
    std::string name;              // saved name, "" if none
    std::string reason;            // "Signal for Help" / "Both hands up"
    double time = 0;               // seconds since start
    std::string clock;             // wall-clock HH:MM:SS
    std::string who() const;
};

struct HelpHand {
    int id;
    int owner;                     // face id, -1 none
    std::array<bool, 5> extended;  // thumb, index, middle, ring, pinky
    std::string gesture;           // stable gesture
    cv::Point2f center, wrist;     // frame pixels
    float size;                    // wrist..middle knuckle, pixels
};

struct HelpFace {
    int id;
    cv::Rect2f box;
    std::string name;
    int facing = -1;               // 1 facing the camera, 0 looking away, -1 unknown
};

class HelpDetector {
public:
    HelpPolicy policy;
    void update(const std::vector<HelpFace>& faces, const std::vector<HelpHand>& hands, double now,
                std::vector<std::string>& events);
    void acknowledge_all() { alerts_.clear(); }
    void reset();
    const std::vector<HelpAlert>& alerts() const { return alerts_; }      // unacknowledged, oldest first
    const std::vector<HelpAlert>& new_alerts() const { return new_alerts_; }
    // People with one Signal for Help given and waiting for the confirming second (face id or -1 - hand id).
    std::map<int, double> pending() const;

private:
    void signal(int owner, int hand_id, const HelpFace* face, double now, std::vector<std::string>& events);
    void raise(const std::string& reason, const HelpFace* face, int face_id, int hand_id, double now);

    std::vector<HelpAlert> alerts_, new_alerts_;
    int next_id_ = 1;
    struct Tucked { double since, last; cv::Point2f start, at; };
    std::map<int, Tucked> tucked_;                  // per hand id
    std::map<int, double> hands_up_;                // per face id: since
    std::map<std::string, double> last_alert_;      // per person key
    std::map<int, std::vector<double>> signals_;    // per person key (face id, or -1 - hand id): times
};

}  // namespace fht
