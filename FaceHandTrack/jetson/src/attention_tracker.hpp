// Waving to call the robot, and where the robot should turn to face that person. Same rules as
// ../mac_native/FaceHandTrack/AttentionTracker.swift, with one change for a robot that moves:
// a wave is measured relative to the waving person's face, so the whole picture shifting while
// the robot turns does not look like a wave.
//
//   wave       an open hand (four fingers up) moving side to side: at least 3 swings, each at least
//              0.5 x the hand size, within 2 s, at about shoulder height or above. Counted per
//              person and hand side, not per tracked hand (a fast hand gets lost and re-found).
//   attention  the person who last waved, or who started addressing the robot when no one had
//              its attention; kept until out of sight 2 s, idle 30 s, or someone else waves
//   turn       horizontal / vertical angle from the camera centre to that person's face
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fht {

struct AttentionPolicy {
    double window = 2.0;
    int min_swings = 3;
    float swing_fraction = 0.5f;
    float raised_below_chin = 0.5f;   // hand centre at most this many face heights below the chin
    double wave_cooldown = 3.0;
    double waving_shown = 1.5;
    double lost_after = 2.0;
    double idle_after = 30.0;
    double horizontal_fov = 69.0;     // degrees: the G1 head camera (RealSense D435i colour)
};

struct AttentionHand {
    int id;
    int owner;                        // face id, -1 none
    std::string handedness;
    std::array<bool, 5> extended;
    std::vector<cv::Point2f> points;  // 21 landmarks, frame pixels
};

struct AttentionFace {
    int id;
    cv::Rect2f box;
    bool addressing;
};

struct AttentionTarget {
    int face_id = -1;                 // -1: no one has the robot's attention
    double yaw = 0, pitch = 0;        // degrees: + = robot's right / up
    std::string reason;               // "waved" / "addressing"
    std::string turn_text() const;
};

int count_swings(const std::vector<float>& xs, float amp);

class AttentionTracker {
public:
    AttentionPolicy policy;
    bool mirrored = false;
    void update(const std::vector<AttentionFace>& faces, const std::vector<AttentionHand>& hands, cv::Size frame,
                double now, std::vector<std::string>& events);
    void reset();
    const AttentionTarget& target() const { return target_; }
    bool waving(int face_id) const { return waving_until_.count(face_id) > 0; }

private:
    struct Sample { double t; float x, size; };
    std::map<std::string, std::vector<Sample>> history_;
    std::map<std::string, double> last_wave_;
    std::map<int, double> waving_until_;
    int target_id_ = -1;
    std::string target_reason_;
    double target_last_seen_ = 0, target_last_active_ = 0;
    std::set<int> was_addressing_;
    AttentionTarget target_;
};

}  // namespace fht
