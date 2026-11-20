// Who is gesturing at the robot (same rules as ../mac_native/FaceHandTrack/PersonLinker.swift):
// each hand is linked to the person (face) it belongs to, and a person who faces the camera and
// holds a gesture is "addressing" the robot. No model: a few comparisons per frame.
//
//   reach  a hand belongs only to a face within arm's reach, measured in face widths
//   size   a person's hand and face appear at a similar scale (a depth cue from one camera)
//   side   a right hand is usually on the person's right of their face
//   one    each person has at most one left and one right hand
//   sticky a hand keeps its person unless another fits clearly better
//   pose   when body poses are known (2+ people in view), a hand whose wrist sits on a person's
//          pose wrist belongs to that person, and a person whose visible wrists are both elsewhere
//          cannot own it: one camera has no depth, the body joins each hand to its person
#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

namespace fht {

struct LinkPolicy {
    float max_reach = 5.0f;         // hand centre to face centre, in face widths
    float hand_face_ratio = 0.6f;   // typical hand size (wrist..middle knuckle) / face box width
    float min_ratio = 0.25f, max_ratio = 2.5f;  // outside this, never the same person
    float size_weight = 2.0f;       // cost per unit of |ln(ratio / hand_face_ratio)|
    float wrong_side_penalty = 1.0f;
    float above_head_penalty = 1.0f;
    float sticky_bonus = 0.75f;
    float max_yaw = 0.35f;          // nose offset from the eye midpoint / eye distance: below = facing
    double hold_seconds = 0.5;      // a gesture held this long while facing = addressing
    double linger_seconds = 1.0;    // keep "addressing" this long after the gesture ends
    float pose_visible = 0.5f;      // pose wrist visibility needed to count
    float pose_match = 1.0f;        // hand wrist within this many hand sizes of a pose wrist: that person's
    float pose_elsewhere = 2.5f;    // ... farther than this from every visible pose wrist: not that person's
    float pose_bonus = 10.0f;       // cost lowered by this for a pose match (decides over the other rules)
    float pose_penalty = 2.0f;      // one visible wrist elsewhere (the other unseen)
};

struct PoseWrist {
    cv::Point2f point;              // frame pixels
    float visibility;               // 0..1
};

struct LinkFace {
    int id;
    cv::Rect2f box;                     // frame pixels
    std::vector<cv::Point2f> landmarks; // 5 YuNet points (empty if unknown this frame)
    bool visible;
};

struct LinkHand {
    int id;
    std::string handedness;             // the person's real hand: "Left" / "Right"
    std::vector<cv::Point2f> points;    // 21 landmarks, frame pixels
    std::string gesture;                // stable gesture, "" = none
};

struct PersonState {
    int face_id = 0;
    std::vector<int> hand_ids;
    int facing = -1;                    // 1 facing, 0 looking away, -1 unknown
    std::string addressing;             // gesture shown to the robot, "" = none
};

float hand_size(const std::vector<cv::Point2f>& p);  // wrist..middle knuckle (or knuckle span x 1.3)
cv::Point2f mean_point(const std::vector<cv::Point2f>& p);

class PersonLinker {
public:
    LinkPolicy policy;
    bool mirrored = false;  // selfie-style frames flip which side of the face a right hand appears on

    static bool yaw(const std::vector<cv::Point2f>& lm, float& out);
    bool cost(const LinkHand& h, const LinkFace& f, float& out) const;
    // poses: per face id, its body's two wrists (only when computed this frame)
    void update(const std::vector<LinkFace>& faces, const std::vector<LinkHand>& hands, double now,
                std::vector<std::string>& events, const std::map<int, std::vector<PoseWrist>>& poses = {});
    // hand wrist distance to the nearest visible pose wrist of face, in hand sizes (-1 if none)
    float pose_distance(const LinkHand& h, int face_id) const;
    void reset();

    const std::map<int, int>& owner() const { return owner_; }  // hand id -> face id
    const std::map<int, PersonState>& people() const { return people_; }

private:
    std::map<int, std::vector<PoseWrist>> poses_;
    std::map<int, int> owner_;
    std::map<int, PersonState> people_;
    std::map<int, bool> facing_;
    struct Held { std::string gesture; double since; };
    std::map<int, Held> held_;
    struct Addr { std::string gesture; double until; };
    std::map<int, Addr> addressing_;
};

}  // namespace fht
