// Fast face path (default; same rules as ../mac_native/FaceHandTrack/FastFaceTracker.swift):
// YuNet on every frame, boxes linked frame to frame by overlap with where each face is predicted
// to be (constant velocity), and SFace fingerprints to give a person their old ID back after they
// were hidden or out of the picture, and to recognise saved names. About 1 ms of YuNet per frame
// plus a fingerprint (~5 ms) only when a face appears, returns or is refreshed, so 3+ faces fit
// easily in a 30 fps frame (EdgeTAM costs ~20-25 ms per face).
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "face_detector.hpp"
#include "face_embedder.hpp"
#include "face_source.hpp"
#include "preprocess.hpp"

namespace fht {

struct FastFacePolicy {
    float keep_score = 0.60f;      // detector confidence to keep following a face
    float start_score = 0.80f;     // ... to start (or revive) a track
    float nms_iou = 0.30f;
    float min_face_px = 16.f;      // in detector input pixels (YuNet finds faces down to ~10 px)
    float match_iou = 0.30f;       // detection <-> predicted track box
    double max_missing = 1.0;      // seconds without a detection -> track is lost (ID kept in the gallery)
    double gallery_seconds = 30.0; // how long a lost ID can come back
    float reid_similarity = 0.55f; // fingerprint cosine needed to give a lost ID back (or a saved name)
    float reid_margin = 0.10f;     // ... and this much above the next-best candidate
    float min_embed_px = 64.f;     // smallest face side (frame pixels) worth a fingerprint
    double refresh_seconds = 1.0;  // re-fingerprint each visible track this often (one track per frame)
    float embed_ema = 0.8f;        // weight of the old fingerprint when refreshing
    double late_reid_seconds = 3.0;   // a new track keeps checking lost faces this long ...
    double late_reid_interval = 0.25; // ... with a fresh fingerprint this often (first views are often blurred)
    int max_tracks = 3;
};

struct FastTrack {
    int id = 0;                  // can change once: a new track that turns out to be a lost face
    double born = 0;
    cv::Rect2f box;              // last detection (frame pixels)
    std::array<cv::Point2f, 5> landmarks{};
    cv::Point2f vel_c{0, 0};     // centre velocity, pixels / second
    cv::Point2f vel_s{0, 0};     // size velocity, pixels / second
    double last_seen = 0, last_embed = -1e9;
    float score = 0;
    bool visible = true;
    bool has_emb = false;
    Embedding emb{};
    std::string name;            // saved name this fingerprint matches
    bool revived = false;        // got a lost face's ID back (no more late checks)
    float late_best = -2;        // closest lost face seen by the late checks (log)
    int late_best_id = -1;
    bool late_logged = false;
    float self_sim = 0;          // latest fresh fingerprint vs the track's own (tuning)
    float name_sim = -2;         // closest saved name (tuning)
    std::string closest_name;
    cv::Rect2f predict(double now) const;
};

struct LostFace {
    int id;
    Embedding emb;
    double lost_at;
    std::string name;
};

struct KnownFace {
    std::string name;
    Embedding emb;
};

class FastFaceTracker {
public:
    FastFaceTracker(FaceEmbedder* embedder, const FastFacePolicy& pol) : emb_(embedder), pol_(pol) {}
    // dets: this frame's detections (>= keep_score); bgr: the frame (for fingerprints).
    void update(const std::vector<FaceDet>& dets, const cv::Mat& bgr, double now, std::vector<std::string>& events);
    const std::vector<FastTrack>& tracks() const { return tracks_; }
    std::vector<FastTrack>& tracks() { return tracks_; }
    std::vector<KnownFace> known;  // saved names
    void apply_names();
    std::string summary() const;   // per visible face: name, closest saved name, self agreement

private:
    bool embeddable(const FaceDet& d) const;
    void lose(size_t i, double now, std::vector<std::string>& events);
    bool known_name(const Embedding& e, std::string& out) const;
    // best lost face for e: index (-1 if not clearly a match), best similarity and its id
    int lost_match(const Embedding& e, float& best, int& best_id) const;

    FaceEmbedder* emb_;  // nullptr: no re-identification
    FastFacePolicy pol_;
    std::vector<FastTrack> tracks_;
    std::vector<LostFace> lost_;
    int next_id_ = 1;
    size_t refresh_next_ = 0;
};

// FaceSource for the fast path: frame upload, YuNet, tracker, saved names.
class FastFacePipeline : public FaceSource {
public:
    FastFacePipeline(const std::string& engines, const FastFacePolicy& pol, bool reid = true,
                     const std::string& names_file = "");
    void process(const cv::Mat& bgr, double now, std::vector<std::string>& events) override;
    std::vector<FaceOut> faces() const override;
    const std::vector<FaceDet>& detections() const override { return dets_; }
    std::string timing(int frames) override;
    const FastFaceTracker& tracker() const { return tracker_; }
    // Saves the fingerprint of face `prefer_id` (else the largest visible face) as `name`.
    std::string save_face(const std::string& name, int prefer_id);
    std::string forget_names();

private:
    void load_names();
    void store_names() const;

    FaceDetector detector_;
    std::unique_ptr<FaceEmbedder> embedder_;
    FastFaceTracker tracker_;
    GpuFrame frame_;
    DetectorParams params_;
    std::vector<FaceDet> dets_;
    double ms_detect_ = 0;
    std::string names_file_;
};

}  // namespace fht
