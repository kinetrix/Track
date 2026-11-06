// Fast face path (default): YuNet on every frame, boxes linked frame to frame by overlap with
// where each face is predicted to be (constant velocity), and SFace embeddings to give a person
// their old ID back after they were hidden or out of the picture. About 1 ms of YuNet per frame
// plus an embedding (~1-2 ms) only when a face appears or a track's fingerprint is refreshed,
// so 3+ faces fit easily in a 30 fps frame (EdgeTAM costs ~20-25 ms per face).
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
    float reid_similarity = 0.55f; // embedding cosine needed to give a lost ID back
    float reid_margin = 0.10f;     // ... and this much above the next-best ID (else ambiguous: new ID)
    float min_embed_px = 64.f;     // smallest face side (frame pixels) worth an embedding; smaller
                                   // faces give fingerprints too alike between people
    double refresh_seconds = 1.0;  // re-embed each visible track this often (one track per frame)
    float embed_ema = 0.8f;        // weight of the old fingerprint when refreshing
    int max_tracks = 3;
};

struct FastTrack {
    int id = 0;
    cv::Rect2f box;              // last detection (frame pixels)
    cv::Point2f vel_c{0, 0};     // centre velocity, pixels / second
    cv::Point2f vel_s{0, 0};     // size velocity, pixels / second
    double last_seen = 0, last_embed = -1e9;
    float score = 0;
    bool visible = true;
    bool has_emb = false;
    Embedding emb{};
    cv::Rect2f predict(double now) const;
};

struct LostFace {
    int id;
    Embedding emb;
    double lost_at;
};

class FastFaceTracker {
public:
    FastFaceTracker(FaceEmbedder* embedder, const FastFacePolicy& pol) : emb_(embedder), pol_(pol) {}
    // dets: this frame's detections (>= keep_score); bgr: the frame (for embeddings).
    void update(const std::vector<FaceDet>& dets, const cv::Mat& bgr, double now, std::vector<std::string>& events);
    const std::vector<FastTrack>& tracks() const { return tracks_; }
    const std::vector<LostFace>& gallery() const { return lost_; }

private:
    bool embeddable(const FaceDet& d) const;
    void lose(size_t i, double now, std::vector<std::string>& events, const char* why);

    FaceEmbedder* emb_;  // nullptr: no re-identification (IDs of lost faces are not given back)
    FastFacePolicy pol_;
    std::vector<FastTrack> tracks_;
    std::vector<LostFace> lost_;
    int next_id_ = 1;
    size_t refresh_next_ = 0;
};

// FaceSource for the fast path: frame upload, YuNet, tracker.
class FastFacePipeline : public FaceSource {
public:
    FastFacePipeline(const std::string& engines, const FastFacePolicy& pol, bool reid = true);
    void process(const cv::Mat& bgr, double now, std::vector<std::string>& events) override;
    std::vector<FaceOut> faces() const override;
    const std::vector<FaceDet>& detections() const override { return dets_; }
    std::string timing(int frames) override;
    const FastFaceTracker& tracker() const { return tracker_; }

private:
    FaceDetector detector_;
    std::unique_ptr<FaceEmbedder> embedder_;
    FastFaceTracker tracker_;
    GpuFrame frame_;
    DetectorParams params_;
    std::vector<FaceDet> dets_;
    double ms_detect_ = 0;
};

}  // namespace fht
