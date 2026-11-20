// Checks the caregiver rules (person_linker, help_detector, attention_tracker) on constructed
// scenes: the same scenes the Mac version's rules were checked on, plus the robot-only changes
// (help signal twice, waves measured relative to the face, un-mirrored camera).
//   g++ -std=c++17 -I../../src $(pkg-config --cflags opencv4) test_rules.cpp
//       ../../src/{person_linker,help_detector,attention_tracker}.cpp $(pkg-config --libs opencv4) -o test_rules
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "attention_tracker.hpp"
#include "help_detector.hpp"
#include "person_linker.hpp"

using namespace fht;

static int fails = 0;
static void check(const char* name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    fails += !ok;
}

// 21 points: wrist at the bottom, middle knuckle `size` above it
static std::vector<cv::Point2f> hand_pts(float cx, float cy, float size) {
    std::vector<cv::Point2f> p(21, cv::Point2f(cx, cy));
    p[0] = {cx, cy + size / 2};
    p[9] = {cx, cy - size / 2};
    p[5] = {cx - size * 0.3f, cy - size / 2};
    p[17] = {cx + size * 0.3f, cy - size / 2};
    return p;
}

static LinkFace face(int id, float cx, float cy, float w, float yaw = 0) {
    const float eye = w * 0.4f;
    return {id, cv::Rect2f(cx - w / 2, cy - w * 0.6f, w, w * 1.2f),
            {{cx - eye / 2, cy - w * 0.15f}, {cx + eye / 2, cy - w * 0.15f}, {cx + yaw * eye, cy},
             {cx - w * 0.2f, cy + w * 0.25f}, {cx + w * 0.2f, cy + w * 0.25f}},
            true};
}

static std::map<int, int> link(PersonLinker& L, std::vector<LinkFace> f, std::vector<LinkHand> h, double t = 0) {
    std::vector<std::string> ev;
    L.update(f, h, t, ev);
    return L.owner();
}

int main() {
    // ---------- whose hand (the robot's camera is not mirrored: a right hand is on the image's left) ----------
    {
        PersonLinker L;
        auto o = link(L, {face(1, 640, 300, 150)},
                      {{1, "Right", hand_pts(480, 450, 95), ""}, {2, "Left", hand_pts(800, 450, 95), ""}});
        check("one person, both hands", o == std::map<int, int>{{1, 1}, {2, 1}});
    }
    {
        PersonLinker L;
        auto o = link(L, {face(1, 400, 300, 140), face(2, 900, 300, 140)},
                      {{1, "Right", hand_pts(260, 470, 90), ""}, {2, "Right", hand_pts(760, 470, 90), ""}});
        check("two people side by side", o == std::map<int, int>{{1, 1}, {2, 2}});
    }
    {
        PersonLinker L;  // B behind (half-size face) just left of A; A's big hand is nearer B's face
        auto o = link(L, {face(1, 800, 330, 160), face(2, 500, 250, 80)}, {{1, "Right", hand_pts(540, 330, 100), ""}});
        check("near person's hand beside a far person's face -> near person", o == std::map<int, int>{{1, 1}});
    }
    {
        PersonLinker L;
        auto o = link(L, {face(1, 800, 330, 160), face(2, 500, 250, 80)}, {{1, "Right", hand_pts(440, 330, 50), ""}});
        check("far person's small hand -> far person", o == std::map<int, int>{{1, 2}});
    }
    {
        PersonLinker L;
        auto o = link(L, {face(1, 100, 100, 60)}, {{1, "Right", hand_pts(1250, 700, 95), ""}});
        check("hand out of reach (5 face widths) -> nobody", o.empty());
        float c1, c2;
        PersonLinker U, M;
        M.mirrored = true;
        LinkHand h{1, "Right", hand_pts(480, 450, 95), ""};
        U.cost(h, face(1, 640, 300, 150), c1);
        M.cost(h, face(1, 640, 300, 150), c2);
        check("side check follows the camera (right hand on the image's left is normal un-mirrored)", c1 < c2);
    }
    {
        PersonLinker L;  // addressing: facing + held 0.5 s -> one event; looking away -> none
        std::vector<std::string> ev, all;
        for (int i = 0; i < 30; ++i) {
            ev.clear();
            L.update({face(1, 640, 300, 150)}, {{1, "Right", hand_pts(480, 450, 95), "Thumbs up"}}, i / 30.0, ev);
            for (auto& e : ev) if (e.find("addressing") != std::string::npos) all.push_back(e);
        }
        check("addressing after 0.5 s, one event", L.people().at(1).addressing == "Thumbs up" && all.size() == 1);
        PersonLinker A;
        for (int i = 0; i < 30; ++i) {
            ev.clear();
            A.update({face(1, 640, 300, 150, 0.6f)}, {{1, "Right", hand_pts(480, 450, 95), "Thumbs up"}}, i / 30.0, ev);
        }
        check("looking away -> not addressing", A.people().at(1).addressing.empty());
    }
    {
        PersonLinker L;  // a held gesture does not carry over when the hand changes person
        std::vector<std::string> ev;
        for (int i = 0; i < 30; ++i) L.update({face(1, 400, 300, 140), face(2, 900, 300, 140)}, {{1, "Right", hand_pts(260, 470, 90), "Pinch"}}, i / 30.0, ev);
        for (int i = 30; i < 40; ++i) L.update({face(1, 400, 300, 140), face(2, 900, 300, 140)}, {{1, "Right", hand_pts(760, 470, 90), "Pinch"}}, i / 30.0, ev);
        check("hold restarts when the hand changes person", L.people().at(2).addressing.empty());
    }

    // ---------- body pose (2+ people): the farther person's hand the size rule gets wrong ----------
    {
        // A near (face 160 px), B a step behind (face 80 px). B's hand (70 px) is raised next to A's
        // face: closer to A, and its size fits A almost as well as B, so the rules alone give it to A.
        std::vector<LinkFace> fs = {face(1, 700, 330, 160), face(2, 420, 260, 80)};
        LinkHand h{1, "Right", hand_pts(590, 330, 70), ""};
        PersonLinker R;
        auto o = link(R, fs, {h});
        check("rules alone: the farther person's hand goes to the nearer person (the bug)", o == std::map<int, int>{{1, 1}});
        PersonLinker P;
        std::vector<std::string> ev;
        std::map<int, LinkPose> poses = {  // wrists: {person's left, person's right}
            {1, {{700, 340}, {{{760, 560}, 0.9f}, {{650, 560}, 0.9f}}}},   // A's hands are down by A's sides
            {2, {{420, 265}, {{{380, 420}, 0.6f}, {{590, 365}, 0.9f}}}}};  // B's right wrist is where the hand's wrist is
        P.update(fs, {h}, 0, ev, poses);
        check("with poses: the hand goes to the farther person whose wrist it is", P.owner() == std::map<int, int>{{1, 2}});
        bool logged = false;
        for (auto& e : ev) logged = logged || e.find("pose wrist") != std::string::npos;
        check("the log says the pose decided it", logged);
        PersonLinker Q;  // only one of A's wrists seen, elsewhere, and no pose for B: rules still decide, A pays a penalty
        Q.update(fs, {h}, 0, ev, {{1, {{700, 340}, {{{760, 560}, 0.1f}, {{650, 560}, 0.9f}}}}});
        float ca, cb;
        Q.cost(h, fs[0], ca);
        Q.cost(h, fs[1], cb);
        check("one wrist of A seen elsewhere: A's cost goes up", ca > 0 && Q.owner().size() == 1);
        PersonLinker S;  // the near person's own hand still goes to them with poses on
        S.update(fs, {{2, "Right", hand_pts(640, 520, 100), ""}}, 0, ev,
                 {{1, {{700, 340}, {{{760, 560}, 0.9f}, {{640, 570}, 0.9f}}}}, {2, {{420, 265}, {{{380, 420}, 0.6f}, {{400, 380}, 0.9f}}}}});
        check("with poses: the near person's own hand stays theirs", S.owner() == std::map<int, int>{{2, 1}});

        // Mac log 20 Nov: A's pose (from a big crop that also held B) put a wrist on B's hand, so both
        // poses claimed it, the bonuses cancelled and the hand flipped between A and B every frame.
        LinkHand hb{3, "Left", hand_pts(590, 330, 70), ""};  // B's left hand raised next to A's face (rules alone: A)
        std::map<int, LinkPose> latched = {
            {1, {{430, 270}, {{{590, 365}, 0.9f}, {{380, 420}, 0.9f}}}},  // A's "pose" is B's body: nose on B
            {2, {{420, 265}, {{{590, 365}, 0.9f}, {{380, 420}, 0.9f}}}}};
        PersonLinker N;
        std::vector<std::string> nev;
        N.update(fs, {hb}, 0, nev, latched);
        check("a pose whose nose is on someone else's face is ignored: B's hand stays B's", N.owner() == std::map<int, int>{{3, 2}});
        bool said = false;
        for (auto& e : nev) said = said || e.find("pose for face #1 ignored") != std::string::npos;
        check("the log says which pose was ignored", said);
        std::map<int, LinkPose> sides = {  // A's RIGHT wrist is on the left hand, B's LEFT wrist is
            {1, {{700, 340}, {{{760, 560}, 0.9f}, {{590, 365}, 0.9f}}}},
            {2, {{420, 265}, {{{594, 362}, 0.9f}, {{380, 420}, 0.9f}}}}};
        PersonLinker M;
        M.update(fs, {hb}, 0, nev, sides);
        check("a left hand is matched only with a pose's left wrist", M.owner() == std::map<int, int>{{3, 2}});
        std::map<int, LinkPose> tie = {  // both poses' left wrists on the hand: neither gets the bonus
            {1, {{700, 340}, {{{598, 365}, 0.9f}, {{650, 560}, 0.2f}}}},
            {2, {{420, 265}, {{{582, 365}, 0.9f}, {{380, 420}, 0.2f}}}}};
        PersonLinker T;
        T.update(fs, {hb}, 0, nev, tie);
        float c1, c2;
        T.cost(hb, fs[0], c1);
        T.cost(hb, fs[1], c2);
        check("two poses claiming a hand equally: no pose bonus for either", c1 > -5 && c2 > -5);

        // Mac log 20 Nov, 72 s: the friend's pose wrists were 2.7 hand sizes off his hand (a noisy
        // pose), the near person's 10.5. A hard cutoff at 2.5 ruled the friend out and the hand
        // went to the near person; the graded cost keeps it with the friend.
        LinkHand hf{4, "Right", hand_pts(420, 396, 45), ""};
        PersonLinker G;
        G.update(fs, {hf}, 0, nev, {{1, {{700, 340}, {{{900, 800}, 0.1f}, {{850, 790}, 0.9f}}}},
                                    {2, {{420, 265}, {{{299, 418}, 0.9f}, {{541, 418}, 0.9f}}}}});
        check("a noisy pose 2.7 hand sizes off doesn't hand the friend's hand to the near person", G.owner() == std::map<int, int>{{4, 2}});
    }

    // ---------- help ----------
    const std::array<bool, 5> TUCK{false, true, true, true, true}, FIST{false, false, false, false, false},
        PALM{true, true, true, true, true};
    int facing = 1;
    auto help_run = [&](const std::vector<std::pair<std::array<bool, 5>, float>>& seq, std::vector<std::string>* evs = nullptr) {
        HelpDetector H;
        std::vector<std::string> ev;
        for (size_t i = 0; i < seq.size(); ++i) {
            HelpHand h{1, 1, seq[i].first, "", {500, 500}, {seq[i].second, 560}, 100};
            H.update({{1, cv::Rect2f(450, 200, 100, 120), "Vignesh", facing}}, {h}, i / 30.0, ev);
        }
        if (evs) *evs = ev;
        return H.alerts().size();
    };
    std::vector<std::pair<std::array<bool, 5>, float>> sig, twice, blur;
    for (int i = 0; i < 15; ++i) sig.push_back({TUCK, 500});
    for (int i = 0; i < 5; ++i) sig.push_back({FIST, 505});
    twice = sig;
    for (int i = 0; i < 20; ++i) twice.push_back({PALM, 500});
    twice.insert(twice.end(), sig.begin(), sig.end());
    for (int i = 0; i < 60; ++i) blur.push_back({(i / 4) % 2 ? FIST : TUCK, 500 + 90 * std::sin(2 * M_PI * 2.5 * i / 30)});
    std::vector<std::string> ev;
    check("one Signal for Help -> no alert yet, asks to repeat", help_run(sig, &ev) == 0 && !ev.empty() &&
                                                                    ev[0].find("1 of 2") != std::string::npos);
    check("Signal for Help twice within 6 s -> alert", help_run(twice) == 1);
    {
        auto late = sig;
        for (int i = 0; i < 210; ++i) late.push_back({PALM, 500});  // 7 s later
        late.insert(late.end(), sig.begin(), sig.end());
        check("second signal 7 s later -> no alert", help_run(late) == 0);
    }
    check("waving, blur flips tucked / fist -> no alert", help_run(blur) == 0);
    facing = 0;
    check("signal twice while looking away (e.g. at another person) -> no alert", help_run(twice) == 0);
    facing = -1;
    check("signal twice, face not seen -> alert", help_run(twice) == 1);
    facing = 1;
    {
        HelpDetector H;
        std::vector<std::string> e2;
        for (int i = 0; i < 70; ++i)
            H.update({{1, cv::Rect2f(450, 200, 100, 120), ""}},
                     {{1, 1, PALM, "Open palm", {400, 300}, {400, 340}, 90}, {2, 1, PALM, "Open palm", {600, 300}, {600, 340}, 90}},
                     i / 30.0, e2);
        check("both hands up 2 s -> alert", H.alerts().size() == 1 && H.alerts()[0].reason == "Both hands up");
        H.acknowledge_all();
        check("acknowledge clears it", H.alerts().empty());
    }

    // ---------- waves and turning ----------
    auto wave_run = [&](bool hand_moves, bool camera_turns, double hz, int frames_kept_mod = 0) {
        AttentionTracker A;
        std::vector<std::string> e3;
        int waves = 0;
        for (int i = 0; i < 60; ++i) {
            if (frames_kept_mod && i % 7 >= 2 && i % 7 <= 4) continue;  // ~40% of frames lost
            const float shift = camera_turns ? 15.f * i : 0.f;  // the whole picture slides as the robot turns
            const float hx = 760 + (hand_moves ? 90 * std::sin(2 * M_PI * hz * i / 30) : 0) - shift;
            AttentionFace f{1, cv::Rect2f(600 - shift, 200, 110, 130), false};
            AttentionHand h{1, 1, "Right", PALM, hand_pts(hx, 280, 110)};
            e3.clear();
            A.update({f}, {h}, cv::Size(1280, 720), i / 30.0, e3);
            for (auto& e : e3) waves += e.find("waved") != std::string::npos;
        }
        return waves;
    };
    check("wave -> robot called", wave_run(true, false, 2.0) >= 1);
    check("wave with 40% of frames lost -> robot called", wave_run(true, false, 2.0, 7) >= 1);
    check("still hand while the robot turns -> no wave", wave_run(false, true, 2.0) == 0);
    check("wave while the robot turns -> still a wave", wave_run(true, true, 2.0) >= 1);
    check("slow drift (0.3 Hz) -> no wave", wave_run(true, false, 0.3) == 0);
    {
        AttentionTracker A;
        std::vector<std::string> e4;
        A.update({{1, cv::Rect2f(1000, 300, 100, 120), true}}, {}, cv::Size(1280, 720), 0.0, e4);
        check("person on the image's right -> robot turns right (un-mirrored)", A.target().face_id == 1 && A.target().yaw > 0 &&
                                                                                    A.target().turn_text().find("turn right") == 0);
    }
    std::printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
