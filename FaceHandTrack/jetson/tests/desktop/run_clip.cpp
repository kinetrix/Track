// Desktop check: run the C++ face path on a folder of .jpg frames and print one line per
// track per frame:  frame id x1 y1 x2 y2 visible iou_pred   (plus "# event" lines).
//   run_clip <frames dir> [fps=15]
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <string>
#include <vector>

#include "face_pipeline.hpp"

using namespace fht;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: run_clip <frames dir> [fps]\n");
        return 1;
    }
    const std::string dir = argv[1];
    const double fps = argc > 2 ? std::atof(argv[2]) : 15.0;
    std::vector<std::string> files;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".jpg") files.push_back(dir + "/" + n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    FacePolicy policy;
    policy.max_tracks = 4;  // same as the Python default, for the comparison
    FacePipeline faces("engines", "fp16", false, policy);
    std::vector<std::string> events;
    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat bgr = cv::imread(files[i]);
        events.clear();
        faces.process(bgr, i / fps, events);
        for (const auto& e : events) std::printf("# %zu %s\n", i, e.c_str());
        for (const auto& t : faces.tracks())
            std::printf("%zu %d %.1f %.1f %.1f %.1f %d %.3f\n", i, t->id, t->box.x, t->box.y, t->box.x + t->box.width,
                        t->box.y + t->box.height, t->visible ? 1 : 0, t->iou_pred);
    }
    return 0;
}
