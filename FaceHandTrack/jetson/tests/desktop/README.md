# Desktop check (no GPU needed)

Builds the Jetson face and hand paths against ONNX Runtime on the CPU instead of TensorRT
(`trt_model_ort.cpp`), with the GPU input kernel's per-pixel code run on the CPU
(`preprocess_cpu.cpp`), runs them on test clips, and compares every frame with the Python
engine (`../../../onnx`).

Needs: g++, OpenCV dev headers, the ONNX Runtime C++ release (onnxruntime-linux-x64-1.20.1),
TensorRT 8.5 headers (github.com/NVIDIA/TensorRT, branch release/8.5, `include/`) and CUDA
runtime headers (pip wheels nvidia-cuda-runtime-cu11, nvidia-cuda-nvcc-cu11), plus the
Python engine's requirements.

```bash
ORT=/path/to/onnxruntime-linux-x64-1.20.1
INC="-isystem $TRT/include -isystem $CUDART/include -isystem $NVCC/include -isystem $ORT/include"
g++ -std=c++17 -O2 $INC -I../../src $(pkg-config --cflags opencv4) run_clip.cpp trt_model_ort.cpp \
    preprocess_cpu.cpp ../../src/{gpu_timer,face_pipeline,face_tracker,edgetam,face_detector,hands,gestures,hand_pipeline}.cpp \
    -o run_clip \
    -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib $(pkg-config --libs opencv4) -lpthread
# ONNX files: yunet_640x384 + palm/handpose (onnx/models) + the converted EdgeTAM graphs (jetson/models)
python3 make_hand_clip.py ../../../../FaceTrack_v1_iOS/conversion/eval/clip_selfie clip_hand   # moving hand
FHT_ONNX_DIR=/path/to/onnx_files ./run_clip ../../../../FaceTrack_v1_iOS/conversion/eval/clip_group > cpp.txt
python3 run_clip_python.py ../../../../FaceTrack_v1_iOS/conversion/eval/clip_group > py.txt
python3 compare.py cpp.txt py.txt
```

Last run (Oct 30, 2026), with the GPU-style input kernel (its bilinear resize differs from
cv::resize by at most 1 grey level on ~2.5% of pixels):
- faces: same track IDs on clip_group, clip_longocc, clip_crowd and the hand clip (same events,
  except one hand-presence score printed 0.17 vs 0.16);
  box IoU 1.000 / 1.000 / 0.959 (smallest crowd faces) / 0.999.
- hands: the 24 labelled hand photos (3 frames each): same Left/Right, gesture and finger count
  on 24/24; moving-hand clip (open palm -> victory, 40 frames): same ID, Left/Right and gesture
  on 40/40, landmarks within 1.2 px.
