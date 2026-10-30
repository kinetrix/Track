# Desktop check (no GPU needed)

Builds the Jetson face path (`src/face_pipeline.cpp`, `face_tracker.cpp`, `edgetam.cpp`,
`face_detector.cpp`) against ONNX Runtime on the CPU instead of TensorRT, runs it on the v1
test clips, and compares every frame with the Python engine (`../../../onnx`).

Needs: g++, OpenCV dev headers, the ONNX Runtime C++ release (onnxruntime-linux-x64-1.20.1),
TensorRT 8.5 headers (github.com/NVIDIA/TensorRT, branch release/8.5, `include/`) and CUDA
runtime headers (pip wheels nvidia-cuda-runtime-cu11, nvidia-cuda-nvcc-cu11), plus the
Python engine's requirements.

```bash
ORT=/path/to/onnxruntime-linux-x64-1.20.1
INC="-isystem $TRT/include -isystem $CUDART/include -isystem $NVCC/include -isystem $ORT/include"
g++ -std=c++17 -O2 $INC -I../../src $(pkg-config --cflags opencv4) run_clip.cpp trt_model_ort.cpp \
    ../../src/{face_pipeline,face_tracker,edgetam,face_detector}.cpp -o run_clip \
    -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib $(pkg-config --libs opencv4) -lpthread
# ONNX files: yunet_640x384.onnx (onnx/models) + the converted EdgeTAM graphs (jetson/models)
FHT_ONNX_DIR=/path/to/onnx_files ./run_clip ../../../../FaceTrack_v1_iOS/conversion/eval/clip_group > cpp.txt
python3 run_clip_python.py ../../../../FaceTrack_v1_iOS/conversion/eval/clip_group > py.txt
python3 compare.py cpp.txt py.txt
```

Last run (Oct 30, 2026): identical track IDs, boxes (IoU 1.0000) and events on clip_group
(36 frames), clip_selfie (40), clip_longocc (60, long occlusion) and clip_crowd (30, 4 faces).
