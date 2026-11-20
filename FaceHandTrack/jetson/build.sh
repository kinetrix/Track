#!/usr/bin/env bash
# Build the Jetson app on the robot (JetPack 5: CUDA 11.4, TensorRT 8.5, OpenCV 4.2, g++ 9).
#   ./build.sh          ->  ./facehandtrack
set -euo pipefail
cd "$(dirname "$0")"
CUDA=/usr/local/cuda
mkdir -p build
# GPU kernels (Orin = sm_87)
"$CUDA/bin/nvcc" -std=c++17 -O3 -arch=sm_87 -Xcompiler -Wno-deprecated-declarations \
    $(pkg-config --cflags opencv4) -c src/preprocess.cu -o build/preprocess.o
# RealSense depth (--camera realsense): librealsense from ROS Noetic's package, or a system install
RS=()
for d in /opt/ros/noetic /usr/local /usr; do
    lib=$(ls "$d"/lib/librealsense2.so "$d"/lib/*/librealsense2.so 2>/dev/null | head -1 || true)
    if [ -f "$d/include/librealsense2/rs.hpp" ] && [ -n "$lib" ]; then
        RS=(-DHAVE_REALSENSE -isystem "$d/include" -L"$(dirname "$lib")" -Wl,-rpath,"$(dirname "$lib")" -lrealsense2)
        echo "librealsense: $lib (depth on)"
        break
    fi
done
[ ${#RS[@]} -eq 0 ] && echo "librealsense not found: built without depth"
g++ -std=c++17 -O3 -march=native -Wall -Wextra -Wno-deprecated-declarations \
    -I"$CUDA/include" $(pkg-config --cflags opencv4) \
    src/*.cpp build/preprocess.o -o facehandtrack \
    -L"$CUDA/lib64" -lnvinfer -lcudart \
    $(pkg-config --libs opencv4) "${RS[@]}" -lpthread
echo "built ./facehandtrack"
