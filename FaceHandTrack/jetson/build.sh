#!/usr/bin/env bash
# Build the Jetson app on the robot (JetPack 5: CUDA 11.4, TensorRT 8.5, OpenCV 4.2, g++ 9).
#   ./build.sh          ->  ./facehandtrack
set -euo pipefail
cd "$(dirname "$0")"
CUDA=/usr/local/cuda
g++ -std=c++17 -O3 -march=native -Wall -Wextra -Wno-deprecated-declarations \
    -I"$CUDA/include" $(pkg-config --cflags opencv4) \
    src/*.cpp -o facehandtrack \
    -L"$CUDA/lib64" -lnvinfer -lcudart \
    $(pkg-config --libs opencv4) -lpthread
echo "built ./facehandtrack"
