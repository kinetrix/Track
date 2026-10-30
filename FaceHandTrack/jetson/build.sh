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
g++ -std=c++17 -O3 -march=native -Wall -Wextra -Wno-deprecated-declarations \
    -I"$CUDA/include" $(pkg-config --cflags opencv4) \
    src/*.cpp build/preprocess.o -o facehandtrack \
    -L"$CUDA/lib64" -lnvinfer -lcudart \
    $(pkg-config --libs opencv4) -lpthread
echo "built ./facehandtrack"
