#!/bin/bash

# Build script for CellFlow CUDA

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo "Build complete! Run ./build/cellflow-cuda to start the application."