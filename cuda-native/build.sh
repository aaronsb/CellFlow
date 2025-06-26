#!/bin/bash

# Build script for CellFlow CUDA

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -b, --build      Build the project"
    echo "  -d, --debug      Build in Debug mode (default: Release)"
    echo "  --clean          Clean build directory"
    echo "  -r, --run        Run the application after building"
    echo "  -j, --jobs NUM   Number of parallel build jobs (default: $(nproc))"
    echo "  -v, --verbose    Enable verbose build output"
    echo ""
    echo "Examples:"
    echo "  $0               # Show this info"
    echo "  $0 --build       # Build in Release mode"
    echo "  $0 --build -d    # Build in Debug mode"
    echo "  $0 --clean       # Clean build directory"
    echo "  $0 --build -r    # Build and run"
    exit 0
}

# Default values
BUILD_TYPE="Release"
DO_BUILD=false
DO_CLEAN=false
RUN_AFTER_BUILD=false
NUM_JOBS=$(nproc)
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -b|--build)
            DO_BUILD=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        -r|--run)
            RUN_AFTER_BUILD=true
            shift
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="VERBOSE=1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# If no arguments provided, show usage
if [ "$DO_BUILD" = false ] && [ "$DO_CLEAN" = false ]; then
    echo "CellFlow CUDA Build Script"
    echo "========================="
    echo ""
    echo "Current configuration:"
    echo "  Build type: $BUILD_TYPE"
    echo "  Parallel jobs: $NUM_JOBS"
    echo ""
    usage
fi

# Clean if requested
if [ "$DO_CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
    echo "Clean complete."
    
    # Exit if only cleaning
    if [ "$DO_BUILD" = false ]; then
        exit 0
    fi
fi

# Exit if not building
if [ "$DO_BUILD" = false ]; then
    exit 0
fi

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake (${BUILD_TYPE} mode)..."
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE

# Build
echo "Building with $NUM_JOBS parallel jobs..."
make -j$NUM_JOBS $VERBOSE

if [ $? -eq 0 ]; then
    echo ""
    echo "Build complete! Executable: ./build/cellflow-cuda"
    
    # Run if requested
    if [ "$RUN_AFTER_BUILD" = true ]; then
        echo ""
        echo "Starting CellFlow CUDA..."
        ./cellflow-cuda
    fi
else
    echo "Build failed!"
    exit 1
fi