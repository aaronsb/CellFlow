# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CellFlow is a high-performance CUDA-based particle life simulation that creates emergent behaviors through force-based particle interactions. It uses CUDA for GPU computation, Qt6 for the UI, and OpenGL for rendering with CUDA-OpenGL interop for maximum performance.

## Development Commands

```bash
# Build the project
cd cuda-native
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Run the application
./cellflow-cuda

# Clean build
rm -rf build
```

## Architecture

### Core Components

1. **CUDA Particle Simulation** (`cuda-native/src/ParticleSimulation.cu`)
   - GPU-accelerated particle physics using CUDA kernels
   - Implements double-buffering for neighbor count tracking
   - Adaptive force calculations based on local density
   - Wraparound boundary conditions in 3D space

2. **Qt6 UI** (`cuda-native/src/MainWindow.cpp`)
   - Parameter controls using Qt6 Widgets
   - Real-time FPS monitoring
   - Preset management (load/save)
   - Particle type color controls

3. **OpenGL Rendering Widget** (`cuda-native/src/CellFlowWidget.cpp`)
   - 3D particle rendering with depth effects
   - GPU proximity graph via CUDA-OpenGL interop
   - Orbital camera with cluster focusing
   - Box selection for camera targeting
   - Depth fade effects on particles and edges

4. **File Structure**
   - `cuda-native/src/`: C++ and CUDA source files
   - `cuda-native/include/`: Header files
   - `cuda-native/shaders/`: GLSL shader snippets (inline in code)
   - `cuda-native/*.json`: Preset configurations
   - `docs/`: Documentation and ADRs

### Key Patterns

- **CUDA-OpenGL Interop**: Proximity graph computed entirely on GPU and written directly to OpenGL VBO
- **Force Matrix**: NxN matrix defines attraction/repulsion between particle types using Gaussian distributions
- **Adaptive Forces**: Previous frame's neighbor counts adjust current forces dynamically
- **Spherical Camera**: Orbital camera using spherical coordinates (pitch/yaw + distance)
- **Coordinate Spaces**: Particles in world space (0 to canvasSize), rendered in centered space (centered around origin)

### Critical Considerations

1. **CUDA Memory Management**: Properly allocate/free device memory, use cudaDeviceSynchronize when needed
2. **CUDA-OpenGL Interop**: Register OpenGL buffers with CUDA using cudaGraphicsGLRegisterBuffer
3. **Coordinate Space Matching**: Camera target must be in centered space (world_pos - canvasSize * 0.5)
4. **Qt6 OpenGL Context**: Use QOpenGLWidget with QOpenGLFunctions for modern OpenGL
5. **Depth Filtering**: Box selection uses NDC depth coordinates to select only front particles

### Parameter Ranges

- Particle Count: 1,000-100,000+
- Particle Types: 2-6
- Universe Size: 5,000-20,000
- Radius: 50-500
- Delta T: 0.0005-0.005
- Friction: 0.01-0.99
- Proximity Distance: 50-500

### Controls

- **Left-click + Drag**: Rotate camera (orbit)
- **Right-click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Shift + Left-click + Drag**: Box select cluster and focus camera
- **Spacebar**: Regenerate force matrix
- **Number Keys 1-8**: Load presets

### Performance Notes

- 50,000 particles with proximity graph: ~25 FPS (RTX 4060 Ti)
- 30,000 particles with edges: ~48 FPS
- GPU proximity graph provides 100-1000x speedup over CPU implementation
- All particle interactions computed on GPU with no CPU bottleneck
