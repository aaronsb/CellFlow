# CellFlow CUDA Native Implementation

High-performance native implementation of CellFlow using CUDA and Qt6 for Linux/Wayland environments.

## Features

- **CUDA-accelerated particle simulation** - Handles 50,000-100,000+ particles
- **Native Qt6 interface** - Optimized for Wayland with OpenGL rendering
- **Real-time parameter adjustment** - All controls from web version
- **Preset support** - Compatible with existing JSON presets
- **High precision controls** - Fine-tuning with increment adjustment

## Requirements

- CUDA Toolkit 11.0+
- Qt6 (Core, Widgets, OpenGL, OpenGLWidgets)
- CMake 3.18+
- OpenGL 3.3+
- NVIDIA GPU with compute capability 7.5+

## Building

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install qt6-base-dev qt6-opengl-dev cmake build-essential

# Build the application
./build.sh

# Or manually:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Running

```bash
./build/cellflow-cuda
```

## Performance

### Video Demonstration

[![CellFlow CUDA Performance Demo](https://img.youtube.com/vi/ZMKAkzZCcAk/maxresdefault.jpg)](https://youtu.be/ZMKAkzZCcAk)

*Click to watch the CUDA implementation in action with 100,000+ particles*

### Measured Performance (RTX 4060 Ti)

![Performance comparison](docs/performance-screenshots.md)

- **100,000 particles**: 13.8 FPS
- **69,500 particles**: 28.4 FPS (optimal for smooth interaction)
- **Performance gain**: 12-17x over WebGPU version

The WebGPU version achieves similar frame rates with only ~8,000 particles.

### Expected Performance (Other GPUs)
- GTX 1080: ~50,000 particles @ 30 FPS
- RTX 3080: ~120,000 particles @ 30 FPS
- RTX 4090: ~200,000 particles @ 30 FPS

## Controls

### Keyboard
- **1-8**: Load presets
- **Space**: Regenerate force matrix
- **Arrow Keys**: Move universe
- **Mouse Wheel**: Fine-tune sliders

### UI Controls
All parameters from the web version are available:
- Particle count and types
- Physics parameters (radius, friction, forces)
- Advanced parameters (force range, bias, LFO)
- Adaptive parameters (balance, multiplier)

## Architecture

- **CUDA Kernels**: Direct port of WGSL compute shaders
- **Double Buffering**: Neighbor count tracking for adaptive forces
- **OpenGL Rendering**: Point-based particle rendering with blending
- **Qt6 Interface**: Native widgets for Wayland compatibility

## Differences from Web Version

- **Performance**: 10-30x more particles
- **Memory**: Direct GPU memory management
- **Precision**: Full float32 throughout
- **Controls**: Native Qt widgets with better responsiveness