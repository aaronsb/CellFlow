# CellFlow

**💪 CUDA Native: 100,000 particles @ 13 FPS**
![CellFlow CUDA - 100,000 particles at 13 FPS](docs/media/Screenshot_20250626_121558.png)

CellFlow is a high-performance particle life simulation that exhibits emergent behaviors resembling biological systems - from single-cell organisms to ecosystems like coral reefs, and occasionally even planetary or galactic formations. Built on CUDA with Qt6, it delivers real-time simulation of tens of thousands of particles with advanced visualization features.

The system builds upon foundational concepts from Clusters by Jeffrey Ventrella and Particle Life by Tom Mohr, introducing significant algorithmic improvements for more organic and complex behaviors.

## Key Features

### Smooth Force Function
The original particle life algorithms use abrupt transitions between attraction and repulsion zones, causing unnatural particle jumps. CellFlow implements a continuous force function with three parameters (attraction, repulsion, and k factor) that creates fluid, lifelike motion without sudden discontinuities.

### GPU-Accelerated Performance
The CUDA implementation enables simulation of 50,000-100,000+ particles in real-time with advanced visualization features:
- **Proximity Graph Visualization**: GPU-computed connection lines between nearby particles reveal community structures
- **Depth-based Effects**: Particles and edges fade with distance for enhanced 3D perception
- **Cluster Focusing**: Box selection (Shift+click+drag) to focus camera on particle clusters
- **Real-time Interaction**: Orbital camera controls with smooth navigation

### Advanced Interaction Systems
- **Force Matrix**: Defines attraction/repulsion relationships between particle types using Gaussian distributions with customizable bias and offset
- **Variable Interaction Radius**: The "ratio" parameter morphs behavior from fluid, liquid-like forms to solid, ossified patterns
- **Neighborhood-Based Physics**: Particles respond to local density with dynamic acceleration or friction
- **Extended Balance Range**: Values beyond 1.0 create exotic behaviors including time-reversal effects and black hole-like structures

## Building and Running

### Requirements
- CUDA-capable GPU (Compute Capability 3.5+)
- CUDA Toolkit 12.0+
- Qt6 (Qt6Core, Qt6Gui, Qt6Widgets, Qt6OpenGL)
- CMake 3.20+
- C++17 compiler

### Build Instructions

```bash
cd cuda-native
mkdir -p build
cd build
cmake ..
make -j$(nproc)
./cellflow-cuda
```

### Controls
- **Left-click + Drag**: Rotate camera (orbit)
- **Right-click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Shift + Left-click + Drag**: Box select cluster and focus camera
- **Spacebar**: Generate new force matrix
- **Number Keys 1-8**: Load preset configurations

See [cuda-native/README.md](cuda-native/README.md) for detailed build instructions and configuration options.

## Technical Documentation

For detailed algorithm explanations and implementation details, see: https://www.youtube.com/watch?v=E8vvSu8PZmI

## Performance

### 🚀 Optimal Performance: 69,500 particles @ 30 FPS
> **CUDA Native** | RTX 4060 Ti | Qt6/Wayland | Real-time interaction

![CellFlow CUDA - 69,500 particles at 30 FPS](docs/media/Screenshot_20250626_121802.png)

**Performance Highlights:**
- 50,000 particles with proximity graph: 25 FPS
- 30,000 particles with edges: 48 FPS
- GPU-accelerated proximity graph (100-1000x speedup over CPU)
- Maintains interactivity even with complex visualizations

### Performance Demo
[![CellFlow CUDA 100k Particles](https://img.youtube.com/vi/ZMKAkzZCcAk/maxresdefault.jpg)](https://youtu.be/ZMKAkzZCcAk)

## Open Source

CellFlow is released as open source to encourage learning, experimentation, and community contributions. The codebase demonstrates advanced CUDA programming techniques including compute shaders, CUDA-OpenGL interop, and real-time particle physics.

## License

See [LICENSE](LICENSE) for details.
