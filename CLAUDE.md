# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CellFlow is a WebGPU-based particle life simulation that creates emergent behaviors through force-based particle interactions. It uses vanilla JavaScript with ES6 modules, WebGPU for GPU computation, and AlpineJS for reactive UI.

## Development Commands

```bash
# Install dependencies (for local development server)
npm install

# Run development server (required for local development due to ES6 module CORS)
npm run dev
# Server runs at http://localhost:3000

# Build TypeScript server
npm run build

# Run production server
npm start
```

## Architecture

### Core Components

1. **WebGPU Pipeline** (`src/js/gpuSetup.js`)
   - Manages WebGPU device, buffers, and pipeline creation
   - Implements double-buffering for neighbor count tracking
   - Handles buffer lifecycle (creation/destruction on parameter changes)

2. **Shaders** (WGSL in `src/shaders/`)
   - `simShader.js`: Compute shader for particle physics simulation
   - `renderShader.js`: Vertex/fragment shaders for particle rendering  
   - `glowShader.js`: Post-processing glow effect

3. **Main Application** (`src/js/main.js`)
   - UI event handling and parameter management
   - Orchestrates simulation loop
   - Handles recording functionality

4. **File Structure**
   - `src/js/`: JavaScript modules
   - `src/shaders/`: WGSL shader files
   - `src/css/`: Stylesheets
   - `presets/`: Preset configurations (JSON)
   - `docs/`: Documentation

### Key Patterns

- **Dynamic Shader Generation**: WGSL code is generated based on particle type count
- **Force Matrix**: NxN matrix defines attraction/repulsion between particle types
- **Adaptive Forces**: Previous frame's neighbor counts adjust current forces
- **Wraparound Space**: Particles wrap at canvas boundaries

### Critical Considerations

1. **WebGPU Buffer Management**: Always properly destroy buffers before recreation
2. **Pipeline Recreation**: Required when particle count or types change
3. **Module Loading**: Requires web server (not file://) due to CORS restrictions
4. **Browser Compatibility**: Requires WebGPU-capable browser (Chrome/Edge with flags)

### Parameter Ranges

- Particle Count: 500-10,000
- Particle Types: 2-6  
- Force Range: 0.1-1.0
- Time Step: 0.0005-0.005
- Friction: 0.01-0.99

### UI Interaction

- Number keys 1-8: Load presets
- Space: Regenerate forces
- X: Toggle UI visibility
- C: Toggle slider clutter
- Mouse wheel on sliders: Fine adjustment