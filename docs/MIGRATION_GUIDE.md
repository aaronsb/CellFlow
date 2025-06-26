# CellFlow GPU Migration Guide

## Overview

This guide provides step-by-step instructions for migrating CellFlow from browser-based WebGPU to alternative GPU computing approaches. Each approach offers different trade-offs between performance, complexity, and compatibility.

## Quick Decision Matrix

| Your Priority | Recommended Approach | Expected Improvement |
|--------------|---------------------|---------------------|
| Minimal code changes | Node-WebGPU | 2x performance |
| Maximum performance | Native Vulkan/CUDA | 10-20x performance |
| Cross-platform desktop | Electron + Native | 5-10x performance |
| Easy deployment | WebAssembly SIMD | 0.5-1x performance |
| Research/experimentation | Hybrid architecture | 15-25x performance |

## Approach 1: Node-WebGPU Migration (Easiest)

### Prerequisites
```bash
npm install @webgpu/node
npm install ws  # for WebSocket communication
```

### Step 1: Extract Core Simulation Logic
Move simulation logic from `gpuSetup.js` to a shared module:

```javascript
// src/core/simulation-core.js
export const simulationShader = /* your existing WGSL shader */;

export class SimulationConfig {
    constructor() {
        this.particleCount = 8000;
        this.numParticleTypes = 6;
        // ... other parameters
    }
}
```

### Step 2: Create Native Runner
```javascript
// src/native/native-runner.js
import { GPU } from '@webgpu/node';
import { simulationShader, SimulationConfig } from '../core/simulation-core.js';

class NativeSimulation {
    async initialize() {
        const gpu = new GPU();
        this.adapter = await gpu.requestAdapter();
        this.device = await this.adapter.requestDevice();
        // Reuse existing shader setup code
    }
}
```

### Step 3: Add Communication Layer
```javascript
// src/native/websocket-bridge.js
import { WebSocketServer } from 'ws';

export class SimulationBridge {
    constructor(simulation) {
        this.simulation = simulation;
        this.wss = new WebSocketServer({ port: 8080 });
        
        this.wss.on('connection', (ws) => {
            ws.on('message', (data) => {
                const msg = JSON.parse(data);
                if (msg.type === 'updateParams') {
                    this.simulation.updateParameters(msg.params);
                }
            });
        });
    }
}
```

### Step 4: Update Frontend
```javascript
// Modify main.js
const ws = new WebSocket('ws://localhost:8080');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderParticles(data.positions, data.types);
};
```

## Approach 2: Vulkan Implementation (High Performance)

### Prerequisites
- Vulkan SDK
- C++ compiler with C++17 support
- CMake

### Step 1: Set Up Project Structure
```
cellflow-vulkan/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── VulkanCompute.h
│   ├── VulkanCompute.cpp
│   └── shaders/
│       └── particle.comp
└── bindings/
    └── node-bindings.cpp
```

### Step 2: Port Shaders to GLSL
```glsl
// shaders/particle.comp
#version 450

struct Particle {
    vec2 pos;
    vec2 vel;
    vec2 acc;
    uint type;
    uint pad;
};

layout(binding = 0, std430) buffer ParticleBuffer {
    Particle particles[];
} particleBuffer;

layout(binding = 1, std430) readonly buffer ForceTable {
    float forces[];
} forceTable;

layout(push_constant) uniform PushConstants {
    float radius;
    float deltaT;
    float friction;
    uint particleCount;
} params;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.particleCount) return;
    
    // Port your WebGPU shader logic here
}
```

### Step 3: Create Vulkan Compute Pipeline
```cpp
// VulkanCompute.cpp
class VulkanCompute {
public:
    void initialize() {
        createInstance();
        selectPhysicalDevice();
        createDevice();
        createComputePipeline();
        createBuffers();
    }
    
    void simulate(float deltaTime) {
        updateUniformBuffer(deltaTime);
        submitComputeCommands();
        retrieveResults();
    }
    
private:
    VkInstance instance;
    VkDevice device;
    VkPipeline computePipeline;
    // ... other Vulkan objects
};
```

### Step 4: Create Node.js Bindings
```cpp
// bindings/node-bindings.cpp
#include <napi.h>
#include "../src/VulkanCompute.h"

class VulkanSimulation : public Napi::ObjectWrap<VulkanSimulation> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "VulkanSimulation", {
            InstanceMethod("initialize", &VulkanSimulation::Initialize),
            InstanceMethod("simulate", &VulkanSimulation::Simulate),
            InstanceMethod("getParticles", &VulkanSimulation::GetParticles)
        });
        
        exports.Set("VulkanSimulation", func);
        return exports;
    }
    
private:
    std::unique_ptr<VulkanCompute> compute;
};
```

## Approach 3: Hybrid Architecture (Maximum Flexibility)

### Architecture Overview
```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│   Web Browser   │ ◄─────────────────► │  Compute Server  │
│   (UI + WebGL)  │     Particles       │  (GPU Backend)   │
└─────────────────┘                     └──────────────────┘
         │                                        │
         │                                        │
         ▼                                        ▼
   ┌───────────┐                          ┌─────────────┐
   │  Canvas   │                          │ GPU (Native)│
   │ Rendering │                          │   Compute   │
   └───────────┘                          └─────────────┘
```

### Step 1: Design Protocol
```typescript
// protocol.ts
interface SimulationMessage {
    type: 'frame' | 'config' | 'command';
    timestamp: number;
    data: any;
}

interface FrameData {
    positions: ArrayBuffer;  // Float32Array buffer
    velocities: ArrayBuffer; // Float32Array buffer
    types: ArrayBuffer;      // Uint8Array buffer
    frameNumber: number;
}
```

### Step 2: Implement Compute Server
```rust
// Rust implementation with wgpu
use wgpu;
use tokio_tungstenite;

struct ComputeServer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    particles: wgpu::Buffer,
}

impl ComputeServer {
    async fn run_simulation_loop(&mut self, tx: Sender<FrameData>) {
        loop {
            self.execute_compute_pass().await;
            let frame_data = self.read_particle_data().await;
            tx.send(frame_data).await;
            tokio::time::sleep(Duration::from_millis(16)).await; // 60 FPS
        }
    }
}
```

### Step 3: Optimize Data Transfer
```javascript
// Frontend optimization for receiving binary data
class OptimizedRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');
        this.setupWebGL();
    }
    
    setupWebGL() {
        // Create vertex buffer for particles
        this.particleBuffer = this.gl.createBuffer();
        
        // Create shader program for instanced rendering
        this.program = this.createShaderProgram(
            vertexShaderSource,
            fragmentShaderSource
        );
    }
    
    renderFrame(frameData) {
        // Decode binary data
        const positions = new Float32Array(frameData.positions);
        
        // Update GPU buffer directly
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.DYNAMIC_DRAW);
        
        // Render using instanced drawing
        this.gl.drawArraysInstanced(
            this.gl.POINTS, 0, 1, positions.length / 2
        );
    }
}
```

## Performance Optimization Tips

### 1. Buffer Management
```javascript
// Reuse buffers instead of creating new ones
class BufferPool {
    constructor(size, length) {
        this.buffers = Array(size).fill(null).map(() => 
            new ArrayBuffer(length)
        );
        this.available = [...this.buffers];
        this.inUse = new Set();
    }
    
    acquire() {
        const buffer = this.available.pop();
        this.inUse.add(buffer);
        return buffer;
    }
    
    release(buffer) {
        this.inUse.delete(buffer);
        this.available.push(buffer);
    }
}
```

### 2. Compression for Network Transfer
```javascript
// Use compression for particle data
import pako from 'pako';

function compressParticleData(positions, types) {
    const data = new ArrayBuffer(positions.byteLength + types.byteLength);
    const view = new Uint8Array(data);
    view.set(new Uint8Array(positions.buffer), 0);
    view.set(new Uint8Array(types.buffer), positions.byteLength);
    
    return pako.deflate(data);
}
```

### 3. Adaptive Quality
```javascript
// Adjust particle count based on performance
class AdaptiveSimulation {
    constructor() {
        this.targetFPS = 60;
        this.particleCount = 8000;
        this.frameTimeHistory = [];
    }
    
    adjustQuality(frameTime) {
        this.frameTimeHistory.push(frameTime);
        if (this.frameTimeHistory.length > 60) {
            this.frameTimeHistory.shift();
        }
        
        const avgFrameTime = this.frameTimeHistory.reduce((a, b) => a + b, 0) 
                           / this.frameTimeHistory.length;
        
        if (avgFrameTime > 17 && this.particleCount > 1000) {
            // Reduce particles if falling below 60 FPS
            this.particleCount = Math.floor(this.particleCount * 0.9);
            this.recreateBuffers();
        } else if (avgFrameTime < 14 && this.particleCount < 100000) {
            // Increase particles if performance is good
            this.particleCount = Math.floor(this.particleCount * 1.1);
            this.recreateBuffers();
        }
    }
}
```

## Testing and Benchmarking

### 1. Create Test Suite
```javascript
// test/performance.test.js
import { NativeSimulation } from '../src/native/native-runner.js';

describe('Performance Tests', () => {
    test('Native vs Browser Performance', async () => {
        const native = new NativeSimulation();
        await native.initialize();
        
        const iterations = 1000;
        const start = performance.now();
        
        for (let i = 0; i < iterations; i++) {
            await native.simulate();
        }
        
        const elapsed = performance.now() - start;
        const fps = (iterations * 1000) / elapsed;
        
        expect(fps).toBeGreaterThan(60);
        console.log(`Native FPS: ${fps.toFixed(1)}`);
    });
});
```

### 2. Profile GPU Usage
```bash
# For NVIDIA GPUs
nvidia-smi dmon -s pucvmet

# For AMD GPUs
rocm-smi --showuse

# For Intel GPUs
intel_gpu_top
```

## Deployment Strategies

### 1. Docker Container
```dockerfile
# Dockerfile for hybrid server
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    build-essential \
    vulkan-tools

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build:native

EXPOSE 8080
CMD ["node", "dist/server.js"]
```

### 2. Electron App
```javascript
// electron-main.js
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');

let mainWindow;
let computeProcess;

app.whenReady().then(() => {
    // Start native compute process
    computeProcess = spawn('./native-compute', ['--port', '8080']);
    
    mainWindow = new BrowserWindow({
        width: 1920,
        height: 1080,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });
    
    mainWindow.loadFile('index.html');
});
```

## Troubleshooting Common Issues

### Issue: GPU Not Detected
```javascript
// Add fallback to CPU
async function initializeCompute() {
    try {
        const gpu = new GPU();
        const adapter = await gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }
        return new GPUCompute(adapter);
    } catch (error) {
        console.warn('GPU not available, falling back to CPU');
        return new CPUCompute();
    }
}
```

### Issue: Memory Leaks
```javascript
// Proper cleanup
class ComputeManager {
    constructor() {
        this.resources = new Set();
    }
    
    createBuffer(size) {
        const buffer = this.device.createBuffer({ size });
        this.resources.add(buffer);
        return buffer;
    }
    
    cleanup() {
        this.resources.forEach(resource => {
            if (resource.destroy) {
                resource.destroy();
            }
        });
        this.resources.clear();
    }
}
```

## Next Steps

1. **Benchmark Your Current Implementation**: Run the provided benchmark scripts to establish baseline performance
2. **Choose Your Approach**: Based on your requirements and the decision matrix
3. **Implement Incrementally**: Start with the simulation core, then add visualization
4. **Test Thoroughly**: Ensure feature parity with the original implementation
5. **Optimize**: Use profiling tools to identify and fix bottlenecks

## Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [wgpu-rs Documentation](https://wgpu.rs/)
- [Node.js N-API Documentation](https://nodejs.org/api/n-api.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)