/**
 * Node.js WebGPU implementation for CellFlow particle simulation
 * This demonstrates how to port the existing WebGPU code to run natively
 * with potentially better performance and no browser limitations
 */

import { GPU } from '@webgpu/node';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

class CellFlowNativeGPU {
    constructor() {
        this.device = null;
        this.particleCount = 8000; // Can handle more particles natively
        this.numParticleTypes = 6;
        
        // Simulation parameters
        this.params = {
            radius: 50.0,
            delta_t: 0.22,
            friction: 0.71,
            repulsion: 50.0,
            attraction: 0.62,
            k: 16.57,
            balance: 0.79,
            forceRange: 0.28,
            forceBias: -0.20,
            ratio: 0.0,
            forceMultiplier: 2.33,
            forceOffset: 1.0,
            canvasWidth: 1920,
            canvasHeight: 1080
        };
        
        // GPU resources
        this.buffers = {};
        this.pipelines = {};
        this.bindGroups = {};
    }

    async initialize() {
        // Request GPU adapter and device
        const gpu = new GPU();
        const adapter = await gpu.requestAdapter({
            powerPreference: 'high-performance'
        });
        
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }
        
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxBufferSize: 256 * 1024 * 1024, // 256MB
                maxComputeWorkgroupSizeX: 256,
                maxComputeWorkgroupSizeY: 256,
                maxComputeWorkgroupSizeZ: 64
            }
        });
        
        console.log('GPU Device initialized:', adapter.name);
        
        // Create buffers
        await this.createBuffers();
        
        // Load and create shaders
        await this.createPipelines();
        
        // Initialize particle data
        this.initializeParticles();
    }

    async createBuffers() {
        const particleStructSize = 32; // 8 floats per particle
        
        // Particle buffer
        this.buffers.particles = this.device.createBuffer({
            size: this.particleCount * particleStructSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false
        });
        
        // Force table buffer
        this.buffers.forceTable = this.device.createBuffer({
            size: this.numParticleTypes * this.numParticleTypes * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        // Simulation parameters buffer
        this.buffers.simParams = this.device.createBuffer({
            size: 14 * 4, // 14 float parameters
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        
        // Radio by type buffer
        this.buffers.radioByType = this.device.createBuffer({
            size: this.numParticleTypes * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        // Neighbor count buffers for ping-pong
        this.buffers.neighborCountsA = this.device.createBuffer({
            size: this.particleCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        this.buffers.neighborCountsB = this.device.createBuffer({
            size: this.particleCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        // Output buffer for reading results back to CPU
        this.buffers.readback = this.device.createBuffer({
            size: this.particleCount * particleStructSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    async createPipelines() {
        // Load the existing shader code
        // In practice, you'd load the actual shader file
        const shaderCode = `
struct Particle {
    pos: vec2f,
    vel: vec2f,
    acc: vec2f,
    ptype: u32,
    pad: u32
}

struct SimParams {
    radius: f32,
    delta_t: f32,
    friction: f32,
    repulsion: f32,
    attraction: f32,
    k: f32,
    balance: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    numParticleTypes: f32,
    ratio: f32,
    forceMultiplier: f32,
    maxExpectedNeighbors: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> forceTable: array<f32>;
@group(0) @binding(2) var<uniform> simParams: SimParams;
@group(0) @binding(3) var<storage, read> radioByType: array<f32>;
@group(0) @binding(4) var<storage, read_write> previousFrameNeighborCounts: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }
    
    let me = particles[i];
    var force = vec2f(0.0);
    
    let myType = me.ptype;
    let effectiveRadius = simParams.radius + (simParams.radius * radioByType[myType] * simParams.ratio);
    
    var neighbors_count = 0u;
    
    // N-body simulation logic here (simplified for example)
    for (var j = 0u; j < arrayLength(&particles); j++) {
        if (i == j) { continue; }
        let other = particles[j];
        var dir = other.pos - me.pos;
        
        // Wrap around boundaries
        dir.x = dir.x - floor(dir.x / simParams.canvasWidth + 0.5) * simParams.canvasWidth;
        dir.y = dir.y - floor(dir.y / simParams.canvasHeight + 0.5) * simParams.canvasHeight;
        
        let dist = length(dir);
        if (dist == 0.0 || dist > effectiveRadius) { continue; }
        
        neighbors_count++;
        
        let r = dist / effectiveRadius;
        let a = forceTable[me.ptype * u32(simParams.numParticleTypes) + other.ptype];
        
        let rep_decay = r * simParams.k;
        let repulsion_ = simParams.repulsion * (1.0 / (1.0 + rep_decay * rep_decay));
        let attraction_ = simParams.attraction * r * r;
        let f = a * (repulsion_ - attraction_) * simParams.forceMultiplier;
        force += normalize(dir) * f;
    }
    
    // Apply adaptive force based on neighbor density
    let prev_neighbors = previousFrameNeighborCounts[i];
    let normalized_density = f32(prev_neighbors) / simParams.maxExpectedNeighbors;
    let clamped_normalized_density = clamp(normalized_density, 0.0, 1.0);
    
    let min_force_mult = mix(1.0, 0.01, simParams.balance);
    let max_force_mult = mix(1.0, 4.0, simParams.balance);
    let adaptive_multiplier = mix(max_force_mult, min_force_mult, clamped_normalized_density);
    
    // Update velocity and position
    var vel = me.vel * simParams.friction;
    vel += force * simParams.delta_t * adaptive_multiplier;
    
    var pos = me.pos + vel * simParams.delta_t;
    
    // Wrap positions
    if (pos.x < 0.0) { pos.x += simParams.canvasWidth; }
    if (pos.x >= simParams.canvasWidth) { pos.x -= simParams.canvasWidth; }
    if (pos.y < 0.0) { pos.y += simParams.canvasHeight; }
    if (pos.y >= simParams.canvasHeight) { pos.y -= simParams.canvasHeight; }
    
    // Write back results
    particles[i].pos = pos;
    particles[i].vel = vel;
    particles[i].acc = force;
    
    previousFrameNeighborCounts[i] = neighbors_count;
}`;

        const shaderModule = this.device.createShaderModule({
            code: shaderCode
        });
        
        // Create compute pipeline
        this.pipelines.compute = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
        
        // Create bind groups
        this.bindGroups.computeA = this.device.createBindGroup({
            layout: this.pipelines.compute.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.particles } },
                { binding: 1, resource: { buffer: this.buffers.forceTable } },
                { binding: 2, resource: { buffer: this.buffers.simParams } },
                { binding: 3, resource: { buffer: this.buffers.radioByType } },
                { binding: 4, resource: { buffer: this.buffers.neighborCountsA } }
            ]
        });
        
        this.bindGroups.computeB = this.device.createBindGroup({
            layout: this.pipelines.compute.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.particles } },
                { binding: 1, resource: { buffer: this.buffers.forceTable } },
                { binding: 2, resource: { buffer: this.buffers.simParams } },
                { binding: 3, resource: { buffer: this.buffers.radioByType } },
                { binding: 4, resource: { buffer: this.buffers.neighborCountsB } }
            ]
        });
    }

    initializeParticles() {
        const particleData = new ArrayBuffer(this.particleCount * 32);
        const floatView = new Float32Array(particleData);
        const uintView = new Uint32Array(particleData);
        
        // Initialize particles with random positions and types
        for (let i = 0; i < this.particleCount; i++) {
            const base = i * 8;
            floatView[base + 0] = Math.random() * this.params.canvasWidth; // posX
            floatView[base + 1] = Math.random() * this.params.canvasHeight; // posY
            floatView[base + 2] = 0; // velX
            floatView[base + 3] = 0; // velY
            floatView[base + 4] = 0; // accX
            floatView[base + 5] = 0; // accY
            uintView[base + 6] = Math.floor(Math.random() * this.numParticleTypes); // type
            uintView[base + 7] = 0; // padding
        }
        
        this.device.queue.writeBuffer(this.buffers.particles, 0, particleData);
        
        // Initialize force table
        const forceTable = new Float32Array(this.numParticleTypes * this.numParticleTypes);
        for (let i = 0; i < forceTable.length; i++) {
            const raw = Math.random() * 2 - 1;
            forceTable[i] = Math.tanh(raw * this.params.forceOffset) * this.params.forceRange + this.params.forceBias;
        }
        this.device.queue.writeBuffer(this.buffers.forceTable, 0, forceTable);
        
        // Initialize radio by type
        const radioByType = new Float32Array(this.numParticleTypes);
        for (let i = 0; i < this.numParticleTypes; i++) {
            radioByType[i] = Math.random() * 2 - 1;
        }
        this.device.queue.writeBuffer(this.buffers.radioByType, 0, radioByType);
        
        // Update simulation parameters
        this.updateSimParams();
    }

    updateSimParams() {
        const simParams = new Float32Array([
            this.params.radius,
            this.params.delta_t,
            this.params.friction,
            this.params.repulsion,
            this.params.attraction,
            this.params.k,
            this.params.balance,
            this.params.canvasWidth,
            this.params.canvasHeight,
            this.numParticleTypes,
            this.params.ratio,
            this.params.forceMultiplier,
            400 // maxExpectedNeighbors
        ]);
        
        this.device.queue.writeBuffer(this.buffers.simParams, 0, simParams);
    }

    async simulate(steps = 1) {
        let useBufferA = true;
        
        for (let step = 0; step < steps; step++) {
            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            
            computePass.setPipeline(this.pipelines.compute);
            computePass.setBindGroup(0, useBufferA ? this.bindGroups.computeA : this.bindGroups.computeB);
            computePass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
            computePass.end();
            
            // Copy results to readback buffer for analysis
            commandEncoder.copyBufferToBuffer(
                this.buffers.particles,
                0,
                this.buffers.readback,
                0,
                this.particleCount * 32
            );
            
            this.device.queue.submit([commandEncoder.finish()]);
            
            useBufferA = !useBufferA;
        }
        
        // Wait for GPU operations to complete
        await this.device.queue.onSubmittedWorkDone();
    }

    async getParticleData() {
        // Map the readback buffer to read particle data
        await this.buffers.readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(this.buffers.readback.getMappedRange());
        const copy = new Float32Array(data);
        this.buffers.readback.unmap();
        
        return copy;
    }

    updateParameters(newParams) {
        Object.assign(this.params, newParams);
        this.updateSimParams();
    }

    destroy() {
        // Clean up GPU resources
        Object.values(this.buffers).forEach(buffer => buffer.destroy());
        this.device.destroy();
    }
}

// Example usage and benchmarking
async function benchmark() {
    console.log('Initializing CellFlow Native GPU...');
    const simulator = new CellFlowNativeGPU();
    
    try {
        await simulator.initialize();
        
        console.log('Running benchmark...');
        const iterations = 1000;
        const startTime = performance.now();
        
        // Run simulation
        await simulator.simulate(iterations);
        
        const endTime = performance.now();
        const totalTime = endTime - startTime;
        const timePerFrame = totalTime / iterations;
        const fps = 1000 / timePerFrame;
        
        console.log(`Benchmark Results:`);
        console.log(`- Particles: ${simulator.particleCount}`);
        console.log(`- Iterations: ${iterations}`);
        console.log(`- Total time: ${totalTime.toFixed(2)}ms`);
        console.log(`- Time per frame: ${timePerFrame.toFixed(2)}ms`);
        console.log(`- Theoretical FPS: ${fps.toFixed(1)}`);
        
        // Get final particle data
        const particleData = await simulator.getParticleData();
        console.log(`Sample particle position: [${particleData[0].toFixed(2)}, ${particleData[1].toFixed(2)}]`);
        
    } catch (error) {
        console.error('Error:', error);
    } finally {
        simulator.destroy();
    }
}

// Export for use as a module
export { CellFlowNativeGPU };

// Run benchmark if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    benchmark();
}