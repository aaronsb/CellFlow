#include "ParticleSimulation.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstring>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Initialize CURAND states
__global__ void initCurandKernel(curandState* states, int count, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Initialize particles kernel
__global__ void initializeParticlesKernel(
    Particle* particles,
    int particleCount,
    int numTypes,
    float canvasWidth,
    float canvasHeight,
    float canvasDepth,
    float spawnRegionSize,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    curandState localState = states[idx];

    // Spawn in central region, centered in the universe
    float spawnWidth = fminf(spawnRegionSize, canvasWidth);
    float spawnHeight = fminf(spawnRegionSize, canvasHeight);
    float spawnDepth = fminf(spawnRegionSize, canvasDepth);

    float offsetX = (canvasWidth - spawnWidth) * 0.5f;
    float offsetY = (canvasHeight - spawnHeight) * 0.5f;
    float offsetZ = (canvasDepth - spawnDepth) * 0.5f;

    particles[idx].pos.x = offsetX + curand_uniform(&localState) * spawnWidth;
    particles[idx].pos.y = offsetY + curand_uniform(&localState) * spawnHeight;
    particles[idx].pos.z = offsetZ + curand_uniform(&localState) * spawnDepth;
    particles[idx].vel.x = 0.0f;
    particles[idx].vel.y = 0.0f;
    particles[idx].vel.z = 0.0f;
    particles[idx].acc.x = 0.0f;
    particles[idx].acc.y = 0.0f;
    particles[idx].acc.z = 0.0f;
    particles[idx].ptype = (unsigned int)(curand_uniform(&localState) * numTypes);
    particles[idx].pad = 0.0f;

    states[idx] = localState;
}

// Main simulation kernel - ported from WGSL shader
__global__ void simulateParticlesKernel(
    Particle* particles,
    const float* forceTable,
    const float* radioByType,
    int* neighborCountsIn,
    int* neighborCountsOut,
    SimulationParams params,
    int particleCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    
    Particle p = particles[idx];
    float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);
    int neighborCount = 0;

    // Calculate forces from all other particles
    for (int j = 0; j < particleCount; j++) {
        if (idx == j) continue;

        Particle other = particles[j];

        // Calculate distance with wrapping in 3D
        float dx = other.pos.x - p.pos.x;
        float dy = other.pos.y - p.pos.y;
        float dz = other.pos.z - p.pos.z;

        // Wrap around boundaries (toroidal space in 3D)
        if (dx > params.canvasWidth * 0.5f) dx -= params.canvasWidth;
        if (dx < -params.canvasWidth * 0.5f) dx += params.canvasWidth;
        if (dy > params.canvasHeight * 0.5f) dy -= params.canvasHeight;
        if (dy < -params.canvasHeight * 0.5f) dy += params.canvasHeight;
        if (dz > params.canvasDepth * 0.5f) dz -= params.canvasDepth;
        if (dz < -params.canvasDepth * 0.5f) dz += params.canvasDepth;

        float distSq = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(distSq + 0.0001f); // Avoid division by zero
        
        // Calculate effective radius
        float radioP = params.radius * (1.0f + radioByType[p.ptype] * params.ratioWithLFO);
        float radioOther = params.radius * (1.0f + radioByType[other.ptype] * params.ratioWithLFO);
        float effectiveRadius = (radioP + radioOther) * 0.5f;
        
        if (dist < effectiveRadius) {
            neighborCount++;
            
            // Get force from force table
            float forceValue = forceTable[p.ptype * params.numParticleTypes + other.ptype];
            
            // Calculate force using the smooth force function from the original
            float r = dist / effectiveRadius;
            float repulsionForce = params.repulsion * expf(-params.k * r * r);
            float attractionForce = params.attraction * r;
            float netForce = repulsionForce - attractionForce;
            
            // Apply force value from table
            netForce *= forceValue;

            // Apply force in direction of other particle (3D)
            float3 forceDir = make_float3(dx / dist, dy / dist, dz / dist);
            totalForce.x += forceDir.x * netForce;
            totalForce.y += forceDir.y * netForce;
            totalForce.z += forceDir.z * netForce;
        }
    }
    
    // Apply adaptive force based on neighbor count
    int prevNeighborCount = neighborCountsIn[idx];
    float avgNeighborCount = (float)(neighborCount + prevNeighborCount) * 0.5f;
    float densityFactor = fminf(avgNeighborCount / (float)params.maxExpectedNeighbors, 1.0f);
    float adaptiveForceFactor = 1.0f - (1.0f - params.balance) * densityFactor;
    
    totalForce.x *= params.forceMultiplier * adaptiveForceFactor;
    totalForce.y *= params.forceMultiplier * adaptiveForceFactor;
    totalForce.z *= params.forceMultiplier * adaptiveForceFactor;

    // Update acceleration
    p.acc = totalForce;

    // Update velocity with friction (3D)
    p.vel.x = p.vel.x * params.friction + p.acc.x * params.delta_t;
    p.vel.y = p.vel.y * params.friction + p.acc.y * params.delta_t;
    p.vel.z = p.vel.z * params.friction + p.acc.z * params.delta_t;

    // Update position (3D)
    p.pos.x += p.vel.x * params.delta_t;
    p.pos.y += p.vel.y * params.delta_t;
    p.pos.z += p.vel.z * params.delta_t;

    // Wrap position (toroidal space in 3D)
    p.pos.x = fmodf(p.pos.x + params.canvasWidth, params.canvasWidth);
    p.pos.y = fmodf(p.pos.y + params.canvasHeight, params.canvasHeight);
    p.pos.z = fmodf(p.pos.z + params.canvasDepth, params.canvasDepth);
    
    // Write back
    particles[idx] = p;
    neighborCountsOut[idx] = neighborCount;
}

// Move particles kernel
__global__ void moveParticlesKernel(
    Particle* particles,
    int particleCount,
    float dx,
    float dy,
    float dz,
    float canvasWidth,
    float canvasHeight,
    float canvasDepth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    particles[idx].pos.x = fmodf(particles[idx].pos.x + dx + canvasWidth, canvasWidth);
    particles[idx].pos.y = fmodf(particles[idx].pos.y + dy + canvasHeight, canvasHeight);
    particles[idx].pos.z = fmodf(particles[idx].pos.z + dz + canvasDepth, canvasDepth);
}

// ParticleSimulation implementation
ParticleSimulation::ParticleSimulation(int particleCount)
    : particleCount(particleCount), numParticleTypes(6), useBufferAasInput(true),
      currentCanvasWidth(8000.0f), currentCanvasHeight(8000.0f), currentCanvasDepth(8000.0f) {
    allocateMemory();
    initializeCurand();
    initializeParticles();
    initializeForceTable();
    initializeRadioByType();
}

ParticleSimulation::ParticleSimulation(int particleCount, float canvasWidth, float canvasHeight)
    : particleCount(particleCount), numParticleTypes(6), useBufferAasInput(true),
      currentCanvasWidth(canvasWidth), currentCanvasHeight(canvasHeight), currentCanvasDepth(canvasHeight) {
    allocateMemory();
    initializeCurand();
    initializeParticles();
    initializeForceTable();
    initializeRadioByType();
}

ParticleSimulation::~ParticleSimulation() {
    freeMemory();
}

void ParticleSimulation::allocateMemory() {
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_particles, particleCount * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_forceTable, MAX_PARTICLE_TYPES * MAX_PARTICLE_TYPES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radioByType, MAX_PARTICLE_TYPES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_neighborCountsA, particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighborCountsB, particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_randStates, particleCount * sizeof(curandState)));
    
    // Initialize neighbor counts to zero
    CUDA_CHECK(cudaMemset(d_neighborCountsA, 0, particleCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_neighborCountsB, 0, particleCount * sizeof(int)));
    
    // Allocate host memory
    h_rawForceTable = std::make_unique<float[]>(MAX_PARTICLE_TYPES * MAX_PARTICLE_TYPES);
    h_radioByType = std::make_unique<float[]>(MAX_PARTICLE_TYPES);
}

void ParticleSimulation::freeMemory() {
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_forceTable));
    CUDA_CHECK(cudaFree(d_radioByType));
    CUDA_CHECK(cudaFree(d_neighborCountsA));
    CUDA_CHECK(cudaFree(d_neighborCountsB));
    CUDA_CHECK(cudaFree(d_randStates));
}

void ParticleSimulation::initializeCurand() {
    int blocks = (particleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initCurandKernel<<<blocks, BLOCK_SIZE>>>(d_randStates, particleCount, time(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ParticleSimulation::initializeParticles() {
    int blocks = (particleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParticlesKernel<<<blocks, BLOCK_SIZE>>>(
        d_particles, particleCount, numParticleTypes,
        currentCanvasWidth, currentCanvasHeight, currentCanvasDepth,
        2000.0f, d_randStates  // Use 2000x2000x2000 spawn region
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ParticleSimulation::initializeParticles(float canvasWidth, float canvasHeight) {
    currentCanvasWidth = canvasWidth;
    currentCanvasHeight = canvasHeight;
    currentCanvasDepth = canvasHeight;  // Default depth to height
    int blocks = (particleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParticlesKernel<<<blocks, BLOCK_SIZE>>>(
        d_particles, particleCount, numParticleTypes,
        canvasWidth, canvasHeight, currentCanvasDepth,
        2000.0f, d_randStates  // Use 2000x2000x2000 spawn region
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ParticleSimulation::updateCanvasDimensions(float canvasWidth, float canvasHeight) {
    currentCanvasWidth = canvasWidth;
    currentCanvasHeight = canvasHeight;
    currentCanvasDepth = canvasHeight;  // Default depth to height
}

void ParticleSimulation::initializeForceTable() {
    // Generate random force table values
    for (int i = 0; i < numParticleTypes * numParticleTypes; i++) {
        h_rawForceTable[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    updateForceTable(0.28f, -0.20f, 1.0f);
}

void ParticleSimulation::updateForceTable(float forceRange, float forceBias, float forceOffset) {
    float forceTable[MAX_PARTICLE_TYPES * MAX_PARTICLE_TYPES];
    
    for (int i = 0; i < numParticleTypes * numParticleTypes; i++) {
        float transformedValue = tanh(h_rawForceTable[i] * forceOffset) * forceRange + forceBias;
        forceTable[i] = fmax(-1.0f, fmin(1.0f, transformedValue));
    }
    
    CUDA_CHECK(cudaMemcpy(d_forceTable, forceTable, 
        numParticleTypes * numParticleTypes * sizeof(float), cudaMemcpyHostToDevice));
}

void ParticleSimulation::initializeRadioByType() {
    for (int i = 0; i < numParticleTypes; i++) {
        h_radioByType[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_radioByType, h_radioByType.get(), 
        numParticleTypes * sizeof(float), cudaMemcpyHostToDevice));
}

void ParticleSimulation::simulate(const SimulationParams& params) {
    int blocks = (particleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    int* neighborCountsIn = useBufferAasInput ? d_neighborCountsA : d_neighborCountsB;
    int* neighborCountsOut = useBufferAasInput ? d_neighborCountsB : d_neighborCountsA;
    
    simulateParticlesKernel<<<blocks, BLOCK_SIZE>>>(
        d_particles, d_forceTable, d_radioByType,
        neighborCountsIn, neighborCountsOut,
        params, particleCount
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    useBufferAasInput = !useBufferAasInput;
}

void ParticleSimulation::getParticleData(std::vector<Particle>& particles) {
    particles.resize(particleCount);
    CUDA_CHECK(cudaMemcpy(particles.data(), d_particles, 
        particleCount * sizeof(Particle), cudaMemcpyDeviceToHost));
}

void ParticleSimulation::setParticleCount(int count) {
    if (count != particleCount) {
        particleCount = count;
        freeMemory();
        allocateMemory();
        initializeCurand();
        initializeParticles();
    }
}

void ParticleSimulation::setNumParticleTypes(int types) {
    numParticleTypes = types;
    initializeForceTable();
    initializeRadioByType();
    initializeParticles();
}

void ParticleSimulation::regenerateForceTable() {
    initializeForceTable();
}

void ParticleSimulation::moveUniverse(float dx, float dy) {
    SimulationParams params;
    int blocks = (particleCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    moveParticlesKernel<<<blocks, BLOCK_SIZE>>>(
        d_particles, particleCount, dx, dy, 0.0f, params.canvasWidth, params.canvasHeight, params.canvasDepth
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ParticleSimulation::rotateRadioByType() {
    // Rotate the radioByType values (shift each value to the next type)
    float temp = h_radioByType[numParticleTypes - 1];
    for (int i = numParticleTypes - 1; i > 0; i--) {
        h_radioByType[i] = h_radioByType[i - 1];
    }
    h_radioByType[0] = temp;
    
    // Upload to device
    CUDA_CHECK(cudaMemcpy(d_radioByType, h_radioByType.get(), 
        numParticleTypes * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> ParticleSimulation::getRadioByType() const {
    std::vector<float> result(numParticleTypes);
    for (int i = 0; i < numParticleTypes; i++) {
        result[i] = h_radioByType[i];
    }
    return result;
}

void ParticleSimulation::setRadioByTypeValue(int index, float value) {
    if (index >= 0 && index < numParticleTypes) {
        h_radioByType[index] = value;
        // Upload to device
        CUDA_CHECK(cudaMemcpy(d_radioByType, h_radioByType.get(), 
            numParticleTypes * sizeof(float), cudaMemcpyHostToDevice));
    }
}