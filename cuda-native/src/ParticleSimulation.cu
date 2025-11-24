#include "ParticleSimulation.cuh"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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

// Proximity graph generation kernel - writes line vertices directly to OpenGL VBO
__global__ void generateProximityGraphKernel(
    const Particle* particles,
    int particleCount,
    float proximityDistanceSq,
    int maxConnectionsPerParticle,
    const ParticleColor* particleColors,
    int numParticleTypes,
    float* lineVertices,
    int* vertexCount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    const Particle& p1 = particles[idx];

    // Temp storage for nearby particles (distance, index)
    struct NearbyParticle {
        float distSq;
        int index;
    };

    // Use shared memory or local array for sorting
    NearbyParticle nearby[32];  // Max 32 to avoid too much local memory
    int nearbyCount = 0;
    int maxConn = min(maxConnectionsPerParticle, 32);

    // Find nearby particles of same type
    for (int j = idx + 1; j < particleCount && nearbyCount < maxConn * 2; j++) {
        const Particle& p2 = particles[j];

        // Only connect particles of same type
        if (p1.ptype != p2.ptype) continue;

        // Calculate distance squared
        float dx = p2.pos.x - p1.pos.x;
        float dy = p2.pos.y - p1.pos.y;
        float dz = p2.pos.z - p1.pos.z;
        float distSq = dx*dx + dy*dy + dz*dz;

        if (distSq < proximityDistanceSq) {
            nearby[nearbyCount].distSq = distSq;
            nearby[nearbyCount].index = j;
            nearbyCount++;
        }
    }

    // Simple insertion sort to get closest particles
    for (int i = 1; i < nearbyCount; i++) {
        NearbyParticle key = nearby[i];
        int j = i - 1;
        while (j >= 0 && nearby[j].distSq > key.distSq) {
            nearby[j + 1] = nearby[j];
            j--;
        }
        nearby[j + 1] = key;
    }

    // Limit to maxConnectionsPerParticle
    int connectionsToWrite = min(nearbyCount, maxConn);

    if (connectionsToWrite > 0) {
        // Atomically allocate space in output buffer
        int writeOffset = atomicAdd(vertexCount, connectionsToWrite * 2);  // 2 vertices per line

        // Get particle color
        const ParticleColor& color = particleColors[p1.ptype % numParticleTypes];

        // Write line vertices (each line needs 2 vertices with pos+color = 6 floats each)
        for (int i = 0; i < connectionsToWrite; i++) {
            const Particle& p2 = particles[nearby[i].index];
            int baseIdx = (writeOffset + i * 2) * 6;  // 6 floats per vertex (pos + color)

            // Vertex 1 (particle p1)
            lineVertices[baseIdx + 0] = p1.pos.x;
            lineVertices[baseIdx + 1] = p1.pos.y;
            lineVertices[baseIdx + 2] = p1.pos.z;
            lineVertices[baseIdx + 3] = color.r;
            lineVertices[baseIdx + 4] = color.g;
            lineVertices[baseIdx + 5] = color.b;

            // Vertex 2 (particle p2)
            lineVertices[baseIdx + 6] = p2.pos.x;
            lineVertices[baseIdx + 7] = p2.pos.y;
            lineVertices[baseIdx + 8] = p2.pos.z;
            lineVertices[baseIdx + 9] = color.r;
            lineVertices[baseIdx + 10] = color.g;
            lineVertices[baseIdx + 11] = color.b;
        }
    }
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

// Proximity graph generation with CUDA-OpenGL interop
void ParticleSimulation::generateProximityGraph(
    unsigned int openglVBO,
    int& outVertexCount,
    float proximityDistance,
    int maxConnectionsPerParticle,
    const std::vector<ParticleColor>& particleColors
) {
    // Register OpenGL VBO with CUDA
    cudaGraphicsResource* cudaVBOResource;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, openglVBO,
        cudaGraphicsMapFlagsWriteDiscard));

    // Map the buffer for CUDA writing
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBOResource, 0));

    float* d_lineVertices;
    size_t numBytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_lineVertices,
        &numBytes, cudaVBOResource));

    // Allocate device memory for vertex count
    int* d_vertexCount;
    CUDA_CHECK(cudaMalloc(&d_vertexCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_vertexCount, 0, sizeof(int)));

    // Copy particle colors to device
    ParticleColor* d_particleColors;
    CUDA_CHECK(cudaMalloc(&d_particleColors, particleColors.size() * sizeof(ParticleColor)));
    CUDA_CHECK(cudaMemcpy(d_particleColors, particleColors.data(),
        particleColors.size() * sizeof(ParticleColor), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    float distSq = proximityDistance * proximityDistance;
    generateProximityGraphKernel<<<gridSize, blockSize>>>(
        d_particles,
        particleCount,
        distSq,
        maxConnectionsPerParticle,
        d_particleColors,
        numParticleTypes,
        d_lineVertices,
        d_vertexCount
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get vertex count
    int h_vertexCount;
    CUDA_CHECK(cudaMemcpy(&h_vertexCount, d_vertexCount, sizeof(int), cudaMemcpyDeviceToHost));
    outVertexCount = h_vertexCount;

    // Cleanup
    CUDA_CHECK(cudaFree(d_vertexCount));
    CUDA_CHECK(cudaFree(d_particleColors));

    // Unmap and unregister
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBOResource, 0));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVBOResource));
}