#ifndef SIMULATION_PARAMS_H
#define SIMULATION_PARAMS_H

#include <cuda_runtime.h>

struct Particle {
    float2 pos;
    float2 vel;
    float2 acc;
    unsigned int ptype;
    unsigned int pad;
};

struct SimulationParams {
    float radius = 50.0f;
    float delta_t = 0.22f;
    float friction = 0.71f;
    float repulsion = 50.0f;
    float attraction = 0.62f;
    float k = 16.57f;
    float balance = 0.79f;
    float canvasWidth = 1920.0f;
    float canvasHeight = 1080.0f;
    int numParticleTypes = 6;
    float ratioWithLFO = 0.0f;
    float forceMultiplier = 2.33f;
    int maxExpectedNeighbors = 400;
    
    // Additional parameters
    float forceRange = 0.28f;
    float forceBias = -0.20f;
    float ratio = 0.0f;
    float lfoA = 0.0f;
    float lfoS = 0.1f;
    float forceOffset = 1.0f;
};

// Color definitions (RGB values 0-1)
struct ParticleColor {
    float r, g, b;
};

constexpr int MAX_PARTICLE_TYPES = 10;
constexpr int BLOCK_SIZE = 256;  // CUDA block size

#endif // SIMULATION_PARAMS_H