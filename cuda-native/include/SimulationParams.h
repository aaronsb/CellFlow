#ifndef SIMULATION_PARAMS_H
#define SIMULATION_PARAMS_H

#include <cuda_runtime.h>

struct Particle {
    float3 pos;
    float3 vel;
    float3 acc;
    unsigned int ptype;
    float pad;  // Changed from uint to float for better alignment
};

struct SimulationParams {
    float radius = 42.07f;
    float delta_t = 0.18f;
    float friction = 0.51f;
    float repulsion = 64.83f;
    float attraction = 3.06f;
    float k = 29.45f;
    float balance = 0.79f;
    float canvasWidth = 8000.0f;   // Large universe independent of viewport
    float canvasHeight = 8000.0f;
    float canvasDepth = 8000.0f;
    float spawnRegionSize = 2000.0f;  // Spawn particles in smaller central region
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
    
    // Rendering parameters
    float pointSize = 10.0f;  // Larger default for bigger universe

    // Depth effect parameters
    float depthFadeStart = 10000.0f;  // Effectively disabled (very far)
    float depthFadeEnd = 15000.0f;    // Effectively disabled (very far)
    float sizeAttenuationFactor = 1000.0f;
    float brightnessMin = 0.4f;

    // Depth-of-field parameters
    float focusDistance = 3000.0f;  // Distance to focal plane
    float apertureSize = 0.0f;       // 0 = everything in focus, higher = more blur

    // Gaussian splatting parameters
    float gaussianSizeScale = 2.0f;      // Size multiplier for Gaussian splats
    float gaussianOpacityScale = 1.0f;   // Opacity multiplier
    float gaussianDensityInfluence = 0.5f; // How much density affects size (0-1)

    // Effect enable/disable flags
    bool enableDepthFade = false;
    bool enableSizeAttenuation = true;
    bool enableBrightnessAttenuation = true;
    bool enableDOF = false;
};

// Color definitions (RGB values 0-1)
struct ParticleColor {
    float r, g, b;
};

constexpr int MAX_PARTICLE_TYPES = 10;
constexpr int BLOCK_SIZE = 256;  // CUDA block size

#endif // SIMULATION_PARAMS_H