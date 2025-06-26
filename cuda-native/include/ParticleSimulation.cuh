#ifndef PARTICLE_SIMULATION_CUH
#define PARTICLE_SIMULATION_CUH

#include "SimulationParams.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <vector>

class ParticleSimulation {
public:
    ParticleSimulation(int particleCount);
    ~ParticleSimulation();
    
    // Initialize particles with random positions and types
    void initializeParticles();
    
    // Initialize force table with random values
    void initializeForceTable();
    
    // Update force table with new parameters
    void updateForceTable(float forceRange, float forceBias, float forceOffset);
    
    // Initialize radio by type array
    void initializeRadioByType();
    
    // Run one simulation step
    void simulate(const SimulationParams& params);
    
    // Get particle data for rendering
    void getParticleData(std::vector<Particle>& particles);
    
    // Get/Set methods
    void setParticleCount(int count);
    int getParticleCount() const { return particleCount; }
    
    void setNumParticleTypes(int types);
    int getNumParticleTypes() const { return numParticleTypes; }
    
    // Force table management
    void regenerateForceTable();
    float* getRawForceTableValues() { return h_rawForceTable.get(); }
    
    // Move universe (for arrow key movement)
    void moveUniverse(float dx, float dy);
    
    // Rotate radius modifiers
    void rotateRadioByType();
    
    // Get/Set radioByType values
    std::vector<float> getRadioByType() const;
    void setRadioByTypeValue(int index, float value);
    
private:
    int particleCount;
    int numParticleTypes;
    
    // Device memory
    Particle* d_particles;
    float* d_forceTable;
    float* d_radioByType;
    int* d_neighborCountsA;
    int* d_neighborCountsB;
    bool useBufferAasInput;
    
    // Host memory
    std::unique_ptr<float[]> h_rawForceTable;
    std::unique_ptr<float[]> h_radioByType;
    
    // CURAND state for random number generation
    curandState* d_randStates;
    
    // Helper methods
    void allocateMemory();
    void freeMemory();
    void initializeCurand();
};

// CUDA kernel declarations
__global__ void initializeParticlesKernel(
    Particle* particles, 
    int particleCount, 
    int numTypes,
    float canvasWidth, 
    float canvasHeight,
    curandState* states
);

__global__ void simulateParticlesKernel(
    Particle* particles,
    const float* forceTable,
    const float* radioByType,
    int* neighborCountsIn,
    int* neighborCountsOut,
    SimulationParams params,
    int particleCount
);

__global__ void moveParticlesKernel(
    Particle* particles,
    int particleCount,
    float dx,
    float dy,
    float canvasWidth,
    float canvasHeight
);

#endif // PARTICLE_SIMULATION_CUH