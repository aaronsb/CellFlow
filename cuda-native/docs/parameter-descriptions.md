# CellFlow Parameter Descriptions

## Particle Parameters

### Count
**Range**: 500-10000  
**Default**: 4000  
**Description**: Total number of particles in the simulation. Higher counts create denser patterns but require more GPU resources.

### Types
**Range**: 2-6  
**Default**: 6  
**Description**: Number of distinct particle types. Each type has unique colors and force relationships with other types.

## Physics Parameters

### Radius
**Range**: 10-100  
**Default**: 50  
**Description**: Interaction radius for particles. Particles only affect each other within this distance. Larger values create longer-range interactions.

### Time
**Range**: 0.01-0.5  
**Default**: 0.22  
**Description**: Simulation time step (delta time). Smaller values are more accurate but slower. Larger values are faster but may become unstable.

### Friction
**Range**: 0.01-0.99  
**Default**: 0.71  
**Description**: Velocity damping factor. Higher values create more viscous movement. Lower values allow particles to maintain momentum longer.

### Repulsion
**Range**: 0.1-100  
**Default**: 50  
**Description**: Base repulsion force between particles. Prevents particles from overlapping and creates personal space.

### Attraction
**Range**: 0.01-1.0  
**Default**: 0.62  
**Description**: Base attraction force modifier. Scales the attraction values in the force matrix. Higher values create stronger clustering.

### K
**Range**: 0.1-50  
**Default**: 16.57  
**Description**: Force scaling constant. Acts as a master force multiplier. Higher values create more energetic systems.

## Advanced Parameters

### F_Range
**Range**: 0.01-1.0  
**Default**: 0.28  
**Description**: Force falloff range. Controls how quickly forces decrease with distance. Lower values create sharper force dropoff.

### F_Bias
**Range**: -1.0-1.0  
**Default**: -0.20  
**Description**: Force curve bias. Negative values favor repulsion at medium distances, positive values favor attraction.

### Ratio
**Range**: -2.0-2.0  
**Default**: 0.0  
**Description**: Base force ratio adjustment. Modifies the balance between attraction and repulsion forces.

### LFOA (LFO Amplitude)
**Range**: -1.0-1.0  
**Default**: 0.0  
**Description**: Low Frequency Oscillator amplitude. Creates pulsing force variations over time. Set to 0 to disable.

### LFOS (LFO Speed)
**Range**: 0.1-10.0  
**Default**: 0.1  
**Description**: Low Frequency Oscillator frequency in Hz. Controls how fast forces pulse when LFO is active.

### F_Offset
**Range**: -1.0-1.0  
**Default**: 0.0  
**Description**: Global force offset. Adds a constant to all forces. Positive values increase overall attraction.

## Rendering Parameters

### Point Size
**Range**: 1.0-10.0  
**Default**: 4.0  
**Description**: Size of particle sprites in pixels. Larger sizes create more coverage but may reduce performance.

### Effect
**Options**: None, Blur, Glow  
**Default**: None  
**Description**: Post-processing visual effect. Blur creates soft edges, Glow adds bright halos to particles.

## Adaptive Parameters

### F_Mult (Force Multiplier)
**Range**: 0.0-5.0  
**Default**: 2.33  
**Description**: Adaptive force multiplier based on local density. Adjusts forces based on neighbor count.

### Balance
**Range**: 0.01-1.5  
**Default**: 0.79  
**Description**: Density adaptation balance. Controls how much neighbor density affects force strength.

### F_Offset (Adaptive)
**Range**: -1.0-1.0  
**Default**: 0.0  
**Description**: Adaptive force offset. Additional force adjustment based on local particle density.

## Controls

### REGEN
Regenerates the force matrix with new random values. Creates entirely new particle behavior patterns.

### RESET
Resets all particle positions to random locations within the current window. Keeps all other settings.

### reX
Rotates the radius modifiers for each particle type. Changes interaction ranges between types.

### Save
Saves current settings and force matrix to a preset file.

### Load
Loads previously saved preset including all parameters and force relationships.

### De-harmonize
Gradually shifts particle colors away from their default values. Creates more color variety.

### Harmonize
Resets particle colors to their default harmonious palette based on golden angle distribution.