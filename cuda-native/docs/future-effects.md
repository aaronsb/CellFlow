# Future Visual Effects for CellFlow

## Planned Effects

### 1. Voronoi with Wave Interference
- Calculate distance to nearest particle of each type
- Create animated wave patterns emanating from each particle
- Generate interference patterns where waves from different types meet
- Mix colors at boundaries based on wave amplitude
- **Use Case**: Creates dynamic territorial visualization with rippling boundaries

### 2. Force Field Visualization
- Render actual force vectors as flowing lines or arrows
- Color intensity based on force strength (red for repulsion, blue for attraction)
- Show force gradients as background fields
- Animate flow direction based on force vectors
- **Use Case**: Makes invisible forces visible, helps understand particle behavior

### 3. Flow Field Streamlines
- Particles emit directional flow based on their velocity
- Create screen-space velocity field texture
- Render streamlines or particle trails following the flow
- Different particle types have different flow "flavors" or colors
- **Use Case**: Shows movement patterns and emergent flows in the system

### 4. Dynamic Web/Thread System
- Create persistent connections when particles get close
- Threads stretch/contract based on distance
- Break when particles move too far apart
- Show energy or information flow along threads
- Different thread properties per particle type pair
- **Use Case**: Visualizes relationships and communication between particles

### 5. Reaction-Diffusion Overlay
- Use particle positions as sources in Gray-Scott or similar RD system
- Run reaction-diffusion equations in fragment shader
- Each particle type emits different "chemicals"
- Creates patterns: coral growth, leopard spots, maze structures
- **Use Case**: Generates complex organic patterns from simple rules

### 6. Territory Tension Rendering
- Voronoi cells that deform based on force pressure
- Cell boundaries shift and ripple where opposing forces meet
- Boundaries glow or pulse with force intensity
- Smooth transitions in contested areas
- **Use Case**: Shows competition and cooperation between particle types

### 7. Phase-Space Visualization
- Render predicted particle paths based on current forces
- Show "potential wells" where particles are attracted
- Visualize "repulsion zones" as exclusion fields
- Ghost trails showing where particles have been
- **Use Case**: Reveals the underlying physics landscape

### 8. Force-Driven Connectivity
- Only connect particles that are attracting each other
- Line thickness/brightness based on force strength
- Animated energy pulses along attraction lines
- Connections break/form as forces change with LFO
- **Use Case**: Shows active relationships in the system

### 9. Interference Pattern Field
- Each particle emits concentric "waves" in screen space
- Overlapping waves create interference (constructive/destructive)
- Use phase differences for moving patterns
- Creates moiré-like effects
- **Use Case**: Beautiful abstract patterns that show particle influence

### 10. Delaunay Network
- Compute Delaunay triangulation of particle positions
- Render as animated mesh with pulsing edges
- Edge properties based on particle type connections
- Lightning or energy effects along edges
- **Use Case**: Creates dynamic neural network appearance

## Implementation Priority
1. Voronoi with Wave Interference (extends current Voronoi concept)
2. Force Field Visualization (directly shows simulation mechanics)
3. Dynamic Web/Thread System (highly requested feature)
4. Flow Field Streamlines (natural extension of particle movement)
5. Others as time permits

## Technical Considerations
- Most effects can use existing particle texture infrastructure
- Should maintain 60 FPS with reasonable particle counts
- Effects should be combinable where it makes sense
- Consider adding effect-specific parameters to UI