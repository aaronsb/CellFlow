# ADR-001: Surface Rendering from Particle Clusters

## Status
PROPOSED

## Context

CellFlow's 3D particle simulation creates emergent patterns where particles cluster together, forming structures that visually suggest surfaces and volumes. Currently, these are rendered as individual billboard sprites with depth-based effects (size/brightness attenuation, DOF).

**The Opportunity:**
When particles cluster densely, they create the visual appearance of surfaces. We want to amplify this effect to make these emergent structures more perceivable and visually compelling.

**Current System Properties:**
- **Particles**: 3D positions, velocities, types (colors)
- **Emergent Behavior**: Particles form dynamic clusters and patterns based on force interactions
- **Temporal**: System evolves over time with complex dynamics
- **Scale**: 1,000-10,000+ particles
- **Performance**: Real-time (30-120fps) on consumer hardware
- **Multi-dimensional Data**:
  - Position (x,y,z)
  - Velocity (direction/magnitude)
  - Particle type (affects forces/color)
  - Local density (emergent from clustering)
  - Temporal coherence (frame-to-frame)

## Problem Statement

How do we render smooth, perceivable surfaces from discrete particle clusters while:
1. Maintaining real-time performance (30+ fps)
2. Preserving the dynamic, emergent nature of the simulation
3. Keeping implementation complexity reasonable
4. Avoiding loss of the "particle life" aesthetic

## Research: Rendering Approaches

### Option 1: Gaussian Splatting ⭐ RECOMMENDED

**Concept:**
Represent each particle as a 3D Gaussian with position, color, opacity, and anisotropic shape. Render by projecting Gaussians to screen space and alpha-blending.

**Relevance to Our System:**
- **Perfect Fit**: Particles already have positions and colors
- **Density Encoding**: Gaussian size/opacity can reflect local particle density
- **Real-time**: Highly optimized for GPU, achieves 60+ fps on modern hardware
- **Smooth Surfaces**: Natural blending creates continuous appearance
- **Temporal**: Gaussians can smoothly interpolate as particles move

**Technical Approach:**
1. **Per-Particle Gaussians**:
   - Position: particle.pos
   - Color: particle type color
   - Opacity: based on local density (query nearby particles)
   - Covariance: ellipsoid shape aligned with velocity direction or local cluster shape

2. **Rendering Pipeline**:
   - Sort particles back-to-front (or use order-independent transparency)
   - Project 3D Gaussians to 2D screen-space Gaussians
   - Rasterize with alpha blending
   - Can use existing CUDA infrastructure for sorting/computation

3. **Adaptive Parameters**:
   - Gaussian size ~ local particle density (more neighbors = larger splat)
   - Opacity ~ cluster coherence (tightly packed = more opaque)
   - Shape ~ velocity field (elongate along motion)

**Pros:**
- ✅ Real-time performance (optimized for GPUs)
- ✅ Natural smooth surfaces from discrete points
- ✅ Preserves particle identity (can still see individual particles if desired)
- ✅ Works with dynamic, moving particles
- ✅ Can leverage CUDA for computation
- ✅ Recent technique with active development/libraries
- ✅ Scales well with particle count

**Cons:**
- ⚠️ Requires sorting for correct blending (manageable with CUDA)
- ⚠️ Need to compute local density (neighborhood queries)
- ⚠️ Additional GPU memory for Gaussian parameters
- ⚠️ More complex than current billboard rendering

**Implementation Complexity:** Medium
**Performance Impact:** Low-Medium (60+ fps expected)

**References:**
- "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)
- CUDA-accelerated implementations available

---

### Option 2: Neural Radiance Fields (NeRF)

**Concept:**
Train a neural network to learn a continuous volumetric scene representation mapping (x,y,z, θ, φ) → (color, density).

**Relevance to Our System:**
- Interesting but **fundamentally mismatched**
- NeRF is designed for static scenes reconstructed from photos
- Our particles are dynamic and change every frame
- Would need to retrain network continuously = computationally prohibitive

**Technical Challenges:**
- Training time: 1-24 hours per scene (incompatible with real-time)
- Inference: 1-30 fps even with optimizations (too slow)
- Dynamic scenes: No good solution for frame-by-frame changes
- Overkill: We don't need view synthesis, just better rendering

**Pros:**
- ✅ Produces photorealistic results (when trained)
- ✅ Continuous representation

**Cons:**
- ❌ Not real-time for training
- ❌ Too slow for dynamic scenes
- ❌ Massive computational overhead
- ❌ Designed for static reconstruction, not particle dynamics
- ❌ Incompatible with our temporal requirements

**Verdict:** ❌ **Not Suitable**
While fascinating, NeRF is the wrong tool for this problem.

---

### Option 3: Metaballs / Marching Cubes

**Concept:**
Define implicit surface from particle field, extract triangle mesh using marching cubes, render as geometry.

**Relevance to Our System:**
- Classic approach for particle→surface
- Creates true geometric surfaces
- We mentioned this in previous work

**Technical Approach:**
1. Rasterize particles into 3D density grid
2. Apply threshold to define surface
3. Run marching cubes to extract mesh
4. Render mesh with standard graphics pipeline

**Pros:**
- ✅ Well-understood, proven technique
- ✅ Creates true surfaces
- ✅ Many optimizations available (GPU marching cubes)
- ✅ Can compute normals for lighting

**Cons:**
- ⚠️ Grid resolution vs performance tradeoff
- ⚠️ Mesh topology changes rapidly (expensive to regenerate)
- ⚠️ Can lose small-scale detail
- ⚠️ May smooth out interesting particle behaviors
- ⚠️ Higher computational cost than splatting
- ⚠️ Harder to preserve particle identity

**Implementation Complexity:** Medium-High
**Performance Impact:** Medium-High (30-60 fps depending on grid resolution)

---

### Option 4: Screen-Space Rendering Techniques

**Concept:**
Use depth buffer and screen-space convolution to smooth/thicken point rendering.

**Options:**
- **Depth-based smoothing**: Bilateral filter on depth + color
- **Screen-space thickness**: Accumulate particle contributions in screen space
- **Point-based splatting**: Render particles as disks, blend in screen space

**Pros:**
- ✅ Very fast (screen-space operations)
- ✅ Low implementation complexity
- ✅ Works with existing rendering pipeline
- ✅ Minimal memory overhead

**Cons:**
- ⚠️ Only smooths visually, doesn't create true surfaces
- ⚠️ View-dependent artifacts
- ⚠️ Less control over surface appearance
- ⚠️ Can look "fake" compared to volumetric approaches

**Implementation Complexity:** Low-Medium
**Performance Impact:** Low (90+ fps)

---

### Option 5: Enhanced Point Cloud Rendering

**Concept:**
Improve current point sprite rendering with better splatting and depth-aware sizing.

**Techniques:**
- Elliptical weighted average (EWA) splatting
- Adaptive point sizes based on local density
- High-quality filtering kernels
- Depth-aware smoothing

**Pros:**
- ✅ Evolutionary improvement over current system
- ✅ Low risk
- ✅ Good performance
- ✅ Keeps particle aesthetic

**Cons:**
- ⚠️ Won't create true surface appearance
- ⚠️ Limited compared to Gaussian splatting
- ⚠️ May not achieve desired "surface" effect

**Implementation Complexity:** Low
**Performance Impact:** Very Low (100+ fps)

---

## Decision

**RECOMMEND: Gaussian Splatting (Option 1)**

### Rationale

Gaussian splatting is the optimal choice because it:

1. **Matches Our Use Case Perfectly**:
   - Designed for real-time rendering from points
   - Handles dynamic scenes naturally
   - Creates smooth surfaces from discrete data
   - Preserves spatial detail

2. **Technical Fit**:
   - Leverages our existing CUDA infrastructure
   - Particles map directly to Gaussians
   - Multi-dimensional data (position, velocity, type) can inform Gaussian parameters
   - Temporal coherence exploitable for optimization

3. **Amplifies Emergent Behavior**:
   - Particle clusters → smooth surfaces (exactly what we want)
   - Density variations → opacity variations (reveals structure)
   - Velocity fields → elongated splats (shows motion)
   - Still can "see" individual particles when sparse

4. **Performance**:
   - Proven real-time capabilities (60-120 fps)
   - Scales with modern GPUs
   - Parallelizable with CUDA

5. **Future-Proof**:
   - Active research area with ongoing optimizations
   - Growing ecosystem of tools/libraries
   - Can start simple, add sophistication incrementally

### Why Not Others?

- **NeRF**: Wrong tool - designed for static scene reconstruction, not dynamic particles
- **Metaballs**: Viable but slower, creates discrete topology changes, harder to implement well
- **Screen-Space**: Too limited - won't achieve true surface appearance
- **Enhanced Points**: Safe but won't deliver the "wow" factor we're seeking

## Implementation Strategy

### Phase 1: Proof of Concept (Minimal Gaussian Splatting)
**Goal**: Verify the approach works with our particle system

1. **Simple Gaussians**:
   - Each particle = isotropic (spherical) Gaussian
   - Fixed size and opacity
   - Use particle type color
   - No density adaptation yet

2. **Basic Rendering**:
   - Back-to-front sorting (use existing CUDA sort)
   - Project to screen space
   - Alpha-blend with existing OpenGL pipeline
   - No advanced optimizations

**Success Criteria**: See smooth blending of particle clusters, maintain 30+ fps

### Phase 2: Adaptive Parameters
**Goal**: Make Gaussians respond to particle dynamics

1. **Local Density Computation**:
   - Use existing neighbor-finding from physics (radius-based)
   - Map neighbor count → Gaussian size/opacity
   - More neighbors = larger, more opaque Gaussians

2. **Anisotropic Shape**:
   - Elongate Gaussians along velocity direction
   - Or align with local particle distribution (PCA)
   - Creates more organic surface appearance

**Success Criteria**: Surfaces appear where particles cluster, transparent where sparse

### Phase 3: Optimizations
**Goal**: Achieve 60+ fps at high particle counts

1. **Efficient Sorting**: Radix sort on CUDA
2. **Culling**: Frustum and occlusion culling of Gaussians
3. **LOD**: Reduce splat quality for distant particles
4. **Caching**: Reuse Gaussian parameters when particles haven't moved much

### Phase 4: Polish
1. **Lighting**: Simple diffuse shading using surface normals from density gradient
2. **User Controls**: Sliders for splat size scale, opacity scale, anisotropy amount
3. **Hybrid Mode**: Toggle between pure points, pure splats, or blend
4. **Temporal Anti-Aliasing**: Smooth frame-to-frame for even better quality

## Alternative Paths

If Gaussian splatting proves problematic:
1. **Fallback**: Enhanced point cloud rendering (Option 5)
2. **Hybrid**: Combine screen-space techniques (Option 4) with improved points
3. **Different**: Explore metaballs (Option 3) with aggressive optimizations

## Questions to Resolve

1. **Sorting Cost**: Is back-to-front sorting acceptable for 10,000 particles every frame?
   - *Initial Answer*: Yes, CUDA radix sort handles this easily

2. **Memory**: Can we afford per-particle Gaussian parameters (6-9 floats)?
   - *Initial Answer*: Yes, trivial compared to simulation data

3. **Transparency**: Order-independent transparency (OIT) vs sorted alpha?
   - *Initial Answer*: Start with sorted, evaluate OIT if artifacts appear

4. **Integration**: How does this interact with existing depth effects (DOF, etc.)?
   - *To investigate*: Render Gaussians into same depth buffer, effects should still work

## Success Metrics

- **Visual**: Emergent particle surfaces clearly visible and compelling
- **Performance**: Maintain 60+ fps at 4000 particles, 30+ fps at 10000
- **Aesthetic**: Preserves "particle life" feel while adding surface perception
- **Control**: User can adjust surface prominence via sliders
- **Compatibility**: Works with existing camera, effects, presets

## References

### Gaussian Splatting
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (2023)
- [Kerbl et al., SIGGRAPH 2023](https://dl.acm.org/doi/10.1145/3592433)
- CUDA implementation: [github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### Related Techniques
- Surface Splatting (Zwicker et al., 2001)
- EWA Volume Splatting (Zwicker et al., 2002)
- High-Quality Point-Based Rendering (Botsch et al., 2005)

### NeRF (for reference, but not recommended)
- [NeRF: Representing Scenes as Neural Radiance Fields](https://www.matthewtancik.com/nerf) (2020)
- Various real-time variants (Instant-NGP, Plenoxels) - still too slow for our use case

## Next Steps

1. **Research existing Gaussian splatting CUDA code** - understand implementation details
2. **Design minimal API** - how does splatting integrate with CellFlowWidget?
3. **Create ADR-002** - detailed technical design for Phase 1 implementation
4. **Prototype** - build Phase 1 proof of concept on feature branch
5. **Evaluate** - measure performance, visual quality, decide go/no-go

---

**Author**: Claude + Aaron
**Date**: 2025-01-23
**Decision**: Pending user approval
