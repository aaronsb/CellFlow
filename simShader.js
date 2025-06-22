export const simShader =/* wgsl */ `
struct Particle {
  pos: vec2f,
  vel: vec2f,
  acc: vec2f,
  ptype: u32,
  pad: u32 // Este padding es importante, el layout de memoria debe ser consistente
};

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
  maxExpectedNeighbors: f32, // ¡Nuevo parámetro para la normalización!

};

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

  var neighbors_count = 0u; // Inicializa el contador de vecinos para la partícula actual 'me'

  for (var j = 0u; j < arrayLength(&particles); j++) {
    if (i == j) { continue; }
    let other = particles[j];
    var dir = other.pos - me.pos;

    dir.x = dir.x - floor(dir.x / simParams.canvasWidth + 0.5) * simParams.canvasWidth;
    dir.y = dir.y - floor(dir.y / simParams.canvasHeight + 0.5) * simParams.canvasHeight;

    let dist = length(dir);

    
    if (dist == 0.0 || dist > effectiveRadius) { continue; }


    neighbors_count++;

    let r = dist / effectiveRadius;//
    let a = forceTable[me.ptype * u32(simParams.numParticleTypes) + other.ptype];
    

    let rep_decay = r * simParams.k; 
    let repulsion_ = simParams.repulsion * (1.0 / (1.0 + rep_decay * rep_decay));

    let attraction_ = simParams.attraction * r*r;
   // let attraction_ = simParams.attraction * (1.0 - r);
    let f = a * (repulsion_ - attraction_) * simParams.forceMultiplier;
    force += normalize(dir) * f;
  }


  // Use previous frame's neighbor count for adaptive_multiplier
  let prev_neighbors = previousFrameNeighborCounts[i];
  let normalized_density = f32(prev_neighbors) / simParams.maxExpectedNeighbors;
  let clamped_normalized_density = clamp(normalized_density, 0.0, 1.0);

  // Calcula min_force_mult y max_force_mult basándose en simParams.forceMultiplier
  // Para forceMultiplier = 0: min_force_mult = 1.0, max_force_mult = 1.0
  // Para forceMultiplier = 1: min_force_mult = 0.0, max_force_mult = 2.0
  let min_force_mult = mix(1.0, 0.01, simParams.balance);
  let max_force_mult = mix(1.0, 4.0, simParams.balance);

  let adaptive_multiplier = mix(max_force_mult, min_force_mult, clamped_normalized_density);
  var vel = me.vel * simParams.friction;
  vel += force * simParams.delta_t * adaptive_multiplier;

  // At the end of the main, store this frame's neighbor count for use in the next frame
  previousFrameNeighborCounts[i] = neighbors_count;


  // **Estrategia 2: Aumentar la repulsión en clústeres densos**
  // Cuando más denso, más repulsión para "descomprimir".
  // let repulsion_boost_factor = mix(1.0, 2.0, clamped_normalized_density); // 1x para poco denso, 2x para muy denso
    //let effective_repulsion = simParams.repulsion * repulsion_boost_factor;
    //let repulsion_ = effective_repulsion * exp(-simParams.k * r * r); // Aplica aquí la repulsión modificada


  // **Estrategia 3: Disminuir la atracción en clústeres densos**
  // Cuando más denso, menos atracción para evitar que se compacten demasiado.
  // let attraction_reduction_factor = mix(1.0, 0.5, clamped_normalized_density); // 1x para poco denso, 0.5x para muy denso
  // let effective_attraction = simParams.attraction * attraction_reduction_factor;
  // let attraction_ = effective_attraction / (r * r + simParams.epsilon); // Aplica aquí la atracción modificada


  // **Estrategia 4: Aumentar la atracción en clústeres poco densos**
  // Para que las partículas sueltas se agrupen más.
  // let attraction_boost_factor = mix(2.0, 1.0, clamped_normalized_density); // 2x para poco denso, 1x para muy denso
  // let effective_attraction = simParams.attraction * attraction_boost_factor;
  // let attraction_ = effective_attraction / (r * r + simParams.epsilon); // Aplica aquí la atracción modificada


  // --- Aplicar la fuerza y actualizar la posición/velocidad ---
  // Mantén la lógica de aplicación de fuerza y movimiento aquí.
  // Si usas una estrategia que modula 'f' directamente, asegúrate de que la 'f' final
  // que sumas a 'force' refleja esos cambios.

  //var vel = me.vel * simParams.friction;
  // Usa simParams.forceMultiplier si no estás usando un adaptive_multiplier.
  // vel += force * simParams.delta_t * simParams.forceMultiplier;
  var pos = me.pos + vel * simParams.delta_t;


  let temp_pos_x = pos.x + simParams.canvasWidth;
  let wrapped_x = temp_pos_x - simParams.canvasWidth * floor(temp_pos_x / simParams.canvasWidth);
  let temp_pos_y = pos.y + simParams.canvasHeight;
  let wrapped_y = temp_pos_y - simParams.canvasHeight * floor(temp_pos_y / simParams.canvasHeight);
  pos = vec2f(wrapped_x, wrapped_y);

  particles[i].pos = pos;
  particles[i].vel = vel;
}
`;