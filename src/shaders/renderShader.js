export function getRenderShaderCode(numParticleTypes, canvasWidth, canvasHeight, particleColors) {
  // Build color array as a constant in the shader for better performance
  let colorArray = 'const particleColors = array<vec3f, ' + numParticleTypes + '>(';
  for (let i = 0; i < numParticleTypes; i++) {
    colorArray += particleColors[i];
    if (i < numParticleTypes - 1) colorArray += ', ';
  }
  colorArray += ');';

  return `
    struct Particle {
      pos: vec2f,
      vel: vec2f,
      acc: vec2f,
      ptype: u32,
      pad: u32
    };

    @group(0) @binding(0) var<storage, read> particles: array<Particle>;

    ${colorArray}

    struct VSOut {
      @builtin(position) pos: vec4f,
      @location(0) color: vec3f
    };

    @vertex
    fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
      // Cada partícula genera 4 vértices
      let particleIndex = i / 4u;
      let subIndex = i % 4u;
      let p = particles[particleIndex];
      
      // Use array lookup instead of if statements
      var c: vec3f;
      if (p.ptype < ${numParticleTypes}u) {
        c = particleColors[p.ptype];
      } else {
        c = vec3f(1.0, 1.0, 0.0); // Amarillo para tipos fuera de rango
      }

      // Definir offsets para los 4 píxeles (en píxeles)
      var offset: vec2f;
      if (subIndex == 0u) { offset = vec2f(0.0, 0.0); }      // Píxel original
      else if (subIndex == 1u) { offset = vec2f(1.0, 0.0); } // +1x
      else if (subIndex == 2u) { offset = vec2f(0.0, 1.0); } // +1y
      else { offset = vec2f(1.0, 1.0); }                     // +1x+1y

      // Calcular posición con offset
      let pixelPos = p.pos + offset;

      // Validar límites para evitar dibujar fuera del canvas
      var out: VSOut;
      if (pixelPos.x < 0.0 || pixelPos.x >= ${canvasWidth}.0 || pixelPos.y < 0.0 || pixelPos.y >= ${canvasHeight}.0) {
        out.pos = vec4f(0.0, 0.0, -1.0, 1.0); // Fuera de pantalla (no visible)
      } else {
        out.pos = vec4f((pixelPos.x / ${canvasWidth}.0) * 2.0 - 1.0,
                        (pixelPos.y / ${canvasHeight}.0) * -2.0 + 1.0,
                        0.0, 1.0);
      }
      out.color = c;

      return out;
    }

    @fragment
    fn fs_main(in: VSOut) -> @location(0) vec4f {
      return vec4f(in.color, 0.8); // Alpha al 50%
    }
  `;
}