// Adaptación WGSL del shader de glow basado en el original GLSL
// Exporta como string, igual que los otros shaders

export const glowShader = `
struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var<uniform> texOffset: vec2f;
@group(0) @binding(3) var<uniform> resolution: vec2f;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let blurRadius: f32 = 15.0;
  let glowIntensity: f32 = 2.5;
  var finalColor: vec4f = vec4f(0.0);

  // Box blur/bloom
  var numSamples: f32 = 0.0;
  for (var x: f32 = -blurRadius; x <= blurRadius; x = x + 1.0) {
    for (var y: f32 = -blurRadius; y <= blurRadius; y = y + 1.0) {
      let sampleUV = uv + vec2f(x * texOffset.x, y * texOffset.y);
      let sampleColor = textureSample(inputTexture, inputSampler, sampleUV);
      let brightness = max(max(sampleColor.r, sampleColor.g), sampleColor.b);
      let threshold = 0.5;
      let bloomFactor = select(0.0, smoothstep(threshold, 1.0, brightness), brightness > threshold);
      let bloomColor = sampleColor * bloomFactor;
      finalColor = finalColor + bloomColor;
      numSamples = numSamples + 1.0;
    }
  }
  finalColor = finalColor / numSamples;

  let origColor = textureSample(inputTexture, inputSampler, uv);
  var outColor = origColor + finalColor * glowIntensity;
  outColor.a = 1.0;
  return outColor;
}
`;
