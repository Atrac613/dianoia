// Shared by the compute shaders of lab_vulkan.c, the Vulkan counterparts of
// the Metal kernels in lab_metal.m (same math, same layouts).
//
// Accumulators are `precise`: SPIR-V lets a driver fuse multiply-adds as it
// likes, differently in different shaders or unrolled iterations, which
// would make a chunk of tokens round differently from single tokens. Metal
// keeps them apart, and so must this for chunking to change no bit.
//
// Buffers are passed as device addresses in push constants
// (VK_KHR_buffer_device_address), so no descriptor sets are needed: each
// dispatch is a pipeline, a push-constant block and a group count.
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Vec4s { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Uints { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Uvec4s { uvec4 v[]; };

// Rows by tokens that one subgroup of a GEMM computes (as in lab_metal.m).
const uint GEMM_ROWS = 4;
const uint GEMM_TOKENS = 8;

// The half at index i of an array of halves read as 32-bit words.
float half_at(Uints words, uint i) {
  const uint word = words.v[i >> 1];
  return unpackHalf2x16((i & 1u) != 0u ? word >> 16 : word & 0xffffu).x;
}

// Q8: four int8 weights per word, widened with their signs.
vec4 q8_word(uint w) {
  return vec4((ivec4(w) << ivec4(24, 16, 8, 0)) >> 24);
}

// Q4 (lab_quantize_q4's layout): weight k of a block is the low nibble of
// byte k and weight k + 16 the high one, worth d * (q - 8). Word k of the
// block's uvec4 holds weights 4k..4k+3 low and 16+4k..19+4k high, so it
// meets inputs x[k] and x[4 + k].
float q4_block_dot(uvec4 w, vec4 x[8]) {
  precise float sum = 0.0;
  for (uint k = 0; k < 4; k++) {
    const uint v = w[k];
    const vec4 lo = vec4(v & 15u, (v >> 8) & 15u, (v >> 16) & 15u, (v >> 24) & 15u) - 8.0;
    const vec4 hi = vec4((v >> 4) & 15u, (v >> 12) & 15u, (v >> 20) & 15u, v >> 28) - 8.0;
    sum += dot(lo, x[k]) + dot(hi, x[4 + k]);
  }
  return sum;
}
