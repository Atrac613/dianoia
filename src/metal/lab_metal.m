// Stage 5b: Metal (GPU) kernels for Dart, called through dart:ffi.
//
// Metal is an Objective-C framework, so this file is the thin layer that
// gives Dart plain C functions to call. Buffers use shared storage: on
// Apple Silicon the CPU and GPU see the same memory, so Dart writes weights
// and inputs through `lab_gpu_contents` and the GPU reads them without a
// copy. Objects are referred to by integer handles.
//
// Two kinds of entry points:
//
// * Measurements (`lab_metal_gemv_q8`, `_batch`, `_touch`): each call is a
//   whole command buffer, committed and waited for.
// * The forward pass: Dart opens a command buffer (`lab_gpu_begin`),
//   encodes one kernel per call (`lab_gpu_enc_*`) and commits it
//   (`lab_gpu_commit`). Dart decides the graph; this layer only records
//   it. The encoder is serial, so each dispatch sees the previous one's
//   writes.
//
// The shader source is compiled at start-up (newLibraryWithSource), so no
// offline Metal compiler is needed to build this library.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

static NSString* const kSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_GROUP = 8;  // one SIMD group (32 threads) per row

// y = W x for Q8 weights: per row, blocks of 32 int8 values share a float
// scale (the layout of lab_quantize_q8). Inputs stay float. Lane l of a
// row's SIMD group handles blocks l, l + 32, ...; consecutive lanes read
// consecutive 32-byte blocks, so a SIMD group reads 1 KB contiguously.
kernel void gemv_q8(device const char* qs [[buffer(0)]],
                    device const float* scales [[buffer(1)]],
                    device const float* x [[buffer(2)]],
                    device float* y [[buffer(3)]],
                    constant uint& rows [[buffer(4)]],
                    constant uint& cols [[buffer(5)]],
                    uint group [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group * ROWS_PER_GROUP + simd;
  if (row >= rows) return;
  const uint blocks = cols / 32;
  device const char4* q4 = (device const char4*)(qs + (ulong)row * cols);
  device const float4* x4 = (device const float4*)x;
  device const float* s = scales + (ulong)row * blocks;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32) {
    float sum = 0.0f;
    for (uint k = 0; k < 8; k++) {
      sum += dot(float4(q4[b * 8 + k]), x4[b * 8 + k]);
    }
    acc += sum * s[b];
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

constant uint GEMM_TOKENS = 8;
constant uint GEMM_ROWS = 4;

struct GemmParams {
  uint rows;
  uint cols;
  uint n;
};

// Y = X W^T for n tokens (X [n, cols], Y [n, rows]), Q8 weights as in
// gemv_q8. A SIMD group computes GEMM_ROWS rows for up to GEMM_TOKENS
// tokens: each weight block is read once per GEMM_TOKENS tokens, and each
// input block once per GEMM_ROWS rows. Every (row, token) is computed with
// gemv_q8's arithmetic in the same order, so results match it bit for bit.
kernel void gemm_q8(device const char* qs [[buffer(0)]],
                    device const float* scales [[buffer(1)]],
                    device const float* x [[buffer(2)]],
                    device float* y [[buffer(3)]],
                    constant GemmParams& p [[buffer(4)]],
                    uint2 group [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (group.x * ROWS_PER_GROUP + simd) * GEMM_ROWS;
  if (row0 >= p.rows) return;
  const uint rn = min(GEMM_ROWS, p.rows - row0);
  const uint t0 = group.y * GEMM_TOKENS;
  const uint tn = min(GEMM_TOKENS, p.n - t0);
  const uint blocks = p.cols / 32;
  const uint cols4 = p.cols / 4;
  device const char4* q4 = (device const char4*)(qs + (ulong)row0 * p.cols);
  device const float4* x4 = (device const float4*)x + (ulong)t0 * cols4;
  device const float* s = scales + (ulong)row0 * blocks;
  float acc[GEMM_ROWS][GEMM_TOKENS];
  for (uint r = 0; r < GEMM_ROWS; r++) {
    for (uint t = 0; t < GEMM_TOKENS; t++) acc[r][t] = 0.0f;
  }
  for (uint b = lane; b < blocks; b += 32) {
    char4 w[GEMM_ROWS][8];
    float scale[GEMM_ROWS];
    for (uint r = 0; r < GEMM_ROWS; r++) {
      if (r >= rn) break;
      for (uint k = 0; k < 8; k++) w[r][k] = q4[(ulong)r * cols4 + b * 8 + k];
      scale[r] = s[(ulong)r * blocks + b];
    }
    for (uint t = 0; t < GEMM_TOKENS; t++) {
      if (t >= tn) break;
      device const float4* xt = x4 + (ulong)t * cols4 + b * 8;
      float4 xv[8];
      for (uint k = 0; k < 8; k++) xv[k] = xt[k];
      for (uint r = 0; r < GEMM_ROWS; r++) {
        if (r >= rn) break;
        float sum = 0.0f;
        for (uint k = 0; k < 8; k++) sum += dot(float4(w[r][k]), xv[k]);
        acc[r][t] += sum * scale[r];
      }
    }
  }
  for (uint r = 0; r < rn; r++) {
    for (uint t = 0; t < tn; t++) {
      const float v = simd_sum(acc[r][t]);
      if (lane == 0) y[(ulong)(t0 + t) * p.rows + row0 + r] = v;
    }
  }
}

// Q4 weights (lab_quantize_q4's layout): per row, 16 bytes of nibbles and
// one half scale per block of 32. Weight k of a block is the low nibble of
// byte k and weight k + 16 the high one, worth d * (q - 8). Word k of the
// block's uint4 holds weights 4k..4k+3 low and 16+4k..19+4k high, so it
// meets inputs x[k] and x[4 + k].
static float q4_block_dot(uint4 w, thread const float4* x) {
  float sum = 0.0f;
  for (uint k = 0; k < 4; k++) {
    const uint v = w[k];
    const float4 lo = float4(v & 15, (v >> 8) & 15, (v >> 16) & 15, (v >> 24) & 15) - 8.0f;
    const float4 hi = float4((v >> 4) & 15, (v >> 12) & 15, (v >> 20) & 15, v >> 28) - 8.0f;
    sum += dot(lo, x[k]) + dot(hi, x[4 + k]);
  }
  return sum;
}

// gemv_q8 for Q4 weights.
kernel void gemv_q4(device const uint4* qs [[buffer(0)]],
                    device const half* scales [[buffer(1)]],
                    device const float* x [[buffer(2)]],
                    device float* y [[buffer(3)]],
                    constant uint& rows [[buffer(4)]],
                    constant uint& cols [[buffer(5)]],
                    uint group [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group * ROWS_PER_GROUP + simd;
  if (row >= rows) return;
  const uint blocks = cols / 32;
  device const uint4* q = qs + (ulong)row * blocks;
  device const half* s = scales + (ulong)row * blocks;
  device const float4* x4 = (device const float4*)x;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32) {
    float4 xv[8];
    for (uint k = 0; k < 8; k++) xv[k] = x4[b * 8 + k];
    acc += q4_block_dot(q[b], xv) * float(s[b]);
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

// gemm_q8 for Q4 weights: GEMM_ROWS rows by GEMM_TOKENS tokens per SIMD
// group, each (row, token) in gemv_q4's arithmetic order.
kernel void gemm_q4(device const uint4* qs [[buffer(0)]],
                    device const half* scales [[buffer(1)]],
                    device const float* x [[buffer(2)]],
                    device float* y [[buffer(3)]],
                    constant GemmParams& p [[buffer(4)]],
                    uint2 group [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (group.x * ROWS_PER_GROUP + simd) * GEMM_ROWS;
  if (row0 >= p.rows) return;
  const uint rn = min(GEMM_ROWS, p.rows - row0);
  const uint t0 = group.y * GEMM_TOKENS;
  const uint tn = min(GEMM_TOKENS, p.n - t0);
  const uint blocks = p.cols / 32;
  const uint cols4 = p.cols / 4;
  device const uint4* q = qs + (ulong)row0 * blocks;
  device const half* s = scales + (ulong)row0 * blocks;
  device const float4* x4 = (device const float4*)x + (ulong)t0 * cols4;
  float acc[GEMM_ROWS][GEMM_TOKENS];
  for (uint r = 0; r < GEMM_ROWS; r++) {
    for (uint t = 0; t < GEMM_TOKENS; t++) acc[r][t] = 0.0f;
  }
  for (uint b = lane; b < blocks; b += 32) {
    uint4 w[GEMM_ROWS];
    float scale[GEMM_ROWS];
    for (uint r = 0; r < GEMM_ROWS; r++) {
      if (r >= rn) break;
      w[r] = q[(ulong)r * blocks + b];
      scale[r] = float(s[(ulong)r * blocks + b]);
    }
    for (uint t = 0; t < GEMM_TOKENS; t++) {
      if (t >= tn) break;
      device const float4* xt = x4 + (ulong)t * cols4 + b * 8;
      float4 xv[8];
      for (uint k = 0; k < 8; k++) xv[k] = xt[k];
      for (uint r = 0; r < GEMM_ROWS; r++) {
        if (r >= rn) break;
        acc[r][t] += q4_block_dot(w[r], xv) * scale[r];
      }
    }
  }
  for (uint r = 0; r < rn; r++) {
    for (uint t = 0; t < tn; t++) {
      const float v = simd_sum(acc[r][t]);
      if (lane == 0) y[(ulong)(t0 + t) * p.rows + row0 + r] = v;
    }
  }
}

// The smallest possible dispatch, for measuring a round trip.
kernel void touch(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i == 0) y[0] += 1.0f;
}

// Sum of v over the threadgroup, returned to every thread. `scratch` holds
// one float per SIMD group.
static float group_sum(float v, threadgroup float* scratch, uint simd, uint lane,
                       uint simds) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[simd] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < simds; s++) total += scratch[s];
  return total;
}

struct NormParams {
  uint cols;
  float eps;
  uint accumulate;
};

// RMSNorm, one threadgroup per row of `cols` values: w * (x * r), with
// x * r rounded first as in the reference. With `accumulate` the result is
// added into out instead of stored (a post-norm added to the residual).
kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant NormParams& p [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  device const float* xr = x + (ulong)row * p.cols;
  device float* yr = out + (ulong)row * p.cols;
  float sum = 0.0f;
  for (uint j = tid; j < p.cols; j += threads) sum += xr[j] * xr[j];
  sum = group_sum(sum, scratch, simd, lane, (threads + 31) / 32);
  const float r = 1.0f / sqrt(sum / p.cols + p.eps);
  for (uint j = tid; j < p.cols; j += threads) {
    const float v = w[j] * (xr[j] * r);
    yr[j] = p.accumulate ? yr[j] + v : v;
  }
}

struct HeadParams {
  uint hd;
  float eps;
  uint x_stride;     // floats from one token's heads to the next
  uint rope_stride;  // floats from one token's RoPE tables to the next
};

// Per (head, token) (one threadgroup of hd / 2 threads each): RMSNorm over
// head_dim with weight w, then the half-split RoPE rotation, dimension j
// pairing with j + hd / 2, from the token's tables [cos(hd / 2), sin(hd / 2)].
kernel void head_norm_rope(device float* x [[buffer(0)]],
                           device const float* w [[buffer(1)]],
                           device const float* rope [[buffer(2)]],
                           constant HeadParams& p [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint j [[thread_index_in_threadgroup]],
                           uint2 size [[threads_per_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const uint pair = p.hd / 2;
  device float* xh = x + (ulong)group.y * p.x_stride + (ulong)group.x * p.hd;
  device const float* table = rope + (ulong)group.y * p.rope_stride;
  const float x1 = xh[j], x2 = xh[j + pair];
  const float sum = group_sum(x1 * x1 + x2 * x2, scratch, simd, lane, (size.x + 31) / 32);
  const float r = 1.0f / sqrt(sum / p.hd + p.eps);
  const float n1 = w[j] * (x1 * r), n2 = w[j + pair] * (x2 * r);
  const float c = table[j], s = table[pair + j];
  xh[j] = n1 * c - n2 * s;
  xh[j + pair] = n2 * c + n1 * s;
}

constant uint MAX_CONV_K = 16;

struct ConvParams {
  uint width;
  uint k;  // at most MAX_CONV_K
  uint n;
};

// The gated short convolution over n tokens, one thread per channel walking
// the tokens in order: z = B * X, a causal depthwise conv of the last k
// values of z (kernel column k - 1 weighting the newest), y = C * conv. The
// k - 1 previous z values come from the history (rows oldest first), which
// is left holding the last k - 1. bcx rows hold B, C, X of in_proj.
kernel void conv(device const float* bcx [[buffer(0)]],
                 device const float* weight [[buffer(1)]],
                 device float* history [[buffer(2)]],
                 device float* y [[buffer(3)]],
                 constant ConvParams& p [[buffer(4)]],
                 uint ch [[thread_position_in_grid]]) {
  if (ch >= p.width) return;
  const uint w = p.width, k = p.k;
  device const float* kw = weight + (ulong)ch * k;
  float past[MAX_CONV_K];
  for (uint r = 0; r + 1 < k; r++) past[r] = history[r * w + ch];
  for (uint t = 0; t < p.n; t++) {
    device const float* row = bcx + (ulong)t * 3 * w;
    const float z = row[ch] * row[2 * w + ch];
    float acc = kw[k - 1] * z;
    for (uint s = 1; s < k; s++) acc += kw[k - 1 - s] * past[k - 1 - s];
    y[(ulong)t * w + ch] = row[w + ch] * acc;
    for (uint r = 0; r + 2 < k; r++) past[r] = past[r + 1];
    past[k - 2] = z;
  }
  for (uint r = 0; r + 1 < k; r++) history[r * w + ch] = past[r];
}

struct AttentionParams {
  uint position;  // of token 0
  uint window;    // 0: every earlier key; else the last `window` keys
  uint heads;
  uint hd;
  uint kv_dim;
  uint group;     // query heads per KV head
  float scale;
};

// Causal attention of n tokens whose keys and values are already cached,
// one threadgroup per (head, token): scores against the keys the token may
// see (one SIMD group per key, lanes across head_dim), softmax, the
// weighted sum of values (one thread per dimension), and the per-head
// output gate sigmoid(gate). Scores live in threadgroup memory, which the
// host sizes for the last token's keys.
kernel void attention(device const float* q [[buffer(0)]],
                      device const float* keys [[buffer(1)]],
                      device const float* values [[buffer(2)]],
                      device const float* gate [[buffer(3)]],
                      device float* out [[buffer(4)]],
                      constant AttentionParams& p [[buffer(5)]],
                      threadgroup float* scores [[threadgroup(0)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint2 size [[threads_per_threadgroup]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const uint threads = size.x;
  const uint head = group.x;
  const uint pos = p.position + group.y;
  const uint start = p.window > 0 && pos >= p.window ? pos - p.window + 1 : 0;
  const uint count = pos - start + 1;
  const ulong qo = ((ulong)group.y * p.heads + head) * p.hd;
  const uint simds = threads / 32;
  const uint kvo = head / p.group * p.hd;
  device const float4* q4 = (device const float4*)(q + qo);
  const uint hd4 = p.hd / 4;
  for (uint i = simd; i < count; i += simds) {
    device const float4* k4 =
        (device const float4*)(keys + (ulong)(start + i) * p.kv_dim + kvo);
    float d = 0.0f;
    for (uint c = lane; c < hd4; c += 32) d += dot(q4[c], k4[c]);
    d = simd_sum(d);
    if (lane == 0) scores[i] = d * p.scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint i = tid; i < count; i += threads) m = max(m, scores[i]);
  m = simd_max(m);
  if (lane == 0) scratch[simd] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = scratch[0];
  for (uint s = 1; s < simds; s++) m = max(m, scratch[s]);
  float sum = 0.0f;
  for (uint i = tid; i < count; i += threads) {
    const float e = exp(scores[i] - m);
    scores[i] = e;
    sum += e;
  }
  // group_sum's barriers also publish the exponentials written above.
  sum = group_sum(sum, scratch, simd, lane, simds);
  const float g = 1.0f / (1.0f + exp(-gate[group.y * p.heads + head]));
  for (uint d = tid; d < p.hd; d += threads) {
    device const float* v = values + (ulong)start * p.kv_dim + kvo + d;
    float acc = 0.0f;
    for (uint i = 0; i < count; i++) acc += scores[i] * v[(ulong)i * p.kv_dim];
    out[qo + d] = (acc / sum) * g;
  }
}

// g = silu(g) * u.
kernel void swiglu(device float* g [[buffer(0)]],
                   device const float* u [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float a = g[i];
  g[i] = (a / (1.0f + exp(-a))) * u[i];
}

kernel void copy_f32(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
  if (i < count) dst[i] = src[i];
}
)MSL";

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gGemvQ8, gGemmQ8, gGemvQ4, gGemmQ4, gTouch, gRmsNorm, gHeadNormRope, gConv,
    gAttention, gSwiglu, gCopy;
static double gLastGpuMs;
static const NSUInteger kRowsPerGroup = 8;
static const NSUInteger kGemmTokens = 8, kGemmRows = 4;

// Buffers stay alive in gStrong until lab_gpu_release; gTable mirrors it
// without reference counting, so a lookup on the encoding path is a load.
// Released handles go to gFree and are handed out again.
enum { kMaxBuffers = 4096 };
static id<MTLBuffer> gStrong[kMaxBuffers];
static __unsafe_unretained id<MTLBuffer> gTable[kMaxBuffers];
static int64_t gFree[kMaxBuffers], gFreeCount, gBufferCount, gLive;
static int gReady;  // lab_gpu_init succeeded

// The command buffer being encoded, and its encoder.
static id<MTLCommandBuffer> gCmd;
static id<MTLComputeCommandEncoder> gEnc;

static id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString* name) {
  NSError* error = nil;
  id<MTLComputePipelineState> state =
      [gDevice newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                             error:&error];
  if (state == nil) NSLog(@"lab_metal: %@: %@", name, error);
  return state;
}

// Returns 1 on success, 0 if there is no Metal device or the shader fails
// to compile (the error is logged).
int64_t lab_gpu_init(void) {
  // Once per process: the buffers of every model share the device.
  if (gReady) return 1;
  @autoreleasepool {
    gDevice = MTLCreateSystemDefaultDevice();
    if (gDevice == nil) return 0;
    gQueue = [gDevice newCommandQueue];
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [gDevice newLibraryWithSource:kSource options:options error:&error];
    if (library == nil) {
      NSLog(@"lab_metal: %@", error);
      return 0;
    }
    gGemvQ8 = pipeline(library, @"gemv_q8");
    gGemmQ8 = pipeline(library, @"gemm_q8");
    gGemvQ4 = pipeline(library, @"gemv_q4");
    gGemmQ4 = pipeline(library, @"gemm_q4");
    gTouch = pipeline(library, @"touch");
    gRmsNorm = pipeline(library, @"rms_norm");
    gHeadNormRope = pipeline(library, @"head_norm_rope");
    gConv = pipeline(library, @"conv");
    gAttention = pipeline(library, @"attention");
    gSwiglu = pipeline(library, @"swiglu");
    gCopy = pipeline(library, @"copy_f32");
    gReady = gGemvQ8 && gGemmQ8 && gGemvQ4 && gGemmQ4 && gTouch && gRmsNorm && gHeadNormRope && gConv &&
             gAttention && gSwiglu && gCopy;
    return gReady;
  }
}

// Copies the device name into out (UTF-8, NUL-terminated).
void lab_gpu_device_name(char* out, int64_t capacity) {
  const char* name = gDevice.name.UTF8String;
  strlcpy(out, name ? name : "?", (size_t)capacity);
}

int64_t lab_gpu_max_buffer_length(void) { return (int64_t)gDevice.maxBufferLength; }

int64_t lab_gpu_max_shared_memory(void) {
  return (int64_t)gDevice.maxThreadgroupMemoryLength;
}

// Bytes the GPU can keep resident without hurting performance.
int64_t lab_gpu_recommended_working_set(void) {
  return (int64_t)gDevice.recommendedMaxWorkingSetSize;
}

// NSProcessInfo's thermal state: 0 nominal, 1 fair, 2 serious, 3 critical.
// Phones throttle from "fair" on, so measurements record it.
int64_t lab_gpu_thermal_state(void) {
  return (int64_t)[NSProcessInfo processInfo].thermalState;
}

int64_t lab_gpu_physical_memory(void) {
  return (int64_t)[NSProcessInfo processInfo].physicalMemory;
}

static int64_t add_buffer(id<MTLBuffer> b) {
  if (b == nil) return -1;
  int64_t handle;
  if (gFreeCount > 0) {
    handle = gFree[--gFreeCount];
  } else if (gBufferCount < kMaxBuffers) {
    handle = gBufferCount++;
  } else {
    return -1;
  }
  gStrong[handle] = b;
  gTable[handle] = b;
  gLive++;
  return handle;
}

// Releases a buffer and frees its handle for reuse. The GPU must be done
// with it: Dart waits for every command buffer it commits. A wrapped
// buffer leaves the memory it wraps alone.
void lab_gpu_release(int64_t handle) {
  if (handle < 0 || handle >= gBufferCount || gStrong[handle] == nil) return;
  gStrong[handle] = nil;
  gTable[handle] = nil;
  gFree[gFreeCount++] = handle;
  gLive--;
}

// Buffers not yet released, for leak checks.
int64_t lab_gpu_live_buffers(void) { return gLive; }

// A shared-storage buffer of `bytes` zeroed bytes; returns its handle, or -1.
int64_t lab_gpu_buffer(int64_t bytes) {
  @autoreleasepool {
    return add_buffer([gDevice newBufferWithLength:(NSUInteger)bytes
                                           options:MTLResourceStorageModeShared]);
  }
}

// A buffer the host reads after every command buffer, such as the logits.
// Apple GPUs share the host's memory, so it is an ordinary buffer.
int64_t lab_gpu_readback_buffer(int64_t bytes) { return lab_gpu_buffer(bytes); }

// Wraps existing memory, such as a mapped file, as a buffer without a copy.
// The address must be page-aligned; the length is rounded up to whole
// pages, which a mapping always covers. Returns the handle, or -1.
int64_t lab_gpu_wrap(void* address, int64_t bytes) {
  @autoreleasepool {
    const NSUInteger page = (NSUInteger)getpagesize();
    if ((uintptr_t)address % page != 0) return -1;
    const NSUInteger length = ((NSUInteger)bytes + page - 1) / page * page;
    return add_buffer([gDevice newBufferWithBytesNoCopy:address
                                                 length:length
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil]);
  }
}

// The CPU address of a buffer's memory.
void* lab_gpu_contents(int64_t handle) { return gTable[handle].contents; }

// Rows per subgroup of decode (lab_vulkan.c); Metal's GEMV has one layout.
void lab_gpu_set_decode_rows(int64_t rows) { (void)rows; }
int64_t lab_gpu_set_q8_group(int64_t invocations) { (void)invocations; return 0; }
int64_t lab_gpu_set_q8_vector(int64_t on) { (void)on; return 0; }
int64_t lab_gpu_register_q8_head(int64_t w, int64_t qs, int64_t rows, int64_t cols, int64_t q4Body) {
  (void)w; (void)qs; (void)rows; (void)cols; (void)q4Body; return 0;
}

// Per-kernel timing (lab_vulkan.c has it); Metal reports none.
void lab_gpu_set_profiling(int64_t on) { (void)on; }

int64_t lab_gpu_profile(double* ms, int64_t* counts, int64_t kinds) {
  (void)ms;
  (void)counts;
  (void)kinds;
  return 0;
}

// GPU execution time of the last waited command buffer, in milliseconds.
double lab_gpu_last_ms(void) { return gLastGpuMs; }

static void encode_gemv(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> gemv,
                        id<MTLBuffer> w, int64_t qsOffset, int64_t scalesOffset, id<MTLBuffer> x,
                        int64_t xOffset, id<MTLBuffer> y, int64_t yOffset, int64_t rows,
                        int64_t cols) {
  const uint32_t r = (uint32_t)rows, c = (uint32_t)cols;
  [enc setComputePipelineState:gemv];
  [enc setBuffer:w offset:(NSUInteger)qsOffset atIndex:0];
  [enc setBuffer:w offset:(NSUInteger)scalesOffset atIndex:1];
  [enc setBuffer:x offset:(NSUInteger)xOffset atIndex:2];
  [enc setBuffer:y offset:(NSUInteger)yOffset atIndex:3];
  [enc setBytes:&r length:sizeof r atIndex:4];
  [enc setBytes:&c length:sizeof c atIndex:5];
  [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + kRowsPerGroup - 1) / kRowsPerGroup, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32 * kRowsPerGroup, 1, 1)];
}

static void finish(id<MTLCommandBuffer> cb) {
  [cb commit];
  [cb waitUntilCompleted];
  gLastGpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

// --- Measurements -----------------------------------------------------------

// One product, one command buffer, waited for: the granularity of calling
// the GPU once per matrix product. qs and scales are separate buffers.
void lab_metal_gemv_q8(int64_t qs, int64_t scales, int64_t x, int64_t y, int64_t rows,
                       int64_t cols) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const uint32_t r = (uint32_t)rows, c = (uint32_t)cols;
    [enc setComputePipelineState:gGemvQ8];
    [enc setBuffer:gTable[qs] offset:0 atIndex:0];
    [enc setBuffer:gTable[scales] offset:0 atIndex:1];
    [enc setBuffer:gTable[x] offset:0 atIndex:2];
    [enc setBuffer:gTable[y] offset:0 atIndex:3];
    [enc setBytes:&r length:sizeof r atIndex:4];
    [enc setBytes:&c length:sizeof c atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + kRowsPerGroup - 1) / kRowsPerGroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * kRowsPerGroup, 1, 1)];
    [enc endEncoding];
    finish(cb);
  }
}

// `count` products in one command buffer and one serial encoder: each
// dispatch finishes before the next starts, as a real layer chain needs,
// but the CPU waits once. The granularity of calling the GPU once per token.
void lab_metal_gemv_q8_batch(const int64_t* qs, const int64_t* scales, const int64_t* rows,
                             const int64_t* cols, int64_t count, int64_t x, int64_t y) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int64_t i = 0; i < count; i++) {
      const uint32_t r = (uint32_t)rows[i], c = (uint32_t)cols[i];
      [enc setComputePipelineState:gGemvQ8];
      [enc setBuffer:gTable[qs[i]] offset:0 atIndex:0];
      [enc setBuffer:gTable[scales[i]] offset:0 atIndex:1];
      [enc setBuffer:gTable[x] offset:0 atIndex:2];
      [enc setBuffer:gTable[y] offset:0 atIndex:3];
      [enc setBytes:&r length:sizeof r atIndex:4];
      [enc setBytes:&c length:sizeof c atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)r + kRowsPerGroup - 1) / kRowsPerGroup, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * kRowsPerGroup, 1, 1)];
    }
    [enc endEncoding];
    finish(cb);
  }
}

// The smallest dispatch, committed and waited for.
void lab_metal_touch(int64_t y) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gTouch];
    [enc setBuffer:gTable[y] offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    finish(cb);
  }
}

// --- The forward pass: begin, encode kernels, commit ------------------------
//
// Every buffer argument is a handle plus a byte offset into it.

// Opens a command buffer and its serial compute encoder. Buffers are not
// retained per command buffer: gStrong keeps them alive.
void lab_gpu_begin(void) {
  @autoreleasepool {
    gCmd = [gQueue commandBufferWithUnretainedReferences];
    gEnc = [gCmd computeCommandEncoder];
  }
}

// Ends encoding and commits. With `wait`, blocks until the GPU is done and
// returns 0 if the command buffer failed (the error is logged), else 1.
int64_t lab_gpu_commit(int64_t wait) {
  @autoreleasepool {
    [gEnc endEncoding];
    gEnc = nil;
    id<MTLCommandBuffer> cb = gCmd;
    gCmd = nil;
    [cb commit];
    if (!wait) return 1;
    [cb waitUntilCompleted];
    gLastGpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    if (cb.status == MTLCommandBufferStatusError) {
      NSLog(@"lab_metal: %@", cb.error);
      return 0;
    }
    return 1;
  }
}

typedef struct {
  uint32_t rows, cols, n;
} GemmParams;

static void encode_matmul(id<MTLComputePipelineState> gemv, id<MTLComputePipelineState> gemm,
                          int64_t w, int64_t qsOffset, int64_t scalesOffset, int64_t x,
                          int64_t xOffset, int64_t y, int64_t yOffset, int64_t rows, int64_t cols,
                          int64_t n) {
  if (n == 1) {
    encode_gemv(gEnc, gemv, gTable[w], qsOffset, scalesOffset, gTable[x], xOffset, gTable[y],
                yOffset, rows, cols);
    return;
  }
  const GemmParams p = {(uint32_t)rows, (uint32_t)cols, (uint32_t)n};
  [gEnc setComputePipelineState:gemm];
  [gEnc setBuffer:gTable[w] offset:(NSUInteger)qsOffset atIndex:0];
  [gEnc setBuffer:gTable[w] offset:(NSUInteger)scalesOffset atIndex:1];
  [gEnc setBuffer:gTable[x] offset:(NSUInteger)xOffset atIndex:2];
  [gEnc setBuffer:gTable[y] offset:(NSUInteger)yOffset atIndex:3];
  [gEnc setBytes:&p length:sizeof p atIndex:4];
  const NSUInteger rowsPerGroup = kRowsPerGroup * kGemmRows;
  [gEnc dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + rowsPerGroup - 1) / rowsPerGroup,
                                         ((NSUInteger)n + kGemmTokens - 1) / kGemmTokens, 1)
       threadsPerThreadgroup:MTLSizeMake(32 * kRowsPerGroup, 1, 1)];
}

// Y = X W^T for n tokens, X [n, cols] and Y [n, rows], with W in Q8 inside
// buffer w at the given byte offsets: gemv_q8 for one token, else gemm_q8.
void lab_gpu_enc_matmul_q8(int64_t w, int64_t qsOffset, int64_t scalesOffset, int64_t x,
                             int64_t xOffset, int64_t y, int64_t yOffset, int64_t rows,
                             int64_t cols, int64_t n) {
  encode_matmul(gGemvQ8, gGemmQ8, w, qsOffset, scalesOffset, x, xOffset, y, yOffset, rows, cols,
                n);
}

// The same for Q4 weights.
void lab_gpu_enc_matmul_q4(int64_t w, int64_t qsOffset, int64_t scalesOffset, int64_t x,
                             int64_t xOffset, int64_t y, int64_t yOffset, int64_t rows,
                             int64_t cols, int64_t n) {
  encode_matmul(gGemvQ4, gGemmQ4, w, qsOffset, scalesOffset, x, xOffset, y, yOffset, rows, cols,
                n);
}

typedef struct {
  uint32_t cols;
  float eps;
  uint32_t accumulate;
} NormParams;

void lab_gpu_enc_rms_norm(int64_t x, int64_t xOffset, int64_t w, int64_t wOffset, int64_t out,
                            int64_t outOffset, int64_t rows, int64_t cols, double eps,
                            int64_t accumulate) {
  const NormParams p = {(uint32_t)cols, (float)eps, (uint32_t)(accumulate != 0)};
  [gEnc setComputePipelineState:gRmsNorm];
  [gEnc setBuffer:gTable[x] offset:(NSUInteger)xOffset atIndex:0];
  [gEnc setBuffer:gTable[w] offset:(NSUInteger)wOffset atIndex:1];
  [gEnc setBuffer:gTable[out] offset:(NSUInteger)outOffset atIndex:2];
  [gEnc setBytes:&p length:sizeof p atIndex:3];
  [gEnc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef struct {
  uint32_t hd;
  float eps;
  uint32_t xStride, ropeStride;
} HeadParams;

// Strides are in floats, between consecutive tokens of x and of rope.
void lab_gpu_enc_head_norm_rope(int64_t x, int64_t xOffset, int64_t xStride, int64_t w,
                                  int64_t wOffset, int64_t rope, int64_t ropeOffset,
                                  int64_t ropeStride, int64_t heads, int64_t hd, int64_t n,
                                  double eps) {
  const HeadParams p = {(uint32_t)hd, (float)eps, (uint32_t)xStride, (uint32_t)ropeStride};
  [gEnc setComputePipelineState:gHeadNormRope];
  [gEnc setBuffer:gTable[x] offset:(NSUInteger)xOffset atIndex:0];
  [gEnc setBuffer:gTable[w] offset:(NSUInteger)wOffset atIndex:1];
  [gEnc setBuffer:gTable[rope] offset:(NSUInteger)ropeOffset atIndex:2];
  [gEnc setBytes:&p length:sizeof p atIndex:3];
  [gEnc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, (NSUInteger)n, 1)
       threadsPerThreadgroup:MTLSizeMake((NSUInteger)hd / 2, 1, 1)];
}

typedef struct {
  uint32_t width, k, n;
} ConvParams;

// k must be at most 16 (MAX_CONV_K in the shader).
void lab_gpu_enc_conv(int64_t bcx, int64_t weight, int64_t weightOffset, int64_t history,
                        int64_t y, int64_t width, int64_t k, int64_t n) {
  const ConvParams p = {(uint32_t)width, (uint32_t)k, (uint32_t)n};
  [gEnc setComputePipelineState:gConv];
  [gEnc setBuffer:gTable[bcx] offset:0 atIndex:0];
  [gEnc setBuffer:gTable[weight] offset:(NSUInteger)weightOffset atIndex:1];
  [gEnc setBuffer:gTable[history] offset:0 atIndex:2];
  [gEnc setBuffer:gTable[y] offset:0 atIndex:3];
  [gEnc setBytes:&p length:sizeof p atIndex:4];
  [gEnc dispatchThreads:MTLSizeMake((NSUInteger)width, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef struct {
  uint32_t position, window, heads, hd, kvDim, group;
  float scale;
} AttentionParams;

// Attention of n tokens at positions position, ... over their KV cache;
// `window` 0 reads every earlier position, otherwise the last `window`
// including the token's own.
void lab_gpu_enc_attention(int64_t q, int64_t keys, int64_t values, int64_t gate, int64_t out,
                             int64_t position, int64_t n, int64_t heads, int64_t kvHeads,
                             int64_t hd, int64_t window) {
  const int64_t last = position + n - 1;
  const int64_t keysOfLast = window > 0 && last >= window ? window : last + 1;
  const AttentionParams p = {(uint32_t)position,
                             (uint32_t)window,
                             (uint32_t)heads,
                             (uint32_t)hd,
                             (uint32_t)(kvHeads * hd),
                             (uint32_t)(heads / kvHeads),
                             (float)(1.0 / sqrt((double)hd))};
  [gEnc setComputePipelineState:gAttention];
  [gEnc setBuffer:gTable[q] offset:0 atIndex:0];
  [gEnc setBuffer:gTable[keys] offset:0 atIndex:1];
  [gEnc setBuffer:gTable[values] offset:0 atIndex:2];
  [gEnc setBuffer:gTable[gate] offset:0 atIndex:3];
  [gEnc setBuffer:gTable[out] offset:0 atIndex:4];
  [gEnc setBytes:&p length:sizeof p atIndex:5];
  [gEnc setThreadgroupMemoryLength:((NSUInteger)keysOfLast * 4 + 15) / 16 * 16 atIndex:0];
  [gEnc dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, (NSUInteger)n, 1)
       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void lab_gpu_enc_swiglu(int64_t g, int64_t u, int64_t count) {
  const uint32_t n = (uint32_t)count;
  [gEnc setComputePipelineState:gSwiglu];
  [gEnc setBuffer:gTable[g] offset:0 atIndex:0];
  [gEnc setBuffer:gTable[u] offset:0 atIndex:1];
  [gEnc setBytes:&n length:sizeof n atIndex:2];
  [gEnc dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void lab_gpu_enc_copy(int64_t src, int64_t srcOffset, int64_t dst, int64_t dstOffset,
                        int64_t count) {
  const uint32_t n = (uint32_t)count;
  [gEnc setComputePipelineState:gCopy];
  [gEnc setBuffer:gTable[src] offset:(NSUInteger)srcOffset atIndex:0];
  [gEnc setBuffer:gTable[dst] offset:(NSUInteger)dstOffset atIndex:1];
  [gEnc setBytes:&n length:sizeof n atIndex:2];
  [gEnc dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
