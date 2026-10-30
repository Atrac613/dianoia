// Stage 5a: native CPU kernels for Dart, called through dart:ffi.
//
// lab_gemv_f32 uses the summation order of lfm3's Dart gemv (eight 4-lane
// accumulators; lane l of accumulator v sums elements with j / 4 % 8 == v
// and j % 4 == l, leftover vectors to accumulator 0, accumulators combined
// as ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7)), lanes as
// (l0 + l1) + (l2 + l3)), and multiply and add stay separate
// (-ffp-contract=off), so its output equals the Dart kernel's bit for bit.
//
// Unlike Dart isolates, native threads share memory, so the rows of one
// product are split across threads with no copying and no messages.
//
// The kernels are written for arm64 NEON. On x86-64 (Windows) each has a
// twin in SSE and AVX2 that does the same float operations in the same
// order, four lanes wide and fused where NEON fuses, so a matrix product
// gives the arm64 result bit for bit. Only expf, from each platform's C
// library, may differ.

#if defined(__aarch64__) || defined(_M_ARM64)
#define LAB_NEON 1
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif !defined(_WIN32)
#include <pthread.h>
#include <sys/auxv.h>
#endif

// A DLL exports only what it is told to.
#ifdef _WIN32
#define LAB_EXPORT __declspec(dllexport)
#else
#define LAB_EXPORT
#endif

// Whether this CPU runs the kernels: they are compiled for the int8 dot
// product (sdot), which arm64 CPUs before ARMv8.2 lack, or on x86-64 for
// AVX2, FMA and F16C (Haswell, Zen and later), so calling any other function
// of this library there may stop the app on an illegal instruction. This
// one only asks the OS or the CPU.
LAB_EXPORT int64_t lab_cpu_supported(void) {
#if defined(__APPLE__)
  int64_t on = 0;
  size_t size = sizeof on;
  return sysctlbyname("hw.optional.arm.FEAT_DotProd", &on, &size, NULL, 0) == 0 && on != 0;
#elif defined(HWCAP_ASIMDDP)
  return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(_M_X64)
  int info[4];
  __cpuid(info, 0);
  if (info[0] < 7) return 0;
  __cpuid(info, 1);
  // FMA, OSXSAVE, AVX and F16C, and the OS saves the 256-bit registers.
  const int ecx = (1 << 12) | (1 << 27) | (1 << 28) | (1 << 29);
  if ((info[2] & ecx) != ecx || (_xgetbv(0) & 6) != 6) return 0;
  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;  // AVX2
#else
  return 0;
#endif
}

// Memory the kernels read: 128-byte aligned (Apple Silicon cache line),
// zeroed. Dart views it with Pointer.asTypedList, so nothing is copied.
LAB_EXPORT void* lab_alloc(int64_t bytes) {
#ifdef _WIN32
  void* p = _aligned_malloc((size_t)bytes, 128);
  if (p == NULL) return NULL;
#else
  void* p = NULL;
  if (posix_memalign(&p, 128, (size_t)bytes) != 0) return NULL;
#endif
  memset(p, 0, (size_t)bytes);
  return p;
}

#ifdef _WIN32
LAB_EXPORT void lab_free(void* p) { _aligned_free(p); }
#else
LAB_EXPORT void lab_free(void* p) { free(p); }
#endif

// Maps a whole file read-only and stores its size in *size. The pages are
// loaded on first touch and stay clean, so the OS can drop and re-read them
// under memory pressure instead of swapping. Returns NULL on failure.
LAB_EXPORT void* lab_map_file(const char* path, int64_t* size) {
#ifdef _WIN32
  // The path is UTF-8; Windows takes UTF-16.
  const int chars = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
  if (chars == 0) return NULL;
  wchar_t* wide = (wchar_t*)malloc((size_t)chars * sizeof(wchar_t));
  if (wide == NULL) return NULL;
  MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, chars);
  const HANDLE file = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  free(wide);
  if (file == INVALID_HANDLE_VALUE) return NULL;
  LARGE_INTEGER length;
  HANDLE mapping = NULL;
  if (GetFileSizeEx(file, &length) && length.QuadPart > 0) {
    mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
  }
  CloseHandle(file);
  if (mapping == NULL) return NULL;
  // The view keeps the mapping, and the mapping the file, open.
  void* p = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(mapping);
  if (p == NULL) return NULL;
  *size = (int64_t)length.QuadPart;
  return p;
#else
  const int fd = open(path, O_RDONLY);
  if (fd < 0) return NULL;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size == 0) {
    close(fd);
    return NULL;
  }
  void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return NULL;
  *size = (int64_t)st.st_size;
  return p;
#endif
}

#ifdef _WIN32
LAB_EXPORT void lab_unmap_file(void* p, int64_t size) {
  (void)size;
  UnmapViewOfFile(p);
}
#else
LAB_EXPORT void lab_unmap_file(void* p, int64_t size) { munmap(p, (size_t)size); }
#endif

// For measuring the cost of a Dart-to-C call by itself.
LAB_EXPORT int64_t lab_nop(int64_t x) { return x + 1; }

#ifdef LAB_NEON
static inline float lane_sum(float32x4_t v) {
  const float32x4_t t = vaddq_f32(v, vrev64q_f32(v));
  return vgetq_lane_f32(t, 0) + vgetq_lane_f32(t, 2);
}

static void gemv_rows(const float* w, const float* x, float* y, int64_t r0,
                      int64_t r1, int64_t cols) {
  const int64_t kv = cols / 4;
  const int64_t main_v = kv - kv % 8;
  for (int64_t i = r0; i < r1; i++) {
    const float* row = w + i * cols;
    float32x4_t a[8];
    for (int l = 0; l < 8; l++) a[l] = vdupq_n_f32(0.0f);
    int64_t j = 0;
    for (; j < main_v; j += 8) {
      for (int l = 0; l < 8; l++) {
        a[l] = vaddq_f32(a[l], vmulq_f32(vld1q_f32(row + 4 * (j + l)),
                                         vld1q_f32(x + 4 * (j + l))));
      }
    }
    for (; j < kv; j++) {
      a[0] = vaddq_f32(a[0], vmulq_f32(vld1q_f32(row + 4 * j), vld1q_f32(x + 4 * j)));
    }
    const float32x4_t v =
        vaddq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])),
                  vaddq_f32(vaddq_f32(a[4], a[5]), vaddq_f32(a[6], a[7])));
    y[i] = lane_sum(v);
  }
}
#else
// (v0 + v1) + (v2 + v3), as the NEON lane_sum.
static inline float lane_sum(__m128 v) {
  const __m128 t = _mm_add_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehl_ps(t, t)));
}

static void gemv_rows(const float* w, const float* x, float* y, int64_t r0,
                      int64_t r1, int64_t cols) {
  const int64_t kv = cols / 4;
  const int64_t main_v = kv - kv % 8;
  for (int64_t i = r0; i < r1; i++) {
    const float* row = w + i * cols;
    __m128 a[8];
    for (int l = 0; l < 8; l++) a[l] = _mm_setzero_ps();
    int64_t j = 0;
    for (; j < main_v; j += 8) {
      for (int l = 0; l < 8; l++) {
        a[l] = _mm_add_ps(a[l], _mm_mul_ps(_mm_loadu_ps(row + 4 * (j + l)),
                                           _mm_loadu_ps(x + 4 * (j + l))));
      }
    }
    for (; j < kv; j++) {
      a[0] = _mm_add_ps(a[0], _mm_mul_ps(_mm_loadu_ps(row + 4 * j), _mm_loadu_ps(x + 4 * j)));
    }
    const __m128 v =
        _mm_add_ps(_mm_add_ps(_mm_add_ps(a[0], a[1]), _mm_add_ps(a[2], a[3])),
                   _mm_add_ps(_mm_add_ps(a[4], a[5]), _mm_add_ps(a[6], a[7])));
    y[i] = lane_sum(v);
  }
}
#endif

// Row chunks per OpenMP thread (see lab_gemm_q8).
enum { CHUNKS_PER_THREAD = 8 };

// Row boundaries for thread t of n, even so every thread gets whole pairs.
static int64_t boundary(int64_t rows, int64_t t, int64_t n) {
  return t == n ? rows : (rows * t / n) & ~(int64_t)1;
}

#ifdef _WIN32
// Windows has no OpenMP runtime to ship, so the kernels use the process's
// thread pool instead: `workers` threads, the caller among them, take
// chunks in turn from a shared counter, so faster cores take more (the
// dynamic schedule of the OpenMP path).
typedef void (*ChunkFn)(const void* job, int64_t chunk, int64_t chunks);

typedef struct {
  ChunkFn fn;
  const void* job;
  int64_t chunks;
  volatile LONG64 next;
} Chunks;

static void take_chunks(Chunks* p) {
  for (int64_t c; (c = InterlockedIncrement64(&p->next) - 1) < p->chunks;) {
    p->fn(p->job, c, p->chunks);
  }
}

static void CALLBACK chunk_worker(PTP_CALLBACK_INSTANCE instance, void* arg, PTP_WORK work) {
  (void)instance;
  (void)work;
  take_chunks((Chunks*)arg);
}

static void parallel_chunks(int64_t workers, int64_t chunks, ChunkFn fn, const void* job) {
  Chunks p = {fn, job, chunks, 0};
  const PTP_WORK work =
      workers > 1 && chunks > 1 ? CreateThreadpoolWork(chunk_worker, &p, NULL) : NULL;
  for (int64_t w = 1; work != NULL && w < workers && w < chunks; w++) SubmitThreadpoolWork(work);
  take_chunks(&p);
  if (work != NULL) {
    WaitForThreadpoolWorkCallbacks(work, FALSE);
    CloseThreadpoolWork(work);
  }
}

typedef struct {
  const float *w, *x;
  float* y;
  int64_t rows, cols;
} Gemv;

static void gemv_chunk(const void* job, int64_t c, int64_t chunks) {
  const Gemv* g = (const Gemv*)job;
  gemv_rows(g->w, g->x, g->y, boundary(g->rows, c, chunks), boundary(g->rows, c + 1, chunks),
            g->cols);
}
#elif !defined(__APPLE__)
typedef struct {
  const float *w, *x;
  float* y;
  int64_t r0, r1, cols;
} Job;

static void* run_job(void* arg) {
  const Job* j = (const Job*)arg;
  gemv_rows(j->w, j->x, j->y, j->r0, j->r1, j->cols);
  return NULL;
}
#endif

// y[0:rows] = w[rows, cols] x, rows split across `threads` threads.
LAB_EXPORT void lab_gemv_f32(const float* w, const float* x, float* y, int64_t rows,
                             int64_t cols, int64_t threads) {
  if (threads <= 1) {
    gemv_rows(w, x, y, 0, rows, cols);
    return;
  }
#ifdef __APPLE__
  dispatch_apply((size_t)threads, DISPATCH_APPLY_AUTO, ^(size_t t) {
    gemv_rows(w, x, y, boundary(rows, (int64_t)t, threads),
              boundary(rows, (int64_t)t + 1, threads), cols);
  });
#elif defined(_WIN32)
  const Gemv g = {w, x, y, rows, cols};
  parallel_chunks(threads, threads, gemv_chunk, &g);
#else
  pthread_t ids[64];
  Job jobs[64];
  const int64_t n = threads > 64 ? 64 : threads;
  for (int64_t t = 0; t < n; t++) {
    jobs[t] = (Job){w, x, y, boundary(rows, t, n), boundary(rows, t + 1, n), cols};
    pthread_create(&ids[t], NULL, run_job, &jobs[t]);
  }
  for (int64_t t = 0; t < n; t++) pthread_join(ids[t], NULL);
#endif
}

// Apple's BLAS, as a reference point. Returns 0 where it is unavailable.
LAB_EXPORT int64_t lab_gemv_accelerate(const float* w, const float* x, float* y,
                                       int64_t rows, int64_t cols) {
#ifdef __APPLE__
  cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)rows, (int)cols, 1.0f, w,
              (int)cols, x, 1, 0.0f, y, 1);
  return 1;
#else
  (void)w; (void)x; (void)y; (void)rows; (void)cols;
  return 0;
#endif
}

// ---------------------------------------------------------------- Q8
//
// llama.cpp's Q8_0 idea in a plain layout: per row, blocks of 32 weights
// share a float scale and store int8 q = round(w / scale), |q| <= 127.
// Native code can widen int8 with shifts and multiply int8 pairs with sdot,
// neither of which Dart's SIMD offers.

enum { Q8_BLOCK = 32 };

LAB_EXPORT void lab_quantize_q8(const float* w, int8_t* qs, float* scales, int64_t rows,
                                int64_t cols) {
  const int64_t blocks = cols / Q8_BLOCK;
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t b = 0; b < blocks; b++) {
      const float* src = w + r * cols + b * Q8_BLOCK;
      float amax = 0.0f;
      for (int k = 0; k < Q8_BLOCK; k++) {
        const float a = src[k] < 0 ? -src[k] : src[k];
        if (a > amax) amax = a;
      }
      const float scale = amax / 127.0f;
      const float inv = scale > 0 ? 1.0f / scale : 0.0f;
      scales[r * blocks + b] = scale;
      for (int k = 0; k < Q8_BLOCK; k++) {
        float q = src[k] * inv;
        q = q < 0 ? q - 0.5f : q + 0.5f;  // round half away from zero
        int v = (int)q;
        qs[r * cols + b * Q8_BLOCK + k] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
      }
    }
  }
}

#ifdef LAB_NEON
// Weights int8, input float: widen each int8 to float and multiply.
static float dot_q8_f32(const int8_t* row, const float* row_scales,
                        const float* x, int64_t blocks) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int64_t b = 0; b < blocks; b++) {
    float32x4_t s = vdupq_n_f32(0.0f);
    for (int h = 0; h < 2; h++) {
      const int8x16_t q = vld1q_s8(row + b * Q8_BLOCK + 16 * h);
      const int16x8_t lo = vmovl_s8(vget_low_s8(q));
      const int16x8_t hi = vmovl_high_s8(q);
      const float* xb = x + b * Q8_BLOCK + 16 * h;
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(xb));
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_high_s16(lo)), vld1q_f32(xb + 4));
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(xb + 8));
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_high_s16(hi)), vld1q_f32(xb + 12));
    }
    acc = vfmaq_n_f32(acc, s, row_scales[b]);
  }
  return lane_sum(acc);
}

// Weights and input both int8 (the input quantized per block once per
// product, as llama.cpp does): sdot multiplies 16 int8 pairs into four
// int32 sums per instruction.
static float dot_q8_sdot(const int8_t* row, const float* row_scales,
                         const int8_t* xq, const float* xs, int64_t blocks) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int64_t b = 0; b < blocks; b++) {
    int32x4_t d = vdupq_n_s32(0);
    d = vdotq_s32(d, vld1q_s8(row + b * Q8_BLOCK), vld1q_s8(xq + b * Q8_BLOCK));
    d = vdotq_s32(d, vld1q_s8(row + b * Q8_BLOCK + 16), vld1q_s8(xq + b * Q8_BLOCK + 16));
    acc = vfmaq_n_f32(acc, vcvtq_f32_s32(d), row_scales[b] * xs[b]);
  }
  return lane_sum(acc);
}
#else
// s plus 16 int8 weights times 16 floats, four lanes at a time in NEON's
// order.
static inline __m128 fma16_i8(__m128 s, __m128i q, const float* xb) {
  s = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(q)), _mm_loadu_ps(xb), s);
  s = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q, 4))),
                   _mm_loadu_ps(xb + 4), s);
  s = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q, 8))),
                   _mm_loadu_ps(xb + 8), s);
  return _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q, 12))),
                      _mm_loadu_ps(xb + 12), s);
}

// Two sdots over a block of 32 int8 pairs: lane l sums bytes 4l..4l+3 of
// both halves. x86 multiplies unsigned by signed bytes, so |x| meets w with
// x's sign; exact while no value is -128, which the quantizers never write.
static inline __m128i sdot_block(__m256i w, __m256i x) {
  const __m256i pairs = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
  const __m256i quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
  return _mm_add_epi32(_mm256_castsi256_si128(quads), _mm256_extracti128_si256(quads, 1));
}

static float dot_q8_f32(const int8_t* row, const float* row_scales,
                        const float* x, int64_t blocks) {
  __m128 acc = _mm_setzero_ps();
  for (int64_t b = 0; b < blocks; b++) {
    __m128 s = _mm_setzero_ps();
    for (int h = 0; h < 2; h++) {
      s = fma16_i8(s, _mm_loadu_si128((const __m128i*)(row + b * Q8_BLOCK + 16 * h)),
                   x + b * Q8_BLOCK + 16 * h);
    }
    acc = _mm_fmadd_ps(s, _mm_set1_ps(row_scales[b]), acc);
  }
  return lane_sum(acc);
}

static float dot_q8_sdot(const int8_t* row, const float* row_scales,
                         const int8_t* xq, const float* xs, int64_t blocks) {
  __m128 acc = _mm_setzero_ps();
  for (int64_t b = 0; b < blocks; b++) {
    const __m128i d = sdot_block(_mm256_loadu_si256((const __m256i*)(row + b * Q8_BLOCK)),
                                 _mm256_loadu_si256((const __m256i*)(xq + b * Q8_BLOCK)));
    acc = _mm_fmadd_ps(_mm_cvtepi32_ps(d), _mm_set1_ps(row_scales[b] * xs[b]), acc);
  }
  return lane_sum(acc);
}
#endif

// One matrix product, for the threads that share its rows.
typedef struct {
  const void* qs;
  const void* scales;
  const float* xs;
  const int8_t* xq;
  const float* xsc;
  float* ys;
  int64_t rows, cols, n, mode;
} Product;

static void q8_rows(const Product* p, int64_t r0, int64_t r1) {
  const int64_t rows = p->rows, cols = p->cols, blocks = cols / Q8_BLOCK;
  for (int64_t r = r0; r < r1; r++) {
    const int8_t* row = (const int8_t*)p->qs + r * cols;
    const float* row_scales = (const float*)p->scales + r * blocks;
    for (int64_t t = 0; t < p->n; t++) {
      p->ys[t * rows + r] =
          p->mode == 1
              ? dot_q8_sdot(row, row_scales, p->xq + t * cols, p->xsc + t * blocks, blocks)
              : dot_q8_f32(row, row_scales, p->xs + t * cols, blocks);
    }
  }
}

#ifdef _WIN32
static void q8_chunk(const void* job, int64_t c, int64_t chunks) {
  const Product* p = (const Product*)job;
  q8_rows(p, boundary(p->rows, c, chunks), boundary(p->rows, c + 1, chunks));
}
#endif

// ys[t * rows + r] = (W xs[t])[r] for t < n, Q8 weights, xs as [n, cols].
// mode 0 keeps the inputs in float; mode 1 quantizes them into xq / xsc
// (n * cols bytes and n * cols / 32 floats of scratch) and uses sdot.
// Rows are split across threads; each row is dotted with every input while
// it is in L1, so a batch streams the weights once. n = 1 is the decode
// GEMV, and a row's result does not depend on n or on the thread split.
LAB_EXPORT void lab_gemm_q8(const int8_t* qs, const float* scales, const float* xs,
                            int8_t* xq, float* xsc, float* ys, int64_t rows, int64_t cols,
                            int64_t n, int64_t threads, int64_t mode) {
  if (mode == 1) lab_quantize_q8(xs, xq, xsc, n, cols);
  const Product p = {qs, scales, xs, xq, xsc, ys, rows, cols, n, mode};
  const int64_t workers = threads < 1 ? 1 : threads;
#ifdef __APPLE__
  dispatch_apply((size_t)workers, DISPATCH_APPLY_AUTO, ^(size_t w) {
    q8_rows(&p, boundary(rows, (int64_t)w, workers), boundary(rows, (int64_t)w + 1, workers));
  });
#else
  // Elsewhere (Android, Linux) OpenMP keeps a pool of threads between
  // calls; built without it, the pragma is ignored and this runs serially.
  // Phones mix fast and slow cores, so rows go out in small chunks to
  // whichever thread is free rather than in one equal share per thread.
  // Windows does the same with its thread pool (parallel_chunks).
  const int64_t chunks = workers == 1 ? 1 : workers * CHUNKS_PER_THREAD;
#ifdef _WIN32
  parallel_chunks(workers, chunks, q8_chunk, &p);
#else
#pragma omp parallel for num_threads((int)workers) schedule(dynamic, 1)
  for (int64_t c = 0; c < chunks; c++) {
    q8_rows(&p, boundary(rows, c, chunks), boundary(rows, c + 1, chunks));
  }
#endif
#endif
}

// The single-vector form used by the stage 5a benchmark.
LAB_EXPORT void lab_gemv_q8(const int8_t* qs, const float* scales, const float* x,
                            int8_t* xq, float* xs, float* y, int64_t rows, int64_t cols,
                            int64_t threads, int64_t mode) {
  lab_gemm_q8(qs, scales, x, xq, xs, y, rows, cols, 1, threads, mode);
}

// ---------------------------------------------------------------- attention
//
// Grouped-query attention for a chunk of n tokens at positions
// pos0 .. pos0 + n - 1, after QK-norm and RoPE (done in Dart):
//
//   q      [n, heads * hd]
//   keys   [>= pos0 + n, kv_heads * hd]   the KV cache, already holding
//   values [>= pos0 + n, kv_heads * hd]   this chunk's rows
//   gate   [n, heads]                      pre-sigmoid output gate
//   out    [n, heads * hd]
//
// Query head h reads KV head h / (heads / kv_heads). window > 0 keeps the
// current token and the window - 1 before it (sliding layers); 0 is full
// causal attention. Every (token, head) pair is independent, so the pairs
// are split across threads. Scores and the softmax are float32, as in the
// float32 reference.

#ifdef LAB_NEON
static float dot_hd(const float* a, const float* b, int64_t hd) {
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
  int64_t d = 0;
  for (; d + 8 <= hd; d += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + d), vld1q_f32(b + d));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
  }
  float s = lane_sum(vaddq_f32(acc0, acc1));
  for (; d < hd; d++) s += a[d] * b[d];
  return s;
}
#else
// The low half of acc is NEON's acc0, the high half acc1.
static float dot_hd(const float* a, const float* b, int64_t hd) {
  __m256 acc = _mm256_setzero_ps();
  int64_t d = 0;
  for (; d + 8 <= hd; d += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
  }
  float s = lane_sum(_mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1)));
  for (; d < hd; d++) s += a[d] * b[d];
  return s;
}
#endif

static void attend_one(const float* q, const float* keys, const float* values,
                       const float* gate, float* out, float* scores, int64_t t,
                       int64_t h, int64_t pos0, int64_t heads, int64_t kv_heads,
                       int64_t hd, int64_t window) {
  const int64_t q_dim = heads * hd, kv_dim = kv_heads * hd;
  const int64_t pos = pos0 + t;
  const int64_t start = window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0;
  const int64_t kvo = (h / (heads / kv_heads)) * hd;
  const float* qv = q + t * q_dim + h * hd;
  const float scale = 1.0f / sqrtf((float)hd);
  float max = -INFINITY;
  for (int64_t s = start; s <= pos; s++) {
    const float v = dot_hd(qv, keys + s * kv_dim + kvo, hd) * scale;
    scores[s - start] = v;
    if (v > max) max = v;
  }
  float sum = 0.0f;
  for (int64_t s = start; s <= pos; s++) {
    const float e = expf(scores[s - start] - max);
    scores[s - start] = e;
    sum += e;
  }
  const float g = 1.0f / (1.0f + expf(-gate[t * heads + h])) / sum;
  float* o = out + t * q_dim + h * hd;
#ifdef LAB_NEON
  for (int64_t d = 0; d < hd; d += 4) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int64_t s = start; s <= pos; s++) {
      acc = vfmaq_n_f32(acc, vld1q_f32(values + s * kv_dim + kvo + d), scores[s - start]);
    }
    vst1q_f32(o + d, vmulq_n_f32(acc, g));
  }
#else
  // Every lane sums on its own, so eight at a time give NEON's four.
  int64_t d = 0;
  for (; d + 8 <= hd; d += 8) {
    __m256 acc = _mm256_setzero_ps();
    for (int64_t s = start; s <= pos; s++) {
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(values + s * kv_dim + kvo + d),
                            _mm256_set1_ps(scores[s - start]), acc);
    }
    _mm256_storeu_ps(o + d, _mm256_mul_ps(acc, _mm256_set1_ps(g)));
  }
  for (; d < hd; d += 4) {
    __m128 acc = _mm_setzero_ps();
    for (int64_t s = start; s <= pos; s++) {
      acc = _mm_fmadd_ps(_mm_loadu_ps(values + s * kv_dim + kvo + d),
                         _mm_set1_ps(scores[s - start]), acc);
    }
    _mm_storeu_ps(o + d, _mm_mul_ps(acc, _mm_set1_ps(g)));
  }
#endif
}

#ifdef _WIN32
typedef struct {
  const float *q, *keys, *values, *gate;
  float* out;
  int64_t tasks, longest, pos0, heads, kv_heads, hd, window;
} Attention;

static void attention_chunk(const void* job, int64_t c, int64_t chunks) {
  const Attention* a = (const Attention*)job;
  float* scores = (float*)malloc((size_t)a->longest * sizeof(float));
  for (int64_t k = a->tasks * c / chunks; k < a->tasks * (c + 1) / chunks; k++) {
    attend_one(a->q, a->keys, a->values, a->gate, a->out, scores, k / a->heads, k % a->heads,
               a->pos0, a->heads, a->kv_heads, a->hd, a->window);
  }
  free(scores);
}
#endif

LAB_EXPORT void lab_attention(const float* q, const float* keys, const float* values,
                              const float* gate, float* out, int64_t n, int64_t pos0,
                              int64_t heads, int64_t kv_heads, int64_t hd, int64_t window,
                              int64_t threads) {
  const int64_t tasks = n * heads;
  const int64_t workers = threads < 1 ? 1 : (threads > tasks ? tasks : threads);
  const int64_t longest = pos0 + n;
#ifdef _WIN32
  // Runs of (token, head) tasks to whichever thread is free, each run with
  // one scores buffer (see lab_gemm_q8).
  const Attention a = {q,     keys,    values, gate,     out, tasks,
                       longest, pos0, heads,  kv_heads, hd,  window};
  const int64_t runs = workers * CHUNKS_PER_THREAD;
  parallel_chunks(workers, runs < tasks ? runs : tasks, attention_chunk, &a);
#elif defined(__APPLE__)
  dispatch_apply((size_t)workers, DISPATCH_APPLY_AUTO, ^(size_t w) {
    float* scores = (float*)malloc((size_t)longest * sizeof(float));
    for (int64_t k = tasks * (int64_t)w / workers; k < tasks * ((int64_t)w + 1) / workers; k++) {
      attend_one(q, keys, values, gate, out, scores, k / heads, k % heads, pos0,
                 heads, kv_heads, hd, window);
    }
    free(scores);
  });
#else
  // One (token, head) task at a time to whichever thread is free (see
  // lab_gemm_q8).
#pragma omp parallel num_threads((int)workers)
  {
    float* scores = (float*)malloc((size_t)longest * sizeof(float));
#pragma omp for schedule(dynamic, 1)
    for (int64_t k = 0; k < tasks; k++) {
      attend_one(q, keys, values, gate, out, scores, k / heads, k % heads, pos0,
                 heads, kv_heads, hd, window);
    }
    free(scores);
  }
#endif
}

// The process's physical footprint in bytes, the figure iOS compares with
// an app's memory limit; on Windows, the working set; -1 where unavailable.
LAB_EXPORT int64_t lab_phys_footprint(void) {
#ifdef __APPLE__
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
    return -1;
  }
  return (int64_t)info.phys_footprint;
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) return -1;
  return (int64_t)counters.WorkingSetSize;
#else
  return -1;
#endif
}

// ---------------------------------------------------------------- Q4
//
// llama.cpp's Q4_0, the format of Liquid's quantization-aware distilled
// LFM3 checkpoint, with nibbles and scales in separate arrays so NEON and
// the GPU load whole aligned blocks. Per row, blocks of 32 weights share an
// IEEE half scale d and store q in 0..15, two per byte: weight k of a block
// in the low nibble of byte k, weight k + 16 in the high nibble (GGUF's
// order). The weight is d * (q - 8): 4.5 bits per weight.

enum { Q4_BLOCK = 32 };

#ifdef LAB_NEON
static inline float half_to_float(uint16_t bits) {
  _Float16 h;
  memcpy(&h, &bits, sizeof h);
  return (float)h;
}

static inline uint16_t float_to_half(float f) {
  const _Float16 h = (_Float16)f;
  uint16_t bits;
  memcpy(&bits, &h, sizeof bits);
  return bits;
}
#else
// F16C, rounding to nearest even as the conversion to _Float16 does.
static inline float half_to_float(uint16_t bits) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

static inline uint16_t float_to_half(float f) {
  return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
}
#endif

// llama.cpp's quantize_row_q4_0_ref: the weight of largest magnitude maps
// to q = 0, so d = that weight / -8.
LAB_EXPORT void lab_quantize_q4(const float* w, uint8_t* qs, uint16_t* scales, int64_t rows,
                                int64_t cols) {
  const int64_t blocks = cols / Q4_BLOCK;
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t b = 0; b < blocks; b++) {
      const float* src = w + r * cols + b * Q4_BLOCK;
      float amax = 0.0f, max = 0.0f;
      for (int k = 0; k < Q4_BLOCK; k++) {
        const float a = src[k] < 0 ? -src[k] : src[k];
        if (a > amax) {
          amax = a;
          max = src[k];
        }
      }
      const float d = max / -8.0f;
      const float id = d != 0.0f ? 1.0f / d : 0.0f;
      scales[r * blocks + b] = float_to_half(d);
      uint8_t* q = qs + (r * cols + b * Q4_BLOCK) / 2;
      for (int k = 0; k < Q4_BLOCK / 2; k++) {
        const int8_t lo = (int8_t)(src[k] * id + 8.5f);
        const int8_t hi = (int8_t)(src[k + 16] * id + 8.5f);
        q[k] = (uint8_t)((lo > 15 ? 15 : lo) | ((hi > 15 ? 15 : hi) << 4));
      }
    }
  }
}

#ifdef LAB_NEON
// Weights Q4, input float: widen each nibble to float and multiply.
static float dot_q4_f32(const uint8_t* row, const uint16_t* row_scales, const float* x,
                        int64_t blocks) {
  const uint8x16_t mask = vdupq_n_u8(0x0F);
  const int8x16_t eight = vdupq_n_s8(8);
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int64_t b = 0; b < blocks; b++) {
    const uint8x16_t v = vld1q_u8(row + b * 16);
    const int8x16_t halves[2] = {
        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight),
        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight),
    };
    float32x4_t s = vdupq_n_f32(0.0f);
    for (int h = 0; h < 2; h++) {
      const int16x8_t lo = vmovl_s8(vget_low_s8(halves[h]));
      const int16x8_t hi = vmovl_high_s8(halves[h]);
      const float* xb = x + b * Q4_BLOCK + 16 * h;
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(xb));
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_high_s16(lo)), vld1q_f32(xb + 4));
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(xb + 8));
      s = vfmaq_f32(s, vcvtq_f32_s32(vmovl_high_s16(hi)), vld1q_f32(xb + 12));
    }
    acc = vfmaq_n_f32(acc, s, half_to_float(row_scales[b]));
  }
  return lane_sum(acc);
}

// Weights Q4, input int8 (quantized per block as for Q8): the nibbles
// become int8 in -8..7 and go through sdot, as llama.cpp's q4_0 x q8_0 dot.
static float dot_q4_sdot(const uint8_t* row, const uint16_t* row_scales, const int8_t* xq,
                         const float* xs, int64_t blocks) {
  const uint8x16_t mask = vdupq_n_u8(0x0F);
  const int8x16_t eight = vdupq_n_s8(8);
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int64_t b = 0; b < blocks; b++) {
    const uint8x16_t v = vld1q_u8(row + b * 16);
    const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
    const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
    int32x4_t d = vdupq_n_s32(0);
    d = vdotq_s32(d, lo, vld1q_s8(xq + b * Q4_BLOCK));
    d = vdotq_s32(d, hi, vld1q_s8(xq + b * Q4_BLOCK + 16));
    acc = vfmaq_n_f32(acc, vcvtq_f32_s32(d), half_to_float(row_scales[b]) * xs[b]);
  }
  return lane_sum(acc);
}
#else
// A block's 16 bytes of nibbles as its 32 weights in -8..7, in order.
static inline __m256i q4_block(const uint8_t* bytes) {
  const __m128i v = _mm_loadu_si128((const __m128i*)bytes);
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m256i q = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), mask),
                                     _mm_and_si128(v, mask));
  return _mm256_sub_epi8(q, _mm256_set1_epi8(8));
}

static float dot_q4_f32(const uint8_t* row, const uint16_t* row_scales, const float* x,
                        int64_t blocks) {
  __m128 acc = _mm_setzero_ps();
  for (int64_t b = 0; b < blocks; b++) {
    const __m256i q = q4_block(row + b * 16);
    __m128 s = fma16_i8(_mm_setzero_ps(), _mm256_castsi256_si128(q), x + b * Q4_BLOCK);
    s = fma16_i8(s, _mm256_extracti128_si256(q, 1), x + b * Q4_BLOCK + 16);
    acc = _mm_fmadd_ps(s, _mm_set1_ps(half_to_float(row_scales[b])), acc);
  }
  return lane_sum(acc);
}

static float dot_q4_sdot(const uint8_t* row, const uint16_t* row_scales, const int8_t* xq,
                         const float* xs, int64_t blocks) {
  __m128 acc = _mm_setzero_ps();
  for (int64_t b = 0; b < blocks; b++) {
    const __m128i d =
        sdot_block(q4_block(row + b * 16), _mm256_loadu_si256((const __m256i*)(xq + b * Q4_BLOCK)));
    acc = _mm_fmadd_ps(_mm_cvtepi32_ps(d), _mm_set1_ps(half_to_float(row_scales[b]) * xs[b]), acc);
  }
  return lane_sum(acc);
}
#endif

static void q4_rows(const Product* p, int64_t r0, int64_t r1) {
  const int64_t rows = p->rows, cols = p->cols, blocks = cols / Q4_BLOCK;
  for (int64_t r = r0; r < r1; r++) {
    const uint8_t* row = (const uint8_t*)p->qs + r * cols / 2;
    const uint16_t* row_scales = (const uint16_t*)p->scales + r * blocks;
    for (int64_t t = 0; t < p->n; t++) {
      p->ys[t * rows + r] =
          p->mode == 1
              ? dot_q4_sdot(row, row_scales, p->xq + t * cols, p->xsc + t * blocks, blocks)
              : dot_q4_f32(row, row_scales, p->xs + t * cols, blocks);
    }
  }
}

#ifdef _WIN32
static void q4_chunk(const void* job, int64_t c, int64_t chunks) {
  const Product* p = (const Product*)job;
  q4_rows(p, boundary(p->rows, c, chunks), boundary(p->rows, c + 1, chunks));
}
#endif

// lab_gemm_q8 for Q4 weights; the same modes, scratch and thread split.
LAB_EXPORT void lab_gemm_q4(const uint8_t* qs, const uint16_t* scales, const float* xs,
                            int8_t* xq, float* xsc, float* ys, int64_t rows, int64_t cols,
                            int64_t n, int64_t threads, int64_t mode) {
  if (mode == 1) lab_quantize_q8(xs, xq, xsc, n, cols);
  const Product p = {qs, scales, xs, xq, xsc, ys, rows, cols, n, mode};
  const int64_t workers = threads < 1 ? 1 : threads;
#ifdef __APPLE__
  dispatch_apply((size_t)workers, DISPATCH_APPLY_AUTO, ^(size_t w) {
    q4_rows(&p, boundary(rows, (int64_t)w, workers), boundary(rows, (int64_t)w + 1, workers));
  });
#else
  const int64_t chunks = workers == 1 ? 1 : workers * CHUNKS_PER_THREAD;
#ifdef _WIN32
  parallel_chunks(workers, chunks, q4_chunk, &p);
#else
#pragma omp parallel for num_threads((int)workers) schedule(dynamic, 1)
  for (int64_t c = 0; c < chunks; c++) {
    q4_rows(&p, boundary(rows, c, chunks), boundary(rows, c + 1, chunks));
  }
#endif
#endif
}
