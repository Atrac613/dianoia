// Stage 5b on Android: the GPU forward pass through Vulkan, exporting the
// same C API as the Metal layer (stage5_native/metal/lab_metal.m), so the
// Dart orchestration (lfm3/lib/src/gpu.dart) drives either.
//
// Design, following the Metal layer:
//
// * Buffers are host-visible and coherent (on a phone's unified memory also
//   device-local), mapped once, so Dart writes inputs and reads logits in
//   place. Shaders get them as device addresses in push constants
//   (VK_KHR_buffer_device_address): no descriptor sets, and each kernel is a
//   pipeline, a push-constant block and a group count.
// * Dart opens a command buffer (lab_gpu_begin), records one dispatch per
//   call (lab_gpu_enc_*), each followed by a barrier so it sees the previous
//   one's writes (Metal's serial encoder), and submits it (lab_gpu_commit).
// * The weights are copied into one buffer: Metal wraps the mapped file
//   without a copy, but Vulkan core cannot.
// * Where the GPU has packed int8 dot products (VK_KHR_shader_integer_dot_
//   product), matrix inputs are quantized to int8 per block and the GEMMs
//   use them, as the CPU path does with sdot: phone GPUs lack the ALU
//   throughput to widen every 4-bit weight to float (Mali-G715 spent 57 ms
//   of a 74 ms token doing that). Inputs are quantized once per command
//   buffer and reused while nothing writes them again.
//
// The shaders (shaders/*.comp) are compiled to SPIR-V at build time and
// included as word arrays (see the plugin's CMakeLists.txt).
//
// The same layer runs on Windows. There the Vulkan loader, vulkan-1.dll,
// comes with the GPU driver rather than with the OS, so it is opened when
// the GPU starts instead of being linked: the CPU kernels in this library
// still load on a machine without it. A discrete GPU's buffers are its own
// memory mapped through the PCIe BAR, which needs Resizable BAR for a model
// to fit.

#ifdef _WIN32
#define VK_NO_PROTOTYPES
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "q8_vector_policy.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#define LOG(...) __android_log_print(ANDROID_LOG_ERROR, "lab_vulkan", __VA_ARGS__)
#else
#define LOG(...) fprintf(stderr, __VA_ARGS__)
#endif

// A DLL exports only what it is told to.
#ifdef _WIN32
#define LAB_EXPORT __declspec(dllexport)
#else
#define LAB_EXPORT
#endif

#ifdef _WIN32
// The entry points this layer calls besides vkCreateInstance, loaded from
// vulkan-1.dll under their usual names once there is an instance.
#define ENTRY_POINTS(X)                                                                   \
  X(vkAllocateCommandBuffers) X(vkAllocateMemory) X(vkBeginCommandBuffer)                 \
  X(vkBindBufferMemory) X(vkCmdBindPipeline) X(vkCmdDispatch) X(vkCmdPipelineBarrier)     \
  X(vkCmdPushConstants) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkCreateBuffer)   \
  X(vkCreateCommandPool) X(vkCreateComputePipelines) X(vkCreateDevice) X(vkCreateFence)   \
  X(vkCreatePipelineLayout) X(vkCreateQueryPool) X(vkCreateShaderModule)                  \
  X(vkDestroyBuffer) X(vkDestroyShaderModule) X(vkEndCommandBuffer)                       \
  X(vkEnumerateDeviceExtensionProperties) X(vkEnumeratePhysicalDevices) X(vkFreeMemory)   \
  X(vkGetBufferMemoryRequirements) X(vkGetDeviceProcAddr) X(vkGetDeviceQueue)             \
  X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties)      \
  X(vkGetQueryPoolResults) X(vkMapMemory) X(vkQueueSubmit) X(vkResetCommandBuffer)        \
  X(vkResetFences) X(vkWaitForFences)

#define DECLARE(name) static PFN_##name name;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
static PFN_vkCreateInstance vkCreateInstance;
ENTRY_POINTS(DECLARE)
#undef DECLARE

// Opens the loader; returns 0 if there is none.
static int load_loader(void) {
  if (vkCreateInstance != NULL) return 1;
  const HMODULE loader = LoadLibraryA("vulkan-1.dll");
  if (loader == NULL) return 0;
  vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)(void (*)(void))GetProcAddress(loader, "vkGetInstanceProcAddr");
  if (vkGetInstanceProcAddr == NULL) return 0;
  vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(NULL, "vkCreateInstance");
  return vkCreateInstance != NULL;
}

// Loads the other entry points through `instance`; returns 0 if any is missing.
static int load_entry_points(VkInstance instance) {
#define LOAD(name) \
  if ((name = (PFN_##name)vkGetInstanceProcAddr(instance, #name)) == NULL) return 0;
  ENTRY_POINTS(LOAD)
#undef LOAD
  return 1;
}
#endif

static const uint32_t kGemvQ8[] =
#include "gemv_q8.inc"
    ;
static const uint32_t kGemmQ8[] =
#include "gemm_q8.inc"
    ;
static const uint32_t kGemvQ4[] =
#include "gemv_q4.inc"
    ;
static const uint32_t kGemmQ4[] =
#include "gemm_q4.inc"
    ;
static const uint32_t kRmsNorm[] =
#include "rms_norm.inc"
    ;
static const uint32_t kHeadNormRope[] =
#include "head_norm_rope.inc"
    ;
static const uint32_t kConv[] =
#include "conv.inc"
    ;
static const uint32_t kAttention[] =
#include "attention.inc"
    ;
static const uint32_t kSwiglu[] =
#include "swiglu.inc"
    ;
static const uint32_t kCopy[] =
#include "copy.inc"
    ;
static const uint32_t kQuantize[] =
#include "quantize.inc"
    ;
static const uint32_t kGemmQ4I8[] =
#include "gemm_q4_i8.inc"
    ;
static const uint32_t kGemmQ8I8[] =
#include "gemm_q8_i8.inc"
    ;
static const uint32_t kGemvQ4I8[] =
#include "gemv_q4_i8.inc"
    ;
static const uint32_t kGemvQ8I8[] =
#include "gemv_q8_i8.inc"
    ;
static const uint32_t kGemvQ8I8Vec[] =
#include "gemv_q8_i8_vec.inc"
    ;

enum {
  kMaxBuffers = 4096,
  kGemvGroup = 128,      // invocations per workgroup of gemv / gemm
  kGemmRows = 4,         // GEMM_ROWS in common.glsl
  kGemmTokens = 8,       // GEMM_TOKENS
  kMaxKeys = 4096,       // MAX_KEYS in attention.comp
  kPushConstantBytes = 128,
  kMaxProfiled = 1024,   // dispatches timed per command buffer
};

// Kernel kinds, in the order lab_gpu_profile reports them (gpu.dart names
// them).
enum { kKindGemvQ8, kKindGemmQ8, kKindGemvQ4, kKindGemmQ4, kKindRmsNorm, kKindHeadNormRope,
       kKindConv, kKindAttention, kKindSwiglu, kKindCopy, kKindQuantize, kKindGemmQ8I8,
       kKindGemmQ4I8, kKindGemvQ8I8, kKindGemvQ4I8, kKindGemvQ8I8Vec, kKinds };

// Rows per subgroup the decode GEMVs are built for (a specialization
// constant); lab_gpu_set_decode_rows picks one.

typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void* contents;
  VkDeviceAddress address;
  VkDeviceSize size;
} Buffer;

static VkInstance gInstance;
static VkPhysicalDevice gPhysical;
static VkDevice gDevice;
static VkQueue gQueue;
static uint32_t gQueueFamily, gMemoryType, gReadbackType, gSubgroupSize;
static VkPhysicalDeviceProperties gProperties;
static VkDeviceSize gMaxBuffer, gHeapSize;
static VkPipelineLayout gLayout;
static VkPipeline gGemvQ8, gGemmQ8, gGemvQ4, gGemmQ4, gRmsNorm, gHeadNormRope, gConv,
    gAttention, gSwiglu, gCopy, gQuantize, gGemmQ8I8, gGemmQ4I8;
static const uint32_t kDecodeRows[] = {1, 2, 4};
static VkPipeline gGemvQ4I8[3], gGemvQ8I8[3][3];
static VkPipeline gGemvQ8I8Vec[3][3];
static int gQ8Vector = -1;  // -1: scoped auto; 0/1: explicit scalar/vector override
static uint64_t gAutoQ8Head[kMaxBuffers];  // registered head address per weight buffer
static const uint32_t kQ8Groups[] = {64, 128, 256};
static int gQ8Group = 1;  // 128 invocations, matching the original shader
static int gDecodeRows = 2;  // index into kDecodeRows; four rows per subgroup
static int gIntDot;  // packed int8 dot products available and in use

// Matrix inputs quantized to int8 (quantize.comp), and which input the
// scratch holds: valid until anything writes that buffer again (its
// generation) or the command buffer ends.
static int64_t gXq = -1, gXs = -1, gXCapacity;
static struct {
  int valid;
  int64_t handle, offset, n, cols;
  uint64_t generation;
} gQuantized;
static uint64_t gGeneration[kMaxBuffers];
static VkCommandPool gPool;
static VkCommandBuffer gCmd;
static VkFence gFence;
static VkQueryPool gQueries;
static int gPending;  // a submitted command buffer not yet waited for
static int gUseGemm = 1;  // decode through the GEMM kernels (see matmul)
// Profiling: a timestamp after every dispatch, summed per kernel kind.
static int gProfiling, gProfiled;
static uint8_t gProfiledKinds[kMaxProfiled];
static double gKindMs[kKinds];
static int64_t gKindCount[kKinds];
static double gLastGpuMs;
static Buffer gBuffers[kMaxBuffers];
static int64_t gBufferCount;
// Handles freed by lab_gpu_release, handed out again before new ones.
static int64_t gFree[kMaxBuffers], gFreeCount, gLive;

// Vulkan 1.1 and 1.2 entry points: Android's libvulkan stubs export only
// 1.0, so these come from the loader at run time.
static PFN_vkGetPhysicalDeviceProperties2 pGetPhysicalDeviceProperties2;
static PFN_vkGetPhysicalDeviceFeatures2 pGetPhysicalDeviceFeatures2;
static PFN_vkGetBufferDeviceAddress pGetBufferDeviceAddress;

#define CHECK(call)                                     \
  do {                                                  \
    const VkResult r_ = (call);                         \
    if (r_ != VK_SUCCESS) {                             \
      LOG("%s failed: %d\n", #call, (int)r_);           \
      return 0;                                         \
    }                                                   \
  } while (0)

static VkPipeline pipeline_with(const uint32_t* code, size_t bytes, const uint32_t* constant);
static VkPipeline pipeline_constants(const uint32_t* code, size_t bytes,
                                     const uint32_t* constants, uint32_t count);

static VkPipeline pipeline(const uint32_t* code, size_t bytes) {
  return pipeline_with(code, bytes, NULL);
}

// A compute pipeline; `constant`, if given, sets specialization constant 0.
static VkPipeline pipeline_with(const uint32_t* code, size_t bytes, const uint32_t* constant) {
  return pipeline_constants(code, bytes, constant, constant ? 1 : 0);
}

// Specialization constants 0 (rows) and optionally 1 (Q8 workgroup size).
static VkPipeline pipeline_constants(const uint32_t* code, size_t bytes,
                                     const uint32_t* constants, uint32_t count) {
  const VkShaderModuleCreateInfo moduleInfo = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
  VkShaderModule module;
  if (vkCreateShaderModule(gDevice, &moduleInfo, NULL, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
  const VkSpecializationMapEntry entries[] = {
      {.constantID = 0, .offset = 0, .size = sizeof(uint32_t)},
      {.constantID = 1, .offset = sizeof(uint32_t), .size = sizeof(uint32_t)}};
  const VkSpecializationInfo specialization = {
      .mapEntryCount = count, .pMapEntries = entries,
      .dataSize = count * sizeof(uint32_t), .pData = constants};
  const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
                .pSpecializationInfo = count ? &specialization : NULL},
      .layout = gLayout};
  VkPipeline result = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(gDevice, VK_NULL_HANDLE, 1, &info, NULL, &result) != VK_SUCCESS) {
    LOG("compute pipeline failed\n");
  }
  vkDestroyShaderModule(gDevice, module, NULL);
  return result;
}

// Returns 1 on success, 0 if there is no usable Vulkan device (logged).
LAB_EXPORT int64_t lab_gpu_init(void) {
  if (gDevice != VK_NULL_HANDLE) return 1;
  const VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                 .pApplicationName = "dart-llm-lab",
                                 .apiVersion = VK_API_VERSION_1_2};
  const VkInstanceCreateInfo instanceInfo = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                             .pApplicationInfo = &app};
#ifdef _WIN32
  if (!load_loader()) {
    LOG("no Vulkan loader (vulkan-1.dll)\n");
    return 0;
  }
#endif
  CHECK(vkCreateInstance(&instanceInfo, NULL, &gInstance));
#ifdef _WIN32
  if (!load_entry_points(gInstance)) {
    LOG("the Vulkan loader lacks an entry point\n");
    return 0;
  }
#endif
  uint32_t count = 1;
  const VkResult enumerated = vkEnumeratePhysicalDevices(gInstance, &count, &gPhysical);
  if ((enumerated != VK_SUCCESS && enumerated != VK_INCOMPLETE) || count == 0) {
    LOG("no Vulkan device\n");
    return 0;
  }

  pGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
      gInstance, "vkGetPhysicalDeviceProperties2");
  pGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(
      gInstance, "vkGetPhysicalDeviceFeatures2");
  if (pGetPhysicalDeviceProperties2 == NULL || pGetPhysicalDeviceFeatures2 == NULL) {
    LOG("Vulkan 1.1 is required\n");
    return 0;
  }
  VkPhysicalDeviceSubgroupProperties subgroup = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceMaintenance3Properties maintenance3 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, .pNext = &subgroup};
  VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            .pNext = &maintenance3};
  pGetPhysicalDeviceProperties2(gPhysical, &properties);
  gProperties = properties.properties;
  gSubgroupSize = subgroup.subgroupSize;
  gMaxBuffer = maintenance3.maxMemoryAllocationSize;
  const VkSubgroupFeatureFlags needed =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  if ((subgroup.supportedOperations & needed) != needed ||
      !(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
      kGemvGroup % gSubgroupSize != 0) {
    LOG("subgroup arithmetic in compute is required\n");
    return 0;
  }

  // Host-visible, coherent memory; device-local too where there is such a
  // type (a phone's unified memory).
  VkPhysicalDeviceMemoryProperties memory;
  vkGetPhysicalDeviceMemoryProperties(gPhysical, &memory);
  const VkMemoryPropertyFlags host =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  gMemoryType = UINT32_MAX;
  for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
    const VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
    if ((flags & host) != host) continue;
    if (gMemoryType == UINT32_MAX || (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      gMemoryType = i;
      gHeapSize = memory.memoryHeaps[memory.memoryTypes[i].heapIndex].size;
      if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) break;
    }
  }
  if (gMemoryType == UINT32_MAX) {
    LOG("no host-visible coherent memory\n");
    return 0;
  }
  // A discrete GPU's memory is its own, which the host reads uncached
  // across PCIe (a token's logits took 150 ms on a Radeon RX 6600 XT), so
  // what the host reads every token goes to the host's cached memory
  // instead, which the GPU writes across PCIe (lab_gpu_readback_buffer).
  gReadbackType = gMemoryType;
  if (gProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    const VkMemoryPropertyFlags cached = host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
      const VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
      if ((flags & cached) == cached && !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        gReadbackType = i;
        break;
      }
    }
  }

  uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(gPhysical, &families, NULL);
  VkQueueFamilyProperties familyProperties[16];
  families = families > 16 ? 16 : families;
  vkGetPhysicalDeviceQueueFamilyProperties(gPhysical, &families, familyProperties);
  gQueueFamily = UINT32_MAX;
  for (uint32_t i = 0; i < families && gQueueFamily == UINT32_MAX; i++) {
    if (familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) gQueueFamily = i;
  }
  if (gQueueFamily == UINT32_MAX || familyProperties[gQueueFamily].timestampValidBits == 0) {
    LOG("no compute queue with timestamps\n");
    return 0;
  }

  const float priority = 1.0f;
  const VkDeviceQueueCreateInfo queueInfo = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                             .queueFamilyIndex = gQueueFamily,
                                             .queueCount = 1,
                                             .pQueuePriorities = &priority};
  // Packed int8 dot products: an extension until Vulkan 1.3.
  VkPhysicalDeviceShaderIntegerDotProductFeatures intDot = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
  VkPhysicalDeviceFeatures2 supported = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                         .pNext = &intDot};
  pGetPhysicalDeviceFeatures2(gPhysical, &supported);
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(gPhysical, NULL, &extensionCount, NULL);
  VkExtensionProperties extensions[512];
  extensionCount = extensionCount > 512 ? 512 : extensionCount;
  vkEnumerateDeviceExtensionProperties(gPhysical, NULL, &extensionCount, extensions);
  int hasIntDotExtension = 0;
  for (uint32_t i = 0; i < extensionCount; i++) {
    if (strcmp(extensions[i].extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) {
      hasIntDotExtension = 1;
    }
  }
  gIntDot = hasIntDotExtension && intDot.shaderIntegerDotProduct;
  const char* enabled[] = {VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME};
  intDot.pNext = NULL;
  VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = gIntDot ? &intDot : NULL,
      .bufferDeviceAddress = VK_TRUE};
  const VkDeviceCreateInfo deviceInfo = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                         .pNext = &features12,
                                         .queueCreateInfoCount = 1,
                                         .pQueueCreateInfos = &queueInfo,
                                         .enabledExtensionCount = gIntDot ? 1 : 0,
                                         .ppEnabledExtensionNames = enabled};
  CHECK(vkCreateDevice(gPhysical, &deviceInfo, NULL, &gDevice));
  vkGetDeviceQueue(gDevice, gQueueFamily, 0, &gQueue);
  pGetBufferDeviceAddress =
      (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(gDevice, "vkGetBufferDeviceAddress");
  if (pGetBufferDeviceAddress == NULL) {
    LOG("Vulkan 1.2 buffer device addresses are required\n");
    return 0;
  }

  const VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                     .size = kPushConstantBytes};
  const VkPipelineLayoutCreateInfo layoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &range};
  CHECK(vkCreatePipelineLayout(gDevice, &layoutInfo, NULL, &gLayout));
#define PIPELINE(name, code) \
  if ((name = pipeline(code, sizeof code)) == VK_NULL_HANDLE) return 0
  PIPELINE(gGemvQ8, kGemvQ8);
  PIPELINE(gGemmQ8, kGemmQ8);
  PIPELINE(gGemvQ4, kGemvQ4);
  PIPELINE(gGemmQ4, kGemmQ4);
  PIPELINE(gRmsNorm, kRmsNorm);
  PIPELINE(gHeadNormRope, kHeadNormRope);
  PIPELINE(gConv, kConv);
  PIPELINE(gAttention, kAttention);
  PIPELINE(gSwiglu, kSwiglu);
  PIPELINE(gCopy, kCopy);
  if (gIntDot) {
    PIPELINE(gQuantize, kQuantize);
    PIPELINE(gGemmQ8I8, kGemmQ8I8);
    PIPELINE(gGemmQ4I8, kGemmQ4I8);
    for (int i = 0; i < 3; i++) {
      gGemvQ4I8[i] = pipeline_with(kGemvQ4I8, sizeof kGemvQ4I8, &kDecodeRows[i]);
      if (gGemvQ4I8[i] == VK_NULL_HANDLE) return 0;
      for (int j = 0; j < 3; j++) {
        if (kQ8Groups[j] < gSubgroupSize || kQ8Groups[j] % gSubgroupSize != 0 ||
            kQ8Groups[j] > gProperties.limits.maxComputeWorkGroupInvocations ||
            kQ8Groups[j] > gProperties.limits.maxComputeWorkGroupSize[0]) continue;
        const uint32_t constants[] = {kDecodeRows[i], kQ8Groups[j]};
        gGemvQ8I8[i][j] = pipeline_constants(kGemvQ8I8, sizeof kGemvQ8I8, constants, 2);
        if (gGemvQ8I8[i][j] == VK_NULL_HANDLE) return 0;
        gGemvQ8I8Vec[i][j] = pipeline_constants(kGemvQ8I8Vec, sizeof kGemvQ8I8Vec, constants, 2);
        if (gGemvQ8I8Vec[i][j] == VK_NULL_HANDLE) return 0;
      }
      if (gGemvQ8I8[i][gQ8Group] == VK_NULL_HANDLE) return 0;
    }
  }
#undef PIPELINE

  const VkCommandPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = gQueueFamily};
  CHECK(vkCreateCommandPool(gDevice, &poolInfo, NULL, &gPool));
  const VkCommandBufferAllocateInfo cmdInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = gPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  CHECK(vkAllocateCommandBuffers(gDevice, &cmdInfo, &gCmd));
  const VkFenceCreateInfo fenceInfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  CHECK(vkCreateFence(gDevice, &fenceInfo, NULL, &gFence));
  const VkQueryPoolCreateInfo queryInfo = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                           .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                           .queryCount = 2 + kMaxProfiled};
  CHECK(vkCreateQueryPool(gDevice, &queryInfo, NULL, &gQueries));
  return 1;
}

// Copies the device name into out (UTF-8, NUL-terminated).
LAB_EXPORT void lab_gpu_device_name(char* out, int64_t capacity) {
  snprintf(out, (size_t)capacity, "%s", gProperties.deviceName);
}

LAB_EXPORT int64_t lab_gpu_max_buffer_length(void) { return (int64_t)gMaxBuffer; }

// What gpu.dart checks maxSeq against: the attention kernel's scores live in
// a fixed shared array of kMaxKeys floats (plus 128 bytes, as on Metal).
LAB_EXPORT int64_t lab_gpu_max_shared_memory(void) { return kMaxKeys * 4 + 128; }

// The heap of the buffers' memory type: on a phone, the shared RAM.
LAB_EXPORT int64_t lab_gpu_recommended_working_set(void) { return (int64_t)gHeapSize; }

// NSProcessInfo's thermal state on Apple; Vulkan has nothing like it.
LAB_EXPORT int64_t lab_gpu_thermal_state(void) { return -1; }

LAB_EXPORT int64_t lab_gpu_physical_memory(void) {
#ifdef _WIN32
  MEMORYSTATUSEX status = {.dwLength = sizeof status};
  return GlobalMemoryStatusEx(&status) ? (int64_t)status.ullTotalPhys : -1;
#else
  return (int64_t)sysconf(_SC_PHYS_PAGES) * (int64_t)sysconf(_SC_PAGESIZE);
#endif
}

static int64_t create_buffer(VkDeviceSize bytes, uint32_t memoryType) {
  if (bytes == 0) return -1;
  int64_t handle;
  if (gFreeCount > 0) {
    handle = gFree[gFreeCount - 1];
  } else if (gBufferCount < kMaxBuffers) {
    handle = gBufferCount;
  } else {
    return -1;
  }
  Buffer* b = &gBuffers[handle];
  const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  if (vkCreateBuffer(gDevice, &info, NULL, &b->buffer) != VK_SUCCESS) return -1;
  VkMemoryRequirements needs;
  vkGetBufferMemoryRequirements(gDevice, b->buffer, &needs);
  if (!(needs.memoryTypeBits & (1u << memoryType))) return -1;
  const VkMemoryAllocateFlagsInfo flags = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                           .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
  const VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                      .pNext = &flags,
                                      .allocationSize = needs.size,
                                      .memoryTypeIndex = memoryType};
  if (vkAllocateMemory(gDevice, &alloc, NULL, &b->memory) != VK_SUCCESS) {
    vkDestroyBuffer(gDevice, b->buffer, NULL);
    return -1;
  }
  if (vkBindBufferMemory(gDevice, b->buffer, b->memory, 0) != VK_SUCCESS ||
      vkMapMemory(gDevice, b->memory, 0, VK_WHOLE_SIZE, 0, &b->contents) != VK_SUCCESS) {
    vkFreeMemory(gDevice, b->memory, NULL);
    vkDestroyBuffer(gDevice, b->buffer, NULL);
    return -1;
  }
  const VkBufferDeviceAddressInfo addressInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b->buffer};
  b->address = pGetBufferDeviceAddress(gDevice, &addressInfo);
  b->size = bytes;
  if (handle == gBufferCount) {
    gBufferCount++;
  } else {
    gFreeCount--;
  }
  gLive++;
  return handle;
}

// A host-visible buffer of `bytes` zeroed bytes; returns its handle, or -1.
LAB_EXPORT int64_t lab_gpu_buffer(int64_t bytes) {
  const int64_t handle = create_buffer((VkDeviceSize)bytes, gMemoryType);
  if (handle >= 0) memset(gBuffers[handle].contents, 0, (size_t)bytes);
  return handle;
}

// The same for a buffer the host reads after every command buffer, such as
// the logits: in the host's cached memory on a discrete GPU.
LAB_EXPORT int64_t lab_gpu_readback_buffer(int64_t bytes) {
  const int64_t handle = create_buffer((VkDeviceSize)bytes, gReadbackType);
  if (handle >= 0) memset(gBuffers[handle].contents, 0, (size_t)bytes);
  return handle;
}

// A buffer holding a copy of `bytes` bytes at `address` (on Metal, a wrap
// without a copy). Returns the handle, or -1.
LAB_EXPORT int64_t lab_gpu_wrap(void* address, int64_t bytes) {
  const int64_t handle = create_buffer((VkDeviceSize)bytes, gMemoryType);
  if (handle >= 0) memcpy(gBuffers[handle].contents, address, (size_t)bytes);
  return handle;
}

LAB_EXPORT void* lab_gpu_contents(int64_t handle) { return gBuffers[handle].contents; }

// GPU execution time of the last waited command buffer, in milliseconds.
LAB_EXPORT double lab_gpu_last_ms(void) { return gLastGpuMs; }

static int wait_pending(void) {
  if (!gPending) return 1;
  gPending = 0;
  if (vkWaitForFences(gDevice, 1, &gFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return 0;
  vkResetFences(gDevice, 1, &gFence);
  uint64_t stamps[2 + kMaxProfiled];
  const uint32_t count = 2 + (uint32_t)gProfiled;
  if (vkGetQueryPoolResults(gDevice, gQueries, 0, count, sizeof stamps, stamps, sizeof stamps[0],
                            VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
    const double ns = gProperties.limits.timestampPeriod;
    gLastGpuMs = (double)(stamps[1] - stamps[0]) * ns / 1e6;
    // Each dispatch's share: from the previous timestamp to its own, which
    // includes the barrier after it.
    for (int i = 0; i < gProfiled; i++) {
      const uint64_t from = i == 0 ? stamps[0] : stamps[2 + i - 1];
      gKindMs[gProfiledKinds[i]] += (double)(stamps[2 + i] - from) * ns / 1e6;
      gKindCount[gProfiledKinds[i]]++;
    }
  }
  return 1;
}

// Destroys a buffer and frees its handle for reuse, after the GPU is done
// with any submitted work. Whatever was keyed by the handle goes with it: a
// cached quantization of its contents and a registered Q8 head.
LAB_EXPORT void lab_gpu_release(int64_t handle) {
  if (handle < 0 || handle >= gBufferCount || gBuffers[handle].buffer == VK_NULL_HANDLE) return;
  wait_pending();
  Buffer* b = &gBuffers[handle];
  vkDestroyBuffer(gDevice, b->buffer, NULL);
  vkFreeMemory(gDevice, b->memory, NULL);  // unmaps it too
  memset(b, 0, sizeof *b);
  gGeneration[handle]++;  // a quantization cached from it goes stale
  if (gQuantized.handle == handle) gQuantized.valid = 0;
  gAutoQ8Head[handle] = 0;
  gFree[gFreeCount++] = handle;
  gLive--;
}

// Buffers not yet released, for leak checks.
LAB_EXPORT int64_t lab_gpu_live_buffers(void) { return gLive; }

// Opens the command buffer that the encode calls record into.
LAB_EXPORT void lab_gpu_begin(void) {
  wait_pending();
  vkResetCommandBuffer(gCmd, 0);
  const VkCommandBufferBeginInfo info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(gCmd, &info);
  vkCmdResetQueryPool(gCmd, gQueries, 0, 2 + kMaxProfiled);
  vkCmdWriteTimestamp(gCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, gQueries, 0);
  gProfiled = 0;
  gQuantized.valid = 0;  // the host may have rewritten any input since
}

// Rows per subgroup of the decode GEMVs: 1, 2 or 4; 0 selects the old
// int8 GEMM for a same-device numerical and performance reference.
LAB_EXPORT void lab_gpu_set_decode_rows(int64_t rows) {
  if (rows == 0) gDecodeRows = -1;
  for (int i = 0; i < 3; i++) {
    if (kDecodeRows[i] == (uint32_t)rows) gDecodeRows = i;
  }
}

// Q8 decode only. Return zero for unsupported sizes, without changing it.
LAB_EXPORT int64_t lab_gpu_set_q8_group(int64_t invocations) {
  for (int i = 0; i < 3; i++) {
    if (invocations == kQ8Groups[i] && gGemvQ8I8[2][i] != VK_NULL_HANDLE) {
      gQ8Group = i;
      return invocations;
    }
  }
  return 0;
}

// -1 restores scoped auto, 0/1 force scalar/vector. Returns 1 on success;
// this means the mode is available, not that auto selected a vector head.
LAB_EXPORT int64_t lab_gpu_set_q8_vector(int64_t on) {
  if (on < -1 || on > 1 || gGemvQ8I8Vec[2][gQ8Group] == VK_NULL_HANDLE) return 0;
  gQ8Vector = (int)on;
  return 1;
}

// Only Dart knows whether the loaded cache's body is Q4. Register its head
// by address, so another model's Q8 products cannot inherit this selection.
LAB_EXPORT int64_t lab_gpu_register_q8_head(int64_t w, int64_t qs, int64_t rows, int64_t cols,
                                           int64_t q4Body) {
#ifdef __ANDROID__
  char manufacturer[PROP_VALUE_MAX] = {0}, model[PROP_VALUE_MAX] = {0};
  __system_property_get("ro.product.manufacturer", manufacturer);
  __system_property_get("ro.product.model", model);
  if (!gIntDot || w < 0 || w >= gBufferCount || qs < 0 ||
      !lab_q8_vector_head_eligible(manufacturer, model, gProperties.deviceName,
                                   q4Body != 0, rows, cols) ||
      (uint64_t)qs + (uint64_t)rows * (uint64_t)cols > gBuffers[w].size ||
      (gBuffers[w].address + (uint64_t)qs) % 16 != 0 ||
      gGemvQ8I8Vec[2][1] == VK_NULL_HANDLE) return 0;
  gAutoQ8Head[w] = gBuffers[w].address + (uint64_t)qs;
  return 1;
#else
  (void)w; (void)qs; (void)rows; (void)cols; (void)q4Body;
  return 0;
#endif
}

// Times every dispatch from now on (1) or stops (0).
LAB_EXPORT void lab_gpu_set_profiling(int64_t on) { gProfiling = on != 0; }

// Copies into ms and counts, per kernel kind (kKinds of them, the order of
// the enum above), the GPU time and dispatches since the last call, and
// starts over. Returns the number of kinds.
LAB_EXPORT int64_t lab_gpu_profile(double* ms, int64_t* counts, int64_t kinds) {
  for (int64_t k = 0; k < kinds && k < kKinds; k++) {
    ms[k] = gKindMs[k];
    counts[k] = gKindCount[k];
    gKindMs[k] = 0;
    gKindCount[k] = 0;
  }
  return kKinds;
}

// Ends recording and submits. With `wait`, blocks until the GPU is done and
// returns 0 if anything failed (logged), else 1.
LAB_EXPORT int64_t lab_gpu_commit(int64_t wait) {
  // The host reads results after the wait.
  const VkMemoryBarrier toHost = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
  vkCmdPipelineBarrier(gCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                       &toHost, 0, NULL, 0, NULL);
  vkCmdWriteTimestamp(gCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, gQueries, 1);
  CHECK(vkEndCommandBuffer(gCmd));
  const VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                               .commandBufferCount = 1,
                               .pCommandBuffers = &gCmd};
  CHECK(vkQueueSubmit(gQueue, 1, &submit, gFence));
  gPending = 1;
  if (!wait) return 1;
  if (!wait_pending()) {
    LOG("waiting for the GPU failed\n");
    return 0;
  }
  return 1;
}

// Records one dispatch and the barrier after it, so the next dispatch sees
// its writes.
static void dispatch(int kind, VkPipeline pipe, const void* params, uint32_t bytes, uint32_t x,
                     uint32_t y) {
  vkCmdBindPipeline(gCmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdPushConstants(gCmd, gLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, params);
  vkCmdDispatch(gCmd, x, y, 1);
  const VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
  vkCmdPipelineBarrier(gCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
  if (gProfiling && gProfiled < kMaxProfiled) {
    vkCmdWriteTimestamp(gCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, gQueries,
                        2 + (uint32_t)gProfiled);
    gProfiledKinds[gProfiled++] = (uint8_t)kind;
  }
}

static uint64_t at(int64_t handle, int64_t offset) {
  return gBuffers[handle].address + (uint64_t)offset;
}

static uint32_t groups(int64_t items, int64_t per) { return (uint32_t)((items + per - 1) / per); }

// Records that a dispatch writes buffer `handle`, so a cached quantization
// of it goes stale.
static void wrote(int64_t handle) { gGeneration[handle]++; }

typedef struct {
  uint64_t x, xq, xs;
  uint32_t blocks, pad;
} QuantizeParams;

// Quantizes n rows of cols floats at x + xOffset into the int8 scratch,
// unless the scratch already holds them.
static void quantize_input(int64_t x, int64_t xOffset, int64_t n, int64_t cols) {
  if (gQuantized.valid && gQuantized.handle == x && gQuantized.offset == xOffset &&
      gQuantized.n == n && gQuantized.cols == cols && gQuantized.generation == gGeneration[x]) {
    return;
  }
  const int64_t values = n * cols;
  if (values > gXCapacity) {
    gXCapacity = values > (1 << 20) ? values : (1 << 20);
    gXq = lab_gpu_buffer(gXCapacity);
    gXs = lab_gpu_buffer(gXCapacity / 32 * 4);
  }
  const QuantizeParams p = {at(x, xOffset), at(gXq, 0), at(gXs, 0), (uint32_t)(values / 32), 0};
  dispatch(kKindQuantize, gQuantize, &p, sizeof p, groups(values / 32, 64), 1);
  gQuantized.valid = 1;
  gQuantized.handle = x;
  gQuantized.offset = xOffset;
  gQuantized.n = n;
  gQuantized.cols = cols;
  gQuantized.generation = gGeneration[x];
}

typedef struct {
  uint64_t qs, scales, xq, xs, y;
  uint32_t rows, cols, n, pad;
} MatmulI8Params;

typedef struct {
  uint64_t qs, scales, x, y;
  uint32_t rows, cols, n, pad;
} MatmulParams;

static void matmul(int kind, VkPipeline gemv, VkPipeline gemm, VkPipeline gemmI8,
                   int64_t w, int64_t qsOffset, int64_t scalesOffset, int64_t x, int64_t xOffset,
                   int64_t y, int64_t yOffset, int64_t rows, int64_t cols, int64_t n) {
  const int64_t subgroups = kGemvGroup / gSubgroupSize;
  wrote(y);
  if (gIntDot) {
    quantize_input(x, xOffset, n, cols);
    const MatmulI8Params p = {at(w, qsOffset), at(w, scalesOffset), at(gXq, 0), at(gXs, 0),
                              at(y, yOffset), (uint32_t)rows, (uint32_t)cols, (uint32_t)n, 0};
    const int q8 = kind == kKindGemvQ8;
    if (n == 1 && gDecodeRows >= 0) {
      const int64_t decodeSubgroups = q8 ? kQ8Groups[gQ8Group] / gSubgroupSize : subgroups;
      const int vector = q8 && (gQ8Vector >= 0 ? gQ8Vector :
          (gDecodeRows == 2 && gQ8Group == 1 && rows == 128000 && cols == 2048 &&
           gAutoQ8Head[w] == at(w, qsOffset)));
      dispatch(q8 ? (vector ? kKindGemvQ8I8Vec : kKindGemvQ8I8) : kKindGemvQ4I8,
               q8 ? (vector ? gGemvQ8I8Vec : gGemvQ8I8)[gDecodeRows][gQ8Group]
                  : gGemvQ4I8[gDecodeRows], &p,
               sizeof p, groups(rows, decodeSubgroups * kDecodeRows[gDecodeRows]), 1);
    } else {
      dispatch(q8 ? kKindGemmQ8I8 : kKindGemmQ4I8, gemmI8, &p, sizeof p,
               groups(rows, subgroups * kGemmRows), groups(n, kGemmTokens));
    }
    return;
  }
  const MatmulParams p = {at(w, qsOffset), at(w, scalesOffset), at(x, xOffset), at(y, yOffset),
                          (uint32_t)rows, (uint32_t)cols, (uint32_t)n, 0};
  // Decode uses the GEMM too: a subgroup then computes kGemmRows rows and
  // applies each loaded input block to all of them. The one-row GEMV
  // reloaded the whole input for every row, about four bytes of input per
  // weight, which Mali's caches could not feed (95 ms per token on a
  // Pixel 8). Each (row, token) is computed the same way in both kernels.
  if (n == 1 && !gUseGemm) {
    dispatch(kind, gemv, &p, sizeof p, groups(rows, subgroups), 1);
  } else {
    dispatch(kind + 1, gemm, &p, sizeof p, groups(rows, subgroups * kGemmRows),
             groups(n, kGemmTokens));
  }
}

// Y = X W^T for n tokens, W in Q8 inside buffer w at the given offsets:
// gemv_q8 for one token, else gemm_q8.
LAB_EXPORT void lab_gpu_enc_matmul_q8(int64_t w, int64_t qsOffset, int64_t scalesOffset,
                                      int64_t x, int64_t xOffset, int64_t y, int64_t yOffset,
                                      int64_t rows, int64_t cols, int64_t n) {
  matmul(kKindGemvQ8, gGemvQ8, gGemmQ8, gGemmQ8I8, w, qsOffset, scalesOffset, x, xOffset, y, yOffset, rows, cols, n);
}

// The same for Q4 weights.
LAB_EXPORT void lab_gpu_enc_matmul_q4(int64_t w, int64_t qsOffset, int64_t scalesOffset,
                                      int64_t x, int64_t xOffset, int64_t y, int64_t yOffset,
                                      int64_t rows, int64_t cols, int64_t n) {
  matmul(kKindGemvQ4, gGemvQ4, gGemmQ4, gGemmQ4I8, w, qsOffset, scalesOffset, x, xOffset, y, yOffset, rows, cols, n);
}

typedef struct {
  uint64_t x, w, out;
  uint32_t cols;
  float eps;
  uint32_t accumulate, pad;
} NormParams;

LAB_EXPORT void lab_gpu_enc_rms_norm(int64_t x, int64_t xOffset, int64_t w, int64_t wOffset,
                                     int64_t out, int64_t outOffset, int64_t rows, int64_t cols,
                                     double eps, int64_t accumulate) {
  const NormParams p = {at(x, xOffset), at(w, wOffset), at(out, outOffset), (uint32_t)cols,
                        (float)eps, accumulate != 0, 0};
  dispatch(kKindRmsNorm, gRmsNorm, &p, sizeof p, (uint32_t)rows, 1);
  wrote(out);
}

typedef struct {
  uint64_t x, w, rope;
  uint32_t hd;
  float eps;
  uint32_t xStride, ropeStride;
} HeadParams;

// The kernel's workgroup is 64 = hd / 2 invocations: head_dim 128.
LAB_EXPORT void lab_gpu_enc_head_norm_rope(int64_t x, int64_t xOffset, int64_t xStride, int64_t w,
                                           int64_t wOffset, int64_t rope, int64_t ropeOffset,
                                           int64_t ropeStride, int64_t heads, int64_t hd, int64_t n,
                                           double eps) {
  const HeadParams p = {at(x, xOffset),  at(w, wOffset),        at(rope, ropeOffset),
                        (uint32_t)hd,    (float)eps,            (uint32_t)xStride,
                        (uint32_t)ropeStride};
  dispatch(kKindHeadNormRope, gHeadNormRope, &p, sizeof p, (uint32_t)heads, (uint32_t)n);
  wrote(x);
}

typedef struct {
  uint64_t bcx, weight, history, y;
  uint32_t width, k, n, pad;
} ConvParams;

LAB_EXPORT void lab_gpu_enc_conv(int64_t bcx, int64_t weight, int64_t weightOffset, int64_t history,
                                 int64_t y, int64_t width, int64_t k, int64_t n) {
  const ConvParams p = {at(bcx, 0),         at(weight, weightOffset), at(history, 0), at(y, 0),
                        (uint32_t)width,    (uint32_t)k,              (uint32_t)n,    0};
  dispatch(kKindConv, gConv, &p, sizeof p, groups(width, 256), 1);
  wrote(history);
  wrote(y);
}

typedef struct {
  uint64_t q, keys, values, gate, out;
  uint32_t position, window, heads, hd, kvDim, group;
  float scale;
  uint32_t pad;
} AttentionParams;

LAB_EXPORT void lab_gpu_enc_attention(int64_t q, int64_t keys, int64_t values, int64_t gate,
                                      int64_t out, int64_t position, int64_t n, int64_t heads,
                                      int64_t kvHeads, int64_t hd, int64_t window) {
  const AttentionParams p = {at(q, 0),
                             at(keys, 0),
                             at(values, 0),
                             at(gate, 0),
                             at(out, 0),
                             (uint32_t)position,
                             (uint32_t)window,
                             (uint32_t)heads,
                             (uint32_t)hd,
                             (uint32_t)(kvHeads * hd),
                             (uint32_t)(heads / kvHeads),
                             (float)(1.0 / sqrt((double)hd)),
                             0};
  dispatch(kKindAttention, gAttention, &p, sizeof p, (uint32_t)heads, (uint32_t)n);
  wrote(out);
}

typedef struct {
  uint64_t g, u;
  uint32_t count, pad;
} SwigluParams;

LAB_EXPORT void lab_gpu_enc_swiglu(int64_t g, int64_t u, int64_t count) {
  const SwigluParams p = {at(g, 0), at(u, 0), (uint32_t)count, 0};
  dispatch(kKindSwiglu, gSwiglu, &p, sizeof p, groups(count, 256), 1);
  wrote(g);
}

typedef struct {
  uint64_t src, dst;
  uint32_t count, pad;
} CopyParams;

LAB_EXPORT void lab_gpu_enc_copy(int64_t src, int64_t srcOffset, int64_t dst, int64_t dstOffset,
                                 int64_t count) {
  const CopyParams p = {at(src, srcOffset), at(dst, dstOffset), (uint32_t)count, 0};
  dispatch(kKindCopy, gCopy, &p, sizeof p, groups(count, 256), 1);
  wrote(dst);
}
