import 'dart:ffi';

/// Whether this host has a GPU the tests run on: Metal on an Apple silicon
/// Mac, Vulkan on an x86-64 Windows PC.
final hostHasGpu =
    Abi.current() == Abi.macosArm64 || Abi.current() == Abi.windowsX64;

const noGpuSkip = 'needs the GPU of an Apple silicon Mac or a Windows PC';

@Native<Int64 Function()>(
  symbol: 'lab_gpu_init',
  assetId: 'package:dianoia/kernels',
)
external int _gpuInit();

/// Whether the GPU starts on this host, which needs [hostHasGpu] and the
/// native kernels. A hosted CI runner can be such a host without a GPU
/// driver (Windows runners have no Vulkan device).
bool get gpuStarts => hostHasGpu && _gpuInit() == 1;

const gpuDoesNotStartSkip = 'the GPU does not start on this host';
