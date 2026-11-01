import 'dart:ffi';

/// Whether this host has a GPU the tests run on: Metal on an Apple silicon
/// Mac, Vulkan on an x86-64 Windows PC.
final hostHasGpu =
    Abi.current() == Abi.macosArm64 || Abi.current() == Abi.windowsX64;

const noGpuSkip = 'needs the GPU of an Apple silicon Mac or a Windows PC';
