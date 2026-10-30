// Builds the native kernels into one code asset, `package:dianoia/kernels`,
// which the `@Native` functions of lib/src/native.dart and lib/src/gpu.dart
// bind to.
//
// * Apple (macOS, iOS): the C kernels (src/c) and the Metal layer
//   (src/metal), whose shader source is compiled when the GPU starts, so no
//   offline Metal compiler is needed.
// * Android: the C kernels, threaded with OpenMP, and the Vulkan layer
//   (src/vulkan), whose shaders arrive precompiled as SPIR-V word arrays
//   (src/vulkan/spirv, written by tool/compile_shaders.sh), so no glslc is
//   needed either.
//
// The kernels use arm64 NEON (sdot), so other targets get no asset and the
// engine runs in Dart alone there.
//
// No FMA contraction anywhere: the float kernels stay comparable, bit for
// bit, with the Dart kernels and across chunkings.
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

const _floatFlags = ['-ffp-contract=off', '-fno-fast-math'];

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    if (code.targetArchitecture != Architecture.arm64) return;
    final builder = switch (code.targetOS) {
      OS.macOS || OS.iOS => CBuilder.library(
        name: 'dianoia_kernels',
        assetName: 'kernels',
        language: Language.objectiveC,
        sources: ['src/c/lab_native.c', 'src/metal/lab_metal.m'],
        frameworks: ['Foundation', 'Metal', 'Accelerate'],
        flags: [
          '-fobjc-arc',
          // Apple M1 and A14 are the first with the sdot instruction.
          code.targetOS == OS.macOS ? '-mcpu=apple-m1' : '-mcpu=apple-a14',
          ..._floatFlags,
        ],
        defines: {'ACCELERATE_NEW_LAPACK': null},
      ),
      OS.android => CBuilder.library(
        name: 'dianoia_kernels',
        assetName: 'kernels',
        sources: ['src/c/lab_native.c', 'src/vulkan/lab_vulkan.c'],
        includes: ['src/vulkan/spirv'],
        libraries: ['vulkan', 'log'],
        flags: [
          '-march=armv8.2-a+dotprod',
          ..._floatFlags,
          // OpenMP gives the C kernels their thread pool (dispatch_apply on
          // Apple), linked statically so nothing else ships.
          '-fopenmp',
          '-static-openmp',
        ],
      ),
      _ => null,
    };
    if (builder == null) return;
    await builder.run(
      input: input,
      output: output,
      logger: Logger('')
        ..level = Level.ALL
        ..onRecord.listen((record) => print(record.message)),
    );
  });
}
