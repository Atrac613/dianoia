// The committed SPIR-V (src/vulkan/spirv) must come from the committed
// shader sources: run tool/compile_shaders.sh after editing a shader.
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:test/test.dart';

void main() {
  test('the SPIR-V word arrays are compiled from the current shaders', () {
    final sources =
        Directory('src/vulkan/shaders')
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.comp') || f.path.endsWith('.glsl'))
            .toList()
          ..sort((a, b) => _name(a).compareTo(_name(b)));
    final expected = [
      for (final f in sources)
        '${sha256.convert(f.readAsBytesSync())}  ${_name(f)}\n',
    ].join();
    expect(
      File('src/vulkan/spirv/SOURCES.sha256').readAsStringSync(),
      expected,
      reason: 'run tool/compile_shaders.sh',
    );
    for (final f in sources.where((f) => f.path.endsWith('.comp'))) {
      final name = _name(f).replaceAll('.comp', '.inc');
      expect(File('src/vulkan/spirv/$name').existsSync(), isTrue);
    }
  });
}

String _name(File f) => f.uri.pathSegments.last;
