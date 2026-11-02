// Optional real-checkpoint packing validation. Exact OCR quality is graded by
// tool/validate_vl_quantization.dart; this test checks preservation of arithmetic.
@Timeout(Duration(minutes: 5))
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

void main() {
  final source = Platform.environment['DIANOIA_VL_DIR'];
  final backbone = Platform.environment['DIANOIA_VL_Q4_BACKBONE'];
  final bundle = Platform.environment['DIANOIA_VL_Q4_BUNDLE'];
  final image = Platform.environment['DIANOIA_VL_Q4_IMAGE'];
  final evidence = Platform.environment['DIANOIA_VL_Q4_EVIDENCE'];
  final records = <String, Object?>{};
  for (final backend in [InferenceBackend.cpu, InferenceBackend.gpu]) {
    test(
      'Q4 packing preserves every OCR logit (${backend.name})',
      () async {
        final manifest =
            jsonDecode(File('$bundle/vl_bundle.json').readAsStringSync())
                as Map;
        expect(manifest['weights'], 'q4');
        final original = await LocalVisionModel.load(
          source!,
          backboneDir: backbone!,
          backend: backend,
          contextLength: 512,
        );
        addTearDown(original.close);
        final packed = await LocalVisionModel.loadBundle(
          bundle!,
          backend: backend,
          contextLength: 512,
        );
        addTearDown(packed.close);
        final input = await ChatImage.fromFile(image!);
        final expected = <String>[], actual = <String>[];
        String digest(Float32List v) => sha256
            .convert(v.buffer.asUint8List(v.offsetInBytes, v.lengthInBytes))
            .toString();
        const prompt = '画像に書かれた文字を、そのまま書き起こしてください。';
        final a = await original
            .chat(maxNewTokens: 32)
            .send(
              prompt,
              images: [input],
              onLogits: (v) => expected.add(digest(v)),
            );
        final b = await packed
            .chat(maxNewTokens: 32)
            .send(
              prompt,
              images: [input],
              onLogits: (v) => actual.add(digest(v)),
            );
        expect(a.generatedTokens, greaterThan(0));
        expect(b.text, a.text);
        expect(b.generatedTokens, a.generatedTokens);
        expect(expected, hasLength(a.generatedTokens + 1));
        expect(actual, orderedEquals(expected));
        records[backend.name] = {
          'source_text': a.text,
          'packed_text': b.text,
          'generated_tokens': b.generatedTokens,
          'source_logit_sha256': expected,
          'packed_logit_sha256': actual,
          'all_logits_byte_identical': true,
        };
        if (evidence != null) {
          File(evidence).writeAsStringSync(
            '${const JsonEncoder.withIndent('  ').convert(records)}\n',
          );
        }
      },
      skip: [source, backbone, bundle, image].any((v) => v == null)
          ? 'set DIANOIA_VL_DIR and DIANOIA_VL_Q4_BACKBONE/BUNDLE/IMAGE'
          : false,
    );
  }
}
