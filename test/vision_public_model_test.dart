// The app-facing API alone, with actual VL weights. Never downloads weights.
@Timeout(Duration(minutes: 10))
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/dianoia.dart';
import 'package:crypto/crypto.dart';
import 'package:image/image.dart' as image;
import 'package:test/test.dart';

void main() {
  final source = Platform.environment['DIANOIA_VL_DIR'];
  final backbone = Platform.environment['DIANOIA_VL_BACKBONE'];
  final validation = Platform.environment['DIANOIA_VL_VALIDATION_DIR'];
  for (final backend in [InferenceBackend.cpu, InferenceBackend.gpu]) {
    test(
      'public VL OCR, streaming and visual follow-up (${backend.name})',
      () async {
        final model = await LocalVisionModel.load(
          source!,
          backboneDir: backbone!,
          backend: backend,
          contextLength: 512,
        );
        addTearDown(model.close);
        final session = model.chat(maxNewTokens: 32);
        final image = ChatImage.rgb(
          File('$validation/reference-ocr-ja/input.rgb').readAsBytesSync(),
          width: 384,
          height: 192,
        );
        final streamed = <String>[];
        final turn = await session.send(
          '画像に書かれた文字を、そのまま書き起こしてください。',
          images: [image],
          onText: (text) {
            streamed.add(text);
            return true;
          },
        );
        expect(model.backend, backend);
        expect(turn.text, '東京 123');
        expect(streamed.last, turn.text);
        for (var i = 0; i < streamed.length; i++) {
          expect(streamed[i].contains('\uFFFD'), isFalse);
          if (i > 0) expect(streamed[i].startsWith(streamed[i - 1]), isTrue);
        }
        expect(turn.reusedTokens, 0);
        // The second turn has no new RGB input; it uses retained visual features.
        final followup = await session.send('もう一度、画像の文字だけを答えてください。');
        expect(followup.text.replaceAll(RegExp(r'\s+'), ''), '東京123');
        expect(session.messages, hasLength(4));
        expect(followup.promptTokens, greaterThan(turn.promptTokens));
      },
      skip: source == null || backbone == null || validation == null
          ? 'set DIANOIA_VL_DIR, DIANOIA_VL_BACKBONE and DIANOIA_VL_VALIDATION_DIR'
          : false,
    );
  }
  final bundle = Platform.environment['DIANOIA_VL_BUNDLE'];
  for (final backend in [InferenceBackend.cpu, InferenceBackend.gpu]) {
    test(
      'standalone VL bundle preserves every OCR logit (${backend.name})',
      () async {
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
        final image = ChatImage.rgb(
          File('$validation/reference-ocr-ja/input.rgb').readAsBytesSync(),
          width: 384,
          height: 192,
        );
        final expected = <String>[], actual = <String>[];
        String digest(Float32List values) => sha256
            .convert(
              values.buffer.asUint8List(
                values.offsetInBytes,
                values.lengthInBytes,
              ),
            )
            .toString();
        final a = await original
            .chat(maxNewTokens: 32)
            .send(
              '画像に書かれた文字を、そのまま書き起こしてください。',
              images: [image],
              onLogits: (v) => expected.add(digest(v)),
            );
        final b = await packed
            .chat(maxNewTokens: 32)
            .send(
              '画像に書かれた文字を、そのまま書き起こしてください。',
              images: [image],
              onLogits: (v) => actual.add(digest(v)),
            );
        expect(a.text, '東京 123');
        expect(b.text, a.text);
        expect(b.generatedTokens, a.generatedTokens);
        expect(expected, hasLength(a.generatedTokens + 1));
        expect(actual, orderedEquals(expected));
      },
      skip:
          source == null ||
              backbone == null ||
              validation == null ||
              bundle == null
          ? 'set VL model/backbone/validation directories and DIANOIA_VL_BUNDLE'
          : false,
    );
  }
  for (final backend in [InferenceBackend.cpu, InferenceBackend.gpu]) {
    test(
      'public PNG and oriented JPEG OCR (${backend.name})',
      () async {
        final model = await LocalVisionModel.loadBundle(
          bundle!,
          backend: backend,
          contextLength: 512,
        );
        addTearDown(model.close);
        final pngPath = '$validation/ocr-ja.png';
        // A camera-shaped JPEG stores a counterclockwise raster and EXIF 6.
        // Decoding must restore the card's original landscape orientation.
        final camera = image.copyRotate(
          image.decodePng(File(pngPath).readAsBytesSync())!,
          angle: -90,
        );
        camera.exif.imageIfd.orientation = 6;
        final jpeg = ChatImage.encoded(image.encodeJpg(camera, quality: 95));
        expect(jpeg.width, 384);
        expect(jpeg.height, 192);
        final images = [
          ChatImage.rgb(
            File('$validation/reference-ocr-ja/input.rgb').readAsBytesSync(),
            width: 384,
            height: 192,
          ),
          await ChatImage.fromFile(pngPath),
          jpeg,
        ];
        final digests = <List<String>>[];
        for (final input in images) {
          final logits = <String>[];
          final turn = await model
              .chat(maxNewTokens: 32)
              .send(
                '画像に書かれた文字を、そのまま書き起こしてください。',
                images: [input],
                onLogits: (v) => logits.add(
                  sha256
                      .convert(
                        v.buffer.asUint8List(v.offsetInBytes, v.lengthInBytes),
                      )
                      .toString(),
                ),
              );
          expect(turn.text, '東京 123');
          expect(logits, hasLength(turn.generatedTokens + 1));
          digests.add(logits);
        }
        expect(
          digests[1],
          orderedEquals(digests[0]),
          reason: 'lossless PNG must preserve RGB logits',
        );
      },
      skip: bundle == null || validation == null
          ? 'set DIANOIA_VL_BUNDLE and DIANOIA_VL_VALIDATION_DIR'
          : false,
    );
  }
}
