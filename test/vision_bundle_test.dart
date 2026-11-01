import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/vision_model_fixture.dart';

Future<List<Float32List>> _answer(LocalVisionModel model) async {
  final logits = <Float32List>[];
  final image = ChatImage.rgb(
    Uint8List(8 * 8 * 3)..fillRange(0, 8 * 8 * 3, 200),
    width: 8,
    height: 8,
  );
  await model
      .chat(maxNewTokens: 3)
      .send(
        'Describe.',
        images: [image],
        onLogits: (rows) => logits.add(Float32List.fromList(rows)),
      );
  return logits;
}

void _same(List<Float32List> a, List<Float32List> b) {
  expect(a.length, b.length);
  for (var i = 0; i < a.length; i++) {
    expect(a[i], orderedEquals(b[i]));
  }
}

void main() {
  group('portable VL bundle', () {
    late Directory root;
    late String source, backbone, bundle;
    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia_vl_bundle');
      await prepareVisionFixture(root);
      source = '${root.path}/model';
      backbone = '${root.path}/backbone';
      bundle = '${root.path}/bundle';
      await packVlBundle(source, backbone, bundle);
    });
    tearDownAll(() => root.deleteSync(recursive: true));

    test(
      'keeps original vision/embedding bytes and native matrix data',
      () async {
        await validateVlBundle(bundle);
        final config = Lfm3VlConfig.load(source);
        final original = SafetensorsFile.open('$source/model.safetensors');
        final packed = SafetensorsFile.open('$bundle/model.safetensors');
        try {
          final shapes = lfm3VisionTensorShapes(config);
          expect(packed.tensors.keys.toSet(), shapes.keys.toSet());
          for (final e in shapes.entries) {
            expect(
              packed.readBf16Bits(e.key, e.value),
              orderedEquals(original.readBf16Bits(e.key, e.value)),
            );
          }
        } finally {
          original.close();
          packed.close();
        }
        final originalLanguage = SafetensorsFile.open(
          '$backbone/model.safetensors',
        );
        final packedLanguage = SafetensorsFile.open(
          '$bundle/language/model.safetensors',
        );
        try {
          expect(
            packedLanguage.tensors.containsKey(
              'model.layers.0.feed_forward.w1.weight',
            ),
            isFalse,
          );
          expect(
            packedLanguage.readBf16Bits('model.embed_tokens.weight', [272, 32]),
            orderedEquals(
              originalLanguage.readBf16Bits('model.embed_tokens.weight', [
                272,
                32,
              ]),
            ),
          );
        } finally {
          originalLanguage.close();
          packedLanguage.close();
        }
        final a = Q8Cache.open(
          '$backbone/lfm3-q8.cache',
          Q8Cache.keyFor(backbone),
        )!;
        final b = Q8Cache.open(
          '$bundle/language/lfm3-q8.cache',
          Q8Cache.keyFor('$bundle/language'),
        )!;
        try {
          expect(b.shapes.toSet(), a.shapes.toSet());
          for (final (name, rows, cols) in a.shapes) {
            final x = a.matrix(name, rows, cols)!,
                y = b.matrix(name, rows, cols)!;
            expect(y.format, x.format);
            expect(y.qsBytes, orderedEquals(x.qsBytes));
            expect(y.scaleBytes, orderedEquals(x.scaleBytes));
          }
        } finally {
          a.close();
          b.close();
        }
      },
    );

    test(
      'public CPU logits are byte-identical to unpacked inference',
      () async {
        final a = await LocalVisionModel.load(
          source,
          backboneDir: backbone,
          backend: InferenceBackend.cpu,
          contextLength: 512,
        );
        final b = await LocalVisionModel.loadBundle(
          bundle,
          backend: InferenceBackend.cpu,
          contextLength: 512,
        );
        try {
          _same(await _answer(a), await _answer(b));
        } finally {
          await a.close();
          await b.close();
        }
      },
    );

    test(
      'relocated bundle runs while original model and backbone are absent',
      () async {
        final hiddenSource = Directory(
          source,
        ).renameSync('${root.path}/hidden-source');
        final hiddenBackbone = Directory(
          backbone,
        ).renameSync('${root.path}/hidden-backbone');
        final moved = Directory(bundle).renameSync('${root.path}/relocated');
        try {
          final model = await LocalVisionModel.loadBundle(
            moved.path,
            backend: InferenceBackend.cpu,
            contextLength: 512,
          );
          try {
            expect(await _answer(model), hasLength(4));
          } finally {
            await model.close();
          }
        } finally {
          moved.renameSync(bundle);
          hiddenSource.renameSync(source);
          hiddenBackbone.renameSync(backbone);
        }
      },
    );

    test('same-size payload corruption fails before model mapping', () async {
      final file = File('$bundle/LICENSE');
      final bytes = file.readAsBytesSync();
      try {
        final changed = Uint8List.fromList(bytes)..[0] ^= 1;
        file.writeAsBytesSync(changed);
        await expectLater(
          LocalVisionModel.loadBundle(bundle, backend: InferenceBackend.cpu),
          throwsFormatException,
        );
        expect(MappedFile.isMapped('$bundle/language/lfm3-q8.cache'), isFalse);
      } finally {
        file.writeAsBytesSync(bytes);
      }
    });

    test(
      'rejects unsupported manifests and unrecorded alternative caches',
      () async {
        final file = File('$bundle/vl_bundle.json');
        final original = file.readAsStringSync();
        try {
          for (final change in <void Function(Map)>[
            (m) => m['version'] = 2,
            (m) => (m['files'] as Map).remove('LICENSE'),
            (m) => (m['files'] as Map)['../outside'] = {
              'bytes': 1,
              'sha256': '0' * 64,
            },
          ]) {
            final value = jsonDecode(original) as Map;
            change(value);
            file.writeAsStringSync(jsonEncode(value));
            await expectLater(validateVlBundle(bundle), throwsFormatException);
          }
        } finally {
          file.writeAsStringSync(original);
        }
        final other = File('$bundle/language/lfm3-q4.cache')
          ..writeAsBytesSync([0]);
        try {
          await expectLater(validateVlBundle(bundle), throwsFormatException);
        } finally {
          other.deleteSync();
        }
      },
    );

    test(
      'existing output and missing cache never publish a partial bundle',
      () async {
        await expectLater(
          packVlBundle(source, backbone, bundle),
          throwsA(isA<FileSystemException>()),
        );
        final saved = File(
          '$backbone/lfm3-q8.cache',
        ).renameSync('$backbone/cache.saved');
        try {
          await expectLater(
            packVlBundle(source, backbone, '${root.path}/missing'),
            throwsStateError,
          );
          expect(Directory('${root.path}/missing').existsSync(), isFalse);
        } finally {
          saved.renameSync('$backbone/lfm3-q8.cache');
        }
      },
    );

    test(
      'missing vision tensor is rejected with source mapping released',
      () async {
        final file = File('$source/model.safetensors');
        final original = file.readAsBytesSync();
        final identityFile = File('$backbone/vl_backbone.json');
        final identity = identityFile.readAsStringSync();
        final st = SafetensorsFile.open(file.path);
        try {
          st.writeSubset(
            '${root.path}/missing-vision.safetensors',
            (name, _) => name != 'model.multi_modal_projector.linear_2.bias',
          );
        } finally {
          st.close();
        }
        try {
          File('${root.path}/missing-vision.safetensors').copySync(file.path);
          final updated = jsonDecode(identity) as Map;
          updated['source_key'] = Q8Cache.keyFor(source);
          identityFile.writeAsStringSync(jsonEncode(updated));
          await expectLater(
            packVlBundle(source, backbone, '${root.path}/bad-vision'),
            throwsFormatException,
          );
          expect(Directory('${root.path}/bad-vision').existsSync(), isFalse);
          expect(fixtureBackboneIsMapped(root), isFalse);
          expect(
            root.listSync().where(
              (e) => e.path.contains('.dianoia-vl-bundle-'),
            ),
            isEmpty,
          );
        } finally {
          file.writeAsBytesSync(original);
          identityFile.writeAsStringSync(identity);
        }
      },
    );

    test(
      'incomplete native cache is rejected instead of dropping weights',
      () async {
        final cache = Q8Cache.open(
          '$backbone/lfm3-q8.cache',
          Q8Cache.keyFor(backbone),
        )!;
        try {
          Q8Cache.write(
            '${root.path}/incomplete.cache',
            Q8Cache.keyFor(backbone),
            {
              for (final (name, rows, cols) in cache.shapes)
                if (name != 'model.layers.0.feed_forward.w1')
                  name: cache.matrix(name, rows, cols)!,
            },
          );
        } finally {
          cache.close();
        }
        final saved = File(
          '$backbone/lfm3-q8.cache',
        ).renameSync('$backbone/cache.saved');
        try {
          File(
            '${root.path}/incomplete.cache',
          ).copySync('$backbone/lfm3-q8.cache');
          await expectLater(
            packVlBundle(source, backbone, '${root.path}/bad-cache'),
            throwsFormatException,
          );
          expect(Directory('${root.path}/bad-cache').existsSync(), isFalse);
          expect(fixtureBackboneIsMapped(root), isFalse);
        } finally {
          File('$backbone/lfm3-q8.cache').deleteSync();
          saved.renameSync('$backbone/lfm3-q8.cache');
        }
      },
    );

    test('Q4 bundle retains native formats and source arithmetic', () async {
      final building = await Lfm3Model.load(
        backbone,
        maxSeq: 512,
        weights: WeightFormat.q4,
        q8CachePath: '$backbone/lfm3-q4.cache',
      );
      await building.close();
      try {
        final out = '${root.path}/q4-bundle';
        await packVlBundle(source, backbone, out, weights: WeightFormat.q4);
        final a = await LocalVisionModel.load(
          source,
          backboneDir: backbone,
          backend: InferenceBackend.cpu,
          contextLength: 512,
        );
        final b = await LocalVisionModel.loadBundle(
          out,
          backend: InferenceBackend.cpu,
          contextLength: 512,
        );
        try {
          _same(await _answer(a), await _answer(b));
        } finally {
          await a.close();
          await b.close();
        }
      } finally {
        File('$backbone/lfm3-q4.cache').deleteSync();
      }
    });
  }, skip: LocalVisionModel.isSupported ? false : 'requires native kernels');
}
