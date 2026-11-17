// Control vectors: their arithmetic and files in pure Dart, and how the CPU
// forward pass applies them, on the synthetic backbone of the vision
// fixture and the native CPU kernels.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/gguf_fixture.dart';
import 'support/gpu_host.dart';
import 'support/vision_model_fixture.dart';

/// Deterministic values in [-1, 1).
Float32List _values(int count, int seed) {
  final random = math.Random(seed);
  return Float32List.fromList([
    for (var i = 0; i < count; i++) random.nextDouble() * 2 - 1,
  ]);
}

double _dot(Float32List a, int offset, Float32List b) {
  var sum = 0.0;
  for (var j = 0; j < b.length; j++) {
    sum += a[offset + j] * b[j];
  }
  return sum;
}

double _norm(Float32List v) => math.sqrt(_dot(v, 0, v));

void main() {
  group('ControlVector', () {
    const h = 8;

    test('add mode adds scale times the direction to every row', () {
      final d = _values(h, 1);
      final v = ControlVector({2: d}, hiddenSize: h, scale: 0.5);
      final rows = _values(3 * h, 2);
      final before = Float32List.fromList(rows);
      v.apply(rows, 3, 2);
      for (var t = 0; t < 3; t++) {
        for (var j = 0; j < h; j++) {
          final scaled = Float32List.fromList([0.5 * d[j]])[0];
          final expected = Float32List.fromList([before[t * h + j] + scaled]);
          expect(rows[t * h + j], expected[0]);
        }
      }
    });

    test('project mode removes the component along the direction', () {
      final d = _values(h, 3);
      final unit = Float32List.fromList([for (final x in d) x / _norm(d)]);
      final rows = _values(2 * h, 4);
      final before = Float32List.fromList(rows);
      ControlVector(
        {0: d},
        hiddenSize: h,
        mode: ControlMode.project,
      ).apply(rows, 2, 0);
      for (var t = 0; t < 2; t++) {
        expect(_dot(rows, t * h, unit), closeTo(0, 1e-6));
        // What is orthogonal to the direction stays.
        final removed = _dot(before, t * h, unit);
        for (var j = 0; j < h; j++) {
          expect(
            rows[t * h + j],
            closeTo(before[t * h + j] - removed * unit[j], 1e-6),
          );
        }
      }

      // Half the scale leaves half the component.
      final half = Float32List.fromList(before);
      ControlVector(
        {0: d},
        hiddenSize: h,
        mode: ControlMode.project,
        scale: 0.5,
      ).apply(half, 2, 0);
      for (var t = 0; t < 2; t++) {
        expect(
          _dot(half, t * h, unit),
          closeTo(_dot(before, t * h, unit) / 2, 1e-6),
        );
      }
    });

    test('leaves layers outside its range or without a direction', () {
      final v = ControlVector(
        {0: _values(h, 5), 1: _values(h, 6), 3: _values(h, 7)},
        hiddenSize: h,
        firstLayer: 1,
        lastLayer: 2,
      );
      expect(
        [for (var l = 0; l < 4; l++) v.steers(l)],
        [false, true, false, false],
      );
      final rows = _values(h, 8);
      final before = Float32List.fromList(rows);
      for (final layer in [0, 2, 3]) {
        v.apply(rows, 1, layer);
      }
      expect(rows, orderedEquals(before));
    });

    test('rejects malformed directions and settings', () {
      final good = _values(h, 9);
      expect(
        () => ControlVector({0: _values(h - 1, 9)}, hiddenSize: h),
        throwsArgumentError,
      );
      expect(
        () =>
            ControlVector({0: Float32List(h)..[3] = double.nan}, hiddenSize: h),
        throwsArgumentError,
      );
      expect(
        () => ControlVector({-1: good}, hiddenSize: h),
        throwsArgumentError,
      );
      expect(
        () => ControlVector({0: good}, hiddenSize: h, scale: double.infinity),
        throwsArgumentError,
      );
      expect(
        () => ControlVector(
          {0: good},
          hiddenSize: h,
          firstLayer: 2,
          lastLayer: 1,
        ),
        throwsArgumentError,
      );
      expect(
        () => ControlVector({0: good}, hiddenSize: h, scale: 1e300),
        throwsArgumentError,
      );
      // A zero direction adds nothing, but has nothing to project out.
      expect(ControlVector({0: Float32List(h)}, hiddenSize: h), isNotNull);
      expect(
        () => ControlVector(
          {0: Float32List(h)},
          hiddenSize: h,
          mode: ControlMode.project,
        ),
        throwsArgumentError,
      );
      expect(
        () =>
            ControlVector({0: good}, hiddenSize: h).apply(Float32List(h), 2, 0),
        throwsRangeError,
      );
    });

    test('fingerprints what changes the arithmetic', () {
      final v = ControlVector({
        0: _values(h, 10),
        1: _values(h, 11),
      }, hiddenSize: h);
      expect(v.withSettings().fingerprint, v.fingerprint);
      expect(v.withSettings(scale: 2).fingerprint, isNot(v.fingerprint));
      expect(
        v.withSettings(mode: ControlMode.project).fingerprint,
        isNot(v.fingerprint),
      );
      expect(v.withSettings(firstLayer: 1).fingerprint, isNot(v.fingerprint));
      // A direction outside the range does not count.
      expect(
        ControlVector(
          {1: _values(h, 11)},
          hiddenSize: h,
          firstLayer: 1,
        ).fingerprint,
        v.withSettings(firstLayer: 1).fingerprint,
      );
    });

    group('LayerSeparation', () {
      Float32List row(List<double> v) => Float32List.fromList(v);

      /// In-sample d', for contrast: rows on the full-sample direction.
      double inSample(List<Float32List> a, List<Float32List> b) {
        final d = LayerSeparation.measure(a, b).direction;
        final norm = _norm(d);
        List<double> p(List<Float32List> rows) => [
          for (final r in rows) _dot(r, 0, d) / norm,
        ];
        (double, double) stats(List<double> x) {
          final m = x.reduce((u, v) => u + v) / x.length;
          final v =
              x.map((e) => (e - m) * (e - m)).reduce((u, v) => u + v) /
              (x.length - 1);
          return (m, v);
        }

        final (ma, va) = stats(p(a));
        final (mb, vb) = stats(p(b));
        return (ma - mb) / math.sqrt((va + vb) / 2);
      }

      test("held-out d' on one dimension", () {
        // Holding out 1 leaves A's mean at 3: direction (5), projection 1.
        final s = LayerSeparation.measure(
          [
            row([1]),
            row([3]),
          ],
          [
            row([-1]),
            row([-3]),
          ],
        );
        expect(s.direction, orderedEquals([4]));
        expect(s.norm, 4);
        expect(s.meanA, closeTo(2, 1e-12));
        expect(s.meanB, closeTo(-2, 1e-12));
        expect(s.sdA, closeTo(math.sqrt2, 1e-12));
        expect(s.sdB, closeTo(math.sqrt2, 1e-12));
        expect(s.separation, closeTo(4 / math.sqrt2, 1e-12));
        expect(s.passes(2), isTrue);
        expect(s.passes(3), isFalse);
      });

      test("held-out d' turns negative when a shared dimension dominates", () {
        // In sample the sets sit at x = ±2 with y = ±5 shared noise, d' =
        // 4 / sqrt 2. Held out, each row's direction leans on the other
        // row's y: (1, 5) of A is projected on (3, -5) - (-2, 0) = (5, -5).
        final a = [
          row([1, 5]),
          row([3, -5]),
        ];
        final b = [
          row([-1, 5]),
          row([-3, -5]),
        ];
        expect(inSample(a, b), closeTo(4 / math.sqrt2, 1e-12));
        final s = LayerSeparation.measure(a, b);
        final pA = [-20 / math.sqrt(50), -16 / math.sqrt(34)];
        final pB = [20 / math.sqrt(50), 16 / math.sqrt(34)];
        final mean = (pA[0] + pA[1]) / 2;
        final sd = (pA[0] - pA[1]).abs() / math.sqrt2;
        expect(s.meanA, closeTo(mean, 1e-12));
        expect(s.meanB, closeTo((pB[0] + pB[1]) / 2, 1e-12));
        expect(s.sdA, closeTo(sd, 1e-12));
        expect(s.sdB, closeTo(sd, 1e-12));
        expect(s.separation, closeTo(2 * mean / sd, 1e-9));
        expect(s.separation!, lessThan(-50));
        expect(s.passes(2), isFalse);
        expect(s.passes(0), isTrue);
      });

      test('pure noise separates in sample but not held out', () {
        final random = math.Random(42);
        double gauss() =>
            math.sqrt(-2 * math.log(1 - random.nextDouble())) *
            math.cos(2 * math.pi * random.nextDouble());
        List<Float32List> rows(int n) => [
          for (var i = 0; i < n; i++)
            Float32List.fromList([for (var j = 0; j < 512; j++) gauss()]),
        ];
        final a = rows(6), b = rows(6);
        expect(inSample(a, b), greaterThan(4));
        final s = LayerSeparation.measure(a, b);
        expect(s.separation!, lessThan(0.5));
        expect(s.passes(2), isFalse);
      });

      test('guards degenerate sets', () {
        // No direction.
        final same = LayerSeparation.measure(
          [
            row([1, 2]),
            row([3, 4]),
          ],
          [
            row([3, 4]),
            row([1, 2]),
          ],
        );
        expect(same.norm, 0);
        expect(same.separation, 0);
        expect(same.passes(0), isFalse);
        // One row in a set cannot be held out.
        final single = LayerSeparation.measure(
          [
            row([1, 0]),
          ],
          [
            row([-1, 0]),
            row([-2, 0]),
          ],
        );
        expect((single.meanA, single.sdA), (null, null));
        expect(single.sdB, isNotNull);
        expect(single.separation, isNull);
        expect(single.passes(2), isFalse);
        expect(single.passes(0), isTrue);
        // A zero held-out direction projects its row to 0: without 2, A's
        // mean is B's. Without 1 the direction is (1) and 1 projects to 1.
        final zero = LayerSeparation.measure(
          [
            row([2]),
            row([1]),
          ],
          [
            row([1]),
            row([1]),
          ],
        );
        expect(zero.meanA, closeTo(0.5, 1e-12));
        expect(zero.sdA, closeTo(math.sqrt1_2, 1e-12));
        // No spread but distinct means: perfectly separated.
        final exact = LayerSeparation.measure(
          [
            row([1, 0]),
            row([1, 0]),
          ],
          [
            row([0, 0]),
            row([0, 0]),
          ],
        );
        expect(exact.separation, double.infinity);
        expect(exact.passes(100), isTrue);
        // Spread at the level of rounding counts as none.
        final rounded = LayerSeparation.measure(
          [
            row([1, 0]),
            row([1, 1e-12]),
          ],
          [
            row([0, 0]),
            row([0, 1e-12]),
          ],
        );
        expect(rounded.separation, double.infinity);
        expect(
          () => LayerSeparation.measure([], [
            row([1]),
          ]),
          throwsArgumentError,
        );
        expect(
          () => LayerSeparation.measure(
            [
              row([1]),
            ],
            [
              row([1, 2]),
            ],
          ),
          throwsArgumentError,
        );
      });

      test("passingFrom and exclusion with negative d'", () {
        LayerSeparation at(double gap) => LayerSeparation.measure(
          [
            row([gap + 1]),
            row([gap - 1]),
          ],
          [
            row([1]),
            row([-1]),
          ],
        );
        final profile = [6.0, -1.0, 5.0, 3.0, 6.0, 7.0];
        final layers = [for (final d in profile) at(d)];
        final measured = [for (final l in layers) l.separation!];
        for (var l = 0; l < profile.length; l++) {
          expect(measured[l].sign, profile[l].sign, reason: 'layer $l');
        }
        final ranked = [
          for (var l = 0; l < profile.length; l++)
            if (layers[l].passes(2)) l,
        ];
        expect(ranked, [
          for (var l = 0; l < profile.length; l++)
            if (measured[l] >= 2) l,
        ]);
        expect(ranked, isNot(contains(1)));
        expect(layers[1].passes(0), isTrue);
        final from = LayerSeparation.passingFrom(layers, 2);
        expect(from, isNotNull);
        for (var l = from!; l < layers.length; l++) {
          expect(measured[l], greaterThanOrEqualTo(2));
        }
        expect(measured[from - 1], lessThan(2));
        expect(LayerSeparation.passingFrom(layers, 1e9), isNull);
      });
    });

    group('files', () {
      late Directory dir;
      setUp(() => dir = Directory.systemTemp.createTempSync('dianoia-cv-'));
      tearDown(() => dir.deleteSync(recursive: true));

      test('save then load gives the same directions', () {
        final v = ControlVector({
          0: _values(h, 12),
          5: _values(h, 13),
        }, hiddenSize: h);
        final path = '${dir.path}/v.safetensors';
        v.save(path);
        expect(File('$path.partial').existsSync(), isFalse);
        final back = ControlVector.load(
          path,
          mode: ControlMode.project,
          scale: 0.25,
          firstLayer: 1,
          lastLayer: 5,
        );
        expect(back.hiddenSize, h);
        expect(back.directions.keys, unorderedEquals([0, 5]));
        for (final l in [0, 5]) {
          expect(back.directions[l], orderedEquals(v.directions[l]!));
        }
        expect(back.mode, ControlMode.project);
        expect(back.scale, 0.25);
        expect((back.firstLayer, back.lastLayer), (1, 5));
      });

      test('rejects tensors that are not directions', () {
        final path = '${dir.path}/bad.safetensors';
        writeSafetensorsF32(path, {
          'direction.1': ([h], _values(h, 14)),
          'other': ([h], _values(h, 15)),
        });
        expect(() => ControlVector.load(path), throwsFormatException);
        writeSafetensorsF32(path, {
          'direction.1': ([2, 4], _values(h, 14)),
        });
        expect(() => ControlVector.load(path), throwsFormatException);
        writeSafetensorsF32(path, {
          'direction.1': ([h], _values(h, 14)),
          'direction.2': ([h + 1], _values(h + 1, 15)),
        });
        expect(() => ControlVector.load(path), throwsFormatException);
      });

      test("reads llama.cpp's GGUF layout", () {
        NativeKernels.load(null, threads: 2);
        final d1 = _values(h, 16), d3 = _values(h, 17);
        final path = '${dir.path}/v.gguf';
        writeTestGguf(
          path,
          {
            'general.architecture': 'controlvector',
            'controlvector.model_hint': 'lfm3',
            'controlvector.layer_count': 2,
          },
          [
            for (final (l, d) in [(1, d1), (3, d3)])
              ('direction.$l', [h], GgufType.f32, d.buffer.asUint8List()),
          ],
        );
        final v = ControlVector.load(path);
        expect(v.hiddenSize, h);
        expect(v.directions[1], orderedEquals(d1));
        expect(v.directions[3], orderedEquals(d3));
        expect(v.directions.keys, unorderedEquals([1, 3]));
      }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');
    });
  });

  group('Lfm3Model control vector', () {
    late Directory root;
    late String backbone;
    late Lfm3Model model;
    // The fixture embeds only tokens 32 to 126; others are zero rows.
    const ids = [40, 50, 60, 70, 80, 90];

    setUpAll(() async {
      root = Directory.systemTemp.createTempSync('dianoia-cv-model-');
      await prepareVisionFixture(root);
      backbone = '${root.path}/backbone';
      model = await _load(backbone);
    });

    tearDownAll(() async {
      await model.close();
      root.deleteSync(recursive: true);
    });

    setUp(() async {
      await model.reset();
      model.setControlVector(null);
    });

    /// Every captured slot of feeding [ids] from scratch, row-major per slot.
    Future<List<Float32List>> captureAll(Lfm3Model m) async {
      final h = m.config.hiddenSize;
      final slots = [
        for (var s = 0; s <= m.config.numLayers; s++)
          Float32List(ids.length * h),
      ];
      await m.forward(
        ids,
        capture: (slot, first, rows, count) =>
            slots[slot].setRange(first * h, (first + count) * h, rows),
      );
      return slots;
    }

    ControlVector vector({
      ControlMode mode = ControlMode.project,
      double scale = 1,
      int first = 0,
      int? last,
    }) {
      final c = model.config;
      return ControlVector(
        {
          for (var l = 0; l < c.numLayers; l++)
            l: _values(c.hiddenSize, 100 + l),
        },
        hiddenSize: c.hiddenSize,
        mode: mode,
        scale: scale,
        firstLayer: first,
        lastLayer: last,
      );
    }

    test('without a vector the logits match a fresh model', () async {
      // Set and clear a vector first: nothing of it may remain.
      model.setControlVector(vector());
      model.setControlVector(null);
      await model.forward(ids);
      final logits = Float32List.fromList(model.logits);

      final fresh = await _load(backbone);
      try {
        await fresh.forward(ids);
        expect(logits, orderedEquals(fresh.logits));
      } finally {
        await fresh.close();
      }
    });

    test('project mode steers in-range layers only', () async {
      final h = model.config.hiddenSize;
      final plain = await captureAll(model);

      // Layer 1 only.
      await model.reset();
      final one = vector(first: 1, last: 1);
      model.setControlVector(one);
      final steered = await captureAll(model);
      expect(steered[0], orderedEquals(plain[0]));
      expect(steered[1], orderedEquals(plain[1]));
      final expected = Float32List.fromList(plain[2]);
      one.apply(expected, ids.length, 1);
      expect(steered[2], orderedEquals(expected));
      final d = one.directions[1]!;
      final unit = Float32List.fromList([for (final x in d) x / _norm(d)]);
      for (var t = 0; t < ids.length; t++) {
        final row = Float32List.sublistView(steered[2], t * h, (t + 1) * h);
        expect(
          _dot(steered[2], t * h, unit).abs(),
          lessThan(1e-5 * _norm(row)),
        );
        expect(
          _dot(plain[2], t * h, unit).abs(),
          greaterThan(1e-3 * _norm(row)),
        );
      }

      // Layers 1 and 2: layer 2 sees the same input, so its output is the
      // output of layer 2 above, steered.
      await model.reset();
      final two = vector(first: 1, last: 2);
      model.setControlVector(two);
      final both = await captureAll(model);
      expect(both[2], orderedEquals(steered[2]));
      final after = Float32List.fromList(steered[3]);
      two.apply(after, ids.length, 2);
      expect(both[3], orderedEquals(after));
      expect(steered[3], isNot(orderedEquals(after)));
    });

    test('the logits come from the steered residual', () async {
      final h = model.config.hiddenSize;
      final last = model.config.numLayers - 1;
      final add = vector(mode: ControlMode.add, scale: 4, first: last);
      final plain = await captureAll(model);
      final residual = Float32List.fromList(
        plain[last + 1].sublist((ids.length - 1) * h),
      );
      add.apply(residual, 1, last);
      final expected = Float32List.fromList(await model.logitsOf(residual));

      await model.reset();
      model.setControlVector(add);
      await model.forward(ids);
      expect(model.logits, orderedEquals(expected));
    });

    test('changes only on an empty sequence', () async {
      final v = vector();
      await model.forward(ids.sublist(0, 2));
      expect(() => model.setControlVector(v), throwsStateError);
      expect(model.controlVector, isNull);
      model.setControlVector(null); // unchanged: allowed

      await model.reset();
      model.setControlVector(v);
      await model.forward(ids.sublist(0, 2));
      // A vector that steers identically is no change.
      model.setControlVector(v.withSettings());
      expect(() => model.setControlVector(null), throwsStateError);
      expect(
        () => model.setControlVector(v.withSettings(scale: 2)),
        throwsStateError,
      );
      // reset keeps the vector.
      await model.reset();
      expect(model.controlVector?.fingerprint, v.fingerprint);
    });

    test('rejects vectors for another model', () {
      final c = model.config;
      expect(
        () => model.setControlVector(
          ControlVector({
            0: Float32List(c.hiddenSize + 1),
          }, hiddenSize: c.hiddenSize + 1),
        ),
        throwsArgumentError,
      );
      expect(
        () => model.setControlVector(
          ControlVector({
            c.numLayers: Float32List(c.hiddenSize),
          }, hiddenSize: c.hiddenSize),
        ),
        throwsArgumentError,
      );
    });

    test('a saved state loads only under its vector', () async {
      final path = '${root.path}/state.bin';
      final v = vector(first: 1);
      model.setControlVector(v);
      await model.forward(ids);
      await model.saveState(path);

      await model.reset();
      model.setControlVector(null);
      expect(await model.loadState(path), isFalse);
      model.setControlVector(v.withSettings(scale: 0.5));
      expect(await model.loadState(path), isFalse);
      model.setControlVector(v);
      expect(await model.loadState(path), isTrue);
      expect(model.history, ids);

      // Continuing from the loaded state matches continuing from scratch.
      await model.forward([100]);
      final continued = Float32List.fromList(model.logits);
      await model.reset();
      await model.forward([...ids, 100]);
      expect(continued, orderedEquals(model.logits));
    });
  }, skip: NativeKernels.isAvailable ? false : 'requires native kernels');

  group(
    'GPU control vector',
    () {
      late Directory root;
      late String backbone;
      late Lfm3Model cpu, gpu;
      const ids = [40, 50, 60, 70, 80, 90];

      setUpAll(() async {
        root = Directory.systemTemp.createTempSync('dianoia-cv-gpu-');
        await prepareVisionFixture(root);
        backbone = '${root.path}/backbone';
        cpu = await _load(backbone);
        gpu = await _load(backbone, gpu: true);
      });

      tearDownAll(() async {
        await gpu.close();
        await cpu.close();
        root.deleteSync(recursive: true);
      });

      setUp(() async {
        for (final m in [cpu, gpu]) {
          await m.reset();
          m.setControlVector(null);
        }
      });

      ControlVector vector(
        ControlMode mode, {
        double scale = 1,
        int first = 0,
      }) {
        final c = gpu.config;
        return ControlVector(
          {
            for (var l = 0; l < c.numLayers; l++)
              l: _values(c.hiddenSize, 200 + l),
          },
          hiddenSize: c.hiddenSize,
          mode: mode,
          scale: scale,
          firstLayer: first,
        );
      }

      /// Every captured slot of feeding [ids] from scratch under [v], and
      /// the logits.
      Future<(List<Float32List>, Float32List)> run(
        Lfm3Model m,
        ControlVector? v,
      ) async {
        await m.reset();
        m.setControlVector(v);
        final h = m.config.hiddenSize;
        final slots = [
          for (var s = 0; s <= m.config.numLayers; s++)
            Float32List(ids.length * h),
        ];
        await m.forward(
          ids,
          capture: (slot, first, rows, count) =>
              slots[slot].setRange(first * h, (first + count) * h, rows),
        );
        return (slots, Float32List.fromList(m.logits));
      }

      /// The largest difference between [a] and [b] relative to the largest
      /// magnitude in [b].
      double relative(Float32List a, Float32List b) {
        var diff = 0.0, scale = 0.0;
        for (var i = 0; i < a.length; i++) {
          diff = math.max(diff, (a[i] - b[i]).abs());
          scale = math.max(scale, b[i].abs());
        }
        return diff / scale;
      }

      for (final mode in ControlMode.values) {
        test('${mode.name} mode agrees with the CPU arithmetic', () async {
          final full = vector(mode, scale: mode == ControlMode.add ? 2 : 1);
          final (plain, _) = await run(gpu, null);
          // One layer at a time: the kernel's output against ControlVector.apply
          // on the same input, the GPU's own unsteered residual.
          for (var l = 0; l < gpu.config.numLayers; l++) {
            final one = full.withSettings(firstLayer: l, lastLayer: l);
            final (steered, _) = await run(gpu, one);
            for (var s = 0; s <= l; s++) {
              expect(steered[s], orderedEquals(plain[s]), reason: 'slot $s');
            }
            final expected = Float32List.fromList(plain[l + 1]);
            one.apply(expected, ids.length, l);
            expect(steered[l + 1], isNot(orderedEquals(plain[l + 1])));
            if (mode == ControlMode.add) {
              // A float32 sum rounds the same in any order.
              expect(
                steered[l + 1],
                orderedEquals(expected),
                reason: 'layer $l',
              );
            } else {
              // The CPU sums the dot product in double, the GPU in float32 in
              // another order.
              expect(
                relative(steered[l + 1], expected),
                lessThan(1e-5),
                reason: 'layer $l',
              );
            }
          }

          // The whole model. The CPU's int8 inputs alone put it a few
          // percent from the GPU on this fixture; steering moves both far
          // more than that, and alike.
          final (cpuPlain, cpuPlainLogits) = await run(cpu, null);
          final (gpuPlain, gpuPlainLogits) = await run(gpu, null);
          final (cpuSlots, cpuLogits) = await run(cpu, full);
          final (gpuSlots, gpuLogits) = await run(gpu, full);
          void agree(
            Float32List g,
            Float32List c,
            Float32List gPlain,
            Float32List cPlain,
            String what,
          ) {
            final apart = relative(g, c);
            final base = math.max(2 * relative(gPlain, cPlain), 1e-2);
            expect(apart, lessThan(base), reason: what);
            expect(apart, lessThan(relative(c, cPlain) / 4), reason: what);
          }

          for (var s = 1; s < cpuSlots.length; s++) {
            agree(gpuSlots[s], cpuSlots[s], gpuPlain[s], cpuPlain[s], 's$s');
          }
          agree(gpuLogits, cpuLogits, gpuPlainLogits, cpuPlainLogits, 'logits');
        });
      }

      test('add mode adds exactly what the CPU adds', () async {
        final last = gpu.config.numLayers - 1;
        final v = vector(ControlMode.add, scale: 3, first: last);
        final (plain, _) = await run(gpu, null);
        final (steered, _) = await run(gpu, v);
        final expected = Float32List.fromList(plain[last + 1]);
        v.apply(expected, ids.length, last);
        expect(steered[last + 1], orderedEquals(expected));
      });

      test('project mode leaves no component along the direction', () async {
        final h = gpu.config.hiddenSize;
        final v = vector(ControlMode.project);
        final (slots, _) = await run(gpu, v);
        for (var l = 0; l < gpu.config.numLayers; l++) {
          final d = v.directions[l]!;
          final unit = Float32List.fromList([for (final x in d) x / _norm(d)]);
          for (var t = 0; t < ids.length; t++) {
            final row = Float32List.sublistView(
              slots[l + 1],
              t * h,
              (t + 1) * h,
            );
            expect(
              _dot(slots[l + 1], t * h, unit).abs(),
              lessThan(1e-5 * _norm(row)),
              reason: 'layer $l token $t',
            );
          }
        }
      });

      test('setting then clearing matches a fresh GPU model', () async {
        gpu.setControlVector(vector(ControlMode.project));
        gpu.setControlVector(null);
        await gpu.forward(ids);
        final logits = Float32List.fromList(gpu.logits);
        await gpu.close();
        gpu = await _load(backbone, gpu: true);
        await gpu.forward(ids);
        expect(logits, orderedEquals(gpu.logits));
      });

      test('a vector replaced after reset takes effect', () async {
        final a = vector(ControlMode.add, scale: 2);
        final b = vector(ControlMode.project, first: 1);
        final (_, onlyB) = await run(gpu, b);
        await run(gpu, a);
        final (_, afterA) = await run(gpu, b);
        expect(afterA, orderedEquals(onlyB));
        // One buffer per vector, released when replaced or cleared.
        final live = GpuForward.liveBuffers;
        await run(gpu, a);
        expect(GpuForward.liveBuffers, live);
        await run(gpu, null);
        expect(GpuForward.liveBuffers, live - 1);
      });
    },
    skip: !NativeKernels.isAvailable
        ? 'requires native kernels'
        : !hostHasGpu
        ? noGpuSkip
        : gpuStarts
        ? false
        : gpuDoesNotStartSkip,
  );
}

Future<Lfm3Model> _load(String backbone, {bool gpu = false}) => Lfm3Model.load(
  backbone,
  maxSeq: 64,
  maxChunk: 4,
  weights: WeightFormat.q8,
  q8CachePath: '$backbone/lfm3-q8.cache',
  gpu: gpu,
);
