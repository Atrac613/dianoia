/// Control vectors: per-layer directions applied to the residual stream
/// leaving a layer, as llama.cpp's `--control-vector` does (activation
/// steering). Experimental; `Lfm3Model.setControlVector` applies one on the
/// CPU path and on the GPU (Metal or Vulkan, the `steer` kernel).
///
/// Layer `l` is the zero-based index of the layer whose output is steered:
/// its direction acts on what `Lfm3Model.forward`'s capture reports in slot
/// `l + 1`, on every token row. Two modes:
///
///  * [ControlMode.add]: `x += scale * d`;
///  * [ControlMode.project]: `x -= scale * (x · u) u`, with `u = d / |d|`;
///    scale 1 removes the component along `d` entirely.
///
/// Files hold the directions only, as F32 tensors `direction.<l>` of hidden
/// size; mode, scale and layer range are chosen when applying. [save] writes
/// safetensors; [ControlVector.load] reads that and llama.cpp's GGUF control
/// vectors. In both, `direction.<l>` steers the output of zero-based layer
/// `l`, which is how llama.cpp's runtime applies a file (llama.cpp never
/// stores layer 0, which this format allows).
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import 'gguf.dart';
import 'safetensors.dart';

/// How a [ControlVector] changes the residual stream.
enum ControlMode {
  /// `x += scale * d`.
  add,

  /// `x -= scale * (x · u) u` with the unit direction `u`.
  project,
}

final class ControlVector {
  /// Directions by zero-based layer, each of [hiddenSize] finite floats;
  /// layers without one are left alone. Only layers from [firstLayer] to
  /// [lastLayer] (inclusive; null: no upper bound) are steered. [scale]
  /// must be finite.
  ///
  /// In [ControlMode.project] a direction of zero norm is an error: it has
  /// no unit direction to remove.
  ControlVector(
    Map<int, Float32List> directions, {
    required this.hiddenSize,
    this.mode = ControlMode.add,
    this.scale = 1.0,
    this.firstLayer = 0,
    this.lastLayer,
  }) : directions = Map.unmodifiable({
         for (final e in directions.entries)
           e.key: Float32List.fromList(e.value),
       }) {
    if (hiddenSize <= 0) {
      throw ArgumentError.value(hiddenSize, 'hiddenSize', 'must be positive');
    }
    if (!scale.isFinite) {
      throw ArgumentError.value(scale, 'scale', 'must be finite');
    }
    if (firstLayer < 0 || (lastLayer != null && lastLayer! < firstLayer)) {
      throw ArgumentError(
        'invalid layer range $firstLayer..${lastLayer ?? ''}',
      );
    }
    for (final MapEntry(key: layer, value: d) in this.directions.entries) {
      if (layer < 0) {
        throw ArgumentError.value(layer, 'layer', 'must not be negative');
      }
      if (d.length != hiddenSize || d.any((v) => !v.isFinite)) {
        throw ArgumentError(
          'direction for layer $layer must contain $hiddenSize finite values',
        );
      }
      final Float32List applied;
      if (mode == ControlMode.add) {
        applied = Float32List(hiddenSize);
        for (var j = 0; j < hiddenSize; j++) {
          applied[j] = scale * d[j];
        }
        if (applied.any((v) => !v.isFinite)) {
          throw ArgumentError('scale $scale overflows layer $layer in float32');
        }
      } else {
        final norm = _norm(d);
        if (norm == 0 || !norm.isFinite) {
          throw ArgumentError(
            'direction for layer $layer has no unit direction to project out',
          );
        }
        applied = Float32List(hiddenSize);
        for (var j = 0; j < hiddenSize; j++) {
          applied[j] = d[j] / norm;
        }
      }
      _applied[layer] = applied;
    }
  }

  /// Reads the directions of a control-vector file: safetensors as [save]
  /// writes it, or llama.cpp's GGUF (which needs the native kernels, for
  /// the mapping). Every tensor must be an F32 vector named `direction.<l>`.
  static ControlVector load(
    String path, {
    ControlMode mode = ControlMode.add,
    double scale = 1.0,
    int firstLayer = 0,
    int? lastLayer,
  }) {
    final raw = _isGguf(path) ? _readGguf(path) : _readSafetensors(path);
    if (raw.isEmpty) throw FormatException('$path holds no directions');
    final sizes = {for (final d in raw.values) d.length};
    if (sizes.length != 1) {
      throw FormatException('$path has directions of sizes $sizes');
    }
    return ControlVector(
      raw,
      hiddenSize: sizes.single,
      mode: mode,
      scale: scale,
      firstLayer: firstLayer,
      lastLayer: lastLayer,
    );
  }

  /// This vector's directions with another mode, scale or layer range.
  ControlVector withSettings({
    ControlMode? mode,
    double? scale,
    int? firstLayer,
    int? lastLayer,
  }) => ControlVector(
    directions,
    hiddenSize: hiddenSize,
    mode: mode ?? this.mode,
    scale: scale ?? this.scale,
    firstLayer: firstLayer ?? this.firstLayer,
    lastLayer: lastLayer ?? this.lastLayer,
  );

  final int hiddenSize;
  final ControlMode mode;
  final double scale;
  final int firstLayer;
  final int? lastLayer;

  /// The directions as given, by zero-based layer.
  final Map<int, Float32List> directions;

  /// Per layer: `scale * d` for add, the unit direction for project.
  final _applied = <int, Float32List>{};

  /// Whether [apply] changes the output of [layer].
  bool steers(int layer) =>
      layer >= firstLayer &&
      (lastLayer == null || layer <= lastLayer!) &&
      _applied.containsKey(layer);

  /// What [apply] uses for [layer], or null for a layer it [steers] not:
  /// `scale * d` in float32 for [ControlMode.add], the unit direction for
  /// [ControlMode.project]. The GPU path uploads these. Do not modify.
  Float32List? appliedVector(int layer) =>
      steers(layer) ? _applied[layer] : null;

  /// The highest layer with a direction, or -1 without any.
  int get maxLayer => directions.keys.fold(-1, math.max);

  /// Steers [rowCount] rows of [hiddenSize] at the start of [rows], the
  /// residual stream leaving [layer], in place. Does nothing for a layer it
  /// does not [steers]. Each row is computed on its own, in a fixed order,
  /// so the result does not depend on how tokens are chunked.
  void apply(Float32List rows, int rowCount, int layer) {
    if (!steers(layer)) return;
    final v = _applied[layer]!;
    final h = hiddenSize;
    if (rowCount < 0 || rowCount * h > rows.length) {
      throw RangeError('$rowCount rows of $h exceed ${rows.length} values');
    }
    if (mode == ControlMode.add) {
      for (var t = 0; t < rowCount; t++) {
        final o = t * h;
        for (var j = 0; j < h; j++) {
          rows[o + j] = rows[o + j] + v[j];
        }
      }
      return;
    }
    for (var t = 0; t < rowCount; t++) {
      final o = t * h;
      var dot = 0.0;
      for (var j = 0; j < h; j++) {
        dot += rows[o + j] * v[j];
      }
      final c = scale * dot;
      for (var j = 0; j < h; j++) {
        rows[o + j] = rows[o + j] - c * v[j];
      }
    }
  }

  /// Writes the directions (not mode, scale or range) to [path] as
  /// safetensors: F32 tensors `direction.<l>` of shape [hiddenSize], in
  /// layer order. Written through a temporary file and a rename.
  void save(String path) {
    final layers = directions.keys.toList()..sort();
    final temp = '$path.partial';
    writeSafetensorsF32(temp, {
      for (final l in layers) 'direction.$l': ([hiddenSize], directions[l]!),
    });
    File(temp).renameSync(path);
  }

  /// Identifies what this vector does to the arithmetic: the mode, scale,
  /// range and the directions of the layers it steers. Equal for vectors
  /// that steer identically.
  late final String fingerprint = () {
    final layers = _applied.keys.where(steers).toList()..sort();
    final out = BytesBuilder(copy: false);
    final scalar = ByteData(8)..setFloat64(0, scale, Endian.little);
    out
      ..add('${mode.name} ${layers.join(',')} '.codeUnits)
      ..add(scalar.buffer.asUint8List());
    for (final l in layers) {
      final d = directions[l]!;
      out.add(d.buffer.asUint8List(d.offsetInBytes, d.lengthInBytes));
    }
    return sha256.convert(out.takeBytes()).toString();
  }();

  static double _norm(Float32List d) {
    var sum = 0.0;
    for (final v in d) {
      sum += v * v;
    }
    return math.sqrt(sum);
  }

  static bool _isGguf(String path) {
    final file = File(path).openSync();
    try {
      final magic = file.readSync(4);
      return magic.length == 4 && String.fromCharCodes(magic) == 'GGUF';
    } finally {
      file.closeSync();
    }
  }

  /// Adds [values] as the direction named [name] to [into].
  static void _put(
    Map<int, Float32List> into,
    String path,
    String name,
    Float32List Function()? values,
  ) {
    final match = RegExp(r'^direction\.(\d+)$').firstMatch(name);
    if (match == null) {
      throw FormatException('$path: unexpected tensor $name');
    }
    if (values == null) {
      throw FormatException('$path: $name is not an F32 vector');
    }
    final layer = int.parse(match[1]!);
    if (into.containsKey(layer)) {
      throw FormatException('$path: two directions for layer $layer');
    }
    into[layer] = values();
  }

  static Map<int, Float32List> _readGguf(String path) {
    final gguf = GgufFile.open(path);
    try {
      final out = <int, Float32List>{};
      for (final t in gguf.tensors.values) {
        final vector = t.type == GgufType.f32 && t.dims.length == 1;
        _put(out, path, t.name, vector ? () => gguf.f32(t) : null);
      }
      return out;
    } finally {
      gguf.close();
    }
  }

  static Map<int, Float32List> _readSafetensors(String path) {
    final st = SafetensorsFile.open(path);
    try {
      final out = <int, Float32List>{};
      for (final MapEntry(key: name, value: info) in st.tensors.entries) {
        final vector = info.dtype == 'F32' && info.shape.length == 1;
        _put(
          out,
          path,
          name,
          vector ? () => st.readF32(name, info.shape) : null,
        );
      }
      return out;
    } finally {
      st.close();
    }
  }
}

/// How well one layer's mean-difference direction separates two sets of
/// residuals (each a row of hidden size, such as the residual leaving that
/// layer at each prompt's last token).
///
/// The direction is `d = mean(A) - mean(B)`. [separation] is the
/// sensitivity index
///
///     d' = (mean proj A - mean proj B) / sqrt((sd A² + sd B²) / 2)
///
/// with sample standard deviations, over leave-one-out projections: each
/// row is projected on the unit direction built without it, from
/// `mean(A without r) - mean(B)` for a row r of A and
/// `mean(A) - mean(B without r)` for a row of B. A direction fitted to few
/// rows in many dimensions separates those rows by construction; held out,
/// a direction that is mostly noise scores near zero or below (it does not
/// generalise), yet project mode would still remove each row's full
/// component along it.
final class LayerSeparation {
  LayerSeparation._(
    this.direction,
    this.norm,
    this.meanA,
    this.sdA,
    this.meanB,
    this.sdB,
    this.separation,
  );

  /// Measures [setA] against [setB]; both non-empty, all rows the same
  /// length. Costs O(rows × hidden size): each held-out mean comes from the
  /// full one, `(n mean - r) / (n - 1)`.
  ///
  /// Degenerate cases:
  /// - a zero full-sample direction: means 0, [separation] 0;
  /// - a set of one row cannot be held out: its mean and deviation are
  ///   null, and so is [separation];
  /// - a zero held-out direction projects its row to 0;
  /// - spreads of zero, or at the level of rounding (1e-9 of the means):
  ///   [separation] is plus or minus infinity by the sign of the gap, or 0
  ///   without one.
  factory LayerSeparation.measure(
    List<Float32List> setA,
    List<Float32List> setB,
  ) {
    if (setA.isEmpty || setB.isEmpty) {
      throw ArgumentError('both sets need at least one row');
    }
    final h = setA.first.length;
    if ([...setA, ...setB].any((r) => r.length != h)) {
      throw ArgumentError('rows of different lengths');
    }
    Float64List mean(List<Float32List> rows) {
      final m = Float64List(h);
      for (final r in rows) {
        for (var j = 0; j < h; j++) {
          m[j] += r[j];
        }
      }
      for (var j = 0; j < h; j++) {
        m[j] /= rows.length;
      }
      return m;
    }

    final ma = mean(setA), mb = mean(setB);
    final direction = Float32List(h);
    var squares = 0.0;
    for (var j = 0; j < h; j++) {
      direction[j] = ma[j] - mb[j];
      squares += direction[j] * direction[j];
    }
    final norm = math.sqrt(squares);
    if (norm == 0) {
      return LayerSeparation._(direction, 0, 0, null, 0, null, 0);
    }

    // Each row of [rows] (whose mean is [own]) projected on the unit
    // direction between [own] without that row and [other], oriented from
    // B to A by [sign] (1 for A's rows, -1 for B's).
    final held = Float64List(h);
    List<double>? heldOut(
      List<Float32List> rows,
      Float64List own,
      Float64List other,
      double sign,
    ) {
      final n = rows.length;
      if (n < 2) return null;
      return [
        for (final r in rows)
          () {
            var squares = 0.0, dot = 0.0;
            for (var j = 0; j < h; j++) {
              final without = (n * own[j] - r[j]) / (n - 1);
              held[j] = sign * (without - other[j]);
              squares += held[j] * held[j];
              dot += r[j] * held[j];
            }
            return squares == 0 ? 0.0 : dot / math.sqrt(squares);
          }(),
      ];
    }

    (double?, double?) stats(List<double>? p) {
      if (p == null) return (null, null);
      final m = p.reduce((a, b) => a + b) / p.length;
      final v =
          p.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) /
          (p.length - 1);
      return (m, math.sqrt(v));
    }

    final (meanA, sdA) = stats(heldOut(setA, ma, mb, 1));
    final (meanB, sdB) = stats(heldOut(setB, mb, ma, -1));
    double? separation;
    if (sdA != null && sdB != null) {
      final pooled = math.sqrt((sdA * sdA + sdB * sdB) / 2);
      final gap = meanA! - meanB!;
      // Rows that are equal within a set (a shallow layer that sees only
      // a shared prompt ending) spread by rounding alone; that is no spread.
      final rounding = 1e-9 * (meanA.abs() + meanB.abs());
      separation = pooled > rounding
          ? gap / pooled
          : gap.abs() <= rounding
          ? 0
          : gap.sign * double.infinity;
    }
    return LayerSeparation._(
      direction,
      norm,
      meanA,
      sdA,
      meanB,
      sdB,
      separation,
    );
  }

  /// `mean(A) - mean(B)` over every row, in float32: the direction a
  /// control vector keeps.
  final Float32List direction;

  /// Its Euclidean norm.
  final double norm;

  /// Mean and sample standard deviation of each set's held-out projections;
  /// null for a set of one row. 0 and null for a zero direction.
  final double? meanA, meanB;
  final double? sdA, sdB;

  /// Held-out d': negative when the direction does not generalise, null
  /// when a set has a single row.
  final double? separation;

  /// Whether the layer belongs in a control vector at [minimum] d': it has
  /// a direction and, unless [minimum] is 0 or less, a measured d' of at
  /// least [minimum].
  bool passes(double minimum) =>
      norm > 0 &&
      (minimum <= 0 || (separation != null && separation! >= minimum));

  /// The first layer from which every later layer [passes] [minimum], or
  /// null when the last layer does not.
  static int? passingFrom(List<LayerSeparation> layers, double minimum) {
    int? first;
    for (var l = layers.length - 1; l >= 0 && layers[l].passes(minimum); l--) {
      first = l;
    }
    return first;
  }
}
