/// Experimental control vectors for the text chat (docs/control_vectors.md
/// in the package): named vector files the user copies into the app's
/// `control_vectors` folder, each `<name>.safetensors` with an optional
/// `<name>.json` of settings.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart' show ControlMode, ControlVector;

/// How a named vector is applied: `{"mode": "project" | "add", "scale":
/// 1.0, "first_layer": 8, "last_layer": null}`, every key optional.
final class SteeringSettings {
  const SteeringSettings({
    this.mode = ControlMode.project,
    this.scale = 1,
    this.firstLayer = 0,
    this.lastLayer,
  });

  /// Without a settings file: project mode at scale 1 on every layer the
  /// file has a direction for, which removes a direction rather than adds
  /// an unbounded amount of it.
  static const defaults = SteeringSettings();

  final ControlMode mode;
  final double scale;
  final int firstLayer;
  final int? lastLayer;

  /// Parses [json]; throws [FormatException] for anything malformed.
  static SteeringSettings parse(String json) {
    final Object? value;
    try {
      value = jsonDecode(json);
    } on FormatException catch (e) {
      throw FormatException('invalid JSON: ${e.message}');
    }
    if (value is! Map<String, Object?>) {
      throw const FormatException('expected a JSON object');
    }
    final map = value;
    const keys = {'mode', 'scale', 'first_layer', 'last_layer'};
    final unknown = map.keys.where((k) => !keys.contains(k)).toList();
    if (unknown.isNotEmpty) {
      throw FormatException('unknown settings: ${unknown.join(', ')}');
    }
    final mode = switch (map['mode']) {
      null => ControlMode.project,
      'project' => ControlMode.project,
      'add' => ControlMode.add,
      final other => throw FormatException(
        'mode must be "project" or "add", not $other',
      ),
    };
    final scale = switch (map['scale']) {
      null => 1.0,
      final num n when n.isFinite => n.toDouble(),
      final other => throw FormatException(
        'scale must be a finite number, not $other',
      ),
    };
    int? layer(String key) => switch (map[key]) {
      null => null,
      final int n when n >= 0 => n,
      final other => throw FormatException(
        '$key must be a layer number, not $other',
      ),
    };
    final first = layer('first_layer') ?? 0;
    final last = layer('last_layer');
    if (last != null && last < first) {
      throw FormatException('last_layer $last is before first_layer $first');
    }
    return SteeringSettings(
      mode: mode,
      scale: scale,
      firstLayer: first,
      lastLayer: last,
    );
  }
}

/// The names of the vectors in [dir] (`<name>.safetensors`), sorted; empty
/// when there is no such folder.
List<String> listControlVectors(String? dir) {
  if (dir == null) return const [];
  final folder = Directory(dir);
  if (!folder.existsSync()) return const [];
  const suffix = '.safetensors';
  return [
    for (final entity in folder.listSync())
      if (entity is File && entity.path.endsWith(suffix))
        entity.uri.pathSegments.last.substring(
          0,
          entity.uri.pathSegments.last.length - suffix.length,
        ),
  ]..sort();
}

/// Loads the vector [name] from [dir] with the settings of `<name>.json`
/// next to it, or [SteeringSettings.defaults] without one. Throws
/// [FormatException] or [FileSystemException] for a file it cannot use.
ControlVector loadControlVector(String dir, String name) {
  final settingsFile = File('$dir/$name.json');
  final settings = settingsFile.existsSync()
      ? SteeringSettings.parse(settingsFile.readAsStringSync())
      : SteeringSettings.defaults;
  try {
    return ControlVector.load(
      '$dir/$name.safetensors',
      mode: settings.mode,
      scale: settings.scale,
      firstLayer: settings.firstLayer,
      lastLayer: settings.lastLayer,
    );
  } on ArgumentError catch (e) {
    throw FormatException('$name: ${e.message}');
  }
}
