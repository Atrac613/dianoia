// The control-vector extensions of LocalModel and LocalVisionModel are
// lab-only: through package:dianoia/dianoia.dart, neither has a
// `controlVector` or `setControlVector`. If dianoia.dart exported
// LocalModelControlVector or LocalVisionModelControlVector, the probe
// extensions below would make those member accesses ambiguous and this file
// would not compile.
import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

extension _Probe on LocalModel {
  String get controlVector => 'probe';
  String setControlVector(Object? vector) => 'probe';
}

extension _VisionProbe on LocalVisionModel {
  String get controlVector => 'probe';
  String setControlVector(Object? vector) => 'probe';
}

String _members(LocalModel model, LocalVisionModel vision) =>
    '${model.controlVector} ${model.setControlVector(null)} '
    '${vision.controlVector} ${vision.setControlVector(null)}';

void main() {
  test('dianoia.dart does not expose control vectors on local models', () {
    // Compiling is the check; referencing the probe keeps it in use.
    expect(_members, isA<Function>());
  });
}
