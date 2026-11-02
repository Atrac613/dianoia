import 'dart:typed_data';

/// The UI never owns inference state or performs model arithmetic.
abstract interface class AudioBackend {
  Future<String> load(String directory);
  Future<void> begin(String? language);
  Future<void> addPcm(Float32List samples);
  Future<String> finish();
  Future<Uint8List> synthesize(String text, String language, String voice);
  Future<void> cancel();
  Future<void> close();
}
