import 'dart:typed_data';
import 'package:dianoia/dianoia.dart';

final class AudioVoiceReply {
  const AudioVoiceReply(this.text, this.wav, this.turn);
  final String text;
  final Uint8List wav;
  final int turn;
}

/// The UI never owns inference state or performs model arithmetic.
abstract interface class AudioBackend {
  Future<String?> installed(String storeDirectory);
  Future<String> install(
    String storeDirectory,
    void Function(InstallProgress) onProgress,
  );
  Future<String> load(String directory);
  Future<void> begin(
    String? language, {
    bool conversation = false,
    bool live = false,
    void Function(String)? onLiveText,
  });
  Future<void> addPcm(Float32List samples);
  Future<String> finish();
  Future<AudioVoiceReply> finishConversation(void Function(String) onText);
  Future<void> acceptConversationReply();
  Future<void> newConversation();
  Future<Uint8List> synthesize(String text, String language, String voice);
  Future<void> cancel();
  Future<void> close();
}
