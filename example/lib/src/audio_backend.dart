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

  /// Starts audio input: transcription by default, a conversation turn, a
  /// live transcript, or ([question]) a spoken question kept for
  /// [askAboutPhoto].
  Future<void> begin(
    String? language, {
    bool conversation = false,
    bool live = false,
    bool question = false,
    void Function(String)? onLiveText,
  });
  Future<void> addPcm(Float32List samples);
  Future<String> finish();
  Future<AudioVoiceReply> finishConversation(void Function(String) onText);
  Future<void> acceptConversationReply();
  Future<void> newConversation();
  Future<Uint8List> synthesize(String text, String language, String voice);

  /// LFM3-Omni only: answers about an encoded PNG or JPEG [image], asked
  /// with the recorded spoken question (when [spoken]) and/or [text]. With
  /// [speak] the reply is interleaved text and speech, else text only.
  /// With [previousTurns] > 0 the question follows that many answered
  /// questions about the previous photo, which is not encoded again, and
  /// [image] may be null; with 0, [image] starts a new conversation. The
  /// reply's turn counts the answered questions about the photo.
  Future<AudioVoiceReply> askAboutPhoto(
    Uint8List? image, {
    String? text,
    required bool spoken,
    required bool speak,
    int previousTurns = 0,
    required void Function(String) onText,
  });
  Future<void> cancel();
  Future<void> close();
}
