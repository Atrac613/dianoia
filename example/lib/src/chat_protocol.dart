/// The messages between the UI isolate and the chat worker isolate
/// (chat_worker.dart). The model, its GPU state and the native kernels live
/// in the worker; the UI only sends text and receives text.
library;

import 'dart:isolate';

/// Where the app looks for the model bundle and where it maps it from.
final class ModelLocation {
  const ModelLocation({required this.dropDir, required this.modelDir});

  /// The bundle's directory name, as `dianoia import-gguf` writes it.
  static const bundleName = 'LFM3-1.7B-QAD-Q4';

  /// Where the bundle is put: the app's Documents on iOS and macOS (Finder
  /// and the Files app can copy into it), its external files directory on
  /// Android (adb push can write there).
  final String dropDir;

  /// Where the model is mapped from: [dropDir] itself, or on Android an
  /// internal copy of it.
  final String modelDir;
}

// UI to worker.

sealed class ChatCommand {
  const ChatCommand();
}

final class SendMessage extends ChatCommand {
  const SendMessage(this.text);
  final String text;
}

/// Ends the reply being generated, keeping what is there.
final class StopReply extends ChatCommand {
  const StopReply();
}

/// Forgets the conversation; the system prompt's state is kept.
final class NewChat extends ChatCommand {
  const NewChat();
}

// Worker to UI.

sealed class ChatEvent {
  const ChatEvent();
}

final class WorkerProgress extends ChatEvent {
  const WorkerProgress(this.status);
  final String status;
}

final class WorkerReady extends ChatEvent {
  const WorkerReady(this.commands, this.backend);
  final SendPort commands;

  /// `GPU` or `CPU`.
  final String backend;
}

/// The worker could not start; [message] says why.
final class WorkerFailed extends ChatEvent {
  const WorkerFailed(this.message);
  final String message;
}

/// The reply so far.
final class ReplyText extends ChatEvent {
  const ReplyText(this.text);
  final String text;
}

final class ReplyDone extends ChatEvent {
  const ReplyDone({
    required this.text,
    required this.tokens,
    required this.firstTokenMs,
    required this.tokensPerSecond,
  });

  final String text;
  final int tokens;
  final int firstTokenMs;
  final double tokensPerSecond;
}

/// The message got no reply; [message] says why.
final class ReplyFailed extends ChatEvent {
  const ReplyFailed(this.message);
  final String message;
}
