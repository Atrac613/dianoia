/// The messages between the UI isolate and the chat worker isolate
/// (chat_worker.dart). The model, its GPU state and the native kernels live
/// in the worker; the UI only sends text and commands and receives events.
library;

import 'dart:isolate';

import 'package:dianoia/dianoia.dart';

/// Where the app keeps models, resolved on the UI isolate (path_provider
/// talks to the platform from there).
final class ModelLocation {
  const ModelLocation({
    required this.storeDir,
    required this.dropDir,
    required this.copyDir,
  });

  /// The [ModelStore] the app downloads into.
  final String storeDir;

  /// Where a developer may put a ready bundle instead (see README.md): the
  /// app's Documents on iOS and macOS, its external files directory on
  /// Android.
  final String dropDir;

  /// Where a bundle from [dropDir] is mapped from: [dropDir] itself, or on
  /// Android an internal copy of it.
  final String copyDir;
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

/// Downloads and builds the model, after the user agreed to.
final class InstallModel extends ChatCommand {
  const InstallModel();
}

/// Stops the download; a later [InstallModel] resumes it.
final class CancelInstall extends ChatCommand {
  const CancelInstall();
}

// Worker to UI.

sealed class ChatEvent {
  const ChatEvent();
}

/// The first event: where commands go.
final class WorkerStarted extends ChatEvent {
  const WorkerStarted(this.commands);
  final SendPort commands;
}

final class WorkerProgress extends ChatEvent {
  const WorkerProgress(this.status);
  final String status;
}

/// No model on the device: the UI should offer to download [spec].
final class ModelNeeded extends ChatEvent {
  const ModelNeeded(this.spec);
  final ModelSpec spec;
}

final class InstallUpdate extends ChatEvent {
  const InstallUpdate(this.stage, this.bytes, this.totalBytes);
  final InstallStage stage;
  final int bytes, totalBytes;
}

/// The install stopped with an error; another [InstallModel] resumes it.
final class InstallFailed extends ChatEvent {
  const InstallFailed(this.message);
  final String message;
}

final class WorkerReady extends ChatEvent {
  const WorkerReady(this.backend);

  /// `GPU` or `CPU`.
  final String backend;
}

/// The model could not be loaded; [message] says why.
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
