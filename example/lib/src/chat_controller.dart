/// The chat's state on the UI side: whether the model is there, being
/// installed or ready, the messages shown, and the worker isolate that runs
/// the model.
library;

import 'dart:io';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import 'chat_protocol.dart';
import 'chat_worker.dart';

enum ChatRole { user, assistant }

final class ChatMessage {
  ChatMessage(this.role, this.text);

  final ChatRole role;
  String text;

  /// For a finished reply: tokens, speed and time to the first token.
  String? stats;

  /// Set when the reply failed.
  bool failed = false;
}

enum Setup { starting, needsModel, installing, loading, ready, failed }

class ChatController extends ChangeNotifier {
  final messages = <ChatMessage>[];

  Setup setup = Setup.starting;

  /// What the worker is doing while loading, or why it failed.
  String status = 'Starting…';

  /// The model to offer while [setup] is [Setup.needsModel].
  ModelSpec? spec;

  /// The install's progress, and why the last attempt failed.
  InstallStage installStage = InstallStage.downloading;
  int installBytes = 0, installTotal = 0;
  String? installError;

  /// `GPU` or `CPU` once the model is loaded.
  String? backend;
  bool replying = false;

  SendPort? _commands;
  Isolate? _worker;
  ReceivePort? _events;

  bool get ready => setup == Setup.ready;

  Future<void> start() async {
    final events = _events = ReceivePort();
    events.listen((event) => _handle(event as ChatEvent));
    _worker = await Isolate.spawn(runChatWorker, (
      events.sendPort,
      await _modelLocation(),
    ));
  }

  /// Downloads and builds the model: only when the user asked to.
  void installModel() {
    installError = null;
    installBytes = 0;
    setup = Setup.installing;
    _commands?.send(const InstallModel());
    notifyListeners();
  }

  void cancelInstall() => _commands?.send(const CancelInstall());

  void send(String text) {
    final commands = _commands;
    if (!ready || commands == null || replying || text.trim().isEmpty) {
      return;
    }
    messages
      ..add(ChatMessage(ChatRole.user, text.trim()))
      ..add(ChatMessage(ChatRole.assistant, ''));
    replying = true;
    commands.send(SendMessage(text.trim()));
    notifyListeners();
  }

  void stop() => _commands?.send(const StopReply());

  void newChat() {
    if (replying) return;
    messages.clear();
    _commands?.send(const NewChat());
    notifyListeners();
  }

  void _handle(ChatEvent event) {
    switch (event) {
      case WorkerStarted(:final commands):
        _commands = commands;
      case WorkerProgress(:final status):
        setup = Setup.loading;
        this.status = status;
      case ModelNeeded(:final spec):
        setup = Setup.needsModel;
        this.spec = spec;
      case InstallUpdate(:final stage, :final bytes, :final totalBytes):
        setup = Setup.installing;
        installStage = stage;
        installBytes = bytes;
        installTotal = totalBytes;
      case InstallFailed(:final message):
        setup = Setup.needsModel;
        installError = message;
      case WorkerReady(:final backend):
        setup = Setup.ready;
        this.backend = backend;
      case WorkerFailed(:final message):
        setup = Setup.failed;
        status = message;
      case ReplyText(:final text):
        messages.last.text = text;
      case ReplyDone(:final text, :final tokens, :final firstTokenMs):
        messages.last
          ..text = text
          ..stats =
              '$tokens tokens · '
              '${event.tokensPerSecond.toStringAsFixed(1)} tok/s · '
              'first token $firstTokenMs ms';
        replying = false;
      case ReplyFailed(:final message):
        // Drop the empty reply; the user's message stays, marked failed.
        if (messages.last.text.isEmpty) messages.removeLast();
        messages.last
          ..failed = true
          ..stats = message;
        replying = false;
    }
    notifyListeners();
  }

  static Future<ModelLocation> _modelLocation() async {
    const name = 'LFM3-1.7B-QAD-Q4';
    if (Platform.isWindows) {
      // The support directory is in the roaming profile, which is no place
      // for a 1 GB model, and Documents is the user's own folder: keep both
      // in the app's directory under Local AppData.
      final local = (await getApplicationCacheDirectory()).path;
      final dir = '$local/$name';
      return ModelLocation(
        storeDir: '$local/models',
        dropDir: dir,
        copyDir: dir,
      );
    }
    final store = '${(await getApplicationSupportDirectory()).path}/models';
    if (Platform.isAndroid) {
      final external = await getExternalStorageDirectory();
      final internal = await getApplicationSupportDirectory();
      return ModelLocation(
        storeDir: store,
        dropDir: '${external!.path}/$name',
        copyDir: '${internal.path}/$name',
      );
    }
    final documents = await getApplicationDocumentsDirectory();
    final dir = '${documents.path}/$name';
    return ModelLocation(storeDir: store, dropDir: dir, copyDir: dir);
  }

  @override
  void dispose() {
    _worker?.kill(priority: Isolate.immediate);
    _events?.close();
    super.dispose();
  }
}
