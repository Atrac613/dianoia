/// The chat's state on the UI side: whether the model is there, being
/// installed or ready, the messages shown, and the worker isolate that runs
/// the model.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import 'chat_protocol.dart';
import 'chat_worker.dart';
import 'example_model.dart';
import 'model_worker_gate.dart';

enum ChatRole { user, assistant }

final class ChatMessage {
  ChatMessage(this.role, this.text, {ImageAttachment? image})
    : image = image == null
          ? null
          : image.thumbnail ?? ImageThumbnail(image.name, null);

  final ImageThumbnail? image;

  final ChatRole role;
  String text;

  /// For a finished reply: tokens, speed and time to the first token.
  String? stats;

  bool pending = false;
  String progress = 'Preparing reply…';

  /// Set when the reply failed.
  bool failed = false;
}

enum Setup { starting, needsModel, installing, loading, ready, failed }

class ChatController extends ChangeNotifier {
  ChatController({this.model = ExampleModel.lfm3});

  final ExampleModel model;
  ModelLocation? location;
  bool _disposed = false;
  final messages = <ChatMessage>[];

  Setup setup = Setup.starting;

  /// What the worker is doing while loading, or why it failed.
  String status = 'Starting…';

  /// The model to offer while [setup] is [Setup.needsModel].
  InstallableModel? spec;

  /// The install's progress, and why the last attempt failed.
  InstallStage installStage = InstallStage.downloading;
  int installBytes = 0, installTotal = 0;
  String? installError;

  /// `GPU` or `CPU` once the model is loaded.
  String? backend;
  bool replying = false;

  SendPort? _commands;
  ReceivePort? _events;

  bool get ready => setup == Setup.ready;

  Future<void> start() async {
    final events = _events = ReceivePort();
    events.listen((event) => _handle(event as ChatEvent));
    final lease = ModelWorkerLease.reserve();
    final exit = ReceivePort();
    try {
      final location = this.location = await _modelLocation();
      await lease.ready;
      if (_disposed) {
        events.close();
        exit.close();
        lease.release();
        return;
      }
      await Isolate.spawn(runChatWorker, (
        events.sendPort,
        location,
        model,
      ), onExit: exit.sendPort);
      unawaited(
        exit.first.then((_) {
          exit.close();
          lease.release();
        }),
      );
    } catch (e) {
      exit.close();
      lease.release();
      events.close();
      if (_disposed) return;
      setup = Setup.failed;
      status = '$e';
      notifyListeners();
    }
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

  bool send(String text, {ImageAttachment? image}) {
    final commands = _commands;
    if (!ready ||
        commands == null ||
        replying ||
        (text.trim().isEmpty && image == null) ||
        (image != null &&
            (!model.acceptsImages ||
                image.bytes.length > ImageAttachment.maxBytes))) {
      return false;
    }
    messages
      ..add(ChatMessage(ChatRole.user, text.trim(), image: image))
      ..add(ChatMessage(ChatRole.assistant, '')..pending = true);
    replying = true;
    commands.send(SendMessage(text.trim(), image: image));
    notifyListeners();
    return true;
  }

  void stop() => _commands?.send(const StopReply());

  void newChat() {
    if (replying) return;
    messages.clear();
    _commands?.send(const NewChat());
    notifyListeners();
  }

  void _handle(ChatEvent event) {
    if (_disposed) {
      // Isolate.spawn may complete after the route has already been popped.
      if (event case WorkerStarted(:final commands)) {
        commands.send(const CloseWorker());
        _events?.close();
      }
      return;
    }
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
      case ReplyProgress(:final status):
        messages.last.progress = status;
      case ReplyText(:final text):
        messages.last.text = text;
      case ReplyDone(:final text, :final tokens, :final firstTokenMs):
        messages.last
          ..pending = false
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
          ..pending = false
          ..failed = true
          ..stats = message;
        replying = false;
    }
    notifyListeners();
  }

  Future<ModelLocation> _modelLocation() async {
    final name = model.directoryName;
    if (Platform.isWindows) {
      // The support directory is in the roaming profile, which is no place
      // for a multi-gigabyte model, and Documents is the user's own folder: keep both
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
    _disposed = true;
    final commands = _commands;
    if (commands != null) {
      commands.send(const CloseWorker());
      _events?.close();
    }
    // Without a command port yet, keep listening for WorkerStarted so a
    // worker that is still starting can also shut down cleanly.
    super.dispose();
  }
}
