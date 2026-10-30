/// The chat's state on the UI side: the messages shown, whether the model
/// is ready or replying, and the worker isolate that runs it.
library;

import 'dart:io';
import 'dart:isolate';

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

class ChatController extends ChangeNotifier {
  final messages = <ChatMessage>[];

  /// What the worker is doing before it is ready, or why it failed.
  String status = 'Starting…';
  bool failed = false;

  /// `GPU` or `CPU` once the model is loaded.
  String? backend;
  bool replying = false;

  SendPort? _commands;
  Isolate? _worker;
  ReceivePort? _events;

  bool get ready => _commands != null;

  Future<void> start() async {
    final events = _events = ReceivePort();
    events.listen((event) => _handle(event as ChatEvent));
    _worker = await Isolate.spawn(runChatWorker, (
      events.sendPort,
      await _modelLocation(),
    ));
  }

  void send(String text) {
    final commands = _commands;
    if (commands == null || replying || text.trim().isEmpty) return;
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
      case WorkerProgress(:final status):
        this.status = status;
      case WorkerReady(:final commands, :final backend):
        _commands = commands;
        this.backend = backend;
      case WorkerFailed(:final message):
        status = message;
        failed = true;
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

  /// See [ModelLocation]. Resolved here because path_provider talks to the
  /// platform through the UI isolate.
  static Future<ModelLocation> _modelLocation() async {
    const name = ModelLocation.bundleName;
    if (Platform.isAndroid) {
      final external = await getExternalStorageDirectory();
      final internal = await getApplicationSupportDirectory();
      return ModelLocation(
        dropDir: '${external!.path}/$name',
        modelDir: '${internal.path}/$name',
      );
    }
    final documents = await getApplicationDocumentsDirectory();
    final dir = '${documents.path}/$name';
    return ModelLocation(dropDir: dir, modelDir: dir);
  }

  @override
  void dispose() {
    _worker?.kill(priority: Isolate.immediate);
    _events?.close();
    super.dispose();
  }
}
