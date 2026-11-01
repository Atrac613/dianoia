/// The chat worker: an isolate that owns the model, so that the forward
/// pass, the download's hashing and the bundle build never block the UI.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';

import 'chat_protocol.dart';
import 'example_model.dart';

const _system =
    'You are a helpful assistant running entirely on this device. '
    'Answer concisely, in the language the user writes in.';

/// `--dart-define=DIANOIA_MODEL_MIRROR=URL` fetches the model from a mirror
/// (example/tool/serve_model.dart, say) instead of Hugging Face.
const _mirror = String.fromEnvironment('DIANOIA_MODEL_MIRROR');

/// The worker isolate's entry point, with the model selected by the UI.
Future<void> runChatWorker((SendPort, ModelLocation, ExampleModel) args) async {
  final (out, location, selected) = args;
  final originalSpec = selected.downloadSpec;
  final spec = _mirror.isEmpty
      ? originalSpec
      : originalSpec.fromMirror(_mirror);
  final store = ModelStore(location.storeDir);
  final commands = ReceivePort();
  out.send(WorkerStarted(commands.sendPort));

  ChatSession? session;
  LocalModel? model;
  LocalVisionModel? visionModel;
  VisionChatSession? visionSession;
  Future<void>? loading, replying, cancelling;
  var closing = false;
  StreamSubscription<InstallProgress>? install;
  var stopRequested = false;
  var replyStarted = false;

  Future<void> open(String dir, {void Function()? onFailed}) async {
    out.send(const WorkerProgress('Loading the model…'));
    try {
      InferenceBackend backend;
      if (selected.isVision) {
        final loaded = await LocalVisionModel.loadBundle(dir);
        visionModel = loaded;
        if (closing) return;
        visionSession = loaded.chat(system: _system);
        backend = loaded.backend;
      } else {
        final loaded = await LocalModel.load(dir);
        model = loaded;
        if (closing) return;
        final chat = loaded.chat(system: _system);
        await chat.warmUp();
        session = chat;
        backend = loaded.backend;
      }
      if (!closing) out.send(WorkerReady(backend.name.toUpperCase()));
    } catch (e) {
      onFailed?.call();
      out.send(WorkerFailed('$e'));
    }
  }

  Future<void> close() async {
    closing = true;
    stopRequested = true;
    if (!replyStarted) visionSession?.cancel();
    try {
      await cancelling;
      await install?.cancel();
      await loading;
      await replying;
      await model?.close();
      await visionModel?.close();
    } finally {
      commands.close();
      Isolate.exit();
    }
  }

  Future<void> cancelInstall() async {
    final subscription = install;
    install = null;
    await subscription?.cancel();
    if (!closing) out.send(ModelNeeded(spec));
  }

  commands.listen((command) {
    if (closing) return;
    switch (command as ChatCommand) {
      case SendMessage(:final text, :final image):
        stopRequested = false;
        replyStarted = false;
        replying = _reply(
          session,
          visionSession,
          text,
          image,
          out,
          () => stopRequested,
          () => replyStarted = true,
        );
      case StopReply():
        stopRequested = true;
        if (!replyStarted) visionSession?.cancel();
      case NewChat():
        session?.clear();
        visionSession?.clear();
      case InstallModel():
        if (cancelling != null) return;
        install ??= store
            .install(spec)
            .listen(
              (p) => out.send(InstallUpdate(p.stage, p.bytes, p.totalBytes)),
              onError: (Object e) {
                install = null;
                out.send(InstallFailed('$e'));
              },
              onDone: () {
                if (install == null || closing) return; // failed or leaving
                install = null;
                loading = open(store.bundlePath(spec));
              },
            );
      case CancelInstall():
        cancelling ??= cancelInstall().whenComplete(() => cancelling = null);
      case CloseWorker():
        unawaited(close());
    }
  });

  try {
    final sideLoaded = _sideLoaded(location, out, vision: selected.isVision);
    if (sideLoaded != null) {
      await (loading = open(
        sideLoaded,
        // A copy that fails its checks is copied again on the next launch.
        onFailed: selected.isVision && location.copyDir != location.dropDir
            ? () => _forgetCopy(location.copyDir)
            : null,
      ));
    } else if (store.isInstalled(spec)) {
      await (loading = open(store.bundlePath(spec)));
    } else {
      out.send(ModelNeeded(spec));
    }
  } catch (e) {
    out.send(WorkerFailed('$e'));
  }
}

/// A bundle a developer copied into [ModelLocation.dropDir], ready to map,
/// or null.
String? _sideLoaded(
  ModelLocation location,
  SendPort out, {
  required bool vision,
}) {
  final cache = vision ? 'vl_bundle.json' : 'lfm3-q4.cache';
  if (!File('${location.dropDir}/$cache').existsSync()) return null;
  if (location.copyDir != location.dropDir) {
    out.send(const WorkerProgress('Copying the model to app storage…'));
    if (vision) {
      _mirrorVisionBundle(location.dropDir, location.copyDir);
    } else {
      _copyBundle(location.dropDir, location.copyDir);
    }
  }
  return location.copyDir;
}

Future<void> _reply(
  ChatSession? session,
  VisionChatSession? visionSession,
  String text,
  ImageAttachment? image,
  SendPort out,
  bool Function() stopRequested,
  void Function() onReplyStarted,
) async {
  try {
    if (image != null && visionSession == null) {
      throw ArgumentError('Images require the VL model.');
    }
    Future<bool> onText(String partial) async {
      onReplyStarted();
      out.send(ReplyText(partial));
      // A timer, not a microtask: it lets the stop command arrive.
      await Future<void>.delayed(Duration.zero);
      return !stopRequested();
    }

    final turn = visionSession != null
        ? await visionSession.send(
            text,
            images: [if (image != null) ChatImage.encoded(image.bytes)],
            onText: onText,
          )
        : await session!.send({
            'role': 'user',
            'content': text,
          }, onText: onText);
    out.send(
      ReplyDone(
        text: turn.text,
        tokens: turn.generatedTokens,
        firstTokenMs: (turn.prefillSeconds * 1000).round(),
        tokensPerSecond: turn.decodeSeconds > 0
            ? turn.generatedTokens / turn.decodeSeconds
            : 0,
      ),
    );
  } on VisionRequestCancelled {
    out.send(const ReplyFailed('Reply stopped.'));
  } on ContextLengthExceeded {
    out.send(
      const ReplyFailed(
        'The conversation no longer fits the model\'s context. '
        'Start a new chat.',
      ),
    );
  } catch (e) {
    out.send(ReplyFailed('$e'));
  }
}

/// Copies nested bundle files into app storage for mapping, skipping files
/// already there at the same size.
void _copyBundle(
  String from,
  String to, {
  bool overwrite = false,
  String? except,
}) {
  Directory(to).createSync(recursive: true);
  for (final file in Directory(
    from,
  ).listSync(recursive: true).whereType<File>()) {
    final relative = file.path.substring(Directory(from).path.length + 1);
    if (relative == except) continue;
    final copy = File('$to/$relative');
    copy.parent.createSync(recursive: true);
    if (!overwrite &&
        copy.existsSync() &&
        copy.lengthSync() == file.lengthSync()) {
      continue;
    }
    file.copySync(copy.path);
  }
}

/// Reuses the VL copy while its manifest matches the dropped bundle's; the
/// loader then checks every file against it. Otherwise replaces the whole
/// copy, so no file of an earlier bundle (such as its other cache) remains.
void _mirrorVisionBundle(String from, String to) {
  final manifest = File('$from/vl_bundle.json').readAsBytesSync();
  final copied = File('$to/vl_bundle.json');
  if (copied.existsSync() && _sameBytes(copied.readAsBytesSync(), manifest)) {
    return;
  }
  final target = Directory(to);
  if (target.existsSync()) target.deleteSync(recursive: true);
  // The manifest goes last: an interrupted copy has none and is redone.
  _copyBundle(from, to, overwrite: true, except: 'vl_bundle.json');
  File('$from/vl_bundle.json').copySync(copied.path);
}

/// Drops the copy's manifest so the next launch copies the bundle again.
void _forgetCopy(String copyDir) {
  final manifest = File('$copyDir/vl_bundle.json');
  if (manifest.existsSync()) manifest.deleteSync();
}

bool _sameBytes(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
