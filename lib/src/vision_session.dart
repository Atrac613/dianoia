/// Image conversations with cached visual features and transactional history.
library;

import 'dart:async';
import 'dart:typed_data';

import 'chat_session.dart';
import 'image_decode.dart';
import 'model.dart';
import 'tokenizer.dart';
import 'tool_calls.dart';
import 'vision_encoder.dart';
import 'vision_image.dart';
import 'vision_prompt.dart';

/// Work before the first visible reply text.
enum VisionPhase { encodingImage, prefilling, generating }

/// Encoder stage counts cover only new images; prior visual features are reused.
final class VisionProgress {
  const VisionProgress(this.phase, {this.completed = 0, this.total = 1});
  final VisionPhase phase;
  final int completed, total;
}

/// Owned, immutable RGB uint8 pixels supplied directly or decoded from PNG/JPEG.
final class ChatImage {
  ChatImage.rgb(Uint8List pixels, {required this.width, required this.height})
    : _pixels = _copy(pixels, width, height);

  ChatImage._owned(this._pixels, this.width, this.height);

  /// Decodes a still PNG/JPEG, applying JPEG EXIF orientation and compositing
  /// alpha over white. Does not resize or apply ICC color conversion. Run in
  /// the inference worker isolate: decoding is synchronous CPU work.
  /// Throws FormatException for invalid, unsupported or over-budget images;
  /// nonpositive limits throw ArgumentError. PNG samples use stored orientation.
  factory ChatImage.encoded(
    Uint8List bytes, {
    int maxEncodedBytes = defaultMaxEncodedImageBytes,
    int maxPixels = defaultMaxDecodedImagePixels,
  }) {
    final decoded = decodeImageRgb(
      bytes,
      maxEncodedBytes: maxEncodedBytes,
      maxPixels: maxPixels,
    );
    return ChatImage._owned(decoded.pixels, decoded.width, decoded.height);
  }

  /// Reads a bounded local file and decodes it like [ChatImage.encoded].
  /// File I/O is asynchronous; decoding runs on the calling isolate. File
  /// extensions are ignored, and URLs are not fetched. I/O errors propagate.
  static Future<ChatImage> fromFile(
    String path, {
    int maxEncodedBytes = defaultMaxEncodedImageBytes,
    int maxPixels = defaultMaxDecodedImagePixels,
  }) async {
    final decoded = await readImageRgb(
      path,
      maxEncodedBytes: maxEncodedBytes,
      maxPixels: maxPixels,
    );
    return ChatImage._owned(decoded.pixels, decoded.width, decoded.height);
  }

  final int width, height;
  final Uint8List _pixels;

  static Uint8List _copy(Uint8List pixels, int width, int height) {
    if (width <= 0 || height <= 0 || pixels.length != width * height * 3) {
      throw ArgumentError(
        'expected positive dimensions and width * height * 3 RGB bytes',
      );
    }
    return Uint8List.fromList(pixels);
  }
}

/// An explicitly cancelled visual turn. Its draft message and streamed reply
/// are not committed to history. Return false from onText to retain a partial
/// reply instead.
final class VisionRequestCancelled implements Exception {
  const VisionRequestCancelled();
  @override
  String toString() => 'VisionRequestCancelled';
}

/// Shared implementation state for one loaded visual model. Not exported by
/// dianoia.dart; applications create sessions through LocalVisionModel.chat.
final class VisionChatRuntime {
  VisionChatRuntime(this.model, this.tokenizer, this.processor, this.encoder);
  final Lfm3Model model;
  final Lfm3Tokenizer tokenizer;
  final Lfm3ImageProcessor processor;
  Lfm3VisionEncoder? encoder;
  Completer<void>? _active;
  Completer<bool>? _waiting;
  Object? _owner;
  bool _cancelled = false, _closing = false;
  Future<void>? _closeFuture;

  bool get isClosed => _closing || model.isClosed;

  VisionChatSession chat({
    String? system,
    List<Map<String, Object?>>? tools,
    int maxNewTokens = 512,
  }) {
    if (isClosed) throw StateError('the visual model is closed');
    if (maxNewTokens <= 0) throw ArgumentError('maxNewTokens must be positive');
    return VisionChatSession._(this, system, tools, maxNewTokens);
  }

  Future<T> run<T>(Object owner, Future<T> Function() operation) async {
    if (isClosed) throw StateError('the visual model is closed');
    if (_active != null) {
      throw StateError('another visual request is in progress');
    }
    final done = Completer<void>();
    _active = done;
    _owner = owner;
    _cancelled = false;
    try {
      return await operation();
    } finally {
      _active = null;
      _owner = null;
      done.complete();
    }
  }

  void cancel(Object owner) {
    if (identical(_owner, owner)) _requestCancel();
  }

  void _requestCancel() {
    _cancelled = true;
    final waiting = _waiting;
    if (waiting != null && !waiting.isCompleted) {
      waiting.completeError(const VisionRequestCancelled());
    }
  }

  /// [callback]'s answer for [text]; a cancellation ends the wait for an
  /// asynchronous one. Only the callback's own future is listened to, so
  /// nothing accumulates across the tokens of a turn.
  Future<bool> waitForText(
    FutureOr<bool> Function(String) callback,
    String text,
  ) async {
    final answer = callback(text);
    if (answer is bool) return answer;
    final waiting = _waiting = Completer<bool>();
    // onLogits or the callback itself may have cancelled already.
    if (_cancelled) waiting.completeError(const VisionRequestCancelled());
    answer.then(
      (value) {
        if (!waiting.isCompleted) waiting.complete(value);
      },
      onError: (Object error, StackTrace stack) {
        if (!waiting.isCompleted) waiting.completeError(error, stack);
      },
    );
    try {
      return await waiting.future;
    } finally {
      _waiting = null;
    }
  }

  void checkCancelled() {
    if (_cancelled || isClosed) throw const VisionRequestCancelled();
  }

  Future<void> close() => _closeFuture ??= _close();

  Future<void> _close() async {
    _closing = true;
    _requestCancel();
    final active = _active;
    if (active != null) await active.future;
    encoder = null;
    await model.close();
  }
}

final class _Attachment {
  _Attachment(this.image, this.tokens);
  ChatImage? image;
  final int tokens;
  Float32List? features;
}

final class _Entry {
  _Entry(this.message, [this.images = const []]);
  final Map<String, Object?> message;
  final List<_Attachment> images;
}

final class VisionChatSession {
  VisionChatSession._(
    this._runtime,
    this._system,
    List<Map<String, Object?>>? tools,
    this.maxNewTokens,
  ) : tools = tools == null ? null : List.unmodifiable(tools.map(_messageCopy));

  final VisionChatRuntime _runtime;
  final String? _system;
  final List<Map<String, Object?>>? tools;
  final int maxNewTokens;
  final _entries = <_Entry>[];
  bool _busy = false;

  /// Read-only message history. Image parts are placeholders; owned visual
  /// features are retained privately until clear() or the session is released.
  List<Map<String, Object?>> get messages =>
      List.unmodifiable(_entries.map((e) => e.message));

  /// Aborts this session's active turn at the next yield boundary. The send
  /// future throws VisionRequestCancelled and history stays unchanged.
  void cancel() {
    _runtime.cancel(this);
  }

  void clear() {
    if (_busy) throw StateError('cannot clear an active visual turn');
    _entries.clear();
  }

  /// Sends a user question, inserting image parts before its text. A text-only
  /// follow-up retains earlier image features without encoding those images
  /// again. Each turn re-prefills language state; reusedTokens is zero.
  Future<ChatTurn> send(
    String text, {
    List<ChatImage> images = const [],
    void Function(Float32List logits)? onLogits,
    FutureOr<bool> Function(String text)? onText,
    void Function(VisionProgress progress)? onProgress,
  }) => sendMessage(
    {
      'role': 'user',
      'content': [
        for (final _ in images) {'type': 'image'},
        {'type': 'text', 'text': text},
      ],
    },
    images: images,
    onLogits: onLogits,
    onText: onText,
    onProgress: onProgress,
  );

  /// Advanced HF-style user/tool message, with decoded images bound to its
  /// image parts in content order. Supports interleaved text/image parts and
  /// tool responses. URLs and file paths in image parts are never fetched.
  ///
  /// Context checks happen before preprocessing/encoding pixels. New history
  /// is committed only after a successful turn or graceful onText stop. Errors
  /// and explicit cancellation leave prior history intact. onText may return
  /// false to commit a partial reply. Cancellation also interrupts waiting on
  /// an asynchronous onText callback; the callback's own future may continue.
  /// onLogits observes a temporary model view: copy it before retaining it.
  /// prefillSeconds includes any newly encoded images.
  Future<ChatTurn> sendMessage(
    Map<String, Object?> message, {
    List<ChatImage> images = const [],
    void Function(Float32List logits)? onLogits,
    FutureOr<bool> Function(String text)? onText,
    void Function(VisionProgress progress)? onProgress,
  }) async {
    if (_busy) throw StateError('this visual session is already generating');
    final role = message['role'];
    if (role != 'user' && role != 'tool') {
      throw ArgumentError('expected a user or tool message');
    }
    final ownedMessage = _messageCopy(message);
    final ownedImages = List<ChatImage>.of(images);
    _busy = true;
    try {
      return await _runtime.run(this, () async {
        final encoder = _runtime.encoder!;
        final attachments = [
          for (final image in ownedImages)
            _Attachment(image, _imageTokens(image)),
        ];
        final draft = _Entry(ownedMessage, attachments);
        final entries = [..._entries, draft];
        final allImages = [for (final entry in entries) ...entry.images];
        final prompt = prepareVisionPrompt(
          [
            if (_system != null) {'role': 'system', 'content': _system},
            for (final entry in entries) entry.message,
          ],
          tokenizer: _runtime.tokenizer,
          config: encoder.config,
          tools: tools,
          imageTokenCounts: [for (final image in allImages) image.tokens],
          enableThinking: false,
        );
        if (prompt.ids.length >= _runtime.model.maxSeq) {
          throw ContextLengthExceeded(prompt.ids.length, _runtime.model.maxSeq);
        }
        final clock = Stopwatch()..start();
        final freshImages = allImages.where((a) => a.features == null).length;
        var encodedImages = 0;
        final stages = encoder.config.vision.numLayers + 2;
        for (final attachment in allImages) {
          if (attachment.features != null) continue;
          onProgress?.call(
            VisionProgress(
              VisionPhase.encodingImage,
              completed: encodedImages * stages,
              total: freshImages * stages,
            ),
          );
          await Future<void>.delayed(Duration.zero);
          _runtime.checkCancelled();
          final image = attachment.image!;
          final input = _runtime.processor.prepareRgb(
            image._pixels,
            height: image.height,
            width: image.width,
          );
          attachment.features = await encoder.encodeAsync(
            input,
            checkCancelled: _runtime.checkCancelled,
            onProgress: (completed, _) => onProgress?.call(
              VisionProgress(
                VisionPhase.encodingImage,
                completed: encodedImages * stages + completed,
                total: freshImages * stages,
              ),
            ),
          );
          encodedImages++;
          attachment.image = null;
        }
        _runtime.checkCancelled();
        onProgress?.call(const VisionProgress(VisionPhase.prefilling));
        _runtime.checkCancelled();
        await prompt.prefill(_runtime.model, [
          for (final image in allImages) image.features!,
        ]);
        _runtime.checkCancelled();
        final prefillSeconds = clock.elapsedMicroseconds / 1e6;
        onLogits?.call(_runtime.model.logits);
        onProgress?.call(const VisionProgress(VisionPhase.generating));
        final generated = <int>[];
        clock.reset();
        while (generated.length < maxNewTokens &&
            _runtime.model.length < _runtime.model.maxSeq) {
          await Future<void>.delayed(Duration.zero);
          _runtime.checkCancelled();
          final token = _argmax(_runtime.model.logits);
          if (token == _runtime.model.config.eosTokenId) break;
          generated.add(token);
          await _runtime.model.step(token);
          _runtime.checkCancelled();
          onLogits?.call(_runtime.model.logits);
          if (onText != null) {
            final decoded = _runtime.tokenizer.decode(generated);
            final complete = decoded.endsWith('\uFFFD')
                ? decoded.substring(0, decoded.length - 1)
                : decoded;
            final keepGoing = await _runtime.waitForText(onText, complete);
            _runtime.checkCancelled();
            if (!keepGoing) break;
          }
        }
        _runtime.checkCancelled();
        final text = _runtime.tokenizer.decode(generated);
        _entries.addAll([
          draft,
          _Entry(_messageCopy({'role': 'assistant', 'content': text})),
        ]);
        return ChatTurn(
          text: text,
          reply: parseAssistantReply(text),
          promptTokens: prompt.ids.length,
          reusedTokens: 0,
          prefillSeconds: prefillSeconds,
          generatedTokens: generated.length,
          decodeSeconds: clock.elapsedMicroseconds / 1e6,
        );
      });
    } finally {
      _busy = false;
    }
  }

  int _imageTokens(ChatImage image) {
    final processor = _runtime.processor;
    final (h, w) = processor.resizedSize(image.height, image.width);
    return (h ~/ processor.patchSize) *
        (w ~/ processor.patchSize) ~/
        (processor.mergeSize * processor.mergeSize);
  }
}

int _argmax(Float32List values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

Map<String, Object?> _messageCopy(Map<String, Object?> value) =>
    _freeze(value) as Map<String, Object?>;

Object? _freeze(Object? value) {
  if (value is Map) {
    final result = <String, Object?>{};
    for (final entry in value.entries) {
      if (entry.key is! String) {
        throw ArgumentError('message keys must be strings');
      }
      result[entry.key as String] = _freeze(entry.value);
    }
    return Map<String, Object?>.unmodifiable(result);
  }
  if (value is List) return List<Object?>.unmodifiable(value.map(_freeze));
  if (value == null ||
      value is String ||
      value is bool ||
      value is int ||
      value is double && value.isFinite) {
    return value;
  }
  throw ArgumentError('unsupported message value ${value.runtimeType}');
}
