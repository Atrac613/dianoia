/// LFM3-Omni conversations shared by the audio worker (questions about a
/// photo) and the chat worker (text chat with images): the turns kept, the
/// bundle check, and one reply from prompt to decoded text or speech.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';

/// The system prompt of text replies about a photo.
const photoTextSystem = 'Respond in text.';

/// [text] with every spelled-out added token ([addedTokenIn] finds them)
/// broken by a word joiner after its first character, so replaying it in a
/// later prompt keeps it as text instead of control tokens.
String plainPromptText(String text, String? Function(String) addedTokenIn) {
  for (var marker = addedTokenIn(text); marker != null;) {
    text = text.replaceAll(marker, '${marker[0]}⁠${marker.substring(1)}');
    marker = addedTokenIn(text);
  }
  return text;
}

/// The projected rows of one encoded image and its token count.
typedef OmniImage = ({Float32List rows, int tokens});

/// One completed user turn: typed [text], [images] and/or the projected rows
/// of a spoken question ([audio]), and the model's [answer].
typedef OmniTurn = ({
  String? text,
  List<OmniImage> images,
  Float32List? audio,
  String answer,
});

/// The turns of one conversation, kept in a worker. Every reply re-renders
/// the whole conversation with the official template: earlier answers as
/// text, images and spoken questions as their cached rows. Saved attachments
/// have stable keys so matching language state can be reused, even when the
/// reply is rendered again. Other tasks may use the backbone between turns.
final class OmniConversation {
  OmniConversation(this.hiddenSize, {this.system});

  final int hiddenSize;

  /// A fixed system prompt, or null to follow the reply mode as photo
  /// questions do.
  final String? system;

  final _turns = <OmniTurn>[];
  final _mediaKeys = Expando<Object>();
  List<OmniTurn> get turns => List.unmodifiable(_turns);

  Float32List _ownRows(Float32List rows) {
    final owned = Float32List.fromList(rows).asUnmodifiableView();
    _mediaKeys[owned] = Object();
    return owned;
  }

  List<OmniImage> _ownImages(List<OmniImage> images) => List.unmodifiable([
    for (final image in images)
      (rows: _ownRows(image.rows), tokens: image.tokens),
  ]);

  /// The messages for the next user turn, which carries [images] (in that
  /// order), a spoken question when [spoken], then [text].
  List<Map<String, Object?>> messages({
    required bool speak,
    String? text,
    List<OmniImage> images = const [],
    bool spoken = false,
  }) {
    Map<String, Object?> user(String? text, int images, bool audio) => {
      'role': 'user',
      'content': [
        for (var i = 0; i < images; i++) {'type': 'image'},
        if (audio) {'type': 'audio'},
        if (text != null && text.isNotEmpty) {'type': 'text', 'text': text},
      ],
    };
    return [
      {
        'role': 'system',
        'content':
            system ?? (speak ? audioConversationSystem : photoTextSystem),
      },
      for (final turn in _turns) ...[
        user(turn.text, turn.images.length, turn.audio != null),
        {'role': 'assistant', 'content': turn.answer},
      ],
      user(text, images.length, spoken),
    ];
  }

  /// Rows of every earlier image, then of [current], in prompt order.
  List<Float32List> imageRows([List<OmniImage> current = const []]) => [
    for (final turn in _turns)
      for (final image in turn.images) image.rows,
    for (final image in current) image.rows,
  ];

  /// Token counts matching [imageRows].
  List<int> imageTokens([List<OmniImage> current = const []]) => [
    for (final turn in _turns)
      for (final image in turn.images) image.tokens,
    for (final image in current) image.tokens,
  ];

  /// Projected rows of earlier spoken questions, then [current] if any, in
  /// prompt order.
  List<Float32List> audio([Float32List? current]) => [
    for (final turn in _turns) ?turn.audio,
    ?current,
  ];

  /// Token counts matching [audio].
  List<int> audioTokens([Float32List? current]) => [
    for (final rows in audio(current)) rows.length ~/ hiddenSize,
  ];

  /// Forgets the turns after the first [count], which the caller never
  /// received (a reply cancelled after it was generated).
  void keep(int count) {
    if (count < 0 || count > _turns.length) {
      throw StateError('The conversation was restarted');
    }
    _turns.removeRange(count, _turns.length);
  }

  void clear() => _turns.clear();

  void add({
    String? text,
    List<OmniImage> images = const [],
    Float32List? audio,
    required String answer,
  }) => _add(
    text: text,
    images: _ownImages(images),
    audio: audio == null ? null : _ownRows(audio),
    answer: answer,
  );

  void _add({
    String? text,
    required List<OmniImage> images,
    Float32List? audio,
    required String answer,
  }) {
    if ((text == null || text.isEmpty) && images.isEmpty && audio == null) {
      throw ArgumentError('A turn needs text, an image or audio');
    }
    for (final image in images) {
      if (image.tokens <= 0 || image.rows.length != image.tokens * hiddenSize) {
        throw ArgumentError('Image rows do not match its token count');
      }
    }
    _turns.add((text: text, images: images, audio: audio, answer: answer));
  }
}

/// The next turn does not fit the language model's context.
final class OmniContextFull implements Exception {
  const OmniContextFull({required this.firstTurn});

  /// True when even the first turn is too long; otherwise earlier turns
  /// fill the context and a new conversation is needed.
  final bool firstTurn;

  @override
  String toString() => firstTurn
      ? 'The message is too long for the context.'
      : 'The conversation no longer fits the context. '
            'Start a new conversation.';
}

/// Loads the bundle's Q4 language model on the GPU when [gpu] and it
/// succeeds, else on the CPU. Returns the model and whether it is on the GPU.
Future<(TextModel, bool)> loadOmniLanguageModel(
  String directory, {
  bool gpu = true,
}) async {
  if (!NativeKernels.isAvailable) {
    throw UnsupportedError('Native inference is unavailable on this device');
  }
  NativeKernels.load(null);
  Future<TextModel> open(bool gpu) => TextModel.load(
    '$directory/language',
    maxSeq: 2048,
    weights: WeightFormat.q4,
    q8CachePath: '$directory/language/lfm3-q4.cache',
    gpu: gpu,
  );
  if (gpu) {
    try {
      return (await open(true), true);
    } catch (_) {
      // Fall back to the CPU kernels.
    }
  }
  return (await open(false), false);
}

/// Encodes images and answers turns of an [OmniConversation] on a loaded
/// Omni bundle at [directory]. Owns the vision encoder, loaded on first use.
final class OmniResponder {
  OmniResponder(
    this.model,
    this.tokenizer,
    this.directory, {
    this.maxImageTokens = 256,
  });

  final TextModel model;
  final BpeTokenizer tokenizer;
  final String directory;

  /// The image budget: a mobile CPU encodes 256 tokens in a few seconds.
  final int maxImageTokens;

  late final Lfm3VlConfig config = loadOmniVisionConfig(
    '$directory/${Lfm3OmniQ4Source.config.name}',
    tokenizer,
  );
  late final _processor = Lfm3ImageProcessor.load(
    directory,
  ).withMaxImageTokens(maxImageTokens);
  Lfm3VisionEncoder? _vision;

  /// The vision encoder, with its matrix products on the GPU when the
  /// language model runs there.
  Lfm3VisionEncoder _loadVision() {
    final vision = loadVisionEncoderGguf(
      '$directory/${Lfm3OmniQ4Source.mmproj.name}',
      config,
    );
    if (model.gpu != null) vision.useGpu();
    return vision;
  }

  /// Decodes a PNG or JPEG and projects it for the language model.
  Future<OmniImage> encodeImage(
    Uint8List encoded, {
    void Function()? check,
    void Function(int completed, int total)? onProgress,
  }) async {
    final vision = _vision ??= _loadVision();
    final rgb = decodeImageRgb(encoded);
    final image = _processor.prepareRgb(
      rgb.pixels,
      height: rgb.height,
      width: rgb.width,
    );
    final rows = await vision.encodeAsync(
      image,
      checkCancelled: check,
      onProgress: onProgress,
    );
    return (rows: rows, tokens: image.imageTokens);
  }

  /// Feeds [chat] plus the next user turn ([images], spoken [audio] rows,
  /// [text]) and decodes the reply: text, or interleaved text and speech
  /// with a [generator]. A reply with text joins [chat]. [check] may throw
  /// to cancel; [onPrefilled] marks the end of the prompt.
  Future<SpeechReply> reply(
    OmniConversation chat, {
    String? text,
    List<OmniImage> images = const [],
    Float32List? audio,
    Lfm3AudioGenerator? generator,
    void Function()? check,
    void Function()? onPrefilled,
    void Function(String text)? onText,
  }) async {
    if (text != null) {
      if (tokenizer.addedTokenIn(text) case final marker?) {
        throw ArgumentError('Messages cannot contain the control text $marker');
      }
    }
    final speak = generator != null;
    final mode = speak ? AudioReplyMode.mixed : AudioReplyMode.text;
    final ownedImages = chat._ownImages(images);
    final ownedAudio = audio == null ? null : chat._ownRows(audio);
    final prompt = prepareOmniPrompt(
      chat.messages(
        speak: speak,
        text: text,
        images: ownedImages,
        spoken: ownedAudio != null,
      ),
      tokenizer: tokenizer,
      config: config,
      imageTokenCounts: chat.imageTokens(ownedImages),
      audioTokenCounts: chat.audioTokens(ownedAudio),
      mode: mode,
    );
    final maxSlots = speak ? 768 : 512;
    if (prompt.ids.length + maxSlots + 1 > model.maxSeq) {
      throw OmniContextFull(firstTurn: chat.turns.isEmpty);
    }
    final imageRows = chat.imageRows(ownedImages);
    final audioRows = chat.audio(ownedAudio);
    await prompt.prefill(
      model,
      images: imageRows,
      audio: audioRows,
      mediaKeys: {
        OmniMediaKind.image: [
          for (final rows in imageRows) chat._mediaKeys[rows],
        ],
        OmniMediaKind.audio: [
          for (final rows in audioRows) chat._mediaKeys[rows],
        ],
      },
    );
    check?.call();
    onPrefilled?.call();
    final reply = await decodeSpeechReply(
      model: model,
      tokenizer: tokenizer,
      generator: generator,
      variant: Lfm3SpeechVariant.omni,
      mode: mode,
      maxSlots: maxSlots,
      check: check,
      onText: onText,
      feed: (id, embedding, {required logits, meanwhile}) async {
        await model.forward(
          [id],
          computeLogits: logits,
          embeddingOverrides: {0: ?embedding},
          meanwhile: meanwhile,
        );
        // A timer, not a microtask: it lets a stop or cancel arrive.
        await Future<void>.delayed(Duration.zero);
      },
    );
    check?.call();
    if (reply.text.isNotEmpty) {
      // An answer may spell a marker with ordinary tokens; replay it as
      // text so later turns still render.
      chat._add(
        text: text,
        images: ownedImages,
        audio: ownedAudio,
        answer: plainPromptText(reply.text, tokenizer.addedTokenIn),
      );
    }
    return reply;
  }

  void close() {
    _vision?.close();
    _vision = null;
  }
}

/// Omni bundle provenance: an installed bundle is fully hashed against its
/// manifest; a prepared research folder must carry the pinned receipt and
/// the original mmproj and metadata.
Future<void> verifyOmniBundle(String dir, void Function() check) async {
  final installed = File('$dir/omni_bundle.json').existsSync();
  if (installed) {
    await validateOmniBundle(
      dir,
      OmniGgufSpec.lfm3OmniQ4.files,
      Lfm3OmniQ4Source.mmproj.name,
      check: check,
    );
  }
  final receipt =
      jsonDecode(await File('$dir/language/omni_source.json').readAsString())
          as Map;
  if (receipt['repository'] != Lfm3OmniQ4Source.repository ||
      receipt['revision'] != Lfm3OmniQ4Source.revision ||
      receipt['metadata_revision'] != Lfm3OmniQ4Source.metadataRevision) {
    throw const FormatException(
      'Choose a prepared official LFM3-Omni Q4_0 model',
    );
  }
  for (final f in [
    ...Lfm3OmniQ4Source.files,
    ...Lfm3OmniQ4Source.metadataFiles,
  ]) {
    final entry = (receipt['files'] as Map)[f.name] as Map?;
    if (entry?['size'] != f.size || entry?['sha256'] != f.sha256) {
      throw const FormatException('Omni source receipt differs');
    }
  }
  for (final (file, spec) in [
    if (!installed) ...[
      (File('$dir/${Lfm3OmniQ4Source.mmproj.name}'), Lfm3OmniQ4Source.mmproj),
      (File('$dir/language/tokenizer.json'), Lfm3OmniQ4Source.tokenizer),
      (File('$dir/omni_config.json'), Lfm3OmniQ4Source.config),
      (File('$dir/processor_config.json'), Lfm3OmniQ4Source.processorConfig),
    ],
  ]) {
    check();
    if (await file.length() != spec.size) {
      throw FormatException('${spec.name} has the wrong size');
    }
    final chunks = file.openRead().map((chunk) {
      check();
      return chunk;
    });
    if ((await sha256.bind(chunks).first).toString() != spec.sha256) {
      throw FormatException('${spec.name} has the wrong hash');
    }
  }
  check();
  final cache = Q8Cache.open(
    '$dir/language/lfm3-q4.cache',
    Q8Cache.keyFor('$dir/language', weights: 'q4'),
  );
  if (cache == null) {
    throw const FormatException(
      'Prepared Omni language cache is missing or stale',
    );
  }
  cache.close();
  check();
}
