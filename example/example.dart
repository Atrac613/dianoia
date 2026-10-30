// A tool-calling conversation with LFM3 on the GPU.
//
//   dart run example/example.dart MODEL_DIR
//
// MODEL_DIR holds config.json, tokenizer.json, model.safetensors and the Q4
// cache lfm3-q4.cache, as written by `dianoia import-gguf` (or the full
// checkpoint, whose cache is then built on the first run).
import 'dart:io';

import 'package:dianoia/dianoia.dart';

Future<void> main(List<String> args) async {
  final modelDir = args.single;
  NativeKernels.load(null);
  final model = await Lfm3Model.load(
    modelDir,
    weights: WeightFormat.q4,
    q8CachePath: '$modelDir/lfm3-q4.cache',
    gpu: true,
  );
  final session = ChatSession(
    model,
    Lfm3Tokenizer.load(modelDir),
    system: 'Answer briefly. Call a tool for live information.',
    tools: [
      {
        'type': 'function',
        'function': {
          'name': 'get_weather',
          'description': 'Get the current weather for a city.',
          'parameters': {
            'type': 'object',
            'properties': {
              'city': {'type': 'string', 'description': 'City name'},
            },
            'required': ['city'],
          },
        },
      },
    ],
  );
  // Feeds the system prompt and tools once; every conversation reuses it.
  await session.warmUp();

  var turn = await session.send({
    'role': 'user',
    'content': 'Is it raining in Kyoto?',
  });
  for (final call in turn.reply.toolCalls) {
    stdout.writeln('tool call: ${call.name} ${call.arguments}');
  }
  turn = await session.send({
    'role': 'tool',
    'content': '{"city": "Kyoto", "condition": "light rain", "temp_c": 14}',
  });
  stdout.writeln('reply: ${turn.reply.content}');
  stdout.writeln(
    'first token after ${(turn.prefillSeconds * 1000).round()} ms, '
    '${(turn.generatedTokens / turn.decodeSeconds).toStringAsFixed(1)} tok/s',
  );
}
