import 'package:flutter/material.dart';

import 'chat_controller.dart';
import 'message_bubble.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.controller});

  final ChatController controller;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _input = TextEditingController();
  final _scroll = ScrollController();

  ChatController get _chat => widget.controller;

  @override
  void initState() {
    super.initState();
    _chat.addListener(_followReply);
  }

  @override
  void dispose() {
    _chat.removeListener(_followReply);
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  /// Keeps the newest text in view while a reply streams in.
  void _followReply() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  void _send() {
    _chat.send(_input.text);
    _input.clear();
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: _chat,
    builder: (context, _) => Scaffold(
      appBar: AppBar(
        title: Text(
          _chat.backend == null ? 'Dianoia' : 'LFM3 · ${_chat.backend}',
        ),
        actions: [
          IconButton(
            tooltip: 'New chat',
            onPressed: _chat.ready && !_chat.replying ? _chat.newChat : null,
            icon: const Icon(Icons.add_comment_outlined),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(child: _chat.ready ? _messages() : _status(context)),
            _composer(context),
          ],
        ),
      ),
    ),
  );

  Widget _messages() => _chat.messages.isEmpty
      ? const Center(child: Text('Say something to the model on this device.'))
      : ListView.builder(
          controller: _scroll,
          padding: const EdgeInsets.all(12),
          itemCount: _chat.messages.length,
          itemBuilder: (context, i) =>
              MessageBubble(message: _chat.messages[i]),
        );

  Widget _status(BuildContext context) => Center(
    child: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (!_chat.failed) const CircularProgressIndicator(),
          const SizedBox(height: 16),
          SelectableText(
            _chat.status,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: _chat.failed ? Theme.of(context).colorScheme.error : null,
            ),
          ),
        ],
      ),
    ),
  );

  Widget _composer(BuildContext context) => Padding(
    padding: const EdgeInsets.fromLTRB(12, 4, 8, 12),
    child: Row(
      children: [
        Expanded(
          child: TextField(
            controller: _input,
            enabled: _chat.ready,
            minLines: 1,
            maxLines: 5,
            textInputAction: TextInputAction.send,
            onSubmitted: (_) => _send(),
            decoration: const InputDecoration(
              hintText: 'Message',
              border: OutlineInputBorder(),
              isDense: true,
            ),
          ),
        ),
        const SizedBox(width: 4),
        _chat.replying
            ? IconButton.filledTonal(
                tooltip: 'Stop',
                onPressed: _chat.stop,
                icon: const Icon(Icons.stop),
              )
            : IconButton.filled(
                tooltip: 'Send',
                onPressed: _chat.ready ? _send : null,
                icon: const Icon(Icons.arrow_upward),
              ),
      ],
    ),
  );
}
