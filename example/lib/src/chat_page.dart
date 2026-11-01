import 'package:flutter/material.dart';

import 'chat_controller.dart';
import 'chat_protocol.dart';
import 'image_attachment.dart';
import 'message_bubble.dart';
import 'model_setup.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.controller, this.pickImage});

  final ChatController controller;
  final Future<ImageAttachment?> Function()? pickImage;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  ImageAttachment? _image;
  bool _picking = false;
  String? _imageError;

  ChatController get _chat => widget.controller;
  bool get _failed => _chat.setup == Setup.failed;

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
    if (_picking) return;
    if (_chat.send(_input.text, image: _image)) {
      _input.clear();
      setState(() {
        _image = null;
        _imageError = null;
      });
    }
  }

  Future<void> _pickImage() async {
    setState(() {
      _picking = true;
      _imageError = null;
    });
    try {
      final image = await (widget.pickImage ?? pickChatImage)();
      if (mounted && image != null) setState(() => _image = image);
    } catch (e) {
      if (mounted) setState(() => _imageError = '$e');
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  void _newChat() {
    _chat.newChat();
    _input.clear();
    setState(() {
      _image = null;
      _imageError = null;
    });
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: _chat,
    builder: (context, _) => Scaffold(
      appBar: AppBar(
        title: Text(
          _chat.backend == null
              ? _chat.model.name
              : '${_chat.model.name} · ${_chat.backend}',
        ),
        actions: [
          IconButton(
            tooltip: 'New chat',
            onPressed: _chat.ready && !_chat.replying && !_picking
                ? _newChat
                : null,
            icon: const Icon(Icons.add_comment_outlined),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: switch (_chat.setup) {
                Setup.ready => _messages(),
                Setup.needsModel || Setup.installing => ModelSetup(chat: _chat),
                _ => _status(context),
              },
            ),
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
          if (!_failed) const CircularProgressIndicator(),
          const SizedBox(height: 16),
          SelectableText(
            _chat.status,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: _failed ? Theme.of(context).colorScheme.error : null,
            ),
          ),
        ],
      ),
    ),
  );

  Widget _composer(BuildContext context) => Padding(
    padding: const EdgeInsets.fromLTRB(12, 4, 8, 12),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (_image case final image?)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: AttachmentPreview(
              image: image,
              onRemove: () => setState(() => _image = null),
            ),
          ),
        if (_imageError case final error?)
          Text(
            error,
            style: TextStyle(color: Theme.of(context).colorScheme.error),
          ),
        Row(
          children: [
            if (_chat.model.isVision)
              IconButton(
                tooltip: 'Attach image',
                onPressed: _chat.ready && !_chat.replying && !_picking
                    ? _pickImage
                    : null,
                icon: _picking
                    ? const SizedBox.square(
                        dimension: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.add_photo_alternate_outlined),
              ),
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
                    onPressed: _chat.ready && !_picking ? _send : null,
                    icon: const Icon(Icons.arrow_upward),
                  ),
          ],
        ),
      ],
    ),
  );
}
