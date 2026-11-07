import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:file_selector/file_selector.dart';
import 'package:dianoia_example/src/image_attachment.dart';

import 'package:dianoia_example/src/chat_controller.dart';
import 'package:dianoia_example/src/chat_page.dart';
import 'package:dianoia_example/src/chat_protocol.dart';
import 'package:dianoia_example/src/example_model.dart';
import 'package:dianoia_example/src/message_bubble.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class _RecordingChat extends ChatController {
  _RecordingChat({super.model = ExampleModel.lfm3Vl}) {
    setup = Setup.ready;
  }
  String? sentText;
  ImageAttachment? sentImage;
  bool accept = true;
  @override
  bool send(String text, {ImageAttachment? image}) {
    if (!accept) return false;
    sentText = text;
    sentImage = image;
    return true;
  }
}

void main() {
  late ImageAttachment image;
  setUp(() {
    image = ImageAttachment(
      'card.png',
      File('../test/fixtures/image_input/gradient.png').readAsBytesSync(),
    );
  });

  test(
    'picker accepts PNG, handles cancel, rejects unsupported and large files',
    () async {
      final picked = await pickChatImage(
        chooseFile: () async => XFile.fromData(image.bytes, name: image.name),
      );
      expect(picked!.bytes, image.bytes);
      expect(await pickChatImage(chooseFile: () async => null), isNull);
      await expectLater(
        pickChatImage(
          chooseFile: () async => XFile.fromData(
            Uint8List.fromList([1, 2, 3]),
            name: 'invalid.png',
          ),
        ),
        throwsFormatException,
      );
      await expectLater(
        pickChatImage(
          chooseFile: () async => XFile.fromData(
            Uint8List(ImageAttachment.maxBytes + 1),
            name: 'large.png',
          ),
        ),
        throwsFormatException,
      );
    },
  );

  test(
    'HEIC photos are converted to upright PNG at pick time',
    () async {
      // 64x48 pixels tagged with EXIF orientation 6 (rotate 90 degrees).
      final heic = File('test/fixtures/rotated_shapes.heic').readAsBytesSync();
      expect(isHeif(heic), isTrue);
      final picked = await pickChatImage(
        chooseFile: () async => XFile.fromData(heic, name: 'IMG_0001.HEIC'),
      );
      expect(picked!.bytes.sublist(1, 4), 'PNG'.codeUnits);
      final buffer = await ui.ImmutableBuffer.fromUint8List(picked.bytes);
      final descriptor = await ui.ImageDescriptor.encoded(buffer);
      expect((descriptor.width, descriptor.height), (48, 64));
      descriptor.dispose();
      buffer.dispose();
      expect(picked.thumbnail!.bytes!.sublist(1, 4), 'PNG'.codeUnits);
    },
    // Flutter decodes HEIC with the platform's codecs, which the Linux and
    // Windows test hosts lack.
    skip: Platform.isMacOS ? false : 'HEIC decoding needs macOS codecs',
  );

  test('unreadable HEIC photos are rejected with a clear message', () async {
    final broken = Uint8List.fromList([
      0, 0, 0, 24, ...'ftypheic'.codeUnits, ...List.filled(12, 0), //
    ]);
    expect(isHeif(broken), isTrue);
    await expectLater(
      pickChatImage(
        chooseFile: () async => XFile.fromData(broken, name: 'broken.heic'),
      ),
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains('cannot read HEIC'),
        ),
      ),
    );
    expect(isHeif(image.bytes), isFalse);
  });

  testWidgets('history retains only a bounded thumbnail and its file name', (
    tester,
  ) async {
    final message = (await tester.runAsync(() async {
      final recorder = ui.PictureRecorder();
      ui.Canvas(recorder).drawColor(const Color(0xFF336699), BlendMode.src);
      final picture = recorder.endRecording();
      final large = await picture.toImage(1024, 512);
      final png = (await large.toByteData(format: ui.ImageByteFormat.png))!;
      final source = Uint8List(2 * 1024 * 1024)
        ..setAll(
          0,
          png.buffer.asUint8List(png.offsetInBytes, png.lengthInBytes),
        );
      large.dispose();
      picture.dispose();
      final picked = (await pickChatImage(
        chooseFile: () async =>
            XFile.fromData(source, name: image.name, path: image.name),
      ))!;
      final message = ChatMessage(ChatRole.user, 'Describe.', image: picked);
      expect(message.image!.name, image.name);
      final thumbnail = message.image!.bytes!;
      expect(identical(thumbnail.buffer, picked.bytes.buffer), isFalse);
      expect(thumbnail.length, lessThan(240 * 240 * 4));
      final codec = await ui.instantiateImageCodec(thumbnail);
      final frame = await codec.getNextFrame();
      expect(frame.image.width, 240);
      expect(frame.image.height, 120);
      frame.image.dispose();
      codec.dispose();
      return message;
    }))!;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: MessageBubble(message: message)),
      ),
    );
    expect(find.byType(Image), findsOneWidget);
    expect(find.text(image.name), findsOneWidget);
    // Attachments supplied without a thumbnail keep only the name in history.
    expect(ChatMessage(ChatRole.user, '', image: image).image!.bytes, isNull);
  });

  testWidgets('select, preview and send an image with a question', (
    tester,
  ) async {
    final chat = _RecordingChat();
    addTearDown(chat.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: ChatPage(controller: chat, pickImage: () async => image),
      ),
    );
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    expect(find.text('card.png'), findsOneWidget);
    expect(find.byType(Image), findsOneWidget);
    await tester.enterText(find.byType(TextField), 'What is here?');
    await tester.tap(find.byTooltip('Send'));
    await tester.pump();
    expect(chat.sentText, 'What is here?');
    expect(chat.sentImage, same(image));
    expect(find.text('card.png'), findsNothing);
  });

  testWidgets('image-only messages and removing an attachment', (tester) async {
    final chat = _RecordingChat();
    addTearDown(chat.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: ChatPage(controller: chat, pickImage: () async => image),
      ),
    );
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    await tester.tap(find.byTooltip('Remove image'));
    await tester.pump();
    expect(find.text('card.png'), findsNothing);
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    await tester.tap(find.byTooltip('Send'));
    await tester.pump();
    expect(chat.sentText, '');
    expect(chat.sentImage, same(image));
  });

  testWidgets('picker cancellation and rejection retain the draft', (
    tester,
  ) async {
    final chat = _RecordingChat()..accept = false;
    addTearDown(chat.dispose);
    var cancel = false;
    await tester.pumpWidget(
      MaterialApp(
        home: ChatPage(
          controller: chat,
          pickImage: () async => cancel ? null : image,
        ),
      ),
    );
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    cancel = true;
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'Keep this');
    await tester.tap(find.byTooltip('Send'));
    await tester.pump();
    expect(find.text('card.png'), findsOneWidget);
    expect(find.text('Keep this'), findsOneWidget);
  });

  testWidgets('picker errors are shown and a new chat clears the attachment', (
    tester,
  ) async {
    final chat = _RecordingChat();
    addTearDown(chat.dispose);
    var fail = true;
    await tester.pumpWidget(
      MaterialApp(
        home: ChatPage(
          controller: chat,
          pickImage: () async {
            if (fail) throw const FormatException('Image too large');
            return image;
          },
        ),
      ),
    );
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    expect(find.textContaining('Image too large'), findsOneWidget);
    fail = false;
    await tester.tap(find.byTooltip('Attach image'));
    await tester.pumpAndSettle();
    await tester.tap(find.byTooltip('New chat'));
    await tester.pump();
    expect(find.text('card.png'), findsNothing);
  });

  testWidgets('text models do not offer image attachments', (tester) async {
    final chat = _RecordingChat(model: ExampleModel.lfm3);
    addTearDown(chat.dispose);
    await tester.pumpWidget(MaterialApp(home: ChatPage(controller: chat)));
    expect(find.byTooltip('Attach image'), findsNothing);
  });

  testWidgets(
    'an image-only user bubble shows the image without typing progress',
    (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: MessageBubble(
              message: ChatMessage(ChatRole.user, '', image: image),
            ),
          ),
        ),
      );
      expect(find.text('card.png'), findsOneWidget);
      expect(find.byType(CircularProgressIndicator), findsNothing);
    },
  );
}
