# FOR_ME: dianoia の全貌

## 研究室から店舗へ

dart-llm-lab は「研究室」だった。GEMV を Dart と Rust と C++ で競わせるところから始めて、量子化を試し、LFM3-1.7B を transformers と層ごとに突き合わせ、最後は iPhone で 29.5 tok/s、Pixel 8 で約 18 tok/s まで来た。

ただ、研究室の道具はそのまま売り物にはならない。ベンチマークのつまみ、`--native-lib=build/native/liblab_native.dylib` のような手作業のパス指定、iOS 用の podspec と Android 用の CMakeLists が別々に転がっている状態。Caverno に組み込み、いずれ pub.dev で公開するには「店舗」が要る。それがこのリポジトリ。

名前の dianoia（ディアノイア）は、プラトンの「線分の比喩」から取った。認識の4段階のうち、仮定から筋道を立てて推し量る思考の段階を指す。洞窟の比喩の続きの話なので Caverno（洞窟）とも世界観が揃い、Caverno にすでにある「Anabasis（上昇）」とも並ぶ。何より、与えられたトークンから一歩ずつ推論するのは、言語モデルがやっていることそのものだ。

## 技術アーキテクチャ: 指揮者と楽団

構造は研究室時代から変わらない。**Dart が指揮者で、ネイティブコードは楽団**。

```
                 Dart (lib/src)
  ┌──────────────────────────────────────────────┐
  │ ChatSession ─ chat_template ─ tokenizer      │  会話・テンプレート・トークン
  │      │                                        │
  │ Lfm3Model ── history / checkpoint / state     │  何を見たか、どこまで再利用できるか
  │      │                                        │
  │  ┌───┴──────────────┐                          │
  │  CPU 経路            GPU 経路 (gpu.dart)       │  計算グラフを組むのは Dart
  │  native.dart        1カーネル = 1 FFI 呼び出し  │
  └──┬──────────────────┬───────────────────────┘
     │ @Native          │ @Native
  ┌──┴──────────────────┴───────────────────────┐
  │ code asset "package:dianoia/kernels"         │  hook/build.dart がビルド
  │  Apple:   src/c + src/metal  (Metal)         │
  │  Android: src/c + src/vulkan (Vulkan)        │
  └──────────────────────────────────────────────┘
```

楽団（C、Metal、Vulkan）は「この行列をこの入力に掛けろ」と言われたことしかしない。どの順番で何を鳴らすか、つまり層の順序、KV キャッシュの位置、どこまでのプロンプトを再利用するかは、すべて Dart の指揮者が決める。

だからモデルを増やすときに書き換えるのは Dart 側だけで済む（はず）。LFM3 のオーディオ対応や汎用 GGUF 対応が「楽団はそのまま、楽譜を増やす」作業になるかどうかが、このパッケージの設計の試金石になる。

## コードベース構造

| 場所 | 役割 |
|---|---|
| `lib/dianoia.dart` | 公開 API の入口（今はラボの全部をそのまま export） |
| `lib/src/model.dart` | `Lfm3Model`: 読み込み、prefill（共通接頭辞の再利用）、チェックポイント、状態ファイル |
| `lib/src/chat_session.dart` | `ChatSession`: system とツール定義を一度だけ流し、会話ごとに再利用 |
| `lib/src/native.dart` | CPU カーネルへの FFI。Q8/Q4 行列、mmap、アテンション |
| `lib/src/gpu.dart` | GPU の順伝播。Metal と Vulkan は同じ C API `lab_gpu_*` を持つので、このファイル1つで両方を駆動する |
| `lib/src/gguf.dart` | Liquid の QAD Q4_0 GGUF の読み込み |
| `src/c/lab_native.c` | CPU カーネル（NEON `sdot`、Apple は GCD、Android は OpenMP） |
| `src/metal/lab_metal.m` | Metal 層。シェーダはソース文字列で持ち、起動時にコンパイル |
| `src/vulkan/` | Vulkan 層、GLSL シェーダ、コンパイル済み SPIR-V |
| `hook/build.dart` | ターゲットごとにカーネルをビルドする build hook |
| `bin/dianoia.dart` | CLI（ラボの `lfm3` コマンドを移植） |
| `tool/lab_parity.sh` | ラボとのビット単位一致チェック |

ファイル名に `lab_` が残っているのは意図的。最初のコミットは「ラボのエンジンを、動作を一切変えずに持ってきた」状態にしたかった。名前の整理は、一致チェックで守られた後続のコミットでやる。

## 使用技術とその選定理由

### FFI プラグインではなく build hooks を選んだ

ラボでは `lab_kernels` という Flutter の FFI プラグインを作っていた。iOS は podspec、Android は CMakeLists.txt と build.gradle。これで iPhone も Pixel も動いた。それでも新リポジトリでは Dart の **build hooks（native assets）** に乗り換えた。

決め手は「1つのパッケージが、Dart CLI でも Flutter アプリでも同じように動く」こと。

- FFI プラグインは Flutter 専用。CLI でカーネルを使うには、別途 `clang` でビルドして `--native-lib=...` とパスを渡す必要があった。ラボの CLI に `--native-lib` と `--gpu-lib` があったのはそのため。
- build hooks なら `dart run`、`dart test`、`dart build cli`、`flutter run` のどれでも `hook/build.dart` が走り、ターゲットに合ったライブラリができる。`@Native` 関数は asset ID（`package:dianoia/kernels`）で勝手に解決される。
- プラットフォームごとのビルド設定（podspec、CMake、gradle）が、Dart で書かれた1つの `build.dart` にまとまる。

トレードオフもある。build hooks は FFI プラグインより新しい仕組みで、Flutter の iOS/Android ビルドでの実績はまだ自分たちの手で確かめていない。ここは次の実機検証で埋める穴だ。

### Vulkan シェーダは「コンパイル済み」で同梱する

Android の Vulkan シェーダは GLSL で書き、`glslc` で SPIR-V にする。ラボでは CMake がビルドのたびに NDK の `glslc` を呼んでいた。

pub.dev で配るなら、利用者に `glslc` を要求したくない。そこで `tool/compile_shaders.sh` で SPIR-V の配列（`.inc`）を作り、リポジトリにコミットすることにした。ビルド時に要るのは C コンパイラだけになる。

代わりに「シェーダを直したのに SPIR-V を作り直し忘れる」事故が起きうる。そこで `SOURCES.sha256` にシェーダソースのハッシュを記録し、`test/shaders_test.dart` で突き合わせる。作り直し忘れはテストが落ちて気づける。ちなみに、作り直した16本の SPIR-V はラボの CMake ビルドの出力とバイト単位で一致した。

### Metal はソースのまま持つ

Metal のシェーダは `newLibraryWithSource` で起動時にコンパイルしている。オフラインの Metal コンパイラ（`xcrun metal`）が要らないので、build hook は Objective-C のファイルを clang に渡すだけで済む。引き換えに GPU の起動のたびにコンパイルが走る。そのコストはまだ単独で測っていない。

## 学びとレッスン

### 遭遇したバグと解決方法

**1. 「tool call が出ていない」と思ったら、自分の grep が消していた**

example を走らせると、返答は出るのに `tool call: get_weather ...` の行が見当たらない。パーサが壊れたかと思って最初の応答を単独で出力してみると、ちゃんと解析できていた。

原因は出力の加工側にあった。`dart run` は最初に `Running build hooks...` と表示するが、**末尾に改行がない**。そのため次の行の出力がこの表示と同じ行にくっつき、`grep -v "^Running"` がその行ごと消していた。

教訓: 「出ていない」と判断する前に、フィルタを通していない生の出力を見ること。

**2. macOS の bash は 3.2**

一致チェックのスクリプトで `${w^^}`（大文字化）を使ったら `bad substitution` で落ちた。macOS 標準の `/bin/bash` は、ライセンスの都合で 2007 年の 3.2 のまま止まっている。`${var^^}` は bash 4 から。macOS 向けのシェルスクリプトでは bash 4 の機能を使わない。

### 潜在的な落とし穴

- **`@Native` のシンボル解決順**: (1) asset ID、(2) 埋め込み側が設定したリゾルバ、(3) プロセス全体、の順。asset が無くても、プロセスに同名のシンボルがあれば黙ってそちらに繋がる。`--native-lib` で別ビルドを開く経路はこの (3) に頼っている。将来、同じ名前のシンボルを持つ別ライブラリ（たとえば別バージョンの dianoia）がプロセスにいると、意図しない方に繋がる可能性がある。C シンボルの接頭辞を `lab_` から固有の名前に変えたい理由の1つがこれ。
- **カーネルの有無は呼んでみないと分からない**: asset が無いプラットフォームで `@Native` 関数を呼ぶと `ArgumentError` が飛ぶ。`NativeKernels.isAvailable` は、副作用のない関数（`lab_phys_footprint`）を試しに呼んで判定している。
- **hook の出力はキャッシュされる**: C や Metal のコードを直しても反映されないように見えたら、`.dart_tool/hooks_runner` を消す。
- **arm64 専用**: カーネルは NEON の `sdot` を前提にしている。x86 の Mac、Linux、Windows ではカーネルが作られず、Dart だけの経路（遅いが正しい）になる。

### 新しく学んだ技術

- **build hooks（`package:hooks` + `native_toolchain_c`）**: `CBuilder.library` に、ソース、フレームワーク、フラグ、リンクするライブラリを渡すだけでターゲット別にビルドしてくれる。Objective-C の場合、`.c` と `.m` を混ぜても拡張子で言語を判断してくれる。iOS/macOS 向けの `-Wl,-encryptable` や、Android 15 の 16KB ページ対応の `-Wl,-z,max-page-size=16384` は向こうが自動で付ける。
- **`@DefaultAsset`**: `library` 宣言に付けると、そのファイルの `@Native` 関数すべての asset ID をまとめて指定できる。`native.dart` と `gpu.dart` の両方に付けて、同じ1つの asset を指すようにした。

### 優れたエンジニアの思考法

**引っ越しは「何も変えない」ことから始める。** コードを新しい家に移すとき、ついでに名前を変え、構造を整理し、API を磨きたくなる。でもそれを同時にやると、何かが壊れたときに原因が引っ越しなのか改装なのか分からなくなる。

だから最初のコミットでは、ラボのコードをほぼそのまま持ってきて、変えたのはライブラリの読み込み方だけにした。そして `tool/lab_parity.sh` で、ラボの CLI と dianoia の CLI に同じプロンプトを流し、隠れ状態・logits・生成トークンが**1バイトも違わない**ことを確かめた。CPU の Q8、Metal の Q8、Metal の Q4、chat と long の2ケースで、24ファイルすべてが一致した。

ビルドフラグは実は変わっている。`-mcpu=native` は `-mcpu=apple-m1` に、Metal 層の `-O2` は `-O3` になった。それでも結果が一致したのは、ラボ時代に `-ffp-contract=off` で浮動小数の計算順を固定し、ビット単位で再現できる設計にしておいたからだ。研究の段階で払った「ビット一致」へのこだわりが、引っ越しの安全網として効いた。

これから API の整理、ライフサイクル（解放、キャンセル、ストリーミング）、実機での build hooks 検証と進む。各段階でこの一致チェックを回し、壊していないことを確かめてから次に進む。

## 実機に持っていく: build hooks は本当にスマホで動くのか

build hooks を選んだとき、一番の不安は「Flutter の iOS/Android ビルドで、本当にカーネルがアプリに入るのか」だった。研究室ではプラグイン（podspec と CMake）で動いていたが、hooks では一度も試していなかった。

そこで研究室のベンチアプリを `tool/device_bench/` に移し、依存先をプラグインから dianoia パッケージに差し替えた。結果:

- iOS: `Runner.app/Frameworks/dianoia_kernels.framework` が自動で入った
- Android: APK に `libdianoia_kernels.so` が入った

そして中身の数字は研究室と完全に同じだった。iPhone 15 Pro Max では、CPU の出力が Mac の CPU とビット単位で一致し、GPU の誤差も KL も研究室の記録と同じ値。Pixel 8 では、KL が平均・p99・最大値まで研究室と同じだった。速度も同等（iPhone GPU の decode 29.4 tok/s、Pixel 8 GPU の warm first token 337 ms）。

### 落とし穴1: libm がリンクされていなかった

APK を実行する前に `llvm-readelf` でライブラリの依存を眺めたら、`expf` が未解決のまま残っていて、`libm.so` が NEEDED に無かった。研究室の CMake ビルドでは、NDK のツールチェーンが黙って `-lm` を付けてくれていた。`native_toolchain_c` はそれをしない。

実行していたら、端末上で `dlopen` が「シンボルが見つからない」で落ちていたはずだ。ビルドの仕組みを乗り換えたときは、**前の仕組みが黙ってやってくれていたこと**を疑うこと。依存ライブラリと未解決シンボルの一覧は、実機で落ちる前に見られる安い検査だ。

### 落とし穴2: CocoaPods の残骸

プラグインが無くなると CocoaPods も要らなくなる。ところが Xcode プロジェクトには Pods への参照が残っていて、ビルドが「ファイルリストが読めない」で落ちた。`pod deintegrate` はこの Mac の古い Ruby 2.6 で異常終了。結局、Flutter のテンプレートから iOS プロジェクトを作り直し、研究室版の設定（ファイル共有、スリープ無効化、iOS 16.4）だけを移した。

## チャットアプリ: 「止める」は意外と難しい

`example/` に、端末上のモデルと話せる簡単なチャットアプリを作った。モデルはワーカー isolate に置き、UI とはメッセージだけをやり取りする。返答はトークンごとに流れてきて、停止ボタンで途中で止められる。

ここで面白い問題があった。停止ボタンを押すと UI から「止めて」というメッセージがワーカーに届く。ところがワーカーは生成ループの中で `await model.step()` を回している。この `await` は、すでに完了した Future を待つだけの**マイクロタスク**で、イベントループに順番を譲らない。だから「止めて」のメッセージはいつまでも届かない。

解決策は、トークンごとに `await Future<void>.delayed(Duration.zero)` を挟むこと。これはマイクロタスクではなく**タイマー**なので、イベントループが一周し、その間にポートのメッセージが処理される。ライブラリ側には `ChatSession.send(onText:)` を足した。トークンごとに途中の文章を渡し、`false` が返ったら止める。非同期のコールバックを渡せば、呼び出し側のイベントループが回る。

iPhone 15 Pro Max でも GPU で動き、日本語の質問に日本語で答えた（最初のトークンまで 309 ms、25.9 tok/s）。Pixel 8 では次のことを確かめた:

- 最初のトークンまで 310 ms
- 前の発言を踏まえた続き（"that city" → 京都）が通じる
- 停止ボタンで約3秒・34トークンのところで止まる

ただし、画面に流しながらの速度は約 12 tok/s で、ベンチ（約 14.5 tok/s）より遅い。トークンごとに画面を描き直していて、その描画が計算と同じ GPU を取り合っているのでは、と疑っている。まだ測っていない。描画の頻度を間引けば戻るかどうかが、次に確かめること。

### ちょっとした教訓: adb のタップ座標

adb でテキストを入力して送信ボタンをタップしたら、何も起きなかった。キーボードが出て入力欄が上に押し上げられ、送信ボタンも一緒に動いていたからだ。画面の操作を自動化するときは、**操作のたびに画面を見直す**。座標は前のスクリーンショットのものでしかない。

## API の整理: 店先と厨房を分ける

研究室から持ってきたままの `package:dianoia/dianoia.dart` は、18個のファイルを全部 export していた。アプリから見ると、`GpuForward.decodeRows` や `setQ8Vector` のような計測用のつまみ、isolate の executor、GGUF の読み込み器まで、全部が同じ棚に並んでいる状態だった。これでは「どれを使えばいいのか」が分からない。

そこで入口を2つに分けた。

- **`dianoia.dart`（店先）**: アプリが使うものだけ。`LocalModel`、`InferenceBackend`、`ChatSession`、`ChatTurn`、`ContextLengthExceeded`、ツール呼び出しの型。
- **`lab.dart`（厨房）**: 中身の全部。CLI、ベンチ、テストが使う。予告なく変わってよい場所。

`LocalModel.load(dir)` は、それまでアプリが自分でやっていた手順をまとめたものだ。バンドルの中の重み（Q4 か Q8）を見つけ、ネイティブカーネルを読み込み、GPU で開けなければ CPU に切り替える。サンプルアプリのワーカーは、この1行と `model.chat(...)` だけで済むようになった。

### 文字列で例外を見分けない

整理の途中で、サンプルアプリが文脈長オーバーを `'$e'.contains('maxSeq')` で見分けているのに気づいた。例外のメッセージは、誰かが文言を直した瞬間に一致しなくなる。そこで `ChatSession.send` に専用の例外 `ContextLengthExceeded` を足した。プロンプトに返答の余地が無ければ、計算を始める前に投げる。失敗したときに会話の履歴を元に戻すのもライブラリ側に移した。呼び出し側が `messages` をいじって後始末する必要はもう無い。

### まだやっていないこと: 片付け

`LocalModel` にはまだ `close()` が無い。GPU のバッファを解放する C 関数が、Metal にも Vulkan にも存在しないからだ。何もしない `close()` を置くと「解放したつもり」の事故を招くので、あえて置かなかった。ネイティブ側に解放の口を作り、実機で確かめるのが次の段になる。
