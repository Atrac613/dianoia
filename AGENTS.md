# AGENTS.md

## Purpose

This repository hosts `dianoia`, on-device LLM inference orchestrated in
Dart with native CPU kernels and Metal or Vulkan GPU kernels. Dart owns the
model and the graph; native code does only arithmetic.

## Working Rules

- Treat `lib/dianoia.dart` as the public API surface; keep implementation
  under `lib/src/`.
- Native sources live in `src/` and are compiled by `hook/build.dart` into
  the code asset `package:dianoia/kernels`. Do not add platform build files
  (podspec, CMakeLists.txt) for the kernels.
- After editing `src/vulkan/shaders/`, run `tool/compile_shaders.sh` and
  commit the regenerated `src/vulkan/spirv/`.
- Run `dart format .`, `dart analyze` and `dart test` after meaningful
  changes. On an Apple silicon Mac with dart-llm-lab checked out next to
  this repository, also run `tool/lab_parity.sh`: changes that are not meant
  to alter arithmetic must keep every dump byte-identical to the lab.
- Never commit model weights, caches (`*.cache`), state files or `build/`.
- Code, comments, docs and commit messages are in English. `FOR_ME.md` is
  the exception: a Japanese narrative of the project for its owner.

## Commits

- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
  `chore:`, `perf:`, `build:`, `ci:`), imperative subject, at most 72
  characters, no trailing period.
- No AI attribution trailers or comments.
- One logical change per commit.
