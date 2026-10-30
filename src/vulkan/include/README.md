# Vulkan headers

`vulkan/` and `vk_video/` are the Khronos
[Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) 1.3.275,
unmodified, as the Android NDK r28 ships them, under the Apache License 2.0
(see each file).

Only the Windows build uses them: there is no Vulkan SDK to find from a build
hook, and `lab_vulkan.c` opens `vulkan-1.dll` at run time, so the headers are
all it needs. Android builds use the NDK's own copy.
