// Packed int8 dot products (SPV_KHR_integer_dot_product), which phone GPUs
// such as Mali-G715 accelerate. The NDK's glslc predates
// GL_EXT_integer_dot_product, so OpSDot is declared directly.
#extension GL_EXT_spirv_intrinsics : require

// The dot product of two words of four signed int8.
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6018, 6019], id = 4450)
int sdot4x8(uint a, uint b, spirv_literal int packed_format);

int sdot(uint a, uint b) { return sdot4x8(a, b, 0); }
