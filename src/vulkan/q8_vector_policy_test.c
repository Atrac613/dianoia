#include <assert.h>
#include "q8_vector_policy.h"

int main(void) {
  assert(lab_q8_vector_head_eligible("Google", "Pixel 8", "Mali-G715", 1, 128000, 2048));
  assert(!lab_q8_vector_head_eligible("Google", "Pixel 8", "Mali-G715", 0, 128000, 2048));
  assert(!lab_q8_vector_head_eligible("Google", "Pixel 8 Pro", "Mali-G715", 1, 128000, 2048));
  assert(!lab_q8_vector_head_eligible("Other", "Pixel 8", "Mali-G715", 1, 128000, 2048));
  assert(!lab_q8_vector_head_eligible("Google", "Pixel 8", "Mali-G710", 1, 128000, 2048));
  assert(!lab_q8_vector_head_eligible("Google", "Pixel 8", "Mali-G715", 1, 7680, 2048));
  assert(!lab_q8_vector_head_eligible("Google", "Pixel 8", "Mali-G715", 1, 128000, 4096));
  return 0;
}
