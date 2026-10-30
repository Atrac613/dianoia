#ifndef LAB_Q8_VECTOR_POLICY_H
#define LAB_Q8_VECTOR_POLICY_H
#include <stdint.h>
#include <string.h>

// The only device/cache/shape combination with paired chat and long-context
// acceptance plus two independent performance runs. Reject near matches.
static inline int lab_q8_vector_head_eligible(const char* manufacturer, const char* model,
                                             const char* gpu, int q4_body,
                                             int64_t rows, int64_t cols) {
  return q4_body && rows == 128000 && cols == 2048 &&
         strcmp(manufacturer, "Google") == 0 && strcmp(model, "Pixel 8") == 0 &&
         strcmp(gpu, "Mali-G715") == 0;
}
#endif
