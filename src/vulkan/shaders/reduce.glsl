// The sum of v over the workgroup, returned to every invocation.
shared float group_partial[64];

float group_sum(float v) {
  v = subgroupAdd(v);
  barrier();
  if (subgroupElect()) group_partial[gl_SubgroupID] = v;
  barrier();
  float total = 0.0;
  for (uint s = 0; s < gl_NumSubgroups; s++) total += group_partial[s];
  return total;
}
