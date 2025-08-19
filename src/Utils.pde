static int[] redistributeProportionally(int newTotal, int[] old) {
  int L = old.length;
  int oldTotal = 0;
  for (int v : old) oldTotal += v;
  if (L == 0) return old.clone();

  // Handle edge cases
  if (newTotal < L) newTotal = L; // at least 1 per layer
  if (oldTotal == 0) {
    // Even split if old was all zeros
    int base = newTotal / L, rem = newTotal % L;
    int[] res = new int[L];
    for (int i = 0; i < L; i++) res[i] = base + (i < rem ? 1 : 0);
    return res;
  }

  // Quotas and floors
  double[] quota = new double[L];
  int[] res = new int[L];
  int sum = 0;
  for (int i = 0; i < L; i++) {
    quota[i] = (double) old[i] * newTotal / (double) oldTotal;
    res[i] = (int) Math.floor(quota[i]);
    if (res[i] < 1) res[i] = 1; // enforce ≥1 per layer
    sum += res[i];
  }

  // Fix over-assign if enforcing ≥1 pushed us over
  while (sum > newTotal) {
    // take 1 from a layer with res[i] > 1 and the smallest fractional part
    int best = -1;
    double bestFrac = Double.POSITIVE_INFINITY;
    for (int i = 0; i < L; i++) {
      double frac = quota[i] - Math.floor(quota[i]);
      if (res[i] > 1 && frac < bestFrac) {
        bestFrac = frac;
        best = i;
      }
    }
    if (best == -1) break; // nothing we can do
    res[best]--; sum--;
  }

  // Distribute remainder to largest fractional parts
  while (sum < newTotal) {
    int best = -1;
    double bestFrac = -1.0;
    for (int i = 0; i < L; i++) {
      double frac = quota[i] - Math.floor(quota[i]);
      if (frac > bestFrac) { bestFrac = frac; best = i; }
    }
    res[best]++; sum++;
    // Slight nudge to avoid picking same index forever (optional)
    quota[best] = Math.floor(quota[best]);
  }

  return res;
}