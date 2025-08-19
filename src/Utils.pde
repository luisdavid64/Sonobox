double[] baselineW;

void initBaseline(int[] layers) {
  int tot = 0; for (int v : layers) tot += v;
  baselineW = new double[layers.length];
  for (int i = 0; i < layers.length; i++) {
    baselineW[i] = (tot == 0) ? 1.0 / layers.length : (double) layers[i] / tot;
  }
}

// Choose the layer to +1: most under target after the increment
int pickForIncrement(int[] layers, double[] w) {
  int total = 0; for (int v : layers) total += v;
  int best = 0;
  double bestErr = Double.POSITIVE_INFINITY; // smaller (more negative) is better
  for (int i = 0; i < layers.length; i++) {
    double target = w[i] * (total + 1);
    double err = (layers[i] + 1) - target;
    if (err < bestErr) { bestErr = err; best = i; }
  }
  return best;
}

// Choose the layer to -1: most over target after the decrement, but keep >= 1
int pickForDecrement(int[] layers, double[] w) {
  int total = 0; for (int v : layers) total += v;
  int best = -1;
  double bestErr = -Double.POSITIVE_INFINITY; // larger is better
  for (int i = 0; i < layers.length; i++) {
    if (layers[i] <= 1) continue; // enforce minimum of 1
    double target = w[i] * (total - 1);
    double err = (layers[i] - 1) - target;
    if (err > bestErr) { bestErr = err; best = i; }
  }
  return best; // -1 means can't decrement without breaking min=1
}

void produceRenderedMessage(String msg) {
  triggerText = msg;
  showText = true;  // Enable the text to be displayed
  textTimer = millis();  // Reset the timer
}