boolean yPressed = false;
boolean xPressed = false;
boolean zPressed = false;
boolean upPressed = false;
boolean downPressed = false;

void keyPressed() {
  // --- your original logic ---
  if (Character.isDigit(key)) {
    int index = Character.getNumericValue(key) - 1;
    if (index >= 0 && index < drivers.size()) {
      float f = 3.0;
      Driver3D driver = drivers.get(index);
      driver.applyFrc(f, f, f);
      println("PLAYING driver[" + index + "] -> " + driver.getName());
    }
  }
  if (key == 'i') {
    renderer.toggleModuleNameDisplay();
  }

  if (key == 'y' || key == 'Y') yPressed = true;
  if (key == 'x' || key == 'X') xPressed = true;
  if (key == 'z' || key == 'Z') zPressed = true;
  if (key == CODED && keyCode == UP) upPressed = true;
  if (key == CODED && keyCode == DOWN) downPressed = true;
  checkModelChanges();

}

void keyReleased() {
  // combo tracking (optional)
  if (key == 'y' || key == 'Y') yPressed = false;
  if (key == 'x' || key == 'X') xPressed = false;
  if (key == 'z' || key == 'Z') zPressed = false;
  if (key == CODED && keyCode == UP) upPressed = false;
  if (key == CODED && keyCode == DOWN) downPressed = false;
}

void checkModelChanges() {
  boolean modelChanged = false;
  if (yPressed && upPressed) {
    int newY = config.dimY + 1;
    config.numNodesPerLayer = redistributeProportionally(newY, config.numNodesPerLayer);
    config.dimY = newY;
    println("Increasing Y dimension to: " + config.dimY);
    modelChanged = true;
  }
  if (yPressed && downPressed) {
    int newY = Math.max(config.numNodesPerLayer.length, config.dimY - 1); // keep ≥1 per layer
    config.numNodesPerLayer = redistributeProportionally(newY, config.numNodesPerLayer);
    config.dimY = newY;
    config.numNodesPerLayer[0] -= 1; // Assuming the second layer is Y
    println("Decreasing Y dimension to: " + config.dimY);
    modelChanged = true;
  }
  if (xPressed && upPressed) {
    config.dimX += 1;
    println("Increasing X dimension to: " + config.dimX);
    modelChanged = true;
  }
  if (xPressed && downPressed) {
    config.dimX -= 1;
    println("Decreasing X dimension to: " + config.dimX);
    modelChanged = true;
  }
  if (zPressed && upPressed) {
    config.dimZ += 1;
    println("Increasing Z dimension to: " + config.dimZ);
    modelChanged = true;
  }
  if (zPressed && downPressed) {
    config.dimZ -= 1;                  
    println("Decreasing Z dimension to: " + config.dimZ);
    modelChanged = true;
  }
  if (modelChanged) {
    checkDimValidity();
    resetModel();
  }
}

void checkDimValidity() {
    if (config.dimX < 1 || config.dimY < 1 || config.dimZ < 1) {
        println("Invalid dimensions detected. Resetting to minimum valid values.");
        config.dimX = Math.max(config.dimX, 1);
        config.dimY = Math.max(config.dimY, 1);
        config.dimZ = Math.max(config.dimZ, 1);
        println("New dimensions: X=" + config.dimX + ", Y=" + config.dimY + ", Z=" + config.dimZ);
    }
}