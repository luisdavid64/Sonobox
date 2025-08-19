boolean yPressed = false;
boolean xPressed = false;
boolean zPressed = false;
boolean upPressed = false;
boolean downPressed = false;
boolean cPressed = false;

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
  

  if (key == 's') {
    String base = "/Users/luisreyes/Sonify/SonoBox/model_configs/";
    String name = config.name + "_x_" + config.dimX + "y_" + config.dimY + "z_" + config.dimZ + ".json";
    config.writeProcessingJson(Paths.get(base + "/" + name));
    println("Saved current configuration to Processing JSON format.");
  }

  if (key == 'c' || key == 'C') cPressed = true;
  if (key == 'y' || key == 'Y') yPressed = true;
  if (key == 'x' || key == 'X') xPressed = true;
  if (key == 'z' || key == 'Z') zPressed = true;
  if (key == CODED && keyCode == UP) upPressed = true;
  if (key == CODED && keyCode == DOWN) downPressed = true;
  checkModelChanges();

}

void keyReleased() {
  // combo tracking (optional)
  if (key == 'c' || key == 'C') cPressed = false;
  if (key == 'y' || key == 'Y') yPressed = false;
  if (key == 'x' || key == 'X') xPressed = false;
  if (key == 'z' || key == 'Z') zPressed = false;
  if (key == CODED && keyCode == UP) upPressed = false;
  if (key == CODED && keyCode == DOWN) downPressed = false;
}

void checkModelChanges() {
  boolean modelChanged = false;
  if (yPressed && upPressed) {
    int i = pickForIncrement(config.numNodesPerLayer, baselineW);
    config.numNodesPerLayer[i] += 1;
    config.dimY += 1;
    modelChanged = true;
  }
  if (yPressed && downPressed) {
    int i = pickForDecrement(config.numNodesPerLayer, baselineW);
    if (i != -1) {
      config.numNodesPerLayer[i] -= 1;
      config.dimY -= 1;
      modelChanged = true;
    } 
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
  
  if (cPressed) {
      
      phy3DModel.interactionType[] vals = phy3DModel.interactionType.values();
      config.interactionType = vals[(config.interactionType.ordinal() + 1) % vals.length];}
      modelChanged = true;
      println("Switching to interaction type: " + config.interactionType);
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
