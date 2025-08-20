boolean yPressed = false;
boolean xPressed = false;
boolean zPressed = false;
boolean upPressed = false;
boolean downPressed = false;
boolean cPressed = false;
boolean dPressed = false;
boolean rPressed = false;

import java.util.List;
import java.util.ArrayList;

void keyPressed() {
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
    String name = config.name
    + "_x_" + config.dimX + "y_" + config.dimY + "z_" + config.dimZ + "_inter_" + config.interactionType + ".json";
    config.writeProcessingJson(Paths.get(base + "/" + name));
    println("Saved current configuration to Processing JSON format.");
  }

  if (key == 'c' || key == 'C') cPressed = true;
  if (key == 'r' || key == 'R') rPressed = true;
  if (key == 'd' || key == 'D') dPressed = true;
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
  if (key == 'r' || key == 'R') rPressed = false;
  if (key == 'd' || key == 'D') dPressed = false;
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

  if (xPressed && (upPressed || downPressed)) {
    config.dimX += (downPressed ? -1 : 1);
    println("Changing X dimension to: " + config.dimX);
    modelChanged = true;
  }

  if (zPressed && (upPressed || downPressed)) {
    config.dimZ += (downPressed ? -1 : 1);
    println("Changing Z dimension to: " + config.dimZ);
    modelChanged = true;
  }

  if (rPressed && (upPressed || downPressed)) {
    config.massRadius += (downPressed ? -1 : 1);
    println("Decreasing mass size to: " + config.dimZ);
    modelChanged = true;
  }
  
  if (cPressed) {
      
      InteractionType[] vals = InteractionType.values();
      config.interactionType = vals[(config.interactionType.ordinal() + 1) % vals.length];
      modelChanged = true;
      produceRenderedMessage(config.interactionType.name());
  }
  if (dPressed && (xPressed || zPressed)) {
    String axis = xPressed ? "X" : "Z";
    var newDrivers = shiftNodeName(config.driverNodes, axis);
    var newListeners = shiftNodeName(config.listenerNodes, axis);
    config.driverNodes = newDrivers;
    config.listenerNodes = newListeners;
    modelChanged = true;
    produceRenderedMessage("Shifted driver and listener nodes on " + axis + " axis.");
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
