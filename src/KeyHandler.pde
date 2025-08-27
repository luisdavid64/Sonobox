boolean yPressed = false;
boolean xPressed = false;
boolean zPressed = false;
boolean upPressed = false;
boolean downPressed = false;
boolean cPressed = false;
boolean dPressed = false;
boolean rPressed = false;
boolean qPressed = false;
boolean fPressed = false;

import java.util.List;
import java.util.ArrayList;

void keyPressed() {
  if (Character.isDigit(key)) {
    int index = Character.getNumericValue(key) - 1;
    if (index >= 0 && index < drivers.size()) {
      float f = 3.0;
      Driver3D driver = drivers.get(index);
      driver.apdplyFrc(f, f, f);
      // driver.releaseADSR();
      // driver.triggerADSR(f,f,f);
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
  if (key == 'q' || key == 'Q') qPressed = true;
  if (key == 'd' || key == 'D') dPressed = true;
  if (key == 'y' || key == 'Y') yPressed = true;
  if (key == 'x' || key == 'X') xPressed = true;
  if (key == 'z' || key == 'Z') zPressed = true;
  if (key == 'f' || key == 'F') fPressed = true;
  if (key == CODED && keyCode == UP) upPressed = true;
  if (key == CODED && keyCode == DOWN) downPressed = true;
  checkModelChanges();

}

void keyReleased() {
  // combo tracking (optional)
  if (key == 'c' || key == 'C') cPressed = false;
  if (key == 'r' || key == 'R') rPressed = false;
  if (key == 'q' || key == 'Q') qPressed = false;
  if (key == 'd' || key == 'D') dPressed = false;
  if (key == 'y' || key == 'Y') yPressed = false;
  if (key == 'x' || key == 'X') xPressed = false;
  if (key == 'z' || key == 'Z') zPressed = false;
  if (key == 'f' || key == 'F') fPressed = false;
  if (key == CODED && keyCode == UP) upPressed = false;
  if (key == CODED && keyCode == DOWN) downPressed = false;
}

void checkModelChanges() {
  boolean modelChanged = false;
  if (yPressed && upPressed) {
    controller.incrementY();
  }
  if (yPressed && downPressed) {
    controller.decrementY();
  }

  if (xPressed && (upPressed || downPressed)) {
    controller.adjustX(upPressed ? 1 : -1);
  }

  if (zPressed && (upPressed || downPressed)) {
    controller.adjustZ(upPressed ? 1 : -1);
  }

  if (rPressed && (upPressed || downPressed)) {
    controller.adjustRadius(upPressed ? 1 : -1);
  }
  
  if (qPressed && (upPressed || downPressed)) {
    controller.adjustResolution(upPressed ? 1 : -1);
  }
  
  if (cPressed) {
      controller.cycleInteractionType();
      produceRenderedMessage(config.interactionType.name());
  }
  if (dPressed && (xPressed || zPressed)) {
    char axis = (xPressed ? 'X' : 'Z');
    controller.shiftDriversListeners(axis);
    produceRenderedMessage("Shifted driver and listener nodes on " + axis + " axis.");
  }

  if (fPressed && (upPressed || downPressed)) {
    controller.adjustGlobalFriction(0, upPressed ? 1.1 : 0.9);
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
