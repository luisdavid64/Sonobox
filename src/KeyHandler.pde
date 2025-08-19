boolean yPressed = false;
boolean upPressed = false;

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
  if (key == CODED && keyCode == UP) upPressed = true;
  checkModelChanges();

}

void keyReleased() {
  // combo tracking (optional)
  if (key == 'y' || key == 'Y') yPressed = false;
  if (key == CODED && keyCode == UP) upPressed = false;
}

void checkModelChanges() {
  if (yPressed && upPressed) {
    println("Y + UP pressed: Resetting model");
    resetModel();
    upPressed = false;
  }
}