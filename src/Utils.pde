void produceRenderedMessage(String msg) {
  triggerText = msg;
  showText = true;  // Enable the text to be displayed
  textTimer = millis();  // Reset the timer
}