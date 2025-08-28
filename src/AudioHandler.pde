public void setAudioClient() {
  if (audioStreamHandler != null) {
    audioStreamHandler.shutdown();
    audioStreamHandler = null;
  }
  audioStreamHandler = miPhyAudioClient.miPhyClassic(44100, 128, 0, 2, phys);
  audioStreamHandler.setListenerAxis(listenerAxis.Y);
  audioStreamHandler.setGain(gain);
  audioStreamHandler.start();
  
  minim = new Minim(this);
  int buffSize = 2048;
  try {
    boolean stereoAvailable = true;
    in = minim.getLineIn(Minim.STEREO, 2048);
    if (in.getFormat().getChannels() != 2) {
      in.close();
      stereoAvailable = false;
      in = minim.getLineIn(Minim.MONO, 2048);
    }
    
    out = minim.getLineOut(stereoAvailable ? Minim.STEREO : Minim.MONO);
  }
  catch (Exception e) {
    println("Error requesting stereo. Falling back to mono.");
    in = minim.getLineIn(Minim.MONO, 2048);
    out = minim.getLineOut(Minim.MONO);
  }
}

public void resetAudioClient() {
  if (audioStreamHandler != null) {
    audioStreamHandler.shutdown();
    audioStreamHandler = null;
    in = null;
    out = null;
  }
  setAudioClient();
}