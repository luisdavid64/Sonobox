package miPhysics.Engine;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;



public class WavWriter {
  private final File file;
  private final int sr, ch;
  private final ByteArrayOutputStream headerBuf = new ByteArrayOutputStream(44);
  private FileOutputStream fos;
  private long samplesWritten = 0;
  private boolean open = false;

  public WavWriter(File file, int sampleRate, int channels) {
    this.file = file; this.sr = sampleRate; this.ch = channels;
  }

  public void start() throws IOException {
    fos = new FileOutputStream(file);
    // write placeholder 44-byte header
    writeHeader(0); 
    open = true;
  }

  public void append(float[][] data, int nframes) throws IOException {
    if (!open) return;
    // interleave and convert float[-1..1] -> 16-bit
    for (int i = 0; i < nframes; i++) {
      for (int c = 0; c < ch; c++) {
        float f = data[Math.min(c, data.length-1)][i]; // duplicate if mono
        int s = (int)Math.max(-32768, Math.min(32767, Math.round(f * 32767f)));
        fos.write(s & 0xFF);
        fos.write((s >>> 8) & 0xFF);
      }
    }
    samplesWritten += nframes;
  }

  public void stop() throws IOException {
    if (!open) return;
    fos.flush();
    // fix up header sizes
    long dataBytes = samplesWritten * ch * 2L;
    fos.getChannel().position(0);
    writeHeader(dataBytes);
    fos.close();
    open = false;
  }

  private void writeHeader(long dataBytes) throws IOException {
    // RIFF header
    ByteBuffer b = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
    b.put("RIFF".getBytes("US-ASCII"));
    b.putInt((int)(36 + dataBytes));
    b.put("WAVE".getBytes("US-ASCII"));
    b.put("fmt ".getBytes("US-ASCII"));
    b.putInt(16);               // PCM fmt chunk size
    b.putShort((short)1);       // PCM
    b.putShort((short)ch);
    b.putInt(sr);
    b.putInt(sr * ch * 2);      // byte rate
    b.putShort((short)(ch * 2));// block align
    b.putShort((short)16);      // bits per sample
    b.put("data".getBytes("US-ASCII"));
    b.putInt((int)dataBytes);
    fos.write(b.array());
  }
}
