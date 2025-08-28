package miPhysics.Engine; 

import java.nio.FloatBuffer;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import miPhysics.Engine.PhysicsContext;
import miPhysics.Engine.Sound.*;

import java.util.logging.Logger;
import java.util.logging.Level;


/**
 * This class allows tapping into the audio stream of miPhysics audio client. 
 *
 */
public class miTappedPhyAudioClient extends miPhyAudioClient {

    /** Callback fired once per processed audio block (on the audio thread). */
    public interface AudioTap {
        /**
         * @param data       snapshot of the current block, shape [channels][frames]
         * @param nframes    number of frames in this block
         * @param sampleRate sample rate
         * @param timeNanos  timestamp passed by the audio server (or -1 if not meaningful)
         */
        void onAudio(float[][] data, int nframes, int sampleRate, long timeNanos);
    }

    private final CopyOnWriteArrayList<AudioTap> taps = new CopyOnWriteArrayList<>();
    private final int sampleRateInt;

    /* ---------- Convenience factories mirroring the base class ---------- */

    public static miTappedPhyAudioClient miPhyJack(
        float sampleRate, int bufS, int inputChannelCount, int outputChannelCount, PhysicsContext c
    ) {
        try {
            return new miTappedPhyAudioClient(sampleRate, inputChannelCount, outputChannelCount, c, bufS, "JACK");
        } catch (Exception e) {
            Logger.getLogger(miTappedPhyAudioClient.class.getName())
                .log(Level.SEVERE, "Could not create a JACK miTappedPhyAudioClient", e);
            return null;
        }
    }

    public static miTappedPhyAudioClient miPhyClassic(
        float sampleRate, int bufS, int inputChannelCount, int outputChannelCount, PhysicsContext c
    ) {
        try {
            return new miTappedPhyAudioClient(sampleRate, inputChannelCount, outputChannelCount, c, bufS, "JavaSound");
        } catch (Exception e) {
            Logger.getLogger(miTappedPhyAudioClient.class.getName())
                .log(Level.SEVERE, "Could not create a JavaSound miTappedPhyAudioClient", e);
            return null;
        }
    }

    /* ------------------------- Constructors ---------------------------- */

    public miTappedPhyAudioClient(
            float sampleRate,
            int inputChannelCount,
            int outputChannelCount,
            PhysicsContext c,
            int bufferSize,
            String serverType
    ) throws Exception {
        super(sampleRate, inputChannelCount, outputChannelCount, c, bufferSize, serverType);
        this.sampleRateInt = Math.round(sampleRate);
    }

    /* --------------------------- API ---------------------------------- */

    public void addTap(AudioTap tap) {
        if (tap != null) taps.add(tap);
    }

    public void removeTap(AudioTap tap) {
        if (tap != null) taps.remove(tap);
    }

    public void clearTaps() {
        taps.clear();
    }

    /* ----------------------- Audio hook logic -------------------------- */

    /**
     * We let the base class render first (super.process), which writes nframes
     * into each FloatBuffer and advances its position. Then we snapshot the
     * just-written region [pos - nframes, pos) for every channel and notify taps.
     */
    @Override
    public boolean process(long time, List<FloatBuffer> inputs, List<FloatBuffer> outputs, int nframes) {
        boolean ok = super.process(time, inputs, outputs, nframes);
        if (!ok || taps.isEmpty()) return ok;

        final int chCount = outputs.size();
        float[][] snap = new float[chCount][nframes];

        for (int ch = 0; ch < chCount; ch++) {
            FloatBuffer buf = outputs.get(ch);
            // super.process used FloatBuffer.put(float[]), which advances position by nframes
            int end = buf.position();
            int start = end - nframes;

            // Defensive clamp in case a provider uses unexpected positions
            if (start < 0) start = 0;
            if (end > buf.limit()) end = buf.limit();

            FloatBuffer ro = buf.asReadOnlyBuffer();
            ro.position(start);
            ro.limit(end);
            int len = ro.remaining(); // should be nframes, but clamp just in case
            ro.get(snap[ch], 0, len);
            if (len < nframes) {
                // pad tail with zeros if short read (rare)
                for (int i = len; i < nframes; i++) snap[ch][i] = 0f;
            }
        }

        // Fire taps (still on audio thread) — keep very lightweight.
        for (AudioTap tap : taps) {
            tap.onAudio(snap, nframes, sampleRateInt, time);
        }

        return true;
    }
}