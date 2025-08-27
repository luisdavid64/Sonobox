package miPhysics.Engine;

/**
 * Driver module (an interaction module that acts upon a Mass)
 * - Legacy: triggerForceRamp(x,y,z,steps) → 0→target linear ramp
 * - New:    ADSR driven by a FULL force vector (direction + peak magnitude)
 */
public class Driver3D extends InOut {

    // ===== Constructors =====
    public Driver3D(Mass m) { super(m); setType(inOutType.DRIVER3D); }
    public Driver3D() { this(null); }

    // ===== Basic I/O API (unchanged) =====
    public void applyPos(Vect3D pos){ this.getMat().setPos(pos); }
    public void applyPos(double x,double y,double z){ this.applyPos(new Vect3D(x,y,z)); }

    public void applyFrc(Vect3D frc){ this.getMat().applyForce(frc); }
    public void applyFrc(double x,double y,double z){ this.applyFrc(new Vect3D(x,y,z)); }

    public void moveDriver(Mass m){ this.connect(m); }

    // ===== Legacy linear ramp (kept for compatibility) =====
    public void triggerForceRamp(double x,double y,double z,int timeSteps){
        m_rampSteps = Math.max(1, timeSteps);
        m_targetForce.set(x,y,z);
        m_currentForce.reset();
        m_steps = 0;
        m_rampActive = true;
    }
    public void triggerForceRamp(Vect3D frc,int timeSteps){
        triggerForceRamp(frc.x, frc.y, frc.z, timeSteps);
    }

    // ===== ADSR (peak-matched; vector gives direction + magnitude) =====
    public enum EnvState { IDLE, ATTACK, DECAY, SUSTAIN, RELEASE }
    public enum Curve    { LINEAR, HANN }

    private EnvState m_state = EnvState.IDLE;
    private Curve    m_curve = Curve.HANN;

    private int    m_atkS = 1, m_decS = 1, m_relS = 1; // durations in SAMPLES
    private double m_sus  = 0.5;                       // 0..1 of peak

    private final Vect3D m_dirUnit = new Vect3D(0,1,0); // unit direction of ADSR force
    private double m_peakMag = 0.0;                     // peak magnitude in Newtons
    private int    m_envStep = 0;                       // step within current stage
    private double m_levelNow = 0.0;                    // current amplitude (N) at boundaries

    /** Configure ADSR (samples; sustain 0..1). */
    public Driver3D setADSR(int attackS, int decayS, double sustain, int releaseS){
        m_atkS=Math.max(1,attackS);
        m_decS=Math.max(1,decayS);
        m_relS=Math.max(1,releaseS);
        m_sus =Math.max(0.0,Math.min(1.0,sustain));
        return this;
    }
    /** Curve for attack/release easing. */
    public Driver3D setCurve(Curve c){ m_curve=(c==null?Curve.HANN:c); return this; }

    /** Start ADSR using a FULL force vector (direction + peak magnitude). */
    public void triggerADSR(Vect3D peak){
        double mag = Math.sqrt(peak.x*peak.x + peak.y*peak.y + peak.z*peak.z);
        if (mag < 1e-12) return;                 // ignore zero
        m_peakMag = mag;
        m_dirUnit.set(peak.x/mag, peak.y/mag, peak.z/mag); // unit dir
        m_state = EnvState.ATTACK;
        m_envStep = 0;
        m_levelNow = 0.0;
        m_rampActive = false;                    // ensure legacy ramp doesn't double-apply
    }
    public void triggerADSR(double x,double y,double z){ triggerADSR(new Vect3D(x,y,z)); }

    /** Go to RELEASE stage (note-off). */
    public void releaseADSR(){
        if (m_state != EnvState.IDLE) { m_state = EnvState.RELEASE; m_envStep = 0; }
    }

    public boolean isADSRActive(){ return m_state != EnvState.IDLE; }

    // ===== Per-tick update =====
    @Override
    public void compute(){
        // --- ADSR path ---
        switch(m_state){
            case IDLE:
                break;

            case ATTACK: {
                double t   = (double)m_envStep / (double)m_atkS;          // 0..1
                double amp = ease(m_curve, t) * m_peakMag;                 // 0 → peak
                applyFrc(m_dirUnit.x*amp, m_dirUnit.y*amp, m_dirUnit.z*amp);
                if (++m_envStep >= m_atkS) {
                    m_state = EnvState.DECAY; m_envStep = 0; m_levelNow = m_peakMag;
                }
            } break;

            case DECAY: {
                double target = m_sus * m_peakMag;
                double t   = (double)m_envStep / (double)m_decS;          // 0..1
                // peak → sustain (linear progression; change ease if desired)
                double amp = lerp(m_peakMag, target, t);
                applyFrc(m_dirUnit.x*amp, m_dirUnit.y*amp, m_dirUnit.z*amp);
                if (++m_envStep >= m_decS) {
                    m_state = EnvState.SUSTAIN; m_levelNow = target;
                }
            } break;

            case SUSTAIN: {
                double amp = m_levelNow;                                   // constant
                applyFrc(m_dirUnit.x*amp, m_dirUnit.y*amp, m_dirUnit.z*amp);
                // remain until releaseADSR()
            } break;

            case RELEASE: {
                double t   = (double)m_envStep / (double)m_relS;          // 0..1
                double amp = (1.0 - ease(m_curve, t)) * m_levelNow;       // sustain → 0
                applyFrc(m_dirUnit.x*amp, m_dirUnit.y*amp, m_dirUnit.z*amp);
                if (++m_envStep >= m_relS) {
                    m_state = EnvState.IDLE; m_levelNow = 0.0;
                }
            } break;
        }

        // --- Legacy 0→target ramp runs ONLY if ADSR is idle ---
        if (m_state == EnvState.IDLE && m_rampActive) {
            m_steps++;
            m_currentForce.set(m_targetForce).mult((double)m_steps/(double)m_rampSteps);
            this.applyFrc(m_currentForce);
            if (m_steps > m_rampSteps) m_rampActive = false;
        }
    }

    // ===== Helpers =====
    private static double ease(Curve c, double t){
        t = (t<=0)?0 : (t>=1)?1 : t;
        return (c==Curve.HANN) ? (0.5 - 0.5*Math.cos(Math.PI*t)) : t; // cosine-in
    }
    private static double lerp(double a,double b,double t){
        t = (t<=0)?0 : (t>=1)?1 : t;
        return a + (b - a) * t;
    }

    // ===== Legacy ramp fields =====
    private final Vect3D m_targetForce = new Vect3D();
    private final Vect3D m_currentForce = new Vect3D();
    private int m_steps = 0, m_rampSteps = 0;
    private boolean m_rampActive = false;
}
