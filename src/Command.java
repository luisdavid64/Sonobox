package miPhysics.Engine;
enum Action {
  APPLY_FORCE_TO_DRIVER,         // payload: index, fx, fy, fz
  SAVE_CONFIG,                   // payload: path to save
  INCREMENT_Y, DECREMENT_Y,
  ADJUST_X,                      // payload: +1 or -1
  ADJUST_Z,                      // payload: +1 or -1
  ADJUST_RADIUS,                 // payload: +1 or -1
  CYCLE_INTERACTION_TYPE,
  SHIFT_DRIVERS_LISTENERS_X,
  SHIFT_DRIVERS_LISTENERS_Z
}

final class Command {
  final Action action;
  final float[] f;     // numeric payload, as needed
  final String s;      //s string payload, as needed
  Command(Action a) { this(a, null, null); }
  Command(Action a, float[] f) { this(a, f, null); }
  Command(Action a, String s) { this(a, null, s); }
  Command(Action a, float[] f, String s) { this.action = a; this.f = f; this.s = s; }
}