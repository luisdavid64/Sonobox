void oscEvent(OscMessage msg) {
  String pattern = msg.addrPattern();
  switch(pattern) {

  case "/msgTrigger":
    triggerText = msg.get(0).stringValue();
    println("received trigger msg::" + triggerText);
    showText = true;  // Enable the text to be displayed
    textTimer = millis();  // Reset the timer
    break;
    
  case "/tissueToggle":
    ArrayList<String> stringIDs = new ArrayList<>();
    ArrayList<Float> forces = new ArrayList<>();
    int i = 0;
    int forceRampSteps = (int)(1. * 44100);
    stringIDs.add(msg.get(0).stringValue());  // firstID
    stringIDs.add(msg.get(2).stringValue());  // secondID
    stringIDs.add(msg.get(4).stringValue());  // thirdID

    forces.add(msg.get(1).floatValue());  // firstForce
    forces.add(msg.get(3).floatValue());  // secondForce
    forces.add(msg.get(5).floatValue());  // thirdForce
        
    for (int j = 0; j < Math.min(model.getDrivers().size(), stringIDs.size()); j++) {
        Driver3D driver = model.getDrivers().get(j);
        driver.moveDriver(model.getMass(stringIDs.get(j)));
        driver.applyFrc(forces.get(j), forces.get(j), forces.get(j));
        println("RECEIVED:: driver " + driver.getName() + " exciting node " + stringIDs.get(j) + " with force " + forces.get(j));
    }
    break;
  
  case "/multipleToggle":
      for (int k = 0; k < msg.arguments().length; k++) {
        float excF = msg.get(k).floatValue();
        Driver3D dr = drivers.get(k);
        dr.applyFrc(excF, excF, excF);
        //println("Driver Name: " + dr.getName());
        //println("excited with: " + excF);
        }
    break;
    
  case "/moveListener":
    String toMassName = msg.get(0).stringValue();
    String listenerName = msg.get(1).stringValue();
    println(" ### LIST MSG RECEIVED ####");
    for(Observer3D obs : model.getObservers()){
      if (obs.getName().equals(listenerName)) {  
            obs.moveObserver(model.getMass(toMassName));
            System.out.println("#####################  OBS moved to mass: " + toMassName);
            println("#######################################");
        }
    }
    break;
    
  case "/inertiaController":
    String setName = msg.get(0).stringValue();
    float setRadius = msg.get(1).floatValue();
    float setMass = msg.get(2).floatValue();  
    println("RECEIVED subset mod::" + setName + " with radius value:: " + setRadius);
    phys.setParamForMassSubset(setName, param.RADIUS, setRadius);
    phys.setParamForMassSubset(setName, param.MASS, setMass);
    break;
  }
  // OSC ModelController API usage
  processOSCModelControllerEvents(pattern, msg);
    
}

void processOSCModelControllerEvents(String pattern, OscMessage msg) {

    // -------------------- CONTROLLER ROUTES (NEW) --------------------
  switch (pattern) {
    case "/dim/y": { // arg: +1 or -1
      int step = asInt(msg, 0, 0);
      if (step > 0) controller.incrementY();
      else if (step < 0) controller.decrementY();
      break;
    }

    case "/dim/x": { // arg: +1 or -1
      controller.adjustX(signOrZero(msg, 0));
      break;
    }

    case "/dim/z": { // arg: +1 or -1
      controller.adjustZ(signOrZero(msg, 0));
      break;
    }

    case "/mass/radius": { // arg: +1 or -1
      controller.adjustRadius(signOrZero(msg, 0));
      break;
    }

    case "/mass/m": { // arg: +1 or -1
      controller.adjustMs(asDoubleArray(msg));
      break;
    }

    case "/spring/k": { // arg: +1 or -1
      controller.adjustKs(asDoubleArray(msg));
      break;
    }

    case "/spring/c": { // arg: +1 or -1
      controller.adjustCs(asDoubleArray(msg));
      break;
    }

    case "/interaction/next": { // no args
      controller.cycleInteractionType();
      break;
    }
    case "/interaction/set": { // no args
      controller.setInteractionType(msg.get(0).stringValue());
      break;
    }

    case "/shiftInOut": { // arg: "X" or "Z"
      char axis = axisChar(msg, 0, 'X');
      controller.shiftDriversListeners(axis);
      break;
    }

    case "friction": {
      float delta = asFloat(msg, 0, 0);
      float mult = asFloat(msg, 1, 1);
      controller.adjustGlobalFriction(delta, mult);
    }

    case "resolution": {
      float mult = asFloat(msg, 0.5, 2);
      controller.adjustResolution(mult);
      break;
    }

    // -------------------- NON-CONTROLLER UTILITIES (OPTIONAL) --------------------

    case "/config/save": { // optional string arg: base path
      String name = (msg.typetag().length() > 0)
        ? msg.get(0).stringValue()
        : "/Users/luisreyes/Sonify/SonoBox/model_configs/config_api.json";
      config.writeProcessingJson(java.nio.file.Paths.get(name));
      System.out.println("Saved current configuration to Processing JSON format.");
      break;
    }
  }
}

int signOrZero(OscMessage m, int i) {
  return Integer.signum(asInt(m, i, 0));
}

int asInt(OscMessage m, int i, int defVal) {
  try { return m.get(i).intValue(); } catch (Exception e) { return defVal; }
}

float asFloat(OscMessage m, int i, float defVal) {
  try { return m.get(i).floatValue(); } catch (Exception e) { return defVal; }
}

char axisChar(OscMessage m, int i, char defVal) {
  try {
    String s = m.get(i).stringValue();
    if (s == null || s.isEmpty()) return defVal;
    char c = Character.toUpperCase(s.charAt(0));
    return (c == 'X' || c == 'Z') ? c : defVal;
  } catch (Exception e) {
    return defVal;
  }
}

double[] asDoubleArray(OscMessage m) {
  String tags = m.typetag();          // e.g., "iii", "fff", "ifs", ...
  int n = tags.length();
  double[] out = new double[n];

  for (int i = 0; i < n; i++) {
    char t = tags.charAt(i);
    try {
      switch (t) {
        case 'i': out[i] = m.get(i).intValue();   break;  // 32-bit int
        case 'f': out[i] = m.get(i).floatValue(); break;  // 32-bit float
        case 'h': out[i] = (double) m.get(i).longValue();  break; // 64-bit int
        case 'd': out[i] = m.get(i).doubleValue();         break; // 64-bit float
        default:  out[i] = Double.NaN; // non-numeric (string, blob, etc.)
      }
    } catch (Exception e) {
      out[i] = Double.NaN;
    }
  }
  return out;
}
