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
    
}
