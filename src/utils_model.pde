ArrayList<String> jsonToList(JSONArray arr) {
  ArrayList<String> list = new ArrayList<String>();
  for (int i = 0; i < arr.size(); i++) {
    list.add(arr.getString(i));
  }
  return list;
}

void subsetsCreation3D(phy3DModel model, ArrayList<String> tissueNodeNames,
                       int[] nodeCounts, int dimX, int dimZ, ArrayList<String> massSubsetNames) {
  println("creating MASS SUBSETS::");

  int[] counts = new int[3];

  // Mass Subset Creation
  for (int t = 0; t < 3; t++) {
    String[] parts = tissueNodeNames.get(t).split("_");
    int startX = Integer.parseInt(parts[1]);
    int startY = Integer.parseInt(parts[2]);
    int startZ = Integer.parseInt(parts[3]);

    phys.createMassSubset(massSubsetNames.get(t));

    int yStart = startY;
    int yEnd = 0;
    for (int i = 0; i <= t; i++) yEnd += nodeCounts[i];

    for (int i = startX; i < dimX; i++) {
      for (int j = yStart; j < yEnd; j++) {
        for (int k = startZ; k < dimZ; k++) {
          String curMass = "m_" + i + "_" + j + "_" + k;
          phys.addMassToSubset(model.getMass(curMass), massSubsetNames.get(t));
          counts[t]++;
        }
      }
    }

    println("# " + massSubsetNames.get(t).replace("Masses", " masses") + ": " + counts[t]);
  }

  // Interaction Subset Creation
  println("creating interactions' subsets");
  String[] springSubsetNames = { "firstSprings", "secondSprings", "thirdSprings" };
  for (String name : springSubsetNames) phys.createInteractionSubset(name);

  int[] springCounts = new int[3];
  ArrayList<Interaction> interactionList = model.getInteractionList();

  for (Interaction interaction : interactionList) {
    String[] parts = interaction.getName().split("_");
    int y2 = Integer.parseInt(parts[5]);

    if (y2 < nodeCounts[0]) {
      phys.addInteractionToSubset(model.getInteraction(interaction.getName()), springSubsetNames[0]);
      springCounts[0]++;
    } else if (y2 < nodeCounts[0] + nodeCounts[1]) {
      phys.addInteractionToSubset(model.getInteraction(interaction.getName()), springSubsetNames[1]);
      springCounts[1]++;
    } else if (y2 < nodeCounts[0] + nodeCounts[1] + nodeCounts[2]) {
      phys.addInteractionToSubset(model.getInteraction(interaction.getName()), springSubsetNames[2]);
      springCounts[2]++;
    }
  }

  println("first springs: " + springCounts[0]);
  println("second springs: " + springCounts[1]);
  println("third springs: " + springCounts[2]);
}

void tissuePhysicalPropertiesInit(phy3DModel model, double[] nodesM, double[] nodesK, ArrayList<String> massSubsets) {
  // Ensure that we have the same number of nodes for masses and springs, and also match mass subsets
  if (nodesM.length != nodesK.length || nodesM.length != massSubsets.size()) {
    println("Error: The number of masses (M), spring constants (K), and mass subsets do not match.");
    return;
  }

  // Loop over the nodes and apply properties for each layer
  for (int i = 0; i < nodesM.length; i++) {
    // Fetch the current mass and stiffness values for this layer
    float currentM = (float) nodesM[i];
    float currentK = (float) nodesK[i];

    // Fetch the mass subset name from the list
    String currentMassSubset = massSubsets.get(i);

    // Print the physical properties being set
    println("Setting physical properties for layer " + (i + 1) + ": M → " + currentM + " | K → " + currentK);

    // Apply mass and stiffness to the corresponding subsets
    phys.setParamForMassSubset(currentMassSubset, param.MASS, currentM);
    phys.setParamForInteractionSubset(currentMassSubset, param.STIFFNESS, currentK);
  }
}

void tissueNodesDefinition3D(phy3DModel model, ArrayList<String> tissueNodeNames) {
  for (String nodeName : tissueNodeNames) {
    Mass mass = model.getMass(nodeName);
    mass.setParam(param.RADIUS, 6);
    println("changing size @ mass name: " + mass.getName());
  }
}

ArrayList<String> getMassSubsets(String modelType) {
  ArrayList<String> massSubsets = new ArrayList<String>();
  if ("1D".equals(modelType)) {
    massSubsets.add("firstMasses");
    massSubsets.add("secondMasses");
    massSubsets.add("thirdMasses");
  } else if ("2D".equals(modelType)) {
    massSubsets.add("firstMasses");
    massSubsets.add("secondMasses");
    massSubsets.add("thirdMasses");
  } else if ("3D".equals(modelType)) {
    massSubsets.add("firstMasses");
    massSubsets.add("secondMasses");
    massSubsets.add("thirdMasses");
  }
  return massSubsets;
}

public static float[] toFloatArray(JSONArray jsonArray) {
    float[] result = new float[jsonArray.size()];
    for (int i = 0; i < jsonArray.size(); i++) {
        Object value = jsonArray.get(i);
        if (value instanceof Number) {
            result[i] = ((Number) value).floatValue();
        } else {
            throw new IllegalArgumentException("JSONArray contains non-numeric value at index " + i);
        }
    }
    return result;
}

String getConfig() {
  String conf_path = System.getenv("BIOSONIX_CONF");
  
  // Fallback to default if not set
  if (conf_path == null) {
    conf_path = "/Users/luisreyes/Sonify/SonoBox/model_configs/config.json";
    println("No Config Found: Using " + conf_path);

  }
  return conf_path;
}

void resetModel() {
  if (model != null) {
    phys.clearModel();
    model = null;
  }
  createModelFromConfig();
}

void createModelFromConfig() {
  String modelType = config.modelDim; // "1D", "2D", or "3D"
  println("IMPLEMENTING A "+ modelType +" TOPOLOGY FOR THE SOUND MODEL");
  int[] nPerLayer = config.numNodesPerLayer; // number of nodes per layer
  double[] nodeM = config.M;
  double[] nodeK = config.K;
  int numNodes = Arrays.stream(nPerLayer).sum(); //Make sure it matches

  ArrayList<String> massSubsets = getMassSubsets(modelType);

  println("with n nodes in the first layer: "+ nPerLayer[0] +" in the second: "+nPerLayer[1]+" in the third: "+nPerLayer[2]);

  float dist = config.dist;
  float massesRadius = config.massRadius;

  // Generic dimensions for all model types
  int dimX = 1, dimZ = 1;
  if (!"1D".equals(modelType)) {
    dimX = config.dimX;
    dimZ = config.dimZ;
  }

  model = new phy3DModel("BioSonix" + modelType, phys.getGlobalMedium());
  model.setDim(dimX, numNodes, dimZ, 1);
  model.setGeometry(dist);
  model.setParams(nodeM[0], nodeK[0]);
  model.setMassRadius(massesRadius);
  model.setModelType(modelType);
  model.generate();
  model.translate(0, -150, 0);

  drivers = model.addDrivers(config.driverNodes);
  listeners = model.addListeners(config.listenerNodes);

  tissueNodeNames = new ArrayList<String>();
  tissueNodeNames.add("m_0_0_0");
  tissueNodeNames.add("m_0_" + nPerLayer[0] + "_0");
  tissueNodeNames.add("m_0_" + (nPerLayer[0] + nPerLayer[1]) + "_0");

  subsetsCreation3D(model, tissueNodeNames, nPerLayer, dimX, dimZ, massSubsets);
  tissuePhysicalPropertiesInit(model, nodeM, nodeK, massSubsets);
  tissueNodesDefinition3D(model, tissueNodeNames);

  phys.mdl().addPhyModel(model);
  println("Drivers initialized: " + (drivers != null));
  println("Listeners initialized: " + (listeners != null));
  
  phys.init();
  
}
