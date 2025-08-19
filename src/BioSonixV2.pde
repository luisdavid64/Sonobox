import java.util.logging.Logger;
import java.util.concurrent.atomic.AtomicBoolean;
import ddf.minim.analysis.FFT;
// for msg reception
import java.util.Arrays;
import java.nio.file.*;
import oscP5.*;
import netP5.*;
OscP5 oscP5;
int port1DModel = 12001;

// rendering
import peasy.*;
PeasyCam cam;
//int baseFrameRate = 60;
float currAudio = 0;
boolean showInstructions = true;

// minim library for recordings
import ddf.minim.*;
import ddf.minim.ugens.*;
int displayRate = 90;
Minim minim;
AudioInput in;
AudioRecorder recorder;
boolean recorded;
AudioOutput out;
FilePlayer player;
boolean isRecording = false;

/*  global physical model object : will contain the model and run calculations. */
import miPhysics.Renderer.*;
import miPhysics.Engine.*;
import miPhysics.Engine.Sound.*;

PhysicsContext phys;
PhyModel mdl;
ModelRenderer renderer;
miPhyAudioClient audioStreamHandler;
Phy3DConfig config; // loading json config parameters
phy3DModel model;

// physical parameters
float friction = 0.25;
float gain = 10;

// generic model creation
int numNodes; // number of nodes in the y direction -- whole model - 26 for 250 mm  model with 10 mm of distance
int dx = 3; // number of nodes in the x direction
int dz = 3; // number of nodes in the z direction
int distance = 2;
float massesRadius = 0.5;
float mass = 0.1;
float k = 1.5;


ArrayList<Driver3D> drivers;
ArrayList<Observer3D> listeners;
ArrayList<String> tissueNodeNames;

// START TRIGGER RENDERING (for sync)
String triggerText = "";
boolean showText = false;
int textTimer = 0;  // To control how long the text will be shown
int displayDuration = 1000;  // show text for this duration (ms)


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


void setup() {

  // Screen & Camera
  size(500, 800, P3D);
  cam = new PeasyCam(this, 100);
  cam.setMinimumDistance(50);
  cam.setMaximumDistance(1000);
  cam.setDistance(400);

  // Physics and config
  String absPath = getConfig();
  println("Loading config from: " + absPath);
  var cfgBuilder = Phy3DConfig.fromProcessingJsonFile(Paths.get(absPath));
  config = cfgBuilder.build();
  phys = new PhysicsContext(44100);
  phys.setGlobalFriction(friction);

  // Create model from config
  createModelFromConfig();
  
  // OSC
  oscP5 = new OscP5(this, port1DModel);

  // Renderer
  renderer = new ModelRenderer(this);
  renderer.displayMasses(true);
  renderer.setColor(massType.MASS3D, 140, 140, 240);
  renderer.setColor(interType.SPRINGDAMPER3D, 135, 70, 70, 255);
  renderer.setStrainGradient(interType.SPRINGDAMPER3D, true, 0.1);
  renderer.setStrainColor(interType.SPRINGDAMPER3D, 105, 100, 200, 255);
  renderer.displayIntersectionVolumes(true);
  renderer.displayForceVectors(true);

  // Audio
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
  // Rendering rate & text setup
  frameRate(displayRate);
  textFont(createFont("Helvetica", 120));

}

void draw() {
  directionalLight(251, 102, 126, 0, -1, 0);
  ambientLight(102, 102, 102);
  background(0);
  stroke(255);
  renderer.renderScene(phys);

  // START TRIGGER DISPLAY:: Show the trigger text for a set duration
  if (showText && millis() - textTimer < displayDuration) {
    fill(255);
    textSize(20);
    text(triggerText, 40, 20);  // Adjusted position and size
  } else {
    showText = false;  // Hide the text after the duration
  }
}

void keyPressed() {
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
}
