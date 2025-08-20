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
import miPhysics.Engine.InteractionConstants.*;

PhysicsContext phys;
PhyModel mdl;
ModelRenderer renderer;
miPhyAudioClient audioStreamHandler;
Phy3DConfig config; // loading json config parameters
phy3DModel model;
ModelController controller;

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
  setAudioClient();
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
