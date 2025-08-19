package miPhysics.Engine;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import java.util.ArrayList;
import com.google.gson.*;

public class Phy3DConfig {
  // ---- Engine core knobs ----
  public String name;
  public int dimX, dimY, dimZ;
  public int neighborSpan;
  public float dist;
  public float massRadius;
  public EnumSet<Bound> bounds;

  // IO (engine-ready lists)
  public List<String> driverNodes;
  public List<String> listenerNodes;

  // ---- Processing schema extras ----
  public String dataDir;
  public boolean useNormData;
  public String modelDim;
  public int numLayers;
  public int[] numNodesPerLayer;
  public double[] K;
  public double[] M;
  public int acousticScalingFactor;
  public List<String> contributingPixels;
  public int[] stiffnessArray;

  private Phy3DConfig(Builder b) {
    this.name = b.name;
    this.dimX = b.dimX; this.dimY = b.dimY; this.dimZ = b.dimZ;
    this.neighborSpan = b.neighborSpan;
    this.dist = b.dist;
    this.massRadius = b.massRadius;
    this.bounds = b.bounds.clone();
    this.driverNodes = List.copyOf(b.driverNodes);
    this.listenerNodes = List.copyOf(b.listenerNodes);

    this.dataDir = b.dataDir;
    this.useNormData = b.useNormData;
    this.modelDim = b.modelDim;
    this.numLayers = b.numLayers;
    this.numNodesPerLayer = b.numNodesPerLayer != null ? b.numNodesPerLayer.clone() : new int[0];
    this.K = b.K != null ? b.K.clone() : new double[0];
    this.M = b.M != null ? b.M.clone() : new double[0];
    this.acousticScalingFactor = b.acousticScalingFactor;
    this.contributingPixels = List.copyOf(b.contributingPixels);
    this.stiffnessArray = b.stiffnessArray != null ? b.stiffnessArray.clone() : new int[0];
  }

  public Builder toBuilder() {
    return new Builder()
        .name(name)
        .dims(dimX, dimY, dimZ)
        .neighborSpan(neighborSpan)
        .dist(dist)
        .massRadius(massRadius)
        .bounds(bounds)
        .drivers(driverNodes)
        .listeners(listenerNodes)
        .dataDir(dataDir)
        .useNormData(useNormData)
        .modelDim(modelDim)
        .numLayers(numLayers)
        .numNodesPerLayer(numNodesPerLayer)
        .K(K)
        .M(M)
        .acousticScalingFactor(acousticScalingFactor)
        .contributingPixels(contributingPixels)
        .stiffnessArray(stiffnessArray);
  }

  public static final class Builder {
    private String name = "variant";
    private int dimX = 3, dimY = 3, dimZ = 3;
    private int neighborSpan = 1;
    private float dist = 1.0f;
    private float massRadius = 3.0f;
    private EnumSet<Bound> bounds = EnumSet.noneOf(Bound.class);
    private List<String> driverNodes = new ArrayList<>();
    private List<String> listenerNodes = new ArrayList<>();

    private String dataDir = "";
    private boolean useNormData = false;
    private String modelDim = "1D";
    private int numLayers = 1;
    private int[] numNodesPerLayer = new int[0];
    private double[] K = new double[0];
    private double[] M = new double[0];
    private int acousticScalingFactor = 1;
    private List<String> contributingPixels = new ArrayList<>();
    private int[] stiffnessArray = new int[0];

    public Builder name(String n){ this.name=n; return this; }
    public Builder dims(int x,int y,int z){ this.dimX=x; this.dimY=y; this.dimZ=z; return this; }
    public Builder neighborSpan(int s){ this.neighborSpan=s; return this; }
    public Builder dist(float d){ this.dist=d; return this; }
    public Builder massRadius(float r){ this.massRadius=r; return this; }
    public Builder bounds(EnumSet<Bound> b){ this.bounds=b.clone(); return this; }
    public Builder addBound(Bound b){ this.bounds.add(b); return this; }
    public Builder drivers(Collection<String> a){ this.driverNodes = new ArrayList<>(a); return this; }
    public Builder listeners(Collection<String> a){ this.listenerNodes = new ArrayList<>(a); return this; }

    public Builder dataDir(String d){ this.dataDir=d; return this; }
    public Builder useNormData(boolean u){ this.useNormData=u; return this; }
    public Builder modelDim(String m){ this.modelDim=m; return this; }
    public Builder numLayers(int n){ this.numLayers=n; return this; }
    public Builder numNodesPerLayer(int[] a){this.numNodesPerLayer = a != null ? a.clone() : new int[0]; return this;}
    public Builder K(double[] arr){ this.K = arr!=null?arr.clone():new double[0]; return this; }
    public Builder M(double[] arr){ this.M = arr!=null?arr.clone():new double[0]; return this; }
    public Builder acousticScalingFactor(int v){ this.acousticScalingFactor=v; return this; }
    public Builder contributingPixels(Collection<String> a){ this.contributingPixels = new ArrayList<>(a); return this; }
    public Builder stiffnessArray(int[] a){ this.stiffnessArray = a!=null?a.clone():new int[0]; return this; }

    public Phy3DConfig build(){ return new Phy3DConfig(this); }
  }

  // ---------- JSON IMPORT ----------
  public static Builder fromProcessingJson(JsonObject root){
    Builder b = new Builder();
    if (root.has("data_dir")) b.dataDir(root.get("data_dir").getAsString());
    if (root.has("use_norm_data")) b.useNormData(root.get("use_norm_data").getAsBoolean());
    String model = root.has("model") ? root.get("model").getAsString() : "1D";
    b.modelDim(model);

    // geometry
    JsonObject g = root.getAsJsonObject("geometry");
    int dx = g.has("dx") ? g.get("dx").getAsInt() : 1;
    int dy = g.has("dy") ? g.get("dy").getAsInt() : 1;
    int dz = g.has("dz") ? g.get("dz").getAsInt() : 1;
    float distance = g.has("distance") ? (float)g.get("distance").getAsDouble() : 1.0f;
    float radius   = g.has("massesRadius") ? (float)g.get("massesRadius").getAsDouble() : 3.0f;
    b.dims(dx,dy,dz).dist(distance).massRadius(radius);
    if (g.has("numLayers")) b.numLayers(g.get("numLayers").getAsInt());
    if (g.has("numNodesPerLayer")) {
      b.numNodesPerLayer(arrayDInt(g.getAsJsonArray("numNodesPerLayer")));
    }

    // parameters
    if (root.has("parameters")){
      JsonObject par = root.getAsJsonObject("parameters");
      if (par.has("K")) b.K(arrayD(par.getAsJsonArray("K")));
      if (par.has("M")) b.M(arrayD(par.getAsJsonArray("M")));
    }

    // sonification_set_up
    if (root.has("sonification_set_up")){
      JsonObject su = root.getAsJsonObject("sonification_set_up");
      if (su.has("acoustic_scaling_factor")) b.acousticScalingFactor(su.get("acoustic_scaling_factor").getAsInt());
      if (su.has("contributing_pixels")) b.contributingPixels(arrayS(su.getAsJsonArray("contributing_pixels")));
      if (su.has("stiffnessArray")) b.stiffnessArray(arrayI(su.getAsJsonArray("stiffnessArray")));
      // drivers/listeners (top-level)
      if (su.has("drivers")) b.drivers(arrayS(su.getAsJsonArray("drivers")));
      if (su.has("listeners")) b.listeners(arrayS(su.getAsJsonArray("listeners")));
    }


    return b;
  }

  public static Builder fromProcessingJsonFile(Path p) {
    try (Reader r = Files.newBufferedReader(p)) {
      JsonObject jo = JsonParser.parseReader(r).getAsJsonObject();
      return fromProcessingJson(jo);
    }
    catch (Exception ex) {
      throw new RuntimeException("Invalid JSON syntax in " + p + ": " + ex.getMessage(), ex);
    }
  }

  // ---------- JSON EXPORT ----------
  public JsonObject toProcessingJson() {
    JsonObject root = new JsonObject();
    root.addProperty("data_dir", dataDir);
    root.addProperty("use_norm_data", useNormData);
    root.addProperty("model", modelDim);

    JsonObject g = new JsonObject();
    g.addProperty("numLayers", numLayers);
    JsonArray nnpl = new JsonArray();
    for (int v : numNodesPerLayer) nnpl.add(v);
    g.add("numNodesPerLayer", nnpl);
    g.addProperty("distance", dist);
    g.addProperty("massesRadius", massRadius);
    g.addProperty("dy", dimY);
    g.addProperty("dx", dimX);
    g.addProperty("dz", dimZ);
    root.add("geometry", g);

    JsonObject par = new JsonObject();
    par.add("K", toJsonArray(K));
    par.add("M", toJsonArray(M));
    root.add("parameters", par);

    JsonObject su = new JsonObject();
    su.addProperty("acoustic_scaling_factor", acousticScalingFactor);
    su.add("contributing_pixels", toJsonArray(contributingPixels));
    su.add("stiffnessArray", toJsonArray(stiffnessArray));
    root.add("sonification_set_up", su);

    // drivers/listeners (top-level)
    root.add("drivers", toJsonArray(driverNodes));
    root.add("listeners", toJsonArray(listenerNodes));

    return root;
  }

  public void writeProcessingJson(Path path) {
    Gson gson = new GsonBuilder().setPrettyPrinting().create();
    try (Writer w = Files.newBufferedWriter(path)) { gson.toJson(toProcessingJson(), w); }
    catch (Exception ex) {
      throw new RuntimeException("Failed to write Processing JSON to " + path + ": " + ex.getMessage(), ex);
    }
  }

  // ---------- Helpers ----------
  private static double[] arrayD(JsonArray arr){
    double[] o = new double[arr.size()];
    for (int i=0;i<arr.size();i++) o[i] = arr.get(i).getAsDouble();
    return o;
  }
  private static int[] arrayI(JsonArray arr){
    int[] o = new int[arr.size()];
    for (int i=0;i<arr.size();i++) o[i] = arr.get(i).getAsInt();
    return o;
  }
  private static int[] arrayDInt(JsonArray arr){
    int[] o = new int[arr.size()];
    for (int i=0;i<arr.size();i++) o[i] = arr.get(i).getAsInt();
    return o;
  }
  private static List<String> arrayS(JsonArray arr){
    List<String> o = new ArrayList<>();
    for (JsonElement e : arr) o.add(e.getAsString());
    return o;
  }
  private static JsonArray toJsonArray(double[] a){ JsonArray ja = new JsonArray(); for (double v:a) ja.add(v); return ja; }
  private static JsonArray toJsonArray(int[] a){ JsonArray ja = new JsonArray(); for (int v:a) ja.add(v); return ja; }
  private static JsonArray toJsonArray(List<String> a){ JsonArray ja = new JsonArray(); for (String v:a) ja.add(v); return ja; }
}
