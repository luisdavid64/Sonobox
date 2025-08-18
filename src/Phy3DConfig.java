package miPhysics.Engine;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

import javax.management.RuntimeErrorException;

import com.google.gson.*;

public final class Phy3DConfig {
  // ---- Engine core knobs ----
  public final String name;
  public final int dimX, dimY, dimZ;   // geometry.dx, geometry.dy, geometry.dz
  public final int neighborSpan;       // connectivity span (kept for future use)
  public final double dist;            // geometry.distance
  public final double massRadius;      // geometry.massesRadius
  public final EnumSet<Bound> bounds;  // optional engine bounds

  // IO (engine-ready lists)
  public final List<String> driverNodes;
  public final List<String> listenerNodes;

  // ---- Processing schema extras (to round‑trip without loss) ----
  public final String dataDir;                 // data_dir
  public final boolean useNormData;            // use_norm_data
  public final String modelDim;                // "1D" | "2D" | "3D"
  public final int numLayers;                  // geometry.numLayers
  public final List<Integer> numNodesPerLayer; // geometry.numNodesPerLayer
  public final double[] K;                     // parameters.K (per-layer stiffness)
  public final double[] M;                     // parameters.M (per-layer mass)
  public final int acousticScalingFactor;      // sonification_set_up.acoustic_scaling_factor
  public final List<String> contributingPixels;// sonification_set_up.contributing_pixels
  public final int[] stiffnessArray;           // sonification_set_up.stiffnessArray

  // For export: original per-dimension drivers/listeners (optional)
  public final List<String> drivers1D, listeners1D;
  public final List<String> drivers2D, listeners2D;
  public final List<String> drivers3D, listeners3D;

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
    this.numNodesPerLayer = List.copyOf(b.numNodesPerLayer);
    this.K = b.K != null ? b.K.clone() : new double[0];
    this.M = b.M != null ? b.M.clone() : new double[0];
    this.acousticScalingFactor = b.acousticScalingFactor;
    this.contributingPixels = List.copyOf(b.contributingPixels);
    this.stiffnessArray = b.stiffnessArray != null ? b.stiffnessArray.clone() : new int[0];

    this.drivers1D = List.copyOf(b.drivers1D);
    this.listeners1D = List.copyOf(b.listeners1D);
    this.drivers2D = List.copyOf(b.drivers2D);
    this.listeners2D = List.copyOf(b.listeners2D);
    this.drivers3D = List.copyOf(b.drivers3D);
    this.listeners3D = List.copyOf(b.listeners3D);
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
        .stiffnessArray(stiffnessArray)
        .drivers1D(drivers1D).listeners1D(listeners1D)
        .drivers2D(drivers2D).listeners2D(listeners2D)
        .drivers3D(drivers3D).listeners3D(listeners3D);
  }

  public static final class Builder {
    private String name = "variant";
    private int dimX = 3, dimY = 3, dimZ = 3;
    private int neighborSpan = 1;
    private double dist = 1.0;
    private double massRadius = 3.0;
    private EnumSet<Bound> bounds = EnumSet.noneOf(Bound.class);
    private List<String> driverNodes = new ArrayList<>();
    private List<String> listenerNodes = new ArrayList<>();

    private String dataDir = "";
    private boolean useNormData = false;
    private String modelDim = "1D";
    private int numLayers = 1;
    private List<Integer> numNodesPerLayer = new ArrayList<>();
    private double[] K = new double[0];
    private double[] M = new double[0];
    private int acousticScalingFactor = 1;
    private List<String> contributingPixels = new ArrayList<>();
    private int[] stiffnessArray = new int[0];
    private List<String> drivers1D = new ArrayList<>(), listeners1D = new ArrayList<>();
    private List<String> drivers2D = new ArrayList<>(), listeners2D = new ArrayList<>();
    private List<String> drivers3D = new ArrayList<>(), listeners3D = new ArrayList<>();

    public Builder name(String n){ this.name=n; return this; }
    public Builder dims(int x,int y,int z){ this.dimX=x; this.dimY=y; this.dimZ=z; return this; }
    public Builder neighborSpan(int s){ this.neighborSpan=s; return this; }
    public Builder dist(double d){ this.dist=d; return this; }
    public Builder massRadius(double r){ this.massRadius=r; return this; }
    public Builder bounds(EnumSet<Bound> b){ this.bounds=b.clone(); return this; }
    public Builder addBound(Bound b){ this.bounds.add(b); return this; }
    public Builder drivers(Collection<String> a){ this.driverNodes = new ArrayList<>(a); return this; }
    public Builder listeners(Collection<String> a){ this.listenerNodes = new ArrayList<>(a); return this; }

    public Builder dataDir(String d){ this.dataDir=d; return this; }
    public Builder useNormData(boolean u){ this.useNormData=u; return this; }
    public Builder modelDim(String m){ this.modelDim=m; return this; }
    public Builder numLayers(int n){ this.numLayers=n; return this; }
    public Builder numNodesPerLayer(Collection<Integer> a){ this.numNodesPerLayer = new ArrayList<>(a); return this; }
    public Builder K(double[] arr){ this.K = arr!=null?arr.clone():new double[0]; return this; }
    public Builder M(double[] arr){ this.M = arr!=null?arr.clone():new double[0]; return this; }
    public Builder acousticScalingFactor(int v){ this.acousticScalingFactor=v; return this; }
    public Builder contributingPixels(Collection<String> a){ this.contributingPixels = new ArrayList<>(a); return this; }
    public Builder stiffnessArray(int[] a){ this.stiffnessArray = a!=null?a.clone():new int[0]; return this; }
    public Builder drivers1D(Collection<String> a){ this.drivers1D = new ArrayList<>(a); return this; }
    public Builder listeners1D(Collection<String> a){ this.listeners1D = new ArrayList<>(a); return this; }
    public Builder drivers2D(Collection<String> a){ this.drivers2D = new ArrayList<>(a); return this; }
    public Builder listeners2D(Collection<String> a){ this.listeners2D = new ArrayList<>(a); return this; }
    public Builder drivers3D(Collection<String> a){ this.drivers3D = new ArrayList<>(a); return this; }
    public Builder listeners3D(Collection<String> a){ this.listeners3D = new ArrayList<>(a); return this; }

    public Phy3DConfig build(){ return new Phy3DConfig(this); }
  }

  // ---------- JSON IMPORT (Processing schema) ----------
  public static Builder fromProcessingJson(JsonObject root){
    Builder b = new Builder();
    // top-level
    if (root.has("data_dir")) b.dataDir(root.get("data_dir").getAsString());
    if (root.has("use_norm_data")) b.useNormData(root.get("use_norm_data").getAsBoolean());
    String model = root.has("model") ? root.get("model").getAsString() : "1D";
    b.modelDim(model);

    // geometry
    JsonObject g = root.getAsJsonObject("geometry");
    int dx = g.has("dx") ? g.get("dx").getAsInt() : 1;
    int dy = g.has("dy") ? g.get("dy").getAsInt() : 1; // for 1D, dy == numNodes
    int dz = g.has("dz") ? g.get("dz").getAsInt() : 1;
    double distance = g.has("distance") ? g.get("distance").getAsDouble() : 1.0;
    double radius   = g.has("massesRadius") ? g.get("massesRadius").getAsDouble() : 3.0;
    b.dims(dx,dy,dz).dist(distance).massRadius(radius);
    if (g.has("numLayers")) b.numLayers(g.get("numLayers").getAsInt());
    if (g.has("numNodesPerLayer")) {
      List<Integer> nl = new ArrayList<>();
      for (JsonElement e : g.getAsJsonArray("numNodesPerLayer")) nl.add(e.getAsInt());
      b.numNodesPerLayer(nl);
    }

    // parameters (arrays per-layer)
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
      if (su.has("1D")) {
        JsonObject s1 = su.getAsJsonObject("1D");
        if (s1.has("drivers")) b.drivers1D(arrayS(s1.getAsJsonArray("drivers")));
        if (s1.has("listeners")) b.listeners1D(arrayS(s1.getAsJsonArray("listeners")));
      }
      if (su.has("2D")) {
        JsonObject s2 = su.getAsJsonObject("2D");
        if (s2.has("drivers")) b.drivers2D(arrayS(s2.getAsJsonArray("drivers")));
        if (s2.has("listeners")) b.listeners2D(arrayS(s2.getAsJsonArray("listeners")));
      }
      if (su.has("3D")) {
        JsonObject s3 = su.getAsJsonObject("3D");
        if (s3.has("drivers")) b.drivers3D(arrayS(s3.getAsJsonArray("drivers")));
        if (s3.has("listeners")) b.listeners3D(arrayS(s3.getAsJsonArray("listeners")));
      }
    }

    // choose engine IO list based on modelDim
    if ("1D".equals(model)) { b.drivers(toEngine1D(b.drivers1D)).listeners(toEngine1D(b.listeners1D)); }
    else if ("2D".equals(model)) { b.drivers(b.drivers2D).listeners(b.listeners2D); }
    else { b.drivers(b.drivers3D).listeners(b.listeners3D); }

    return b;
  }

  public static Builder fromProcessingJsonFile(Path p) {
    try (Reader r = Files.newBufferedReader(p)) {
      JsonObject jo = JsonParser.parseReader(r).getAsJsonObject();
      return fromProcessingJson(jo);
    }   
    catch (Exception ex) {
      // Common causes: comments (// ...), trailing commas, unquoted keys
      throw new RuntimeException("Invalid JSON syntax in " + p + ": " + ex.getMessage(), ex);
    }
  }

  // ---------- JSON EXPORT (Processing schema) ----------
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

    // Export all 3 blocks so the file remains portable
    JsonObject s1 = new JsonObject();
    s1.add("drivers", toJsonArray(drivers1D));
    s1.add("listeners", toJsonArray(listeners1D));
    su.add("1D", s1);
    JsonObject s2 = new JsonObject();
    s2.add("drivers", toJsonArray(drivers2D));
    s2.add("listeners", toJsonArray(listeners2D));
    su.add("2D", s2);
    JsonObject s3 = new JsonObject();
    s3.add("drivers", toJsonArray(drivers3D));
    s3.add("listeners", toJsonArray(listeners3D));
    su.add("3D", s3);

    root.add("sonification_set_up", su);
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
  private static List<String> arrayS(JsonArray arr){
    List<String> o = new ArrayList<>();
    for (JsonElement e : arr) o.add(e.getAsString());
    return o;
  }
  private static JsonArray toJsonArray(double[] a){ JsonArray ja = new JsonArray(); for (double v:a) ja.add(v); return ja; }
  private static JsonArray toJsonArray(int[] a){ JsonArray ja = new JsonArray(); for (int v:a) ja.add(v); return ja; }
  private static JsonArray toJsonArray(List<String> a){ JsonArray ja = new JsonArray(); for (String v:a) ja.add(v); return ja; }

  // Map 1D labels like "m_3" -> engine 3D ids "m_1_3_1"
  private static List<String> toEngine1D(List<String> ids){
    return ids.stream().map(s -> {
      // parse integer after 'm_'
      int idx = Integer.parseInt(s.replace("m_", "").trim());
      return "m_1_"+idx+"_1"; // X=1, Z=1; adjust if needed
    }).toList();
  }
}
