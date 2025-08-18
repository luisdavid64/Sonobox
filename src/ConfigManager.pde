import processing.data.JSONObject;

public class ConfigManager {
  JSONObject config;

  public ConfigManager(String filepath) {
    config = loadJSONObject(filepath);
  }

  public String getString(String key) {
    return config.getString(key);
  }

  public boolean getBoolean(String key) {
    return config.getBoolean(key);
  }

  public int getInt(String key) {
    return config.getInt(key);
  }

  public float getFloat(String key) {
    return config.getFloat(key);
  }

  // --- Nested Access Methods ---

  public JSONObject getGeometry() {
    return config.getJSONObject("geometry");
  }

  public int getLayerNodeCount(String layerKey) {
    return getGeometry().getInt(layerKey); // e.g., "firstLayerNumNodes"
  }

  public JSONObject getGeometryForModel(String model) {
    return getGeometry().getJSONObject(model); // "1D", "2D", "3D"
  }

  public JSONObject getSonificationSetup() {
    return config.getJSONObject("sonification_set_up");
  }

  public JSONObject getSonificationForModel(String model) {
    return getSonificationSetup().getJSONObject(model);
  }

  public JSONObject getParametersForExperiment() {
    String exp = config.getString("surgery");  // e.g. "US-first-trial"
    return config.getJSONObject("parameters").getJSONObject(exp);
  }

  public float getParameter(String paramKey) {
    return getParametersForExperiment().getFloat(paramKey); // e.g. "firstK", "thirdM"
  }

  public JSONArray getParameterList(String paramKey) {
    return getParametersForExperiment().getJSONArray(paramKey);
  }

  public String[] getContributingPixels() {
    JSONArray arr = getSonificationSetup().getJSONArray("contributing_pixels");
    String[] pixels = new String[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
      pixels[i] = arr.getString(i);
    }
    return pixels;
  }
  
  public String[] getStiffnessArray() {
    JSONArray arr = getSonificationSetup().getJSONArray("stiffnessArray");
    String[] stiffnessArray = new String[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
      stiffnessArray[i] = arr.getString(i);
    }
    return stiffnessArray;
  }
  
  public String[] getDriverNames(String modelType) {
    JSONArray arr = getSonificationForModel(modelType).getJSONArray("drivers");
    String[] drivers = new String[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
      drivers[i] = arr.getString(i);
    }
    return drivers;
  }
  
  public String[] getListenerNames(String modelType) {
    JSONArray arr = getSonificationForModel(modelType).getJSONArray("listeners");
    String[] listeners = new String[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
      listeners[i] = arr.getString(i);
    }
    return listeners;
  }

  public int getDx(String modelType) {
    return getGeometryForModel(modelType).getInt("dx");
  }

  public int getDz(String modelType) {
    return getGeometryForModel(modelType).getInt("dz");
  }

  public float getMassesRadius(String modelType) {
    return getGeometryForModel(modelType).getFloat("massesRadius");
  }
    // Returns the model type string: "1D", "2D", or "3D"
  public String getModelType() {
    return config.getString("model");
  }
  
  // Returns the surgery string: e.g., "US-first-trial"
  public String getSurgery() {
    return config.getString("surgery");
  }
  
  // Returns the full geometry object for a given model
  public JSONObject getGeometry(String modelType) {
    return getGeometry().getJSONObject(modelType);
  }
  
  // Returns int from geometry of a model
  public int getIntGeometry(String modelType, String key) {
    return getGeometry(modelType).getInt(key); // depending on the modelType
  }
  
  public int getIntGlobalGeometry(String key) {
  return config.getJSONObject("geometry").getInt(key);
  }
  
  // Returns float from geometry of a model
  public float getFloatGeometry(String modelType, String key) {
    return getGeometry(modelType).getFloat(key);
  }
  
  // Returns parameters for a specific model & surgery
  public JSONObject getParameters(String surgery) {
    return config.getJSONObject("parameters").getJSONObject(surgery);
  }
  
  public float getParameter(String surgery, String key) {
    return config.getJSONObject("parameters")
                 .getJSONObject(surgery)
                 .getFloat(key);
  }
  
  // Drivers (for a model)
  public String[] getDriverList(String modelType) {
    JSONArray arr = getSonificationForModel(modelType).getJSONArray("drivers");
    String[] drivers = new String[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
      drivers[i] = arr.getString(i);
    }
    return drivers;
  }
  
  // Listeners (for a model)
  public String[] getListenerList(String modelType) {
    JSONArray arr = getSonificationForModel(modelType).getJSONArray("listeners");
    String[] listeners = new String[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
      listeners[i] = arr.getString(i);
    }
    return listeners;
  }

  int[] getNumNodesPerLayerArray(String modelType) {
      JSONArray jsonArray = getGeometry().getJSONArray("numNodesPerLayer");
      int[] result = new int[jsonArray.size()];
      for (int i = 0; i < jsonArray.size(); i++) {
          result[i] = (int) jsonArray.getInt(i); // Use getDouble for proper conversion
      }
      return result;
  }

  double[] getKArray(String modelType, String surgery) {
      JSONArray jsonArray = getParameterList("K");
      double[] result = new double[jsonArray.size()];
      for (int i = 0; i < jsonArray.size(); i++) {
          result[i] = jsonArray.getDouble(i);  // Use getDouble for proper conversion
      }
      return result;
  }

  double[] getMArray(String modelType, String surgery) {
      JSONArray jsonArray = getParameterList("M");
      double[] result = new double[jsonArray.size()];
      for (int i = 0; i < jsonArray.size(); i++) {
          result[i] = jsonArray.getDouble(i);  // Use getDouble for proper conversion
      }
      return result;
  }
}
