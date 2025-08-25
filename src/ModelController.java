package miPhysics.Engine;

import miPhysics.Engine.Phy3DConfig;

import java.util.ArrayList;
import java.util.List;

import miPhysics.Engine.InteractionConstants.*;


public class ModelController {
  
    public interface ResetHook {
        void run();
    } // e.g., calls resetModel()
    
    public interface MessageSink {
        void show(String msg);
    } // e.g., calls produceRenderedMessage(msg)

    final Phy3DConfig config;
    final ResetHook resetHook; // required to re-init after changes
    final MessageSink messages; // optional status messages (can be null)
    double[] baselineW;
    int baseDimX, baseDimY, baseDimZ;
    float baselineDist = 1;

    public ModelController(Phy3DConfig config, ResetHook resetHook, MessageSink messages) {
        this.config = config;
        this.resetHook = resetHook;
        this.messages = messages;
    }

    public void initBaseline(int[] layers) {
        int tot = 0;
        for (int v : layers)
            tot += v;
        baselineW = new double[layers.length];
        for (int i = 0; i < layers.length; i++) {
            baselineW[i] = (tot == 0) ? 1.0 / layers.length : (double) layers[i] / tot;
        }
        baselineDist = config.dist;
        baseDimX = config.dimX;
        baseDimY = config.dimY;
        baseDimZ = config.dimZ;
    }

    // --- Public high-level API: performs action and handles post-change duties ---
    public void incrementY() {
        int i = pickForIncrement(config.numNodesPerLayer, baselineW);
        config.numNodesPerLayer[i] += 1;
        config.dimY += 1;
        onModelChanged();
    }

    public void decrementY() {
        int i = pickForDecrement(config.numNodesPerLayer, baselineW);
        if (i != -1) {
            config.numNodesPerLayer[i] -= 1;
            config.dimY -= 1;
            onModelChanged();
        }
    }

    public void adjustX(int delta) {
        config.dimX += delta;
        System.out.println("Changing X dimension to: " + config.dimX);
        onModelChanged();
    }

    public void adjustZ(int delta) {
        config.dimZ += delta;
        System.out.println("Changing Z dimension to: " + config.dimZ);
        onModelChanged();
    }

    public void adjustRadius(int delta) {
        config.massRadius += delta;
        System.out.println("Mass radius now: " + config.massRadius);
        onModelChanged();
    }

    public void adjustMs(double[] incrementsM) {
        double[] results = new double[config.M.length];
        for (int i = 0; i < results.length; i++) {
            results[i] = config.M[i] + incrementsM[i];
        }
        config.M = results;
        System.out.println("Changing M");
        onModelChanged();
    }

    public void adjustKs(double[] incrementsK) {
        double[] results = new double[config.K.length];
        for (int i = 0; i < results.length; i++) {
            results[i] = config.K[i] + incrementsK[i];
        }
        config.K = results;
        System.out.println("Changing K");
        onModelChanged();
    }

    public void adjustCs(double[] incrementsC) {
        double[] results = new double[config.C.length];
        for (int i = 0; i < results.length; i++) {
            results[i] = config.C[i] + incrementsC[i];
        }
        config.C = results;
        System.out.println("Changing C");
        onModelChanged();
    }

    public void cycleInteractionType() {
        InteractionType[] vals = InteractionType.values();
        config.interactionType = vals[(config.interactionType.ordinal() + 1) % vals.length];
        if (messages != null)
            messages.show(config.interactionType.name());
        onModelChanged();
    }

    public void shiftDriversListeners(char axis) {
        // axis: 'X' or 'Z'
        String ax = (axis == 'X' || axis == 'x') ? "X" : "Z";
        var newDrivers = shiftNodeName(config.driverNodes, ax);
        var newListeners = shiftNodeName(config.listenerNodes, ax);
        config.driverNodes = newDrivers;
        config.listenerNodes = newListeners;
        if (messages != null)
            messages.show("Shifted driver and listener nodes on " + ax + " axis.");
        onModelChanged();
    }

    public void adjustResolution(int delta) {
        int oldY = config.dimY;
        float newDist = config.dist + delta;
        if (newDist <= 0f) newDist = 1f; // guard; choose your own minimum

        config.dist = newDist;

        // Inverse scaling: keep overall size consistent with the initial setup
        float scale = newDist / baselineDist;

        config.dimY = 0;
        for(int i = 0; i < config.numNodesPerLayer.length; i++) {
            config.numNodesPerLayer[i] = (int) Math.max(1, Math.round(baseDimY * scale * baselineW[i]));
            config.dimY += config.numNodesPerLayer[i];
        }
        //sum numNodesPerLayer into dimY
        System.err.println("New dimensions: X=" + config.dimX + ", Y=" + config.dimY + ", Z=" + config.dimZ);

        onModelChanged();
    }

    // --- Internal helpers ---
    private void onModelChanged() {
        enforceDimMinimums();
        if (resetHook != null)
            resetHook.run();
    }

    private void enforceDimMinimums() {
        if (config.dimX < 1 || config.dimY < 1 || config.dimZ < 1) {
            System.out.println("Invalid dimensions detected. Resetting to minimum valid values.");
            config.dimX = Math.max(config.dimX, 1);
            config.dimY = Math.max(config.dimY, 1);
            config.dimZ = Math.max(config.dimZ, 1);
            System.out.println("New dimensions: X=" + config.dimX + ", Y=" + config.dimY + ", Z=" + config.dimZ);
        }
    }

    private int pickForIncrement(int[] layers, double[] w) {
        int total = 0;
        for (int v : layers)
            total += v;
        int best = 0;
        double bestErr = Double.POSITIVE_INFINITY; // smaller (more negative) is better
        for (int i = 0; i < layers.length; i++) {
            double target = w[i] * (total + 1);
            double err = (layers[i] + 1) - target;
            if (err < bestErr) {
                bestErr = err;
                best = i;
            }
        }
        return best;
    }

    private int pickForDecrement(int[] layers, double[] w) {
        int total = 0;
        for (int v : layers)
            total += v;
        int best = -1;
        double bestErr = -Double.POSITIVE_INFINITY; // larger is better
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] <= 1)
                continue; // enforce minimum of 1
            double target = w[i] * (total - 1);
            double err = (layers[i] - 1) - target;
            if (err > bestErr) {
                bestErr = err;
                best = i;
            }
        }
        return best; // -1 means can't decrement without breaking min=1
    }

    private List<String> shiftNodeName(List<String> nodeNames, String dim) {
        List<String> shiftedNames = new ArrayList<String>(nodeNames.size());
        for (String nodeName : nodeNames) {
            String[] parts = nodeName.split("_");
            if (parts.length != 4) {
                shiftedNames.add(nodeName); // ignore unexpected names
                continue;
            }
            int i = Integer.parseInt(parts[1]);
            int j = Integer.parseInt(parts[2]);
            int k = Integer.parseInt(parts[3]);
            if ("X".equals(dim)) {
                int newI = (i + 1) % config.dimX;
                shiftedNames.add("m_" + newI + "_" + j + "_" + k);
            } else if ("Z".equals(dim)) {
                int newK = (k + 1) % config.dimZ;
                shiftedNames.add("m_" + i + "_" + j + "_" + newK);
            } else {
                shiftedNames.add(nodeName); // no shift if unknown dim
            }
        }
        return shiftedNames;
    }
}
