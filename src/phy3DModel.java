package miPhysics.Engine;

import java.util.*;
import miPhysics.Engine.*;

public class phy3DModel extends PhyModel {
  private boolean m_generated = false;

  // private int numNodes = 3;
  // private int numInteractions = numNodes + 1;

  // dimension of the model
  private int m_dimX = 1;
  private int m_dimY = 1;
  private int m_dimZ = 1;
  private int m_neighbors = 1;
  private String m_modelType = "3D";

  // private int excitedNode = 0;
  private String m_mLabel = "m";
  private String m_iLabel = "i";

  private double massSize = 20.; // radius
  private double mass = 1.0;
  private double stiffness = 0.001;
  // private double dist = 62;
  private double m_dist = 1;
  private double m_l0 = 1;
  private MassIDAdapter m_mIDAdapter = new MassIDAdapter();

  public enum interactionType {
    FIRST,
    SECOND,
    CHECKERED,
    DILATED2,
    // CUSTOM,
    CLIQUE,
  }

  private interactionType m_iOrder = interactionType.FIRST;
  private static final int[][] OFFSETS_FIRST = { { +1, 0, 0 }, { 0, +1, 0 }, { 0, 0, +1 } };
  private static final int[][] OFFSETS_SECOND = { { 0, +1, +1 }, { +1, 0, +1 }, { +1, +1, 0 }, { +1, +1, +1 } };
  private static final int[][] OFFSETS_CHECKERBOARD_EVEN = { { +1, +1, 0 }, { +1, 0, +1 }, { 0, +1, +1 } };
  private static final int[][] OFFSETS_DILATED2 = { { +2, 0, 0 }, { 0, +2, 0 }, { 0, 0, +2 } }; // “skip” springs

  private EnumSet<Bound> bCond;

  public phy3DModel(String name, Medium m) {
    super(name, m);
    bCond = EnumSet.noneOf(Bound.class);
    System.out.println(bCond);
  }

  public void init() {
    super.init();
    if (!m_generated) {
      System.out.println("The TopoCreator model has not yet been generated !");
      System.exit(-1);
    }
  }

  public void setDim(int dx, int dy, int dz, int span) {
    m_dimX = dx;
    m_dimY = dy;
    m_dimZ = dz;
    m_neighbors = span;
  }

  public void setGeometry(double d) { // vr: constraint for interaction elements: the distance between nodes and the
                                      // initial length is the same
    m_dist = d;
    m_l0 = d;
  }

  public void setMassRadius(double s) {
    massSize = s;
  }

  public void setParams(double M, double K) {
    mass = M;
    stiffness = K;
  }

  public ArrayList<Driver3D> addDrivers(List<String> InNodes) {
    InNodes = m_mIDAdapter.adapt(InNodes, m_modelType);
    ArrayList<Driver3D> drivers = new ArrayList<>(); // Initialize the drivers list
    int count = 1;
    for (String drvNode : InNodes) {
      String driverName = "driver_" + count;
      Driver3D driver = this.addInOut(driverName, new Driver3D(), drvNode);
      drivers.add(driver);
      System.out.println(this.getName() + ": creating driver with name: " + driverName + " at node: " + drvNode
          + " stored as:: " + driver);
      count += 1;
    }
    return drivers; // Return the drivers list
  }

  public ArrayList<Observer3D> addListeners(List<String> OutNodes) {
    OutNodes = m_mIDAdapter.adapt(OutNodes, m_modelType);
    ArrayList<Observer3D> listeners = new ArrayList<>(); // Initialize the listeners list
    int count = 1;
    for (String obsNode : OutNodes) {
      String obsName = "listener_" + count;
      Observer3D listener = this.addInOut(obsName, new Observer3D(filterType.HIGH_PASS), obsNode);
      listeners.add(listener);
      System.out.println(this.getName() + ": creating listener with name: " + obsName + " at node: " + obsNode
          + " stored as:: " + listener);
      count += 1;
    }
    return listeners; // Return the listeners list
  }

  public void generate() {
    // System.out.println(this.getName() + ": creating mass elements with naming
    // pattern: " + m_mLabel + "_[X]_[Y]_[Z]");
    // System.out.println(this.getName() + ": creating interaction elements with
    // naming pattern: " + m_iLabel + "_[X1]_[Y1]_[Z1]_[X2]_[Y2]_[Z2]");

    String masName;
    Vect3D X0, U1;
    System.out.println(this.getName() + ": creating mass elements with naming pattern: "
        + m_mLabel + "_[X]_[Y]_[Z]");

    for (int i = 0; i < m_dimX; i++) {
      for (int j = 0; j < m_dimY; j++) {
        for (int k = 0; k < m_dimZ; k++) {

          X0 = new Vect3D(i * m_dist, j * m_dist, k * m_dist);
          masName = m_mLabel + "_" + (i + "_" + j + "_" + k);

          if (((i == 0) || (i == (m_dimX - 1))) && ((j == 0) || (j == m_dimY - 1)) && ((k == 0) || (k == m_dimZ - 1))) {
            this.addMass(masName, new Ground3D(1., new Vect3D(X0)));
          } else {
            {
              this.addMass(masName, new Mass3D(mass, massSize, X0));
            }
          }
        }
      }
    }

    System.out.println(this.getName() + ": creating interaction elements with naming pattern: "
        + m_iLabel + "_[X1]_[Y1]_[Z1]_[X2]_[Y2]_[Z2] where 1 is the mass downstream and 2 in the one upstream");

    // add the springs to the model: length, stiffness, connected mats
    String masName1, masName2;
    int idx = 0, idy = 0, idz = 0;
    switch (m_iOrder) {
      case FIRST:
        generateOffsetGrid(OFFSETS_FIRST);
        break;
      case SECOND:
        generateOffsetGrid(OFFSETS_SECOND);
        System.out.println("phy3DModel: generating SECOND order interactions");
        break;
      case CHECKERED:
        generateCheckerboardWithFrame();
        System.out.println("phy3DModel: generating CHECKERED order interactions");
        break;
      case DILATED2:
        generateDilated2WithFrame();
        System.out.println("phy3DModel: generating DILATED2 order interactions"); // <>// //<>// //<>//
        break;
      case CLIQUE:
        generateCliques();
        System.out.println("phy3DModel: generating CLIQUE order interactions");
        break;
      default:
        System.out.println("phy3DModel: generating CUSTOM order interactions"); // <>// //<>// //<>//
    } // <>//
    for (int i = 0; i < m_dimX; i++) { // <>//
      for (int j = 0; j < m_dimY; j++) {
        for (int k = 0; k < m_dimZ; k++) {

          masName1 = m_mLabel + "_" + (i + "_" + j + "_" + k);

          if (m_iOrder == interactionType.FIRST || m_iOrder == interactionType.SECOND) {
            applyOffsetsAt(i, j, k, masName1, OFFSETS_FIRST);
          }
          if (m_iOrder == interactionType.SECOND) {
            applyOffsetsAt(i, j, k, masName1, OFFSETS_SECOND); // <>// //<>// //<>//
          }
        }
      }
    }
    // <>// //<>// //<>//
    m_generated = true;
  }

  // <>//
  public void addInteractions(int idx, int idy, int idz, int i, int j, int k, String masName1, int a, int b, int c,
      int mult) {
    Vect3D X0, U1; // <>//
    String masName2; // <>//
    if ((idx < m_dimX) && (idy < m_dimY) && (idz < m_dimZ)) { // <>// //<>// //<>//
      if ((idx >= 0) && (idy >= 0) && (idz >= 0)) { // <>//
        if (!((idx == i) && (idy == j) && (idz == k))) { // <>//
          U1 = new Vect3D(a, b, c);
          // if (j > m_dimY-3)
          // m_l0 = m_l0 * 0.5;
          double d = U1.norm() * m_l0;
          masName2 = m_mLabel + "_" + (idx + "_" + idy + "_" + idz);
          String ln = m_iLabel + "_" + (idx + "_" + idy + "_" + idz) + "_" + (i + "_" + j + "_" + k);
          if ((j == m_dimY - 2) || (j == 0)) {
            addInteraction(ln, new Spring3D(mult * d, stiffness), masName1, masName2);
          } else {
            addInteraction(ln, new Spring3D(mult * d, stiffness), masName1, masName2);
          }
        }
      }
    }
  }

  public void addBoundaryCondition(Bound b) {
    bCond.add(b);
  }

  private void applyBoundaryConditions() {

    if (bCond.contains(Bound.X_LEFT)) {
      for (int j = 0; j < m_dimY; j++) {
        for (int k = 0; k < m_dimZ; k++) {
          String name = m_mLabel + ("_0_" + j + "_" + k);
          System.out.println("changing to fix point mass:: " + name);
          this.changeToFixedPoint(name);
        }
      }
    }
    if (bCond.contains(Bound.X_RIGHT)) {
      for (int j = 0; j < m_dimY; j++) {
        for (int k = 0; k < m_dimZ; k++) {
          String name = m_mLabel + ("_" + (m_dimX - 1) + "_" + j + "_" + k);
          System.out.println("changing to fix point mass:: " + name);
          this.changeToFixedPoint(name);
        }
      }
    }

    if (bCond.contains(Bound.Y_LEFT)) {
      for (int i = 1; i < m_dimX - 1; i++) {
        for (int k = 1; k < m_dimZ - 1; k++) {
          String name = m_mLabel + ("_" + i + "_" + 0 + "_" + k);
          System.out.println("changing to fix point mass:: " + name);
          this.changeToFixedPoint(name);
        }
      }
    }
    if (bCond.contains(Bound.Y_RIGHT)) {
      for (int i = 1; i < m_dimX - 1; i++) {
        for (int k = 1; k < m_dimZ - 1; k++) {
          String name = m_mLabel + ("_" + i + "_" + (m_dimY - 1) + "_" + k);
          System.out.println("changing to fix point mass:: " + name);
          this.changeToFixedPoint(name);
        }
      }
    }

    if (bCond.contains(Bound.Z_LEFT)) {
      for (int i = 1; i < m_dimX - 1; i++) {
        for (int j = 0; j < m_dimY; j++) {
          String name = m_mLabel + ("_" + i + "_" + j + "_" + 0);
          System.out.println("changing to fix point mass:: " + name);
          this.changeToFixedPoint(name);
        }
      }
    }
    if (bCond.contains(Bound.Z_RIGHT)) {
      for (int i = 1; i < m_dimX - 1; i++) {
        for (int j = 0; j < m_dimY; j++) {
          String name = m_mLabel + ("_" + i + "_" + j + "_" + (m_dimZ - 1));
          System.out.println("changing to fix point mass:: " + name);
          this.changeToFixedPoint(name);
        }
      }
    }

    if (bCond.contains(Bound.FIXED_CORNERS)) {
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          for (int k = 0; k < 2; k++) {
            String name = m_mLabel + ("_" + (i * (m_dimX - 1)) + "_" + (j * (m_dimY - 1)) + "_" + (k * (m_dimZ - 1)));
            System.out.println("changing to fix point mass:: " + name);
            this.changeToFixedPoint(name);
          }
        }
      }
    }

    if (bCond.contains(Bound.FIXED_CENTRE)) {
      String name = m_mLabel
          + ("_" + (Math.floor(m_dimX / 2)) + "_" + (Math.floor(m_dimY / 2)) + "_" + (Math.floor(m_dimZ / 2)));
      System.out.println("changing to fix point mass:: " + name);
      this.changeToFixedPoint(name);
    }
  }

  public int setParam(param p, double val) {
    switch (p) {
      case MASS:
        this.mass = val;
        break;
      case RADIUS:
        this.massSize = val;
        break;
      case STIFFNESS:
        this.stiffness = val;
        break;
      default:
        System.out.println("Cannot apply param " + val + " for "
            + this + ": no " + p + " parameter");
        break;
    }
    ArrayList<Mass> masses = getMassList();
    ArrayList<Interaction> interactions = getInteractionList();
    for (Mass o : masses)
      o.setParam(p, val);
    for (Interaction i : interactions)
      i.setParam(p, val);
    return 0;
  }

  public double getParam(param p) {
    switch (p) {
      case MASS:
        return this.mass;
      case RADIUS:
        return this.massSize;
      case STIFFNESS:
        return this.stiffness;
      default:
        System.out.println("No " + p + " parameter found in " + this);
        return 0.;
    }
  }

  public void setModelType(String modelType) {
    this.m_modelType = modelType;
  }

  public void setInteractionType(interactionType it) {
    this.m_iOrder = it;
  }

  private int lin(int i, int j, int k) {
    return (i * m_dimY + j) * m_dimZ + k;
  }

  private void addIfValid(int i2, int j2, int k2, int i, int j, int k, String masName1, int mult) {
    if (i2 < 0 || j2 < 0 || k2 < 0 || i2 >= m_dimX || j2 >= m_dimY || k2 >= m_dimZ)
      return;
    // forward-only guard
    if (lin(i2, j2, k2) <= lin(i, j, k))
      return;

    int dx = Integer.compare(i2, i);
    int dy = Integer.compare(j2, j);
    int dz = Integer.compare(k2, k);

    // flags like you already use (1 if axis is involved)
    int fx = dx != 0 ? 1 : 0;
    int fy = dy != 0 ? 1 : 0;
    int fz = dz != 0 ? 1 : 0;

    addInteractions(i2, j2, k2, i, j, k, masName1, fx, fy, fz, mult);
  }

  private void applyOffsetsAt(int i, int j, int k, String masName1, int[][] offsets) {
    for (int[] d : offsets) {
      int max = Math.max(d[0], Math.max(d[1], d[2]));
      addIfValid(i + d[0], j + d[1], k + d[2], i, j, k, masName1, max);
    }
  }

  private void generateOffsetGrid(int[][] offsets) {
    String masName1, masName2;
    int idx = 0, idy = 0, idz = 0;
    for (int i = 0; i < m_dimX; i++) {
      for (int j = 0; j < m_dimY; j++) {
        for (int k = 0; k < m_dimZ; k++) {

          masName1 = m_mLabel + "_" + (i + "_" + j + "_" + k);
          applyOffsetsAt(i, j, k, masName1, offsets);
        }
      }
    }
  }

  private boolean isBoundary(int i, int j, int k) {
    return (i == 0 || j == 0 || k == 0 ||
        i == m_dimX - 1 || j == m_dimY - 1 || k == m_dimZ - 1);
  }

  private void generateCheckerboardWithFrame() {
    for (int i = 0; i < m_dimX; i++) {
      for (int j = 0; j < m_dimY; j++) {
        for (int k = 0; k < m_dimZ; k++) {
          String masName1 = m_mLabel + "_" + i + "_" + j + "_" + k;
          // 1) boundary wireframe (axis-aligned frame)
          if (isBoundary(i, j, k)) {
            applyOffsetsAt(i, j, k, masName1, OFFSETS_FIRST);
          }
          // 2) interior + boundary diagonals on checkerboard-even cells
          if (((i + j + k) & 1) == 0) {
            applyOffsetsAt(i, j, k, masName1, OFFSETS_CHECKERBOARD_EVEN);
          }
        }
      }
    }
  }

  private void generateDilated2WithFrame() {
    for (int i = 0; i < m_dimX; i++)
      for (int j = 0; j < m_dimY; j++)
        for (int k = 0; k < m_dimZ; k++) {
          String name = m_mLabel + "_" + i + "_" + j + "_" + k;

          // 1) sparse dilated interior links
          applyOffsetsAt(i, j, k, name, OFFSETS_DILATED2);

          // 2) boundary wireframe (connectivity around the hull)
          if (isBoundary(i, j, k)) {
            applyOffsetsAt(i, j, k, name, OFFSETS_FIRST);
          }
        }
  }

  private void generateCliques() {
    // This generates isolated 2x2x2 (or 2x2 if dimZ==1) cliques,
    // connecting each clique to its neighbors by a single edge.
    int cliqueX = (m_dimX + 1) / 2;
    int cliqueY = (m_dimY + 1) / 2;
    int cliqueZ = (m_dimZ + 1) / 2;

    // 1. Generate intra-clique full connections (all pairs within each 2x2x2 block)
    for (int cx = 0; cx < cliqueX; cx++) {
      for (int cy = 0; cy < cliqueY; cy++) {
        for (int cz = 0; cz < cliqueZ; cz++) {
          // Gather all valid nodes in this clique
          ArrayList<int[]> nodes = new ArrayList<>();
          for (int dx = 0; dx < 2; dx++) {
            for (int dy = 0; dy < 2; dy++) {
              for (int dz = 0; dz < ((m_dimZ > 1) ? 2 : 1); dz++) {
                int i = cx * 2 + dx;
                int j = cy * 2 + dy;
                int k = cz * 2 + dz;
                if (i < m_dimX && j < m_dimY && k < m_dimZ) {
                  nodes.add(new int[] { i, j, k });
                }
              }
            }
          }
          // Fully connect all pairs within the clique
          for (int a = 0; a < nodes.size(); a++) {
            int[] n1 = nodes.get(a);
            String masName1 = m_mLabel + "_" + n1[0] + "_" + n1[1] + "_" + n1[2];
            for (int b = a + 1; b < nodes.size(); b++) {
              int[] n2 = nodes.get(b);
              String masName2 = m_mLabel + "_" + n2[0] + "_" + n2[1] + "_" + n2[2];
              // Use the difference as the direction vector
              int dx = n2[0] - n1[0];
              int dy = n2[1] - n1[1];
              int dz = n2[2] - n1[2];
              addInteractions(n2[0], n2[1], n2[2], n1[0], n1[1], n1[2], masName1, dx, dy, dz,
                  Math.max(Math.abs(dx), Math.max(Math.abs(dy), Math.abs(dz))));
            }
          }
        }
      }
    }

    // 2. Connect each clique to its neighbors by a single edge (from the
    // boundary-most node in this clique to the boundary-most node in the neighbor
    // clique)
    for (int cx = 0; cx < cliqueX; cx++) {
      for (int cy = 0; cy < cliqueY; cy++) {
        for (int cz = 0; cz < cliqueZ; cz++) {
          // For each direction, connect the boundary-most node of this clique to the
          // boundary-most node of the neighbor clique
          // X neighbor clique
          if (cx + 1 < cliqueX) {
            // Use only one node per clique face for X direction
            int i1 = Math.min((cx + 1) * 2 - 1, m_dimX - 1); // rightmost in this clique
            int i2 = Math.min((cx + 1) * 2, m_dimX - 1); // leftmost in neighbor clique
            int j = Math.min(cy * 2, m_dimY - 1);
            int k = Math.min(cz * 2, m_dimZ - 1);
            String masName1 = m_mLabel + "_" + i1 + "_" + j + "_" + k;
            String masName2 = m_mLabel + "_" + i2 + "_" + j + "_" + k;
            addInteractions(i2, j, k, i1, j, k, masName1, i2 - i1, 0, 0, Math.abs(i2 - i1));
          }
          // Y neighbor clique
          if (cy + 1 < cliqueY) {
            // Use only one node per clique face for Y direction
            int j1 = Math.min((cy + 1) * 2 - 1, m_dimY - 1); // topmost in this clique
            int j2 = Math.min((cy + 1) * 2, m_dimY - 1); // bottommost in neighbor clique
            int i = Math.min(cx * 2, m_dimX - 1);
            int k = Math.min(cz * 2, m_dimZ - 1);
            String masName1 = m_mLabel + "_" + i + "_" + j1 + "_" + k;
            String masName2 = m_mLabel + "_" + i + "_" + j2 + "_" + k;
            addInteractions(i, j2, k, i, j1, k, masName1, 0, j2 - j1, 0, Math.abs(j2 - j1));
          }
          // Z neighbor clique (if 3D)
          if (m_dimZ > 1 && cz + 1 < cliqueZ) {
            // Use only one node per clique face for Z direction
            int k1 = Math.min((cz + 1) * 2 - 1, m_dimZ - 1); // frontmost in this clique
            int k2 = Math.min((cz + 1) * 2, m_dimZ - 1); // backmost in neighbor clique
            int i = Math.min(cx * 2, m_dimX - 1);
            int j = Math.min(cy * 2, m_dimY - 1);
            String masName1 = m_mLabel + "_" + i + "_" + j + "_" + k1;
            String masName2 = m_mLabel + "_" + i + "_" + j + "_" + k2;
            addInteractions(i, j, k2, i, j, k1, masName1, 0, 0, k2 - k1, Math.abs(k2 - k1));
          }
        }
      }
    }
  }


  public void clearInOutLabels() {
    m_inOutLabels.clear();
  }

  private List<String> currentDriverNodes() {
    ArrayList<String> nodes = new ArrayList<>();
    for (Driver3D d : this.getDrivers()) {
      nodes.add(d.getMat().getName());
    }
    return nodes;
  }
}
