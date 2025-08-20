package miPhysics.Engine;

import java.util.*;
import miPhysics.Engine.*;
import miPhysics.Engine.InteractionConstants.*;

public class phy3DModel extends PhyModel {
  private boolean m_generated = false;

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

  private InteractionType m_iOrder = InteractionType.FIRST;

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
        Phy3DTopologyBuilder.generateOffsetGrid(this,InteractionConstants.OFFSETS_FIRST);
        break;
      case SECOND:
        Phy3DTopologyBuilder.generateOffsetGrid(this, InteractionConstants.OFFSETS_FIRST);
        Phy3DTopologyBuilder.generateOffsetGrid(this, InteractionConstants.OFFSETS_SECOND);
        System.out.println("phy3DModel: generating SECOND order interactions");
        break;
      case CHECKERED:
        Phy3DTopologyBuilder.generateCheckerboardWithFrame(this, InteractionConstants.OFFSETS_FIRST, InteractionConstants.OFFSETS_CHECKERBOARD_EVEN);
        System.out.println("phy3DModel: generating CHECKERED order interactions");
        break;
      case DILATED2:
        Phy3DTopologyBuilder.generateDilated2WithFrame(this, InteractionConstants.OFFSETS_DILATED2, InteractionConstants.OFFSETS_FIRST);
        System.out.println("phy3DModel: generating DILATED2 order interactions"); // <>// //<>// //<>//
        break;
      case CLIQUE:
        Phy3DTopologyBuilder.generateCliques(this);
        System.out.println("phy3DModel: generating CLIQUE order interactions");
        break;
      default:
        System.out.println("phy3DModel: generating CUSTOM order interactions"); // <>// //<>// //<>//
    } 
    m_generated = true;
  }

  public void addInteractions(int idx, int idy, int idz, int i, int j, int k, String masName1, int a, int b, int c,
      int mult) {
    Vect3D X0, U1;
    String masName2;
    if ((idx < m_dimX) && (idy < m_dimY) && (idz < m_dimZ)) {
      if ((idx >= 0) && (idy >= 0) && (idz >= 0)) { 
        if (!((idx == i) && (idy == j) && (idz == k))) {
          U1 = new Vect3D(a, b, c);
          double d = U1.norm() * m_l0;
          masName2 = m_mLabel + "_" + (idx + "_" + idy + "_" + idz);
          String ln = m_iLabel + "_" + (idx + "_" + idy + "_" + idz) + "_" + (i + "_" + j + "_" + k);
          if ((j == m_dimY - 2) || (j == 0)) {
            addInteraction(ln, new Spring3D(mult * d, stiffness), masName1, masName2);
          } else {
            addInteraction(ln, new Spring3D(mult * d, stiffness), masName1, masName2); //<>//
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

  public void setInteractionType(InteractionType it) {
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

  public void applyOffsetsAt(int i, int j, int k, String masName1, int[][] offsets) {
    for (int[] d : offsets) {
      int max = Math.max(d[0], Math.max(d[1], d[2]));
      addIfValid(i + d[0], j + d[1], k + d[2], i, j, k, masName1, max);
    }
  }

  public void clearInOutLabels() {
    m_inOutLabels.clear();
  }

  public int getDimX() { return m_dimX; }
  public int getDimY() { return m_dimY; }
  public int getDimZ() { return m_dimZ; }
  public String getMassLabel() { return m_mLabel; }
}
