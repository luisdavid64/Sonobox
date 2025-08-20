package miPhysics.Engine;

import java.util.*;

public class Phy3DTopologyBuilder {
    public static boolean isBoundary(phy3DModel model, int i, int j, int k) {
        return (i == 0 || j == 0 || k == 0 ||
            i == model.getDimX() - 1 || j == model.getDimY() - 1 || k == model.getDimZ() - 1);
    }


    public static void generateOffsetGrid(phy3DModel model, int[][] offsets) {
        String masName1;
        for (int i = 0; i < model.getDimX(); i++) {
            for (int j = 0; j < model.getDimY(); j++) {
                for (int k = 0; k < model.getDimZ(); k++) {
                    masName1 = model.getMassLabel() + "_" + i + "_" + j + "_" + k;
                    model.applyOffsetsAt(i, j, k, masName1, offsets);
                }
            }
        }
    }

    public static void generateCheckerboardWithFrame(phy3DModel model, int[][] offsetsFirst, int[][] offsetsCheckerboard) {
        for (int i = 0; i < model.getDimX(); i++) {
            for (int j = 0; j < model.getDimY(); j++) {
                for (int k = 0; k < model.getDimZ(); k++) {
                    String masName1 = model.getMassLabel() + "_" + i + "_" + j + "_" + k;
                    // 1) boundary wireframe (axis-aligned frame)
                    if (isBoundary(model, i, j, k)) {
                        model.applyOffsetsAt(i, j, k, masName1, offsetsFirst);
                    }
                    // 2) interior + boundary diagonals on checkerboard-even cells
                    if (((i + j + k) & 1) == 0) {
                        model.applyOffsetsAt(i, j, k, masName1, offsetsCheckerboard);
                    }
                }
            }
        }
    }

    public static void generateDilated2WithFrame(phy3DModel model, int[][] offsetsDilated2, int[][] offsetsFirst) {
        for (int i = 0; i < model.getDimX(); i++)
            for (int j = 0; j < model.getDimY(); j++)
                for (int k = 0; k < model.getDimZ(); k++) {
                    String name = model.getMassLabel() + "_" + i + "_" + j + "_" + k;
                    // 1) sparse dilated interior links
                    model.applyOffsetsAt(i, j, k, name, offsetsDilated2);
                    // 2) boundary wireframe (connectivity around the hull)
                    if (isBoundary(model, i, j, k)) {
                        model.applyOffsetsAt(i, j, k, name, offsetsFirst);
                    }
                }
    }

    public static void generateCliques(phy3DModel model) {
        int dimX = model.getDimX(), dimY = model.getDimY(), dimZ = model.getDimZ();
        String massLabel = model.getMassLabel();

        int cliqueX = (dimX + 1) / 2;
        int cliqueY = (dimY + 1) / 2;
        int cliqueZ = (dimZ + 1) / 2;

        // 1. Generate intra-clique full connections (all pairs within each 2x2x2 block)
        for (int cx = 0; cx < cliqueX; cx++) {
            for (int cy = 0; cy < cliqueY; cy++) {
                for (int cz = 0; cz < cliqueZ; cz++) {
                    ArrayList<int[]> nodes = new ArrayList<>();
                    for (int dx = 0; dx < 2; dx++) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dz = 0; dz < ((dimZ > 1) ? 2 : 1); dz++) {
                                int i = cx * 2 + dx;
                                int j = cy * 2 + dy;
                                int k = cz * 2 + dz;
                                if (i < dimX && j < dimY && k < dimZ) {
                                    nodes.add(new int[] { i, j, k });
                                }
                            }
                        }
                    }
                    for (int a = 0; a < nodes.size(); a++) {
                        int[] n1 = nodes.get(a);
                        String masName1 = massLabel + "_" + n1[0] + "_" + n1[1] + "_" + n1[2];
                        for (int b = a + 1; b < nodes.size(); b++) {
                            int[] n2 = nodes.get(b);
                            int dx = n2[0] - n1[0];
                            int dy = n2[1] - n1[1];
                            int dz = n2[2] - n1[2];
                            model.addInteractions(n2[0], n2[1], n2[2], n1[0], n1[1], n1[2], masName1, dx, dy, dz,
                                    Math.max(Math.abs(dx), Math.max(Math.abs(dy), Math.abs(dz))));
                        }
                    }
                }
            }
        }

        // 2. Connect each clique to its neighbors by a single edge
        for (int cx = 0; cx < cliqueX; cx++) {
            for (int cy = 0; cy < cliqueY; cy++) {
                for (int cz = 0; cz < cliqueZ; cz++) {
                    // X neighbor clique
                    if (cx + 1 < cliqueX) {
                        int i1 = Math.min((cx + 1) * 2 - 1, dimX - 1);
                        int i2 = Math.min((cx + 1) * 2, dimX - 1);
                        int j = Math.min(cy * 2, dimY - 1);
                        int k = Math.min(cz * 2, dimZ - 1);
                        String masName1 = massLabel + "_" + i1 + "_" + j + "_" + k;
                        model.addInteractions(i2, j, k, i1, j, k, masName1, i2 - i1, 0, 0, Math.abs(i2 - i1));
                    }
                    // Y neighbor clique
                    if (cy + 1 < cliqueY) {
                        int j1 = Math.min((cy + 1) * 2 - 1, dimY - 1);
                        int j2 = Math.min((cy + 1) * 2, dimY - 1);
                        int i = Math.min(cx * 2, dimX - 1);
                        int k = Math.min(cz * 2, dimZ - 1);
                        String masName1 = massLabel + "_" + i + "_" + j1 + "_" + k;
                        model.addInteractions(i, j2, k, i, j1, k, masName1, 0, j2 - j1, 0, Math.abs(j2 - j1));
                    }
                    // Z neighbor clique (if 3D)
                    if (dimZ > 1 && cz + 1 < cliqueZ) {
                        int k1 = Math.min((cz + 1) * 2 - 1, dimZ - 1);
                        int k2 = Math.min((cz + 1) * 2, dimZ - 1);
                        int i = Math.min(cx * 2, dimX - 1);
                        int j = Math.min(cy * 2, dimY - 1);
                        String masName1 = massLabel + "_" + i + "_" + j + "_" + k1;
                        model.addInteractions(i, j, k2, i, j, k1, masName1, 0, 0, k2 - k1, Math.abs(k2 - k1));
                    }
                }
            }
        }
    }
}
