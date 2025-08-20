package miPhysics.Engine;

public class InteractionConstants {

    public enum InteractionType {
        FIRST,
        SECOND,
        CHECKERED,
        DILATED2,
        CLIQUE,
        // CUSTOM,
    }

    public static final int[][] OFFSETS_FIRST = { { +1, 0, 0 }, { 0, +1, 0 }, { 0, 0, +1 } };
    public static final int[][] OFFSETS_SECOND = { { 0, +1, +1 }, { +1, 0, +1 }, { +1, +1, 0 }, { +1, +1, +1 } };
    public static final int[][] OFFSETS_CHECKERBOARD_EVEN = { { +1, +1, 0 }, { +1, 0, +1 }, { 0, +1, +1 } };
    public static final int[][] OFFSETS_DILATED2 = { { +2, 0, 0 }, { 0, +2, 0 }, { 0, 0, +2 } }; // “skip” springs

}