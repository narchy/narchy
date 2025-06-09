package jcog.nn.spiking0.ca;

import java.util.Arrays;

public class CodiCA extends CA {

    public static final int FIRING_THRESHOLD =
        31;
        //64;

    private int CAChanged;
    private boolean SignalingInited;

    static float dendriteThreshold =
        2;
        //10;

    static float inputBias =
        1;
        //0.1f;

    public static class CodiCell {
        public int Type;
        int Chromo;
        int Gate;
        public float Activ;
        final float[] IOBuf = new float[6];
        //float[] weights = null;

        public void clearIO() {
            Arrays.fill(IOBuf, 0);
        }
    }

    public final CodiCell[][][] cell;

    int neuronGrid =
        2;
        //3;
        //4;

    private static final int BLANK = 0;
    private static final int NEURONSEED = 1;
    public static final int NEURON = 1;
    public static final int AXON = 2;
    public static final int DEND = 4;
    private static final int AXON_SIG = 2;
    private static final int DEND_SIG = 4;

    //    private static final int SIG_COL = 5;
//    private static final int ERROR_COL = 13;
    private final int[][][] ColorSpace;
//    private int ActualColor;

    public CodiCA(int w, int h , int d) {
        super(w, h, d);


        ColorSpace = new int[sizeX][sizeY][sizeZ];
        cell = new CodiCell[sizeX][sizeY][sizeZ];
        for (int ix = 0; ix < sizeX; ix++)
            for (int iy = 0; iy < sizeY; iy++)
                for (int iz = 0; iz < sizeZ; iz++)
                    cell[ix][iy][iz] = new CodiCell();
    }


    @Override
    protected void init() {
        step = 0;
        CAChanged = 1;
        SignalingInited = false;
        uninitialized = true;
        for (int ix = 0; ix < sizeX; ix++)
            for (int iy = 0; iy < sizeY; iy++)
                for (int iz = 0; iz < sizeZ; iz++) {
                    ColorSpace[ix][iy][iz] = 0;
                    CodiCell cell = this.cell[ix][iy][iz];
                    cell.Type = 0;
                    cell.Activ = 0;
                    cell.clearIO();

                    cell.Chromo = random.nextInt(256);

                    if ((ix + 1) % 2 * (iy % 2) == 1)
                        cell.Chromo = cell.Chromo & ~3 | 12;
                    if (ix % 2 * ((iy + 1) % 2) == 1)
                        cell.Chromo = cell.Chromo & ~12 | 3;

                    /* seeds neurons only every NxN cells TODO verify this is what it actually means */
                    if (neuronGrid > 1) {
                        if (ix % neuronGrid + iy % neuronGrid != 0)
                            cell.Chromo &= ~192;
                    }

                    if (cell.Chromo >>> 6 == NEURONSEED)
                        if (random.nextInt(sizeX) < ix / 2)
                            cell.Chromo &= ~192;


                    if (cell.Chromo >>> 6 == NEURONSEED)
                        cell.Chromo = cell.Chromo & 192 | (cell.Chromo & 63) % 4;
                }
    }

    protected void Kicking() {


        CodiCell[][][] ca = cell;
        for (int iz = 0; iz < sizeZ; iz++)
            for (int iy = 0; iy < sizeY; iy++)
                for (int ix = 0; ix < sizeX; ix++) {
                    var caio = ca[ix][iy][iz].IOBuf;
                    caio[4] = iz == sizeZ - 1 ? 0 : ca[ix][iy][iz + 1].IOBuf[4];
                    caio[2] = iy == sizeY - 1 ? 0 : ca[ix][iy + 1][iz].IOBuf[2];
                    caio[0] = ix == sizeX - 1 ? 0 : ca[ix + 1][iy][iz].IOBuf[0];
                }


        for (int iz = sizeZ - 1; iz >= 0; iz--)
            for (int iy = sizeY - 1; iy >= 0; iy--)
                for (int ix = sizeX - 1; ix >= 0; ix--) {
                    var caio = ca[ix][iy][iz].IOBuf;
                    caio[5] = iz != 0 ? ca[ix][iy][iz - 1].IOBuf[5] : 0;
                    caio[3] = iy != 0 ? ca[ix][iy - 1][iz].IOBuf[3] : 0;
                    caio[1] = ix != 0 ? ca[ix - 1][iy][iz].IOBuf[1] : 0;
                }
    }

    private void reset() {
        SignalingInited = true;
        CodiCell[][][] ca = cell;
        for (int iz = 0; iz < sizeZ; iz++)
            for (int iy = 0; iy < sizeY; iy++)
                for (int ix = 0; ix < sizeX; ix++) {
                    CodiCell c = ca[ix][iy][iz];
                    c.clearIO();
                    c.Activ = 0;
                    if (c.Type == NEURON)
                        c.Activ = random.nextFloat(FIRING_THRESHOLD);
                }
    }

    @Override
    public void next() {
        if (step == 0)
            init();
        step++;
        if (CAChanged == 1)
            grow();
        else {
            if (!SignalingInited)
                reset();
            forward();
        }
    }

    protected int grow() {
        CAChanged = 0;
        for (int iz = 0; iz < sizeZ; iz++)
            for (int iy = 0; iy < sizeY; iy++)
                for (int ix = 0; ix < sizeX; ix++) {


                    CodiCell ca = cell[ix][iy][iz];
                    var caio = ca.IOBuf;
                    switch (ca.Type) {
                        case BLANK:

                            if (ca.Chromo >>> 6 == NEURONSEED) {
                                ca.Type = NEURON;
                                CAChanged = 1;

                                ca.Gate = (ca.Chromo & 63) % 6;
                                for (int i = 0; i < 6; i++)
                                    caio[i] = DEND_SIG;
                                caio[ca.Gate] = AXON_SIG;
                                caio[ca.Gate % 2 * -2 + 1 + ca.Gate] = AXON_SIG;
                                break;
                            }

                            int InputSum = 0;
                            for (int i = 0; i < 6; i++)
                                InputSum += caio[i];
                            if (InputSum == 0) break;

                            int result = 0;
                            for (int i = 0; i < 6; i++)
                                result += (int)caio[i] & AXON_SIG;

                            InputSum = result;
                            if (InputSum == AXON_SIG) {
                                ca.Type = AXON;
                                CAChanged = 1;
                                for (int i = 0; i < 6; i++)
                                    if (caio[i] == AXON)
                                        ca.Gate = i;
                                for (int i = 0; i < 6; i++)
                                    caio[i] = (ca.Chromo >>> i & 1) != 0 ? AXON_SIG : 0;
                                break;
                            }
                            if (InputSum > AXON_SIG) {
                                for (int i = 0; i < 6; i++)
                                    caio[i] = 0;
                                break;
                            }

                            int sum = 0;

                            for (int v = 0; v < 6; v++)
                                sum += (int)caio[v] & DEND_SIG;

                            InputSum = sum;
                            if (InputSum == DEND_SIG) {
                                CAChanged = 1;
                                ca.Type = DEND;
                                for (int i = 0; i < 6; i++)
                                    if (caio[i] != 0)
                                        ca.Gate = i % 2 * -2 + 1 + i;
                                for (int i = 0; i < 6; i++)
                                    caio[i] = (ca.Chromo >>> i & 1) != 0 ? DEND_SIG : 0;
                                break;
                            }

                            for (int i = 0; i < 6; i++)
                                caio[i] = 0;
                            break;
                        case NEURON:
                            for (int i = 0; i < 6; i++)
                                caio[i] = DEND_SIG;
                            caio[ca.Gate] = AXON_SIG;
                            caio[ca.Gate % 2 * -2 + 1 + ca.Gate] = AXON_SIG;
                            break;
                        case AXON:
                            for (int i = 0; i < 6; i++)
                                caio[i] = (ca.Chromo >>> i & 1) != 0 ? AXON_SIG : 0;
                            break;
                        case DEND:
                            for (int i = 0; i < 6; i++)
                                caio[i] = (ca.Chromo >>> i & 1) != 0 ? DEND_SIG : 0;
                            break;
                    }
                }
        Kicking();
        return CAChanged;
    }

    protected int forward() {
        for (int iz = 0; iz < sizeZ; iz++)
            for (int iy = 0; iy < sizeY; iy++)
                for (int ix = 0; ix < sizeX; ix++) {
                    forward(cell[ix][iy][iz]);
                }
        Kicking();
        return 0;
    }

    private static void forward(CodiCell ca) {
        float[] caio = ca.IOBuf;
        int gate = ca.Gate;
        switch (ca.Type) {
            case BLANK:
                break;
            case NEURON: {
                int otherGate = gate % 2 * -2 + 1 + gate;
                double InputSum = //1 +
                        inputBias +
                        caio[0] +
                        caio[1] +
                        caio[2] +
                        caio[3] +
                        caio[4] +
                        caio[5] -
                        caio[gate] -
                        caio[otherGate];

                ca.clearIO();

                ca.Activ += InputSum;

                if (ca.Activ > FIRING_THRESHOLD) {
                    caio[otherGate] = caio[gate] = 1;
                    ca.Activ = 0;
                }
                break;
            }
            case AXON:
                float g = caio[gate];
                for (int i = 0; i < 6; i++)
                    caio[i] = g;
                ca.Activ = g != 0 ? 1 : 0;
                break;
            case DEND: {
                double InputSum = 0;
                for (int i = 0; i < 6; i++)
                    InputSum += caio[i];

                InputSum = Math.min(InputSum, dendriteThreshold);

                ca.clearIO();

                caio[gate] = (float) InputSum;
                ca.Activ = InputSum != 0 ? 1 : 0;
                break;
            }
        }
    }

//    @Override
//    protected void DrawCA(Graphics g) {
//        if (bFirstStart) {
//            g.setColor(Color.black);
//            g.fillRect(Offset, Offset,
//                    sizeX * CellWidthPx, sizeY * CellHeightPx);
//            bFirstStart = false;
//        }
//
//        int PosX = Offset - CellWidthPx;
//        int iz = 0;
//        for (int ix = 0; ix < sizeX; ix++) {
//            PosX += CellWidthPx;
//            int PosY = Offset - CellHeightPx;
//            for (int iy = 0; iy < sizeY; iy++) {
//                PosY += CellHeightPx;
//                CodiCell ca = cell[ix][iy][iz];
//                if (ca.Type != 0) {
//                    if (ca.Activ != 0) {
//                        ActualColor = ca.Type != NEURON ? 5 : 1;
//                    } else
//                        switch (ca.Type) {
//                            case NEURON:
//                                ActualColor = 1;
//                                break;
//                            case AXON:
//                                ActualColor = 2;
//                                break;
//                            case DEND:
//                                ActualColor = 4;
//                                break;
//                            default:
//                                ActualColor = 13;
////                                System.out.println("__" + ca.Type + "__");
//                                break;
//                        }
//                    if (ColorSpace[ix][iy][iz] != ActualColor) {
//                        ColorSpace[ix][iy][iz] = ActualColor;
//                        switch (ActualColor) {
//                            case NEURON:
//                                g.setColor(Color.white);
//                                break;
//                            case AXON:
//                                g.setColor(Color.red);
//                                break;
//                            case DEND:
//                                g.setColor(Color.green);
//                                break;
//                            case SIG_COL:
//                                g.setColor(Color.yellow);
//                                break;
//                            default:
//                                g.setColor(Color.blue);
//                        }
//                        g.fillRect(PosX, PosY, CellWidthPx, CellHeightPx);
//                    }
//                }
//            }
//        }
//    }


}