//package spacegraph.space3d.test;
//
//import com.google.common.primitives.Doubles;
//import com.jogamp.opengl.GL2;
//import jcog.Util;
//import jcog.cluster.NeuralGasMap;
//import jcog.exe.Loop;
//import jcog.signal.FloatRange;
//import jcog.math.StreamingNormalizer;
//import jcog.net.attn.MeshMap;
//import jcog.pri.Prioritized;
//import org.jctools.queues.MpscArrayQueue;
//import spacegraph.SpaceGraph;
//import spacegraph.space3d.SimpleSpatial;
//import spacegraph.space3d.phys.Collidable;
//import spacegraph.video.Draw;
//import spacegraph.video.Draw3D;
//
//import java.util.List;
//import java.util.Queue;
//
///**
// * a tool for realtime multidimensional data visualization
// * input modes:
// * UDPeer net
// * -each data model has its own network
// * <p>
// * stdio CSV (TODO)
// * <p>
// * visualization models:
// * 3D gasnet pointcloud
// */
//public class Dimensionaleyez extends SimpleSpatial {
//
//    static final int THREE_D = 3;
//    static final int IN = 5;
//    final NeuralGasMap n = new NeuralGasMap(IN, 64, THREE_D);
//    final FloatRange scale = new FloatRange(10, 0.5f, 300f);
//
//    final Queue<double[]> queue = new MpscArrayQueue<>(1024);
//
//    private final StreamingNormalizer s;
//
//    public Dimensionaleyez(String id) {
//        super(id);
//
//        s = new StreamingNormalizer(IN);
//
//        MeshMap<Integer, List<Float>> m = MeshMap.get(id, this::accept);
//
//        new Loop() {
//
//            @Override
//            public boolean next() {
//                double[] x;
//                while ((x = queue.poll()) != null) {
//                    float[] df = Util.toFloat(x);
//                    x = Util.toDouble(s.normalize(df, df, Prioritized.EPSILON));
//
//
//
//
//                    n.put(x);
//                }
//
//
//                n.update();
//                return true;
//
//            }
//        }.fps(20f);
//    }
//
//
//
//    private void accept(Integer k, List v) {
//        double[] da = Doubles.toArray(v);
//        queue.add(da);
//    }
//
//    @Override
//    public void renderRelative(GL2 gl, Collidable body, float dtS) {
//
//        float s = scale.floatValue();
//        n.forEachCentroid(n -> {
//            float[] d = n.center();
//            if (d == null)
//                return;
//
//            float d0 = d[0];
//            if (d0 != d0) {
//
//                return;
//            }
//            float y = d[1] * s;
//            float z = d.length > 2 ? d[2] * s : 0;
//
//            float last = ((float) n.getEntry(n.getDimension() - 1) + 1f) / 2f;
//
//            gl.glPushMatrix();
//            float x = d0 * s;
//            gl.glTranslatef(x, y, z);
//            float p = 0.3f + (float) (0.7f / (1f + n.localError()));
//
//            float sat = 0.5f;
//            float hue = (n.id %10)/10f;
//            float bri = 0.5f;
//
//            Draw.hsb(gl, hue, sat, p, bri);
//            float size = last;
//            Draw3D.glut.glutSolidCube(1f * size);
//            gl.glPopMatrix();
//        });
//    }
//
//    public static void main(String[] s) {
//        SpaceGraph.window(new Dimensionaleyez("d1"), 1000, 800);
//
//
//    }
//}