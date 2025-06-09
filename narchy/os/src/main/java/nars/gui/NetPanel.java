//package nars.gui;
//
//import com.google.common.base.Joiner;
//import jcog.net.UDPeer;
//import nars.InterNAR;
//import nars.NAR;
//import spacegraph.space2d.container.Bordering;
//import spacegraph.space2d.widget.text.AbstractLabel;
//import spacegraph.space2d.widget.text.BitmapLabel;
//
//public class NetPanel extends Bordering {
//
//    private final AbstractLabel status;
//
//    public NetPanel(NAR n, InterNAR net) {
//
//        north(new BitmapLabel(net.peer.name() + ' ' + net.peer.addr));
//        this.status = new BitmapLabel();
//
//        set(new DurSurface<>(status, n) {
//            @Override protected void update() {
//                status.text(
//                    //net.peer.summary().replace(", (", "\n") + "\n" +
//                    Joiner.on('\n').join(net.peer.them.stream().map(UDPeer.UDProfile::toString).iterator())
//                );
//            }
//        });
//
//    }
//
////    public static void main(String[] args) {
////        int N = 4;
////        SpaceGraph.window(new Gridding(IntStream.range(0, N).mapToObj(z -> {
////            float cpuFPS = 10f, netFPS = cpuFPS/2;
////            NAR n = NARS.realtime(cpuFPS).withNAL(1, 8).get();
////            InterNAR i = new InterNAR(n);
////
////            i.runFPS(cpuFPS);
////            n.startFPS(netFPS);
////            //i.peer.them.setCapacity(3);
////
////            if (z == 0) {
////                new DeductiveChainTest(n, 4, 8000, DeductiveChainTest.inh);
////            }
////            return new Gridding(NARui.taskView(n), new NetPanel(n, i));
////        }).collect(toList())), 800, 800);
////    }
//}
