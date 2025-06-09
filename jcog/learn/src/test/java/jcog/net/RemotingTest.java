//package jcog.net;
//
//import org.gridkit.nanocloud.CloudFactory;
//import org.gridkit.nanocloud.RemoteNode;
//import org.gridkit.nanocloud.VX;
//import org.junit.jupiter.api.Disabled;
//import org.junit.jupiter.api.Test;
//
//import java.lang.management.ManagementFactory;
//import java.net.InetAddress;
//import java.net.UnknownHostException;
//import java.util.concurrent.Callable;
//
///**
// * https://github.com/jenkinsci/remoting/blob/master/README.md
// * https://github.com/gridkit/nanocloud/blob/vicluster-0.8/docs/NanoCloud_Tutorial.md
// * */
//@Disabled
//public class RemotingTest {
//
//    static final String ALL_NODES = "**";
//
//    @Test void test1() {
//        // Let's create simple cloud where slaves will run on same box with master
//        var cloud = CloudFactory.createCloud();
//
//
//        cloud.node(ALL_NODES).x(VX.TYPE).setLocal();
//
//        // This line says that 'node1' should exists
//        // all initialization are lazy and asynchronous
//        // so this line will not trigger any process creation
//        cloud.node("node1");
//
//        // two stars match any node name
//
//        // let our node to say hello
//        cloud.node(ALL_NODES).exec((Callable<Void>) () -> {
//            String jvmName = ManagementFactory.getRuntimeMXBean().getName();
//            System.out.println("My name is '" + jvmName + "'. Hello!");
//            return null;
//        });
//    }
//    public static void main(String[] args) {
//    //@Test void test2() {
//
//        var cloud = CloudFactory.createCloud();
//
//        RemoteNode rn = RemoteNode.at(cloud.node("**"));
//        rn.useSimpleRemotingForLegacyEngine();
//
//        String host = "ml";
//        rn.setRemoteAccount("me");
//        rn.setRemoteJavaExec("/home/me/jdk/bin/java");
//        rn.setProp("debug", "true");
//
//        cloud.node(host).exec((Runnable) () -> {
//            try {
//                System.out.println(InetAddress.getLocalHost().getCanonicalHostName());
//            } catch (UnknownHostException e) {
//                e.printStackTrace();
//            }
//            //String jvmName = ManagementFactory.getRuntimeMXBean().getName();
//            System.getProperties().forEach((k, v) ->
//                System.out.println(k + "\t" + v)
//            );
//        });
//    }
//}
