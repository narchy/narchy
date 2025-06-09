//package nars.perf;
//
//import nars.NARS;
//import nars.Op;
//import nars.nal.nal1.NAL1Test;
//import nars.nal.nal2.NAL2Test;
//import nars.nal.nal3.NAL3Test;
//import nars.nal.nal5.NAL5Test;
//import nars.nal.nal6.NAL6Test;
//import nars.nal.nal8.NAL8Test;
//import nars.term.util.builder.MegaHeapTermBuilder;
//import nars.term.util.builder.SeparateInterningTermBuilder;
//import org.openjdk.jmh.annotations.*;
//
//import static nars.perf.NARBenchmarks.runTests;
//
//@State(Scope.Benchmark)
//public class InterningTermBuilderBenchmark {
//
//    /**
//     * 0 = heap TermBuilder
//     */
//    @Param("32768")
//    private String termBuilderInterningSize;
//
//    @Param({
//            "0",
////            "3",
////            "4",
//            "5",
//            "6",
//            "7",
////            "8",
////            "9",
//            "15"
//    })
//    private String termBuilderInterningVolume;
//
//    @Benchmark
//    @BenchmarkMode(Mode.SingleShotTime)
//    //, jvmArgsPrepend = "-Xint"
//    //            ,jvmArgsPrepend = {"-XX:+UnlockDiagnosticVMOptions", "-XX:+UseMulAddIntrinsic"}
//    @Fork(1)
//    @Threads(1)
//    @Warmup(iterations = 1)
//    @Measurement(iterations = 4)
//    public static void testInterning() {
//
//        runTests(true, NARS::tmp,
//                NAL1Test.class,
//                NAL2Test.class,
//                NAL3Test.class,
//                NAL5Test.class,
//                NAL6Test.class,
//                NAL8Test.class
//        );
//    }
//
//    @Setup
//    public void init() {
//
//        int v = Integer.parseInt(termBuilderInterningVolume);
//        if (v == 0)
//            Op.terms = Op.terms_;
//        else {
//            switch (termBuilderInterningSize) {
//                default -> {
//                    int cap = Integer.parseInt(termBuilderInterningSize);
//                    Op.terms = new SeparateInterningTermBuilder(
//                            cap/2, cap, v, MegaHeapTermBuilder.the);
//                }
//            }
//        }
//    }
//
//
//
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
