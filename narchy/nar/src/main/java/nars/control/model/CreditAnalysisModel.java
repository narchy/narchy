//package nars.control.model;
//
//import jcog.data.list.Lst;
//import nars.NAR;
//import nars.control.Cause;
//
//import java.util.Arrays;
//import java.util.Comparator;
//
///**
// * doesnt change priorities (flat) so that it can get a fair analysis of the recorded statistics
// * TODO let wrap any other Should
// */
//public final class CreditAnalysisModel extends CreditModel {
//
//    int reportPeriod = 400;
//    int cycle = 0;
//    int numTop = 50;
//    Analysis[] a = new Analysis[0];
//
//    static class Analysis {
//
//        double valueSum = 0;
//        private final Cause cause;
//
//        Analysis(Cause cause) {
//            this.cause = cause;
//        }
//        //TODO traffic accumulator
//        //TODO gather instrumentation if enabled
//
//
//        public void clear() {
//            valueSum = 0;
//        }
//
//        public void print() {
//            System.out.println(valueSum + "\t" + causeStr());
//        }
//
//        private String causeStr() {
//            return cause.toString();
//        }
//    }
//
//
//
//    @Override
//    public void update(NAR n) {
//
//
//        Lst<Cause> causes = n.control.why;
//        int s = causes.size();
//        if (a.length != s) {
//            a = Arrays.copyOf(a, s); //resize
//            for (int i = 0; i < s; i++) {
//                if (a[i]==null)
//                    a[i] = new Analysis(causes.get(i));
//            }
//        }
//
//
//        if ((++cycle % reportPeriod) == 0) {
//            System.out.println("GOAL:");
//            report(new float[] { 0, 0, 1, 0 }, numTop, n);
//            System.out.println("BELIEF:");
//            report(new float[] { 0, 1, 0, 0 }, numTop, n);
//
//            credits.fill(0);
//        }
//    }
//
//    public void report(float[] wants, int numTop, NAR n) {
//        Lst<Cause> causes = n.control.why;
//        int s = Math.min(causes.size(), credits.length()/dim - 1); //HACK
//
//        for (int i = 0; i < s; i++) {
//            short why = a[i].cause.id;
//            double v = 0;
//            for (int j = 0; j < wants.length; j++)
//                v += wants[j] * ((double)credit(why, j));
//            a[i].valueSum = v;
//        }
//
//        Analysis[] aSorted = a.clone();
//        Arrays.sort(aSorted, sorter);
//
//        for (int i = 0; i < numTop; i++)
//            aSorted[i].print();
//
//    }
//
//    private static final Comparator<Analysis> sorter = Comparator.comparingDouble(z -> -z.valueSum);
//
//}