//package nars.control.model;
//
//import jcog.Skill;
//import jcog.Util;
//import jcog.data.list.Lst;
//import jcog.signal.tensor.AtomicFloatVector;
//import nars.NAR;
//import nars.control.Cause;
//import nars.control.MetaGoal;
//
//import java.util.Arrays;
//
///**
// * credit assignment tracing
// */
//@Skill("Reinforcement_learning")
//public class CreditControlModel extends CreditModel {
//
//    public interface Governor {
//        void accept(double[] want, Lst<Cause> causes, NAR n);
//    }
//
//    public final Governor governor;
//
//    private final float[][] credit = new float[0][0];
//    private float[] creditMin;
//    private float[] creditMax;
//    private float[] creditRange;
//
//    /**
//     * the indices of this array correspond to the ordinal() value of the MetaGoal enum values
//     * TODO convert to AtomicFloatArray or something where each value is volatile
//     */
//    public transient double[] want = new double[MetaGoal.values().length];
//
//    public CreditControlModel(Governor governor) {
//        this.governor = governor;
//    }
//
//    /**
//     * sets the desired level for a particular MetaGoal.
//     * the value may be positive or negative indicating
//     * its desirability or undesirability.
//     * the absolute value is considered relative to the the absolute values
//     * of the other MetaGoal's
//     */
//    public void want(MetaGoal g, float v) {
//        want[g.ordinal()] = v;
//    }
//
//
//    /**
//     * default linear adder
//     *
//     * @param updater responsible for reading the 'value' computed here and assigning an effective
//     *                priority to it
//     **/
//    @Override
//    public void update(NAR n) {
//
//        Lst<Cause> causesList = n.control.why;
//        int cc = causesList.size();
//        if (cc == 0)
//            return;
//
//        double[] want = this.want;
//        if (Util.equals(Util.sumAbs(want), 0))
//            return; //flat metagoal early exit
//
////        Util.normalizeCartesian(want); //TODO this may differ from the array computed in Should if the want is modified in the meantime by another thread, so it should be shared
//
//        float[][] credit = this.credit;
//        float[] min, max, range;
//        int creditLenPrev = credit.length;
//        int ww = want.length;
//        if (creditLenPrev != cc) {
//            credit = new float[cc][ww];
//            min = creditMin = new float[ww];
//            max = creditMax = new float[ww];
//            range = creditRange = new float[ww];
//        } else {
//            min = creditMin;
//            max = creditMax;
//            range = creditRange;
//        }
//
//        Cause[] causes = causesList.array();
//
//        Arrays.fill(min, Float.POSITIVE_INFINITY);
//        Arrays.fill(max, Float.NEGATIVE_INFINITY);
//        AtomicFloatVector C = this.credits;
//        int dim = CreditModel.dim;
//        int wn = want.length;
//        for (int i = 0; i < cc; i++) {
//            float[] ci = credit[i];
//            commit(C, i*dim, ci);
//            for (int w = 0; w < wn; w++) {
//                float c = ci[w];
//                if (c == c) {
//                    min[w] = Math.min(min[w], c);
//                    max[w] = Math.max(max[w], c);
//                }
//            }
//        }
//        for (int i = 0; i < wn; i++) {
//            float r = Math.max(Math.abs(max[i]), Math.abs(min[i]));
//            if (r < Float.MIN_NORMAL)
//                r = Float.POSITIVE_INFINITY;
//            range[i] = r;
//        }
//
//        for (int i = 0; i < cc; i++) {
//            double v = 0;
//            float[] ci = credit[i];
//            for (int w = 0; w < wn; w++)
//                v = jcog.Util.fma(want[w], ((double) ci[w]) / range[w], v); //v += want[w] * (ci[w]/range[w]);
//
//            causes[i].value((float) v);
//        }
//
//        governor.accept(want, causesList, n);
//    }
//
//    public static void commit(AtomicFloatVector credit, int from, float[] target) {
//        int n = target.length;
//        int f = from;
//        for (int i = 0; i < n; i++)
//            target[i] = credit.getAndZero(f++);
//    }
//
//}