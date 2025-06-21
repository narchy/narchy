package jcog.tensor.rl.dqn;

import jcog.agent.Agent;
import jcog.agent.Policy;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.tensor.DeltaPredictor;
import jcog.tensor.rl.dqn.replay.Experience;
import jcog.tensor.rl.dqn.replay.Replay;
import org.eclipse.collections.api.block.function.primitive.IntIntToObjectFunction;
import org.jetbrains.annotations.Nullable;

import java.util.Random;

import static jcog.Util.sumAbs;

@Deprecated public class PolicyAgent extends Agent {

    public final Policy policy;
    public final Random rng = new XoRoShiRo128PlusRandom();
    public final RandomBits RNG = new RandomBits(rng);

    /** "epsilon" curiosity/exploration parameter.
     * note: this is in addition to curiosity which is applied in AbstractGoalConcept */
    private final FloatRange explore = new FloatRange(0.0f, 0, 1);

    @Override
    public String toString() {
        return super.toString() + "(" + policy.getClass() + ")";
    }

    /**
     * "surprise": last iteration's learning loss
     */
    public double errMean, errMin, errMax;

    /** TODO move to subclass */
    @Deprecated Replay replay;

    private transient double[] xPrev;

    public PolicyAgent(int numInputs, int numActions, IntIntToObjectFunction<? extends Policy> policy) {
        this(numInputs, numActions, policy.value(numInputs, numActions));
    }

    public PolicyAgent(int numInputs, int numActions, Policy policy) {
        super(numInputs, numActions);
        this.policy = policy;
        this.policy.clear(rng);
    }


//    public static final DiffableFunction dqnOutputActivation =
//        LinearActivation.the;
//        //SigmoidActivation.the;
//        //SigLinearActivation.the;
//        //new SigLinearActivation(2, -2, +2);
//        //TanhActivation.the;


    //    public static Agent DQN_LSTM(int inputs, int actions) {
//        return new ValuePredictAgent(inputs, actions, (i, a) ->
//                new PredictorPolicy(new LSTM(i, a, 1)));
//    }


    public Agent replay(Replay r) {
        this.replay = r;
        return this;
    }

    /**
     * TODO parameter to choose individual, or batch
     */
    @Override
    public synchronized void apply(double[] inputPrev, double[] actionPrev /* TODO */, float reward, double[] x, double[] actionNext) {

        if (reward==reward)
            reward = reward(reward);

        double[] xPrev = this.xPrev;
        if (xPrev == null)
            xPrev = this.xPrev = x.clone();

        run(new Experience(
                replay != null ? replay.t : 0,
                xPrev,
                actionPrev,
                reward,
                x), actionNext, 1);

        if (replay != null)
            replay.run(this, actionPrev, reward, x, xPrev, actionNext);

        System.arraycopy(x, 0, xPrev, 0, x.length);

        policy.update();
    }

    /** option to preprocess reward before input */
    protected float reward(float reward) {
        return reward;
        //return rewardFn.valueOf(reward);
    }
    //FloatToFloatFunction rewardFn = new Percentilizer(100);


    /** @return dq[] */
    public synchronized double[] run(Experience e, @Nullable double[] action, float alpha) {

//        DeltaPredictor p = (DeltaPredictor)(
////            (policy instanceof DirectPolicy D) ?
////                D.p :
//                (policy instanceof QPolicy Q ? Q.p : null));

        double errBefore = /*p!=null ? p.deltaSum : */Double.NaN;

        double[] actionNext = policy.learn(e, alpha);

        if (action != null)
            System.arraycopy(actionNext, 0, action, 0, actionNext.length);

        return delta(action, null, errBefore, actionNext);
    }

    /** computes delta and err for some impl's */
    @Deprecated @Nullable private double[] delta(@Nullable double[] action, @Nullable DeltaPredictor p, double errBefore, double[] actionNext) {
        double[] dq = null;

        switch (policy) {
//            case QPolicy qPolicy -> {
//                dq = qPolicy.dq;
//                err(dq);
//            }
//            case QPolicySimul Q -> {
//                dq = Q.q.dq;
//                err(dq);
//            }
//            case BranchedPolicy bb -> {
//                dq = new double[bb.actions.length];
//                for (int i = 0; i < dq.length; i++) {
//                    Policy aa = bb.actions[i];
//                    if (aa instanceof QPolicySimul)
//                        dq[i] = ((QPolicySimul) aa).q.dq[0];
//                    else if (aa instanceof QPolicy1D)
//                        dq[i] = sumAbs(((QPolicy1D) aa).q.dq);
//                    else
//                        throw new TODO();
//                }
//                err(dq);
//            }
            case null, default -> {
            }
        }
        return dq;
    }

    @Override
    protected void actionFilter(double[] actionNext) {
        float e = explore.floatValue();
        if (e <= Float.MIN_NORMAL) return;

        for (int i = 0; i < actionNext.length; i++) {
            if (RNG.nextBoolean(e))
                actionNext[i] = RNG.nextFloat();
        }
    }

    protected void err(double[] qd) {
        double errTotal = sumAbs(qd);
        errMean = errTotal / qd.length;
        double errMin = Double.POSITIVE_INFINITY, errMax = Double.NEGATIVE_INFINITY;
        for (int i = 0, qdLength = qd.length; i < qdLength; i++) {
            double x = qd[i];
            double xAbs = Math.abs(x);
            errMin = Math.min(errMin, xAbs);
            errMax = Math.max(errMax, xAbs);
        }
        this.errMin = errMin;
        this.errMax = errMax;
    }


//    public static Agent DDPG(int i, int o) {
//        return new DDPG(i, (i + o)*2, o).agent();
//    }

//    public static Agent TD3(int i, int o) {
//        return new TD3(i, (i + o)*2, o).agent();
//    }

//    @Deprecated public static class DigitizedPredictAgent extends ValuePredictAgent {
//
//        float actionContrast =
//            1;
//            //2;
//
//        private final int actionDigitization;
//
//        public DigitizedPredictAgent(int actionDigitization, int numInputs, int numActions, IntIntToObjectFunction<Predictor> p) {
//            super(numInputs, numActions, (int i, int o)->
//                    new PredictorPolicy(p.value(i,o*actionDigitization)));
//            assert (actionDigitization > 1);
//            this.actionDigitization = actionDigitization;
//        }
//
//        @Override
//        public synchronized void apply(double[] action, float reward, double[] input, double[] qNext) {
////            System.out.println(Str.n2(action));
//            final double[] aa = split(action);
////            System.out.println(Str.n2(aa));
//            double[] qNextTmp = new double[qNext.length * actionDigitization];
//            super.apply(aa, reward, input, qNextTmp);
//
//            join(qNextTmp, qNext);
////            joinSoftmax(actionNextTmp, qNext);
//
////            System.out.println(Str.n2(actionNextTmp));
////            System.out.println(Str.n2(qNext));
////            System.out.println();
//        }
//
//        public double[] split(double[] x) {
//            double[] y = new double[x.length * actionDigitization];
//            for (int i = 0, j = 0; i < x.length; i++) {
//                double I = x[i];
//
//                assertUnitized(I);
//
//                Digitize digitizer =
//                    Digitize.FuzzyNeedle;
//                    //Digitize.BinaryNeedle;
//                float ii = (float) I;
//                for (int d = 0; d < actionDigitization; d++)
//                    y[j + d] = digitizer.digit(ii, d, actionDigitization);
//
//                //contrast exponent curve
//                if (actionContrast!=1) {
//                    for (int d = 0; d < actionDigitization; d++)
//                        y[j + d] = Math.pow(y[j + d], actionContrast);
//                }
//
//                //make each action's components sum to 1
//                Util.normalize(y, j, j + actionDigitization, 0, Util.max(j, j+actionDigitization, y));
//
//                j += actionDigitization;
//            }
//            return y;
//        }
//
//        //final static double thresh = Math.pow(Float.MIN_NORMAL, 4);
//
//        public double[] join(double[] y, double[] tgt) {
//            for (int i = 0, k = 0; k < tgt.length; ) {
//                tgt[k++] = undigitizeWeightedMean(y, i);
//                i += actionDigitization;
//            }
//            return tgt;
//        }
//
//        /**
//         * digital -> analog
//         */
//        protected double undigitizeWeightedMean(double[] y, int i) {
//            double x = 0, sum = 0;
//            for (int d = 0; d < actionDigitization; d++) {
//                double D = y[i + d];
//                //D = Fuzzy.unpolarize(D);
//                //D = Util.unitize(D);
//                if (D > 1)
//                    Util.nop();
//
//                D = Math.max(0, D);
//
//                //D = Math.max(0, D)/max;
//                //D = Util.normalize(D, min, max);
//                float value = ((float) d) / (actionDigitization - 1);
//                x += value * D;
//                sum += D;
//            }
//            if (sum > Float.MIN_NORMAL)
//                x /= sum;
//            else
//                x = 0.5f;
//            return x;
//        }
//
//        public double[] joinSoftmax(double[] y, double[] tgt) {
//            final DecideSoftmax decide = new DecideSoftmax(0.1f, rng);
//            for (int i = 0, k = 0; k < tgt.length; ) {
//                int index = decide.applyAsInt(Util.toFloat(y, i, i + actionDigitization));
//                tgt[k++] = ((float) index) / (actionDigitization - 1);
//                i += actionDigitization;
//            }
//            return tgt;
//        }
//
//    }


//    private static class NormalizeLayer implements MLP.LayerBuilder {
//        private final int o;
//
//        public NormalizeLayer(int i) {
//            this.o = i;
//        }
//        @Override
//        public int size() {
//            return o;
//        }
//        @Override
//        public AbstractLayer valueOf(int i) {
//            return new StatelessLayer(i, o) {
//
//                @Override
//                public double[] delta(SGDLayer.SGDWeightUpdater updater, double[] dx, float pri) {
//                    return dx;
//                }
//
//                @Override
//                public double[] forward(double[] x, RandomBits rng) {
//                    //return Util.normalize(x.clone());
//                    return Util.normalize(x.clone(), 0, Util.max(x));
//                }
//            };
//        }
//
//    }
//    private static class SplitPolarLayer implements MLP.LayerBuilder {
//        private final int o;
//
//        public SplitPolarLayer(int i) {
//            this.o = i*2;
//        }
//        @Override
//        public int size() {
//            return o;
//        }
//        @Override
//        public AbstractLayer valueOf(int i) {
//            return new StatelessLayer(i, o) {
//
//                @Override
//                public double[] delta(SGDLayer.SGDWeightUpdater updater, double[] dx, float pri) {
//                    //TODO
//                    return null;
//
////                    int n = dx.length;
////                    assert(n==o*2);
////                    double[] dy = new double[n/2];
////                    for (int i = 0, o = 0; i < n; i++) {
////                        double PN = dx[i];
////                        double P, N;
////                        if (PN >= 0) {
////                            P = +PN; N = 0;
////                        } else{
////                            N = -PN; P = 0;
////                        }
////                        dy[o++] = P; dy[o++] = N;
////                    }
////                    return dy;
//                }
//
//                @Override
//                public double[] forward(double[] x, RandomBits rng) {
//                    return PolarValuePredictAgent.split(x);
//                }
//            };
//        }
//
//    }
//
//    /** includes a prediction normalization step. which is not entirely necessary. not yet sure if helpful. probably unstable */
//    @Deprecated private static class PolarPredictorPolicy extends PredictorPolicy {
//        public PolarPredictorPolicy(Predictor p) {
//            super(p);
//        }
//
//        @Override
//        public double[] predict(double[] x) {
//            double[] y = super.predict(x);
//            for (int i = 0; i < y.length; i+=2) {
//                double p = y[i];
//                double n = y[i+1];
//                double s = p + n;
//                if (s > Float.MIN_NORMAL) {
//                    y[i] = p / s;
//                    y[i + 1] = n / s;
//                }
//            }
//            return y;
//        }
//    }
}