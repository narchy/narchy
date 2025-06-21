//package jcog.tensor.rl.dqn;
//
//import jcog.Is;
//import jcog.TODO;
//import jcog.Util;
//import jcog.signal.FloatRange;
//import jcog.tensor.DeltaPredictor;
//import jcog.tensor.Predictor;
//
//import java.util.concurrent.atomic.AtomicBoolean;
//import java.util.function.Supplier;
//
//import static jcog.Util.clampSafe;
//
//
///**
// * DQN
// * https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
// */
//@Is({"Q-learning", "State-action-reward-state-action"})
//public class QPolicy extends PredictorPolicy {
//
//
//    /** "gamma" discount factor: importance of future rewards
//     *  https://en.wikipedia.org/wiki/Q-learning#Discount_factor */
//    public final FloatRange plan = new FloatRange(0.99f, 0, 1);
//
//
//    /** NaN to disable */
//    private static final float deltaClamp =
//        Float.NaN;
//        //2;
//        //1;
//        //10;
//
//    /** TODO move into separate impls of the update function */
//    public final AtomicBoolean sarsaOrQ = new AtomicBoolean(false);
//
//    /**
//     * https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de
//     * https://github.com/BY571/Munchausen-RL/blob/master/M-DQN.ipynb
//     * experimental
//     */
//    public final AtomicBoolean munchausen = new AtomicBoolean(false);
//
//    /** munchausen alpha ? */
//    private static final float m_alpha = 0.9f;
//
//    /** munchausen entropy tau ? */
//    private static final double entropy_tau =
//            //1;
//            0.03;
//
//    /** munchausen minimum ? */
//    private static final float lo = -1;
//
//    public transient double[] dq;
//
//
//
//
//    public QPolicy(Predictor p) {
//        super(()->p, null);
//    }
//
//    public QPolicy(Supplier<Predictor> p) {
//        this(p, p);
//    }
//
//    public QPolicy(Supplier<Predictor> p, Supplier<Predictor> pAct) {
//        super(p, pAct);
//    }
//
//    private transient double rewardPrev = Double.NaN;
//
//    /**
//     * Q update function
//     * @see https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e
//     */
//    @Override public synchronized double[] learn(double[] xPrev, double[] action, double reward, double[] x, float pri) {
//        if (dq == null || dq.length!=action.length) dq = new double[action.length];
//
//        double[] qPrev = p.get(xPrev).clone(); //TODO is clone() necessary?
//        double[] qNext = pTarget.get(x).clone();     //TODO is clone() necessary?
//
//        float alphaPri = pri * learn.floatValue();
//        //float alphaQ = pri * learn.floatValue(), alphaPri = 1;
//        //float alphaQ = (float) Math.sqrt(pri * learn.floatValue()), alphaPri = alphaQ; //balanced
//
//        double gamma = plan.doubleValue();
//        int n = action.length;
//
//        boolean m = munchausen.getOpaque(); if (m) sarsaOrQ.set(false); //HACK
//
//        boolean sarsaOrQ = this.sarsaOrQ.getOpaque();
//        double qNextMax = sarsaOrQ ?
//                Double.NaN /* computed below */ : Util.max(qNext);
//        double gammaQNextMax = gamma * qNextMax;
//
//        double logsumNext = m ? Util.logsumexp(qNext, -qNextMax, 1/entropy_tau)*entropy_tau : Double.NaN;
//
//        double qPrevMax = m ? Util.max(qPrev) : Double.NaN;
//        double logsumPrev = m ? Util.logsumexp(qPrev, -qPrevMax, 1/entropy_tau)*entropy_tau : Double.NaN;
//
//        double rewardPrev = this.rewardPrev;
//        this.rewardPrev = reward;
//
//        for (int a = 0; a < n; a++) {
//            double aa = action[a];
//
//            double qPrevA = qPrev[a], qNextA = qNext[a];
//
//            /* estimate of optimal future value */
//            double gq = m ?
//                gqMunch(gamma, qNextMax, logsumNext, qPrevMax, logsumPrev, qPrevA, qNextA, false) :
//                (sarsaOrQ ? gamma * qNextA : gammaQNextMax);
//
//            //gq *= aa; //EXPERIMENTAL
//
//            dq[a] = (aa * (reward + gq - qPrevA));
//
////            if (contrastive) {
////                dq[a] += ((1 - aa) * ((1 - reward) - gq - qPrevA))/(n-1);
////                //dq[a] += +alphaQ * ((1 - aa) * ((1 - reward) - gq - qPrevA));
////                //dq[a] += +alphaQ * ((1 - aa) /* TODO refine */ * ((1 - reward) - gq - qPrevA)) / (action.length - 1f);
////                //dq[a] += +alphaQ * ((1 - aa) /* TODO refine */ * ((1 - reward) - qPrevA)) / (action.length - 1f);
////                //dq[a] += +alphaQ * ((1 - aa) / (action.length - 1f) /* TODO refine */ * (-(1 - reward) + gq - qPrevA));
////                //dq[a] += -alphaQ * ((1 - aa) / (action.length - 1f) /* TODO refine */ * ((1 - reward) + gq - qPrevA));
////            }
//
//            //dq[a] = alphaQ * (aa * (reward + gq) - qPrevA);
//            //dq[a] = alphaQ * aa * (reward) + (gq - qPrevA);
//            //dq[a] = alphaQ * aa * ((reward*action[a]) + gq - qPrevA); //fair proportion of reward, assuming sum(action)=1
//        }
//
//        if (p instanceof DeltaPredictor D) {
//            if (deltaClamp == deltaClamp) {
//                //System.out.println("dq min..max: " + n4(Util.min(dq)) + ".." + n4(Util.max(dq)));
//                clampSafe(dq, -deltaClamp, +deltaClamp);
//                //Util.normalizePolar(dq, tdErrClamp); //TODO this may only work if tdErrClamp=1
//            }
//
//            //System.out.println(n4(dq));
//            D.putDelta(dq, alphaPri);
//        } else
//            throw new TODO("d.put(plus(q,dq), learnRate) ?");
//
//        //System.out.println(n4(qNext));
//        return pTarget != p ? predict(x) : qNext;
////        return qNext;
//    }
//
//
//    private double gqMunch(double gamma, double qMaxNext, double logsumNext, double qMaxPrev, double logsumPrev, double qPrevA, double qNextA, boolean terminal) {
//
//        // Get predicted Q values (for next states) from target model to calculate entropy term with logsum
//        double mNext = qNextA - qMaxNext - logsumNext;
//
//        double mPrev = qPrevA - qMaxPrev - logsumPrev;
//
//        return m_alpha * clampSafe(mPrev, lo, 0) +
//                (terminal ? 0 : (gamma * qNextA * (qNextA - mNext)));
//    }
//
//
////    private static double qMax(double[] q) {
////        int qMaxIndex = Util.argmax(q);
////        if (qMaxIndex == -1)
////            return q[0];
////        else
////            return q[qMaxIndex];
////    }
//
//}