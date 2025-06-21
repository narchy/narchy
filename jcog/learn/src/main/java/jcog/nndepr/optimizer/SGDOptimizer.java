package jcog.nndepr.optimizer;

import jcog.Util;
import jcog.nndepr.layer.LinearLayer;
import jcog.signal.FloatRange;

import java.util.concurrent.atomic.AtomicBoolean;

import static jcog.Util.fma;

/**
 * 'vanilla' stochastic gradient descent (SGD), with optional:
 * momentum
 * L1-regularization weight decay
 */
public class SGDOptimizer extends BatchWeightUpdater {

    /**
     * gradient momentum.  The coefficient for how much of the previous delta is applied to each weight.
     * In theory, prevents local minima stall.
     * <p>
     * TODO FloatRange
     */
    public final FloatRange dwMomentum = new FloatRange(0, 0, 1);


    /**
     * QUASI-HYPERBOLIC MOMENTUM (QHM)
     * https://arxiv.org/pdf/1810.06801.pdf
     * 1=disabled
     */
    public final FloatRange nu = new FloatRange(0.7f, 0, 1);

    private transient float alpha;

    public final AtomicBoolean momentumAccel = new AtomicBoolean(false);

    public static final float WEIGHT_DECAY_DEFAULT =
        //1.5E-3f;
        1E-4f;
        //2.0E-3f;
        //1.0E-2f;
        //0; //disabled

    public final FloatRange weightDecay = FloatRange.unit(WEIGHT_DECAY_DEFAULT);

    @Deprecated
    public static final double wEpsilon =
            1.0E-24;
    //0;
    //1;
    //0.5;
    //1.0E-1;
    //2.0E-2;
    //8.0E-3;

//    /** whether weight decay is scaled by L1(delta) */
//    private static final boolean weightDecayDX = false;


    public SGDOptimizer(float dwMomentum) {
        this.dwMomentum.set(dwMomentum);
    }

    @Override
    public void reset(int weights, float alpha) {
        super.reset(weights, alpha);
        this.alpha = alpha;
    }

    @Override
    protected void updateWeights(LinearLayer l, double[] dW, double[] dWPrev, double[] W) {
        double pri = this.alpha;
        float _weightDecayRate = this.weightDecay.floatValue();
        boolean weightDecaying = _weightDecayRate > 0;
        double wL1 = weightDecaying ? Util.sumAbs(W) : 0;
        double weightDecayRate =
            //1 / (1 + pri * _weightDecayRate * wL1/W.length);
            pri * _weightDecayRate / (wEpsilon + wL1);
        double negWeightDecayRate = -weightDecayRate;
            //1 - pri * _weightDecayRate / (wEpsilon + wL1);

//        final double a = 1 - pri * _weightDecayRate / (wEpsilon + wL1);
//        final double b = 1 / (1 + pri * _weightDecayRate * wL1/W.length);
//        System.out.println((a-b) + "\t" + a + "\t" + b);

        float nu = this.nu.asFloat();

        float dwMomentum = this.dwMomentum.asFloat();
        boolean momentum = dwMomentum > 0;

        double antiMomentum = momentumAccel.get() ? 1 : (1 - dwMomentum);

        int n = l.ins() * l.outs();
        for (int io = 0; io < n; io++) {
            double dwP = dWPrev[io];
            double dwN = pri * dW[io];

            double dw = momentum ? fma(dwMomentum, dwP, antiMomentum * dwN) : dwN;

            dWPrev[io] = dw;

            /* QHM */
            if (momentum)
                dw = Util.lerpSafe(nu, dwN, dw);

            double wP = W[io];

            //if (weightDecaying) dw = fma(wP, weightDecayRate, dw);
            //W[io] = wP + dw;

//        wP += dw;
//        if (weightDecaying) wP *= weightDecayRate;
//        W[io] = wP;

//            if (weightDecaying) wP = fma(wP, weightDecayRate, dw);
//            else wP += dw;
//            W[io] = wP;

            if (weightDecaying) dw = fma(wP, negWeightDecayRate, dw);

            W[io] = wP + dw;

        }
    }
}