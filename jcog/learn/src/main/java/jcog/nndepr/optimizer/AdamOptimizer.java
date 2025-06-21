package jcog.nndepr.optimizer;

import jcog.nndepr.layer.LinearLayer;
import jcog.signal.FloatRange;
import jcog.util.ArrayUtil;

import static jcog.Util.fma;
import static jcog.Util.lerpSafe;

/** TODO https://arxiv.org/pdf/1709.07417.pdf */
public class AdamOptimizer extends BatchWeightUpdater {

    public final FloatRange
        beta1 = FloatRange.unit(0.9f),
        beta2 = FloatRange.unit(0.99f);

    //        private double beta1PowerT = Double.NaN, beta2PowerT = Double.NaN;

    public final FloatRange epsilon = FloatRange.unit(
        1E-8f
        //1.0E-4
        //1.0E-12f
        //0.5f
    );

    public float pri;

    /** mean - running estimate of gradient */
    private double[] momentM = ArrayUtil.EMPTY_DOUBLE_ARRAY;

    /** variance - running estimate of gradient^2 */
    private double[] momentV = ArrayUtil.EMPTY_DOUBLE_ARRAY;

    private final boolean minimizing = false;
    private int pAbs;


    /**
     * https://arxiv.org/pdf/1711.05101.pdf
     * https://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html
     * https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/torch/optim/_multi_tensor/adamw.py
     */
    public final FloatRange weightDecay = FloatRange.unit(
        0
        //1E-5f
        //2.5E-4f
        //0.5f;
        //0.1f;
        //0.05f;
        //0.001f;
    );


    //        protected void minimizeDeltas(Map<VariableNode, TensorNode> var) {
//            minimizing = true;
//            imize(var);
//        }
//
//        protected void maximizeDeltas(Map<VariableNode, TensorNode> var) {
//            minimizing = false;
//            imize(var);
//        }

    @Override
    public void reset(int weights, float alpha) {
        super.reset(weights,alpha);
        if (momentM.length != weights) {
//                beta1PowerT = beta1;
//                beta2PowerT = beta2;
            momentM = new double[weights];
            momentV = new double[weights];
        }
        this.pAbs = 0;
        this.pri = alpha;
    }


    @Override
    protected void updateWeights(LinearLayer l, double[] dW, double[] dWPrev, double[] W) {
        int pAbs = this.pAbs;
        double[] mm = this.momentM, vv = this.momentV;

        double pri = this.pri;


        var _weightDecay = this.weightDecay.doubleValue();
        double weightDecay = _weightDecay!=0 ?
                pri * _weightDecay
                //pri * Math.min(1, Util.sumAbs(dW) / dW.length) * this.weightDecay
                //pri * this.weightDecay * Util.sumAbs(dW) / dW.length
            : 0;

        double weightDecayFactor = 1 - weightDecay;

        double alphaT =
                pri * (minimizing ? -1 : +1)
                //* Math.sqrt(1 - beta2) / (1 - beta1)
                ;

        double beta1 = this.beta1.doubleValue(), beta2 = this.beta2.doubleValue(), epsilon = this.epsilon.doubleValue();
        int n = l.ins() * l.outs();
        for (int pRel = 0; pRel < n; pRel++, pAbs++) {
            double g = dW[pRel];

            //update the moving averages of the gradient: beta1 * mm[pAbs] + (1-beta1)*g
            double m = mm[pAbs] =
                lerpSafe(beta1, g, mm[pAbs]);

            //update the moving averages of the squared gradient: beta2 * vv[pAbs] + (1-beta2)*(g*g)
            double v = vv[pAbs] =
                lerpSafe(beta2, g * g, vv[pAbs]);

            double dw =
                m / (Math.sqrt(v) + epsilon);

            double Wprev = W[pRel];
            W[pRel] = fma(dw, alphaT, Wprev * weightDecayFactor);

//                    double m_cap = m/(1-(Math.pow(beta1,t)))		#calculates the bias-corrected estimates
//                            v_cap = v/(1-(Math.pow(beta2,t))		#calculates the bias-corrected estimates

        }
        this.pAbs = pAbs;

//            if (this.pAbs == 0) {
//                beta1PowerT *= beta1PowerT;
//                beta2PowerT *= beta2PowerT;
//            }
    }

    public final AdamOptimizer alpha(float v) {
        pri = v;
        return this;
    }

    public final AdamOptimizer momentum(double b1, double b2) {
        this.beta1.set(b1);
        this.beta2.set(b2);
        return this;
    }

    public final AdamOptimizer epsilon(double eps) {
        this.epsilon.set(eps);
        return this;
    }

    public final WeightUpdater weightDecay(float w) {
        this.weightDecay.set(w);
        return this;
    }
}