package jcog.tensor.deprtensor;

import jcog.math.FloatSupplier;
import jcog.signal.FloatRange;

import java.util.List;

import static jcog.Util.fma;
import static jcog.Util.lerpSafe;


public class OptimizeSGDMomentum extends Optimize {
    protected final FloatRange momentum;
    protected final FloatRange weightDecay;

    protected double[][] velocity;

    public OptimizeSGDMomentum(FloatSupplier lr, float momentum) {
        this(lr, momentum, 1.0E-7f);
    }

    public OptimizeSGDMomentum(FloatSupplier lr, float momentum, float weightDecay) {
        super(lr);
        this.momentum = new FloatRange(momentum, 0, 1);
        this.weightDecay = FloatRange.unit(weightDecay);
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.learningRate.getAsDouble();
        double learningRateNeg = -learningRate;
        double momentum = this.momentum.asFloat();
        double weightDecay = this.weightDecay.floatValue();

        int P = parameters.size();
        if (velocity == null) velocity = new double[P][];

//        double gradientClip = this.gradientClip;
//        boolean clipVelocity = gradientClip!=Double.POSITIVE_INFINITY;
//        double gradientClipDivLR = gradientClip / learningRate;

        for (int p = 0; p < P; p++) {
            var pp = parameters.get(p);
            double[][] D = pp.data, G = pp.grad;

            int I = D.length, J = D[0].length;

            double[] V = velocity[p] == null ?
                (velocity[p] = new double[I * J]) : this.velocity[p];

            int idx = 0;
            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++, idx++) {
                    double x = D[i][j];
                    double gij =
                        fma(x, weightDecay, G[i][j]);
                        //dij * weightDecayValue + G[i][j];

                    double v =
                        lerpSafe(momentum, gij, V[idx]); //balanced
                        //fma(V[idx], momentum, gij); //momentum * v[idx] + gij;

//                    if (clipVelocity)
//                        v = clamp(v, -gradientClipDivLR, +gradientClipDivLR);

                    V[idx] = v;
                    D[i][j] = fma(v, learningRateNeg, x);
                }
            }
        }
    }
}