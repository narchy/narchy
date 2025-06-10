package jcog.tensor.deprtensor;

import jcog.Util;
import jcog.math.FloatSupplier;

import java.util.List;

public class OptimizeADAM extends OptimizeWithMemory {
    private final double beta1, beta2;
    private final double epsilon;
    private final double weightDecay; // Weight decay coefficient

    private final static boolean speedLimit = false;

    /** ADAM Atan2 https://arxiv.org/pdf/2407.05872 */
    final boolean atan2;

    public OptimizeADAM(FloatSupplier learningRate) {
        this(learningRate, 0.9f, 0.999f, 1.0E-2,
                //NaN //atan2
                1E-8f
        );
    }

    public OptimizeADAM(FloatSupplier learningRate, double beta1, double beta2, double weightDecay, double epsilon) {
        super(learningRate, 2);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.atan2 = (epsilon!=epsilon);
        this.weightDecay = weightDecay;
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.learningRate.getAsDouble();

        double weightDecay = this.weightDecay * learningRate;
        double weightDecayFactor = 1 - weightDecay;

        int P = parameters.size();
        int data = 0;
        for (int p = 0; p < P; p++) {
            update(parameters.get(p), data, learningRate, weightDecayFactor);
            data+=2;
        }
    }

    private void update(Tens0r t, int data, double learningRate, double weightDecayFactor) {
        double b1 = beta1, b2 = beta2;
        double[][] D = t.data, G = t.grad;
        int I = D.length, J = D[0].length;
        double[][] mm = data(data, I, J), vv = data(data+1, I, J);
        double nb1 = 1 - b1, nb2 = 1 - b2;

        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                double g = G[i][j];
                double m = b1 * mm[i][j] + nb1 * g;
//                if (!Double.isFinite(m))
//                    m = 0;
                mm[i][j] = m;
                double v = b2 * vv[i][j] + nb2 * g * g;
//                if (!Double.isFinite(v))
//                    v = 0;
                vv[i][j] = v;

                double dw = atan2 ?
                    //lambda m, v: a * jnp.arctan2(m, b * jnp.sqrt(v)), a=b=1
                    Math.atan2(m, Math.sqrt(v))
                    :
                    m / (Math.sqrt(v) + epsilon);

                if (!speedLimit) {
                    D[i][j] = (-dw * learningRate) + (D[i][j] * weightDecayFactor);
                } else {
                    double dij = D[i][j];
                    double next = (-dw) + (dij * weightDecayFactor);
                    double delta = next - dij;
                    delta = Util.clampSafe(delta, -learningRate, +learningRate);
                    D[i][j] += delta;
                }
            }
        }
    }

}
