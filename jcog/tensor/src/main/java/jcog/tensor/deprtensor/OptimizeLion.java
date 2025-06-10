package jcog.tensor.deprtensor;

import jcog.math.FloatSupplier;

import java.util.List;

public class OptimizeLion extends OptimizeWithMemory {
    private final double beta1, beta2;
    private final double weightDecay;
    private final double epsilon;

    public OptimizeLion(FloatSupplier learningRate) {
        this(learningRate, 0.9, 0.999, 1E-8, 4);
    }

    public OptimizeLion(FloatSupplier learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        super(learningRate, 2);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double b1 = beta1, b2 = beta2;
        double learningRate = this.learningRate.getAsDouble();

        double weightDecay = this.weightDecay * learningRate;
        double weightDecayFactor = 1 - weightDecay;

        int P = parameters.size();
        int data = 0;
        for (int p = 0; p < P; p++) {
            var t = parameters.get(p);

            double[][] D = t.data, G = t.grad;
            int I = D.length, J = D[0].length;
            double[][] mm = data(data++, I, J), vv = data(data++, I, J);

            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double g = G[i][j];
                    double m = (mm[i][j] = b1 * mm[i][j] + (1 - b1) * g);
                    double v = (vv[i][j] = b2 * vv[i][j] + (1 - b2) * g * g);

                    double update = m / (Math.sqrt(v) + epsilon);
                    double signUpdate = Tens0r.signum(update);
                    double adjustedUpdate = signUpdate * learningRate;

                    D[i][j] = D[i][j] * weightDecayFactor - adjustedUpdate;
                    mm[i][j] = m * b2 + (1 - b2) * g;  // Update momentum term
                }
            }
        }
    }
}
