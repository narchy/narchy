package jcog.tensor.deprtensor;

import jcog.math.FloatSupplier;

import java.util.List;

public class OptimizeYellowFin extends OptimizeWithMemory {
    private double beta1, beta2;
    private double epsilon;
    private double mu; // Momentum
    private double lr; // Learning rate

    public OptimizeYellowFin(FloatSupplier learningRate) {
        this(learningRate, 0.9f, 0.999f, 1E-8f);
    }

    public OptimizeYellowFin(FloatSupplier learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate, 2);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.mu = 0.0; // Initial momentum
        this.lr = learningRate.getAsDouble(); // Initial learning rate
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double b1 = beta1, b2 = beta2;
        double lr = this.lr;
        double mu = this.mu;

        int P = parameters.size();
        int data = 0;

        adjustLrAndMu(parameters);

        for (int p = 0; p < P; p++) {
            var t = parameters.get(p);

            double[][] D = t.data, G = t.grad;
            final int I = D.length, J = D[0].length;
            double[][] mm = data(data++, I, J), vv = data(data++, I, J);


            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double g = G[i][j];
                    double m = (mm[i][j] = b1 * mm[i][j] + (1 - b1) * g);
                    double v = (vv[i][j] = b2 * vv[i][j] + (1 - b2) * g * g);


                    double adjustedLr = lr / (Math.sqrt(v) + epsilon);
                    double dw = m * adjustedLr;

                    D[i][j] = D[i][j] - dw + mu * dw;
                }
            }
        }
    }

    private void adjustLrAndMu(List<Tens0r> parameters) {
        // Compute average squared gradient and average gradient
        double sumSquaredGradient = 0;
        double sumGradient = 0;

        int count = 0, data = 0;
        int P = parameters.size();
        for (int p = 0; p < P; p++) {
            var t = parameters.get(p);
            double[][] D = t.data, G = t.grad;
            final int I = D.length, J = D[0].length;
            double[][] mm = data(data++, I, J), vv = data(data++, I, J);
            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double m = mm[i][j];
                    double v = vv[i][j];
                    sumSquaredGradient += v;
                    sumGradient += Math.abs(m);
                }
            }
            count += I * J;
        }

        double meanSquaredGradient = sumSquaredGradient / count;
        double meanGradient = sumGradient / count;

        // Gradient variance and mean gradient estimation
        double gradientVariance = meanSquaredGradient - (meanGradient * meanGradient);

        // Adjust the learning rate based on gradient variance
        // Here, lr should be inversely proportional to the variance (more variance, smaller steps)
        if (gradientVariance > 0) {
            lr = 0.001 / Math.sqrt(gradientVariance);
        }

        // Adjust the momentum based on the estimated curvature
        // If the gradients are less varied (lower variance), we can afford to be more aggressive (higher momentum)
        mu = 0.9 + 0.1 * (1 - gradientVariance / (1 + gradientVariance));  // Simplified adaptive adjustment

        // Check and constrain lr and mu within practical limits
        lr = Math.max(0.0001, Math.min(lr, 0.01));  // Constrain learning rate to practical values
        mu = Math.max(0.85, Math.min(mu, 0.999));  // Constrain momentum to practical values
    }

}
