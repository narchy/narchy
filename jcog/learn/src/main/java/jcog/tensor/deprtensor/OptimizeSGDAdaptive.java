package jcog.tensor.deprtensor;

import jcog.math.FloatSupplier;

import java.util.List;

public class OptimizeSGDAdaptive extends Optimize {
    private final FloatSupplier lr;
    private final double weightDecay;
    private final double eps;
    private final boolean l1Orl2;

    public OptimizeSGDAdaptive(FloatSupplier lr) {
        this(lr, 1E-3, false);
    }

    public OptimizeSGDAdaptive(FloatSupplier lr, double weightDecay, boolean l1Orl2) {
        this(lr, weightDecay, l1Orl2, 1);
    }

    public OptimizeSGDAdaptive(FloatSupplier lr, double weightDecay, boolean l1Orl2, double eps) {
        super(lr);
        this.lr = lr;
        this.l1Orl2 = l1Orl2;
        this.weightDecay = weightDecay;
        this.eps = eps;
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.lr.getAsDouble();

        double decay = weightDecay;
        int wd = decay <= 0 ? 0 : (l1Orl2 ? 1 : 2);

        for (var t : parameters) {
            double[][] d = t.data, g = t.grad;
            int I = d.length, J = d[0].length;
            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double decayTerm = switch (wd) {
                        case 1 -> decay * Tens0r.signum(d[i][j]);
                        case 2 -> decay * d[i][j];
                        default -> 0;
                    };
                    double gij = g[i][j];
                    double gradNorm = Math.abs(gij);
                    double adaptiveLearningRate = learningRate / (gradNorm + eps);
                    d[i][j] -= adaptiveLearningRate * (gij + decayTerm);
                }
            }
        }
    }
}
