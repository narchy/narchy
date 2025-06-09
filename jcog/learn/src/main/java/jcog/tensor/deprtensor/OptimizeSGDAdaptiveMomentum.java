package jcog.tensor.deprtensor;

import jcog.data.list.Lst;
import jcog.math.FloatSupplier;

import java.util.List;

import static jcog.Util.clampSafe;

public class OptimizeSGDAdaptiveMomentum extends Optimize {
    private final FloatSupplier lr;
    private final double beta1;
    private final double eps;
    private final double momentum;
    private List<Tens0r> velocities;
    private List<Tens0r> movingAvgGrads;
    private double weightDecayFactor;

    static final double maxWeightDecayFactor =
            //1.0;
            1e-1;

    static final double minWeightDecayFactor =
            1e-6;
    private final double weightDecay;

    public OptimizeSGDAdaptiveMomentum(FloatSupplier lr, double momentum, double beta1, double weightDecay, double eps) {
        super(lr);
        this.lr = lr;
        this.momentum = momentum;
        this.beta1 = beta1;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.lr.getAsDouble();
        if (velocities == null || movingAvgGrads == null) {
            velocities = new Lst<>();
            movingAvgGrads = new Lst<>();
            for (var t : parameters) {
                velocities.add(Tens0r.newTensorSizeOf(t));
                movingAvgGrads.add(Tens0r.newTensorSizeOf(t));
            }
        }

        double beta1 = this.beta1;
        double eps = this.eps;
        double wd = this.weightDecay;
        boolean decaying = wd > Double.MIN_NORMAL;

        for (int p = 0; p < parameters.size(); p++) {
            var D = parameters.get(p);
            var velocity = velocities.get(p);
            var movingAvgGrad = movingAvgGrads.get(p);
            double[][] d = D.data, g = D.grad, v = velocity.data, m = movingAvgGrad.data;
            int I = d.length, J = d[0].length;
            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double gij = g[i][j];
                    double mjk = (m[i][j] = beta1 * m[i][j] + (1 - beta1) * gij);
                    double gradNorm = Math.abs(mjk);
                    double adaptiveLearningRate = learningRate / (gradNorm + eps);
                    double weightDecayTerm = decaying ? decay(wd, mjk) * d[i][j] : 0;
                    d[i][j] -=
                            (v[i][j] =
                                    momentum * v[i][j] + adaptiveLearningRate * (gij + weightDecayTerm));
                }
            }
        }
    }

    private double decay(double weightDecay, double movingAvgGrad) {
        double gradNorm = weightDecay * Math.abs(movingAvgGrad);
        return clampSafe(gradNorm, minWeightDecayFactor, maxWeightDecayFactor);
    }
}
