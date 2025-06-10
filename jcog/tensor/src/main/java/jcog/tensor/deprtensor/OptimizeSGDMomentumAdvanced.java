package jcog.tensor.deprtensor;

import jcog.math.FloatSupplier;
import jcog.signal.FloatRange;

import java.util.List;

public class OptimizeSGDMomentumAdvanced extends OptimizeSGDMomentum {
    private final FloatRange nesterov;
    private final FloatRange amsgrad;
    private double[][] maxVelocity;

    public OptimizeSGDMomentumAdvanced(FloatSupplier lr, float momentum, float weightDecay, float nesterov, float amsgrad) {
        super(lr, momentum, weightDecay);
        this.nesterov = new FloatRange(nesterov, 0, 1);
        this.amsgrad = new FloatRange(amsgrad, 0, 1);
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.learningRate.getAsDouble();
        double momentumValue = this.momentum.asFloat();
        double weightDecayValue = this.weightDecay.floatValue();
        double nesterovValue = this.nesterov.asFloat();
        double amsgradValue = this.amsgrad.asFloat();

        int P = parameters.size();
        if (velocity == null) velocity = new double[P][];
        if (maxVelocity == null) maxVelocity = new double[P][];

        for (int p = 0; p < P; p++) {
            var pp = parameters.get(p);
            double[][] D = pp.data, G = pp.grad;

            int I = D.length, J = D[0].length;

            if (velocity[p] == null) velocity[p] = new double[I * J];
            if (maxVelocity[p] == null) maxVelocity[p] = new double[I * J];
            double[] v = this.velocity[p];
            double[] maxV = this.maxVelocity[p];

            int idx = 0;
            for (int i = 0; i < I; i++) {
                for (int j = 0; j < J; j++) {
                    double grad = G[i][j];

                    // Apply weight decay
                    grad += weightDecayValue * D[i][j];

                    // Update velocity
                    v[idx] = momentumValue * v[idx] + grad;

                    // Apply AMSGrad
                    if (amsgradValue > 0) {
                        maxV[idx] = Math.max(maxV[idx], Math.abs(v[idx]));
                        double adaptedVelocity = (1 - amsgradValue) * v[idx] + amsgradValue * maxV[idx] * Math.signum(v[idx]);
                        v[idx] = adaptedVelocity;
                    }

                    // Apply Nesterov momentum
                    double update;
                    if (nesterovValue > 0) {
                        update = momentumValue * v[idx] + nesterovValue * grad;
                    } else {
                        update = v[idx];
                    }

                    // Update parameters
                    D[i][j] -= learningRate * update;

                    idx++;
                }
            }
        }
    }
}