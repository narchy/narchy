package jcog.tensor.deprtensor;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.math.FloatSupplier;

import java.util.List;

/**
 * unstable; can't be used continously because sumGradSquares explodes
 * TODO extend OptimizeWithMemory
 */
public class OptimizeAdaGrad extends Optimize {
    private final FloatSupplier lr;
    private final double baseWeightDecay, eps;
    private final boolean l1Orl2;
    private List<Tens0r> sumGradSquares;

    public OptimizeAdaGrad(FloatSupplier lr, boolean l1Orl2, double baseWeightDecay) {
        this(lr, baseWeightDecay, l1Orl2, 1e-2);
    }

    public OptimizeAdaGrad(FloatSupplier lr, double baseWeightDecay, boolean l1Orl2, double eps) {
        super(lr);
        this.lr = lr;
        this.eps = eps;
        this.baseWeightDecay = baseWeightDecay;
        this.l1Orl2 = l1Orl2;
        this.sumGradSquares = null;
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.lr.getAsDouble();
        if (sumGradSquares == null) {
            sumGradSquares = new Lst<>();
            for (var t : parameters)
                sumGradSquares.add(Tens0r.newTensorSizeOf(t));
        }

        double decay = baseWeightDecay;
        int wd = decay <= 0 ? 0 : (l1Orl2 ? 1 : 2);

        for (int i = 0; i < parameters.size(); i++) {
            var t = parameters.get(i);
            double[][] d = t.data, g = t.grad;
            double[][] gs = sumGradSquares.get(i).data;

            int I = d.length, J = d[0].length;
            for (int j = 0; j < I; j++) {
                for (int k = 0; k < J; k++) {
                    double gsjk = (gs[j][k] += Util.sqr(g[j][k]));
                    double gwjk_sqrt = Math.sqrt(gsjk) + eps;

                    double adaptiveLearningRate = learningRate / gwjk_sqrt;

                    double decayTerm;
                    if (wd == 0) decayTerm = 0;
                    else {
                        double adaptiveWeightDecay = decay / gwjk_sqrt;
                        decayTerm = switch (wd) {
                            case 1 -> adaptiveWeightDecay * Tens0r.signum(d[j][k]); // L1 Decay
                            case 2 -> adaptiveWeightDecay * d[j][k];             // L2 Decay
                            default -> 0;
                        };
                    }

                    d[j][k] -= adaptiveLearningRate * (g[j][k] + decayTerm);
                }
            }
        }
    }
}
