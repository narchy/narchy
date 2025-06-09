package jcog.tensor.deprtensor;

import jcog.Util;
import jcog.data.list.Lst;

import java.util.List;

/**
 * untested
 */
public class OptimizeAdaDelta extends Optimize {
    private final double decayRate, eps;
    private final double baseWeightDecay;
    private final boolean l1Orl2;
    private List<Tens0r> gradAccumulator, updateAccumulator;

    public OptimizeAdaDelta(boolean l1Orl2, double baseWeightDecay, double eps) {
        this(l1Orl2, baseWeightDecay, 0.99f, eps);
    }

    public OptimizeAdaDelta(boolean l1Orl2, double baseWeightDecay, double decayRate, double eps) {
        super(() -> 0);
        this.decayRate = decayRate;
        this.eps = eps;
        this.baseWeightDecay = baseWeightDecay;
        this.l1Orl2 = l1Orl2;
        this.gradAccumulator = null;
        this.updateAccumulator = null;
    }

    @Override
    public void step(List<Tens0r> parameters) {
        if (gradAccumulator == null || updateAccumulator == null) {
            gradAccumulator = new Lst<>();
            updateAccumulator = new Lst<>();
            for (var t : parameters) {
                gradAccumulator.add(Tens0r.newTensorSizeOf(t));
                updateAccumulator.add(Tens0r.newTensorSizeOf(t));
            }
        }

        double decay = baseWeightDecay;
        int wd = decay <= 0 ? 0 : (l1Orl2 ? 1 : 2);

        for (int i = 0; i < parameters.size(); i++) {
            var t = parameters.get(i);
            double[][] d = t.data, g = t.grad;
            double[][] ga = gradAccumulator.get(i).data;
            double[][] ua = updateAccumulator.get(i).data;

            int I = d.length, J = d[0].length;
            for (int j = 0; j < I; j++) {
                for (int k = 0; k < J; k++) {
                    ga[j][k] = decayRate * ga[j][k] + (1 - decayRate) * Util.sqr(g[j][k]);
                    double gwjk_sqrt = Math.sqrt(ga[j][k]) + eps;
                    double uxjk_sqrt = Math.sqrt(ua[j][k]) + eps;

                    double adaptiveRate = Math.sqrt(uxjk_sqrt / gwjk_sqrt);

                    double decayTerm;
                    if (wd == 0) decayTerm = 0;
                    else {
                        double adaptiveWeightDecay = decay / gwjk_sqrt;
                        decayTerm = switch (wd) {
                            case 1 -> adaptiveWeightDecay * Math.signum(d[j][k]); // L1 Decay
                            case 2 -> adaptiveWeightDecay * d[j][k];               // L2 Decay
                            default -> 0;
                        };
                    }

                    double update = adaptiveRate * (g[j][k] + decayTerm);
                    d[j][k] -= update;

                    // Update running average of squared updates
                    ua[j][k] = decayRate * ua[j][k] + (1 - decayRate) * Util.sqr(update);
                }
            }
        }
    }
}
