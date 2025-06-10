package jcog.tensor.deprtensor;

import jcog.math.FloatSupplier;

import java.util.List;

public class OptimizeSGD extends Optimize {

    public OptimizeSGD(FloatSupplier lr) {
        super(lr);
    }

    @Override
    public void step(List<Tens0r> parameters) {
        double learningRate = this.learningRate.getAsDouble();
        for (var t : parameters) {
            double[][] d = t.data, g = t.grad;
            int I = d.length, J = d[0].length;
            for (int i = 0; i < I; i++)
                for (int j = 0; j < J; j++)
                    d[i][j] -= learningRate * g[i][j];
        }
    }

}
