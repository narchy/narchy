package jcog.activation;

/**
 * https://arxiv.org/abs/2108.12943
 */
public class GCUActivation implements DiffableFunction {

    @Override
    public double valueOf(double x) {
        return x * Math.cos(x);
    }

    @Override
    public double derivative(double x) {
        return Math.cos(x) - x * Math.sin(x);
    }

}
