package jcog.activation;

public class TanhActivation implements DiffableFunction
{
    public static final TanhActivation the = new TanhActivation();

    @Override
    public double valueOf(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double y) {
        return 1 - y * y;
    }

}