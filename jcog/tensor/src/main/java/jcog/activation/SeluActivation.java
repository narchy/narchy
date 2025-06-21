package jcog.activation;

/**
 * TODO subclass EluActivation
 * https://ml-explained.com/blog/activation-functions-explained#scaled-exponential-linear-unit-selu*/
public class SeluActivation implements DiffableFunction {

    public static DiffableFunction the = new SeluActivation();

    private SeluActivation() { }

    static final double alpha = 1.67326, lambda = 1.0507;

    @Override public double valueOf(double x) {
        return lambda * (x >= 0 ? x : alpha * Math.expm1(x));
    }

    @Override
    public double derivative(double x) {
        return lambda * (x > 0 ? 1 : alpha * Math.exp(x));
    }

}
