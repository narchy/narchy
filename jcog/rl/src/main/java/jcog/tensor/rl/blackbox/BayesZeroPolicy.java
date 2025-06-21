package jcog.tensor.rl.blackbox;

import jcog.optimize.BayesOptimizer;

import java.util.Arrays;

public class BayesZeroPolicy extends BlackboxPolicy {

    public final BayesOptimizer b;


    /** may be more useful for an 'exploit' mode more than in 'explore' */
    private static final boolean refineContinuously = false;

    public BayesZeroPolicy(int numOutputs, int capacity, int periodMin, int periodMax) {
        super(numOutputs, 1, periodMin, periodMax);

        double[] lower = new double[numOutputs], upper = new double[numOutputs];
        Arrays.fill(upper, 1);

        this.b = new BayesOptimizer(capacity, lower, upper);
    }

    @Override
    protected double[] next() {
        return b.next();
    }

    @Override
    protected void commitIndividual(double[] actualActions, double reward) {
        b.put(actualActions, reward);
    }

    @Override
    protected void commitPopulation(double[] rewards) {
    }

    @Override
    protected void update(double[] policy) {
        if (refineContinuously)
            b.optimize(policy);
    }

    @Override
    protected double[] best() {
        return b.best();
    }
}
