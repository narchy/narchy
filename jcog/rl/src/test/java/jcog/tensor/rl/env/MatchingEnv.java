package jcog.tensor.rl.env;

import jcog.random.XoRoShiRo128PlusRandom;

import java.util.random.RandomGenerator;

public class MatchingEnv implements SyntheticEnv {
    private final int dim;
    private double[] currentState;
    private final RandomGenerator rng = new XoRoShiRo128PlusRandom();

    public MatchingEnv(int dim) {
        this.dim = dim;

    }

    @Override
    public int stateDimension() {
        return dim;
    }

    @Override
    public int actionDimension() {
        return dim;
    }

    @Override
    public boolean isDiscreteActionSpace() {
        return false;
    }

    @Override
    public double[] reset(RandomGenerator rng) {
        currentState = rng.doubles(dim, -1.0, 1.0).toArray();
        return currentState;
    }

    @Override
    public StepResult step(double[] action) {
        double mse = 0;
        for (var i = 0; i < dim; i++) {
            mse += Math.pow(currentState[i] - action[i], 2);
        }
        var reward = -mse / dim;
        // For matching task, each step is an independent trial, so "done" is true.
        // The next state is a new random state.
        var nextState = rng.doubles(dim, -1.0, 1.0).toArray();
        // currentState = nextState; // Not needed as it's reset effectively
        return new StepResult(nextState, reward);
    }

    @Override
    public StepResult step(int action) {
        throw new UnsupportedOperationException("MatchingEnv uses continuous actions.");
    }
}
