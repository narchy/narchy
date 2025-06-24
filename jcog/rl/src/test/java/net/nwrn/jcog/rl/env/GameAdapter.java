package net.nwrn.jcog.rl.env;

import java.util.random.RandomGenerator;

/**
 * A simplified game interface for testing RL agents.
 */
public interface GameAdapter {
    /**
     * @return The number of observation dimensions.
     */
    int observationDim();

    /**
     * @return The number of action dimensions.
     */
    int actionDim();

    /**
     * Resets the environment to a starting state.
     * @param rng Random number generator for stochastic resets.
     * @return The initial observation.
     */
    double[] reset(RandomGenerator rng);

    /**
     * Takes an action in the environment.
     * @param action The action to take.
     * @return The result of taking the action.
     */
    StepResult step(double[] action);

    /**
     * @return Whether the environment uses continuous actions (true) or discrete actions (false).
     */
    boolean isContinuousActions();

    record StepResult(double[] observation, float reward, boolean done) {
    }
}
