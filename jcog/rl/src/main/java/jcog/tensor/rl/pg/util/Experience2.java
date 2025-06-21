package jcog.tensor.rl.pg.util;

import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

/**
 * Represents a single step of interaction with the environment.
 * Includes oldLogProb for PPO's importance sampling calculation.
 */
public record Experience2(Tensor state, double[] action, double reward, Tensor nextState, boolean done, @Nullable Tensor oldLogProb) {
    public Experience2(Tensor state, double[] action, double reward, Tensor nextState, boolean done) {
        this(state, action, reward, nextState, done, null);
    }
}
