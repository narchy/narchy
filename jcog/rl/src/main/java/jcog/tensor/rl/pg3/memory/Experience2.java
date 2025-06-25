package jcog.tensor.rl.pg3.memory; // Corrected package

import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

/**
 * Represents a single step of interaction with an environment, often referred to as a transition.
 * This record stores all necessary information for many reinforcement learning algorithms.
 *
 * @param state The state observed from the environment.
 * @param action The action taken in the {@code state}.
 * @param reward The reward received after taking the {@code action} in the {@code state}.
 * @param nextState The next state observed after the transition. Can be null if the episode terminated.
 * @param done A boolean flag indicating whether the episode terminated after this transition.
 *             True if terminated, false otherwise.
 * @param oldLogProb The log probability of the {@code action} under the policy that generated it.
 *                   This is particularly used by algorithms like PPO for importance sampling.
 *                   May be null if not applicable (e.g., for DDPG or if action was from a buffer).
 */
public record Experience2(
    Tensor state,
    double[] action,
    double reward,
    Tensor nextState,
    boolean done,
    @Nullable Tensor oldLogProb
) {
    /**
     * Constructor for experiences where the log probability of the action is not available or not needed.
     *
     * @param state The state observed.
     * @param action The action taken.
     * @param reward The reward received.
     * @param nextState The next state observed.
     * @param done True if the episode terminated, false otherwise.
     */
    public Experience2(Tensor state, double[] action, double reward, Tensor nextState, boolean done) {
        this(state, action, reward, nextState, done, null);
    }
}
