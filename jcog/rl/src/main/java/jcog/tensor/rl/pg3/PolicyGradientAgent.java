package jcog.tensor.rl.pg3;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg3.configs.PPOAgentConfig;
import jcog.tensor.rl.pg3.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg3.networks.ValueNet;

/**
 * Core interface for Policy Gradient (PG) reinforcement learning agents.
 * This interface defines the common functionality expected from all PG agent implementations,
 * facilitating interaction with environments and training loops.
 *
 * Implementations are expected to handle their own internal state, including networks,
 * optimizers, and memory buffers, configured typically via constructor injection.
 */
public interface PolicyGradientAgent {

    /**
     * Selects an action based on the current state observation.
     * The method of action selection may vary depending on whether the agent is in training
     * or evaluation mode, and whether a deterministic or stochastic action is requested.
     *
     * @param state The current state representation, typically a {@link Tensor}.
     * @param deterministic If true, the agent should select the action that it currently
     *                      believes to be optimal (e.g., the mean of a policy distribution).
     *                      If false, the agent should sample an action from its policy distribution,
     *                      allowing for exploration.
     * @return An array of doubles representing the selected action. The dimensionality
     *         should match the action space of the environment.
     */
    double[] selectAction(Tensor state, boolean deterministic);

    /**
     * Records an experience transition, which consists of the state, action, reward,
     * next state, and done flag.
     * This method is responsible for storing the experience in the agent's memory buffer.
     * Depending on the agent's learning strategy (e.g., on-policy vs. off-policy) and
     * buffer status (e.g., full), this method might also trigger a learning update.
     *
     * @param experience The {@link Experience2} tuple representing the transition.
     *                   For PPO agents, this experience should include the log probability
     *                   of the action taken (oldLogProb).
     */
    void recordExperience(Experience2 experience);

    /**
     * Performs a learning update step for the agent.
     * This typically involves sampling data from the memory buffer, computing gradients
     * based on the agent's algorithm (e.g., REINFORCE, VPG, PPO), and updating
     * the parameters of its internal networks (policy, value function).
     *
     * @param totalSteps The total number of environment steps taken so far. This can be
     *                   used for scheduling purposes, such as annealing learning rates or
     *                   other hyperparameters, or for logging. May not be used by all agents.
     */
    void update(long totalSteps);

    /**
     * Retrieves the policy network (actor) of the agent.
     * The returned object is generally an instance of a class from {@code jcog.tensor.Models}
     * (e.g., {@link GaussianPolicyNet})
     * that can be applied to a state tensor to produce action parameters or a distribution.
     * Exposing this allows for inspection and direct use if needed (e.g., by a GUI or for analysis).
     *
     * @return The policy network object. Type is {@code Object} for interface flexibility;
     *         concrete implementations will return specific network types.
     */
    Object getPolicy();

    /**
     * Retrieves the value function network (critic) of the agent, if applicable.
     * Some policy gradient agents, like basic REINFORCE, do not use an explicit value function network.
     * For agents like VPG or PPO, this returns the network that estimates state values.
     *
     * @return The value function network object (e.g., {@link ValueNet}),
     *         or {@code null} if the agent does not use one.
     */
    Object getValueFunction();

    /**
     * Gets the configuration object used by the agent.
     * This object encapsulates all hyperparameters and structural settings for the agent,
     * its networks, and memory. (e.g., {@link PPOAgentConfig}).
     *
     * @return The agent's configuration object. Type is {@code Object} for interface flexibility.
     */
    Object getConfig();

    /**
     * Indicates whether the agent is currently in training mode.
     * In training mode, agents may perform exploration, update networks, and accumulate gradients.
     * In evaluation mode (not training), agents typically act deterministically or with minimal exploration
     * and do not update their parameters.
     *
     * @return {@code true} if the agent is in training mode, {@code false} otherwise.
     */
    boolean isTrainingMode();

    /**
     * Sets the agent's operational mode.
     *
     * @param training {@code true} to set the agent to training mode,
     *                 {@code false} for evaluation/inference mode.
     *                 This should also propagate the mode to internal networks.
     */
    void setTrainingMode(boolean training);

    /**
     * Clears any accumulated experiences from the agent's memory buffer.
     * This is often called after a learning update for on-policy agents or to reset state.
     */
    void clearMemory();

    /**
     * Returns the total number of learning updates that have been performed by the agent.
     * This can be useful for tracking training progress.
     *
     * @return The count of learning updates.
     */
    long getUpdateCount();

}
