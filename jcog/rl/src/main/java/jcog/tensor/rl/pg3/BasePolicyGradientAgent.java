package jcog.tensor.rl.pg3;

import jcog.agent.Agent;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import org.jetbrains.annotations.Nullable;

/**
 * An abstract base class for implementing {@link PolicyGradientAgent}.
 * This class provides default implementations for managing the training mode status
 * and the update count. Concrete agent implementations should extend this class
 * and implement the core algorithmic logic.
 *
 * <p>Subclasses are responsible for initializing their specific networks, optimizers,
 * memory buffers, and configurations. They must also override {@link #setTrainingMode(boolean)}
 * to propagate the training mode to their internal neural network components.</p>
 */
public abstract class BasePolicyGradientAgent extends Agent implements PolicyGradientAgent {

    /**
     * Flag indicating whether the agent is in training mode. Defaults to true.
     * Subclasses should use and respect this flag in their {@code selectAction}
     * and {@code recordExperience} methods.
     */
    protected boolean trainingMode = true;

    /**
     * Counter for the number of learning updates performed.
     * Incremented by calling {@link #incrementUpdateCount()}.
     */
    protected long updateCount = 0;
    @Nullable protected Tensor lastLogProb = null; // Cache for the log probability of the last action taken

    public BasePolicyGradientAgent(int i, int o) {
        super(i, o);
    }

    /**
     * {@inheritDoc}
     * <p>This implementation returns the current value of the protected {@code trainingMode} field.</p>
     */
    @Override
    public boolean isTrainingMode() {
        return this.trainingMode;
    }

    /**
     * {@inheritDoc}
     * <p>This implementation sets the protected {@code trainingMode} field.
     * Subclasses <strong>must</strong> override this method to also propagate the training mode
     * to their internal neural networks (e.g., policy and value function networks) by calling
     * their respective {@code train(boolean)} methods.</p>
     * <pre>{@code
     * @Override
     * public void setTrainingMode(boolean training) {
     *     super.setTrainingMode(training); // Sets the flag in BasePolicyGradientAgent
     *     this.policy.train(training);
     *     if (this.valueFunction != null) {
     *         this.valueFunction.train(training);
     *     }
     *     if (!training) { // Optional: clear memory when switching to evaluation
     *         this.memory.clear();
     *     }
     * }
     * }</pre>
     */
    @Override
    public void setTrainingMode(boolean training) {
        this.trainingMode = training;
        // Subclasses MUST override to propagate to their networks.
    }

    /**
     * {@inheritDoc}
     * <p>This implementation returns the current value of the protected {@code updateCount} field.</p>
     */
    @Override
    public long getUpdateCount() {
        return this.updateCount;
    }

    /**
     * Increments the learning update counter.
     * Concrete agent implementations should call this method after successfully
     * performing a learning update (e.g., at the end of their {@code update} method).
     */
    protected void incrementUpdateCount() {
        this.updateCount++;
    }

    // Abstract methods from PolicyGradientAgent (selectAction, recordExperience, update,
    // getPolicy, getValueFunction, getConfig, clearMemory) must be implemented by concrete subclasses.

    /**
     * Selects an action and potentially returns its log probability.
     * This method should be implemented by subclasses that need to provide log_prob for PPO/SAC.
     * For other agents, it can simply call selectAction and return null for logProb.
     *
     * @param state The current state tensor.
     * @param deterministic Whether to select action deterministically.
     * @return An ActionWithLogProb object containing the action and its log probability (or null if not applicable).
     */
    protected abstract ActionWithLogProb selectActionWithLogProb(Tensor state, boolean deterministic);


    @Override
    public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
        Tensor currentStateTensor = Tensor.row(input); // Current state s_t+1

        // Select action for the current state (s_t+1) -> a_t+1
        // Use deterministic=false during training for exploration, true for evaluation could be a refinement.
        // For now, let's assume 'isTrainingMode()' implies exploration if agent supports it.
        ActionWithLogProb nextActionDetails = selectActionWithLogProb(currentStateTensor, !isTrainingMode());
        double[] nextActionArray = nextActionDetails.action();
        System.arraycopy(nextActionArray, 0, actionNext, 0, Math.min(nextActionArray.length, actionNext.length));

        Tensor currentLogProb = nextActionDetails.logProb(); // Log prob of a_t+1

        if (isTrainingMode() && inputPrev != null && actionPrev != null) {
            Tensor prevStateTensor = Tensor.row(inputPrev); // Previous state s_t
            Tensor prevActionTensor = Tensor.row(actionPrev); // Previous action a_t

            // Retrieve the cached log probability of a_t (actionPrev)
            Tensor oldLogProbForPrevAction = this.lastLogProb;

            // Assume done = false as per user guidance
            boolean done = false;

            Experience2 experience = new Experience2(
                prevStateTensor,
                prevActionTensor, // Storing action tensor directly
                reward,
                currentStateTensor,
                done,
                oldLogProbForPrevAction // This is log_prob(a_t | s_t)
            );
            recordExperience(experience);
        }

        // Cache the log probability of the action *just taken* (actionNext) for the *next* call to apply.
        this.lastLogProb = currentLogProb;
    }

    /**
     * Helper record to bundle action and its log probability.
     * Used by selectActionWithLogProb.
     */
    protected record ActionWithLogProb(double[] action, @Nullable Tensor logProb) {
    }
}
