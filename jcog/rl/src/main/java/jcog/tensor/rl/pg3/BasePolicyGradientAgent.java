package jcog.tensor.rl.pg3;

// Removed unused imports for Tensor and Experience2 as they are not directly used in this abstract class's code.
// Subclasses will import them as needed.

import jcog.agent.Agent;
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

    @Override
    public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {

    }
}
