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
    /**
     * Stores the log probability of the last action taken.
     * This is primarily used by agents like PPO, where the log probability of the action
     * under the policy it was sampled from (old_log_prob) is needed for the loss calculation.
     * It should be set by the agent after an action is selected and before it's returned/applied.
     */
    @Nullable protected Tensor lastActionLogProb = null;

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
     * Integrates the agent with an environment loop that uses the `jcog.agent.Agent` interface.
     *
     * This method is called by the environment (e.g., via RLPlayer) at each step.
     * It receives:
     * - `inputPrev` (s_{t-1}): The state from the previous timestep. Null if this is the first step.
     * - `actionPrev` (a_{t-1}): The action taken in `inputPrev`. Null if this is the first step.
     * - `reward` (r_{t-1}): The reward received after taking `actionPrev` in `inputPrev`.
     * - `input` (s_t): The current state, resulting from `actionPrev`. This is the state the agent must act upon.
     *                  If null, it might signify the end of an episode (terminal state s_t).
     * - `actionNext` (a_t): An output parameter; the agent must fill this array with the action it selects for `input` (s_t).
     *
     * The method performs two main functions:
     * 1. If `inputPrev` is not null, it records the experience transition (s_{t-1}, a_{t-1}, r_{t-1}, s_t, done).
     *    The `done` flag is assumed to be `false` as per user guidance.
     *    For PPO, `lastActionLogProb` (which corresponds to `logProb(a_{t-1}|s_{t-1})`) is included.
     * 2. It selects a new action `a_t` based on the current state `input` (s_t), and places it in `actionNext`.
     *    For PPO, it also computes and stores `logProb(a_t|s_t)` in `this.lastActionLogProb` for the next transition.
     */
    @Override
    public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
        // Assumption: 'done' is always false as per user guidance.
        final boolean done = false;

        // 1. Record the experience from the PREVIOUS step (s_{t-1} -> s_t)
        if (inputPrev != null && actionPrev != null) {
            Tensor prevTensorState = Tensor.row(inputPrev);
            // actionPrev is double[], but Experience2 expects the raw action for storage.
            // The 'oldLogProb' in Experience2 is this.lastActionLogProb, which was computed when actionPrev was chosen.
            Tensor currentTensorState = (input != null) ? Tensor.row(input) : null; // s_t can be null if episode ends

            Experience2 experience = new Experience2(
                prevTensorState,
                actionPrev, // Storing the raw double[] action
                reward,
                currentTensorState, // This is nextState in the context of (s_{t-1}, a_{t-1})
                done,       // `done` is false based on guidance. If s_t is terminal, this would be true.
                this.lastActionLogProb // This is log_prob(a_{t-1} | s_{t-1})
            );
            this.recordExperience(experience); // Let the agent's specific logic handle it
        }

        // 2. Select and output the NEXT action (a_t) based on the current state (s_t)
        if (input == null) {
            // Current state is null, typically means episode is over, cannot select a new action.
            // Or, if actionNext must be filled, fill with a default (e.g., zeros) or throw error.
            // For now, we assume the environment handles calls appropriately when input is null.
            // If an action is still expected, this part needs refinement.
            // Clearing lastActionLogProb as there's no "next" action from this state.
            this.lastActionLogProb = null;
            // Potentially fill actionNext with zeros or a default if required by the environment.
             if (actionNext != null) {
                 java.util.Arrays.fill(actionNext, 0.0);
             }
            return;
        }

        Tensor currentTensorState = Tensor.row(input);
        double[] selectedActionArray;

        // Agent-specific action selection and logProb handling
        if (this instanceof PPOAgent ppoAgent) {
            PPOAgent.ActionWithLogProb awlp = ppoAgent.selectActionWithLogProb(currentTensorState, !this.isTrainingMode());
            selectedActionArray = awlp.action();
            this.lastActionLogProb = awlp.logProb(); // Store for the *next* call to apply
        } else {
            // For agents not producing a logProb (like Reinforce, VPG in their basic form for this interface)
            selectedActionArray = this.selectAction(currentTensorState, !this.isTrainingMode());
            this.lastActionLogProb = null; // Ensure it's null if not set by the agent type
        }

        if (actionNext == null) {
            throw new IllegalArgumentException("actionNext output array cannot be null when input state is not null.");
        }
        if (selectedActionArray.length != actionNext.length) {
            throw new IllegalArgumentException("Selected action length " + selectedActionArray.length +
                                               " does not match output array length " + actionNext.length);
        }

        System.arraycopy(selectedActionArray, 0, actionNext, 0, selectedActionArray.length);
    }
}
