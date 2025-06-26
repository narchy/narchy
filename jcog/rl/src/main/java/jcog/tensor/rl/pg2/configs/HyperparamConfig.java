package jcog.tensor.rl.pg2.configs;

import jcog.signal.FloatRange;
import jcog.signal.IntRange;

import java.util.Objects;

/**
 * Configuration for common hyperparameters used across various Policy Gradient agents.
 * These parameters control aspects of the learning process such as discount rates,
 * learning rates, and algorithm-specific settings like PPO clipping or SAC entropy.
 *
 * <p>Fields are {@code public final} as records are immutable, facilitating GUI access and inspection.
 * Uses {@link FloatRange} and {@link IntRange} for numeric parameters to support GUI rendering,
 * validation, and definition of sensible min/max/default values.</p>
 *
 * @param gamma Discount factor for future rewards. Typically close to 1.0 (e.g., 0.99).
 * @param lambda Parameter for Generalized Advantage Estimation (GAE). Typically between 0.9 and 1.0.
 * @param entropyBonus Coefficient for the entropy bonus term in the policy loss, encouraging exploration.
 * @param ppoClip Clipping parameter for PPO's surrogate objective. Limits the policy update step.
 * @param tau Target network update rate (for soft updates in algorithms like DDPG or SAC). (1-tau)*target + tau*source.
 * @param policyLR Learning rate for the policy network optimizer.
 * @param valueLR Learning rate for the value network optimizer.
 * @param epochs Number of optimization epochs to run on a collected batch of data (used in PPO).
 * @param policyUpdateFreq Frequency of policy updates relative to critic updates (e.g., in DDPG/SAC, update policy every N critic updates).
 * @param normalizeAdvantages Whether to normalize advantages before using them in the policy loss.
 * @param normalizeReturns Whether to normalize returns (often targets for value function).
 * @param learnableAlpha For SAC, whether the entropy temperature alpha is a learnable parameter.
 */
public record HyperparamConfig(
    FloatRange gamma,
    FloatRange lambda,
    FloatRange entropyBonus,
    FloatRange ppoClip,
    FloatRange tau,
    FloatRange policyLR,
    FloatRange valueLR,
    IntRange epochs,
    IntRange policyUpdateFreq,
    boolean normalizeAdvantages,
    boolean normalizeReturns,
    boolean learnableAlpha
) {

    /** Default discount factor (gamma): 0.99. Range [0.0, 1.0]. */
    public static final FloatRange DEFAULT_GAMMA = new FloatRange(0.9f, 0.0f, 1.0f);
    /** Default GAE lambda: 0.95. Range [0.0, 1.0]. */
    public static final FloatRange DEFAULT_LAMBDA = new FloatRange(0.5f, 0.0f, 1.0f);
    /** Default entropy bonus coefficient: 0.01. Range [0.0, 1.0]. */
    public static final FloatRange DEFAULT_ENTROPY_BONUS = new FloatRange(0.01f, 0.0f, 1.0f);
    /** Default PPO clipping epsilon: 0.2. Range [0.01, 0.5]. */
    public static final FloatRange DEFAULT_PPO_CLIP = new FloatRange(0.2f, 0.01f, 0.5f);
    /** Default target network soft update rate (tau): 0.005. Range [0.0, 1.0]. */
    public static final FloatRange DEFAULT_TAU = new FloatRange(0.005f, 0.0f, 1.0f);
    /** Default policy learning rate: 3e-4. Range [1e-6, 1e-2]. */
    public static final FloatRange DEFAULT_POLICY_LR = new FloatRange(3e-4f, 1e-6f, 1e-2f);
    /** Default value function learning rate: 1e-3. Range [1e-6, 1e-2]. */
    public static final FloatRange DEFAULT_VALUE_LR = new FloatRange(1e-3f, 1e-6f, 1e-2f);
    /** Default number of PPO epochs: 10. Range [1, 100]. */
    public static final IntRange DEFAULT_EPOCHS = new IntRange(1, 1, 100);
    /** Default policy update frequency (e.g., for DDPG/SAC): 1 (update policy as often as critic). Range [1, 1000]. */
    public static final IntRange DEFAULT_POLICY_UPDATE_FREQ = new IntRange(1, 1, 1000);
    /** Default for normalizing advantages: true. */
    public static final boolean DEFAULT_NORMALIZE_ADVANTAGES = true;
    /** Default for normalizing returns: false. */
    public static final boolean DEFAULT_NORMALIZE_RETURNS = false;
    /** Default for learnable alpha (SAC entropy temperature): false. */
    public static final boolean DEFAULT_LEARNABLE_ALPHA = false;

    /**
     * Validates hyperparameter configuration.
     * Ensures that all range-based parameters are within their expected bounds and that
     * learning rates and other critical values are positive.
     */
    public HyperparamConfig {
        Objects.requireNonNull(gamma, "gamma cannot be null");
        Objects.requireNonNull(lambda, "lambda cannot be null");
        Objects.requireNonNull(entropyBonus, "entropyBonus cannot be null");
        Objects.requireNonNull(ppoClip, "ppoClip cannot be null");
        Objects.requireNonNull(tau, "tau cannot be null");
        Objects.requireNonNull(policyLR, "policyLR cannot be null");
        Objects.requireNonNull(valueLR, "valueLR cannot be null");
        Objects.requireNonNull(epochs, "epochs cannot be null");
        Objects.requireNonNull(policyUpdateFreq, "policyUpdateFreq cannot be null");

        if (gamma.floatValue() < 0.0f || gamma.floatValue() > 1.0f) throw new IllegalArgumentException("gamma must be in [0, 1]");
        if (lambda.floatValue() < 0.0f || lambda.floatValue() > 1.0f) throw new IllegalArgumentException("lambda must be in [0, 1]");
        // entropyBonus can be 0
        if (tau.floatValue() < 0.0f || tau.floatValue() > 1.0f) throw new IllegalArgumentException("tau must be in [0, 1]");
        if (policyLR.floatValue() <= 0) throw new IllegalArgumentException("policyLR must be positive");
        if (valueLR.floatValue() <= 0) throw new IllegalArgumentException("valueLR must be positive");
        if (epochs.intValue() <= 0) throw new IllegalArgumentException("epochs must be positive for PPO-like algorithms");
        if (policyUpdateFreq.intValue() <= 0) throw new IllegalArgumentException("policyUpdateFreq must be positive");
        if (ppoClip.floatValue() <= 0) throw new IllegalArgumentException("ppoClip must be positive for PPO");
    }

    /**
     * Creates a {@code HyperparamConfig} with default values for all hyperparameters.
     */
    public HyperparamConfig() {
        this(
            DEFAULT_GAMMA, DEFAULT_LAMBDA, DEFAULT_ENTROPY_BONUS, DEFAULT_PPO_CLIP, DEFAULT_TAU,
            DEFAULT_POLICY_LR, DEFAULT_VALUE_LR, DEFAULT_EPOCHS, DEFAULT_POLICY_UPDATE_FREQ,
            DEFAULT_NORMALIZE_ADVANTAGES, DEFAULT_NORMALIZE_RETURNS, DEFAULT_LEARNABLE_ALPHA
        );
    }

    // Note: "Wither" methods are omitted for brevity but can be added if needed
    // for more convenient modification of individual parameters while maintaining immutability.
    // Example: public HyperparamConfig withGamma(FloatRange gamma) { return new HyperparamConfig(gamma, ...); }
}
