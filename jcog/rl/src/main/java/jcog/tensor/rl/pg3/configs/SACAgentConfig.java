package jcog.tensor.rl.pg3.configs;

import jcog.Util;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;

/**
 * Configuration for the Soft Actor-Critic (SAC) Agent.
 *
 * <p>SAC is an off-policy actor-critic algorithm designed for continuous action spaces that aims to
 * maximize a trade-off between expected return and policy entropy. Higher entropy encourages exploration.
 *
 * <ul>
 *   <li>Actor Network (Policy): Typically a stochastic policy (e.g., Gaussian) that outputs parameters
 *       for an action distribution. Actions are often squashed (e.g., by tanh) to a bounded range.</li>
 *   <li>Critic Networks (Q-Value): SAC usually employs two Q-networks (and their targets) to mitigate
 *       positive bias in Q-value estimation. It takes the minimum of the two Q-target values during updates.</li>
 *   <li>Value Network (Optional, for some SAC variants): Some SAC versions also learn a state-value function V.
 *       The variant implemented here will focus on the Q-networks and policy.</li>
 *   <li>Temperature Parameter (alpha): Controls the stochasticity of the policy by weighting the entropy term
 *       in the objective function. Alpha can be fixed or learned automatically.</li>
 *   <li>Target Networks: Time-delayed copies of the critic (and actor, in some variants) that are updated slowly.</li>
 *   <li>Replay Buffer: Stores past experiences for off-policy learning.
 * </ul>
 */
public record SACAgentConfig(
    HyperparamConfig commonHyperparams, // Includes gamma, learning rates for shared components etc.
    NetworkConfig policyNetworkConfig, // For the actor/policy
    NetworkConfig qNetworkConfig,      // For the Q-value critics
    // Optimizer configs are often part of NetworkConfig, but can be separate if more control is needed.
    // For SAC, we need optimizers for policy, Q-functions, and potentially alpha.
    OptimizerConfig policyOptimizerConfig,
    OptimizerConfig qOptimizerConfig,
    @Nullable OptimizerConfig alphaOptimizerConfig, // Only if learning alpha

    MemoryConfig memoryConfig,         // For replay buffer settings
    @Nullable PerConfig perConfig, // Optional: Configuration for Prioritized Experience Replay


    Float tau,                         // Rate for soft updating target Q-networks (e.g., 0.005)
    @Nullable Float initialAlpha,      // Initial value for temperature parameter alpha. If null, fixed alpha from commonHyperparams might be used or default.
    @Nullable Float targetEntropy,     // Target entropy. If null or not learning alpha, this is not used.
                                       // If learning alpha, set to -action_dim typically.
    Boolean learnAlpha,                // Whether to learn alpha automatically.
    Integer policyUpdateFreq,          // How often to update the policy and alpha (if learned). (e.g., every 2 critic updates)
    Integer targetUpdateFreq,          // How often to update the target Q-networks. (e.g., every 2 critic updates, often same as policyUpdateFreq)

    String notes
) {
    public static final float DEFAULT_TAU = 0.005f;
    public static final float DEFAULT_INITIAL_ALPHA = 0.2f; // A common starting point if fixed or learning
    public static final int DEFAULT_POLICY_UPDATE_FREQ = 2;
    public static final int DEFAULT_TARGET_UPDATE_FREQ = 2;

    /**
     * Configuration for Prioritized Experience Replay (PER).
     * (Copied from DDPGAgentConfig for consistency)
     */
    public record PerConfig(
        double alpha,
        double beta0,
        int betaAnnealingSteps
    ) {
        public static final double DEFAULT_ALPHA = 0.6;
        public static final double DEFAULT_BETA0 = 0.4;
        public static final int DEFAULT_BETA_ANNEALING_STEPS = 1_000_000;

        public PerConfig {
            Util.validate(alpha, a -> a >= 0, "PER alpha must be non-negative.");
            Util.validate(beta0, b -> b >= 0, "PER beta0 must be non-negative.");
            Util.validate(betaAnnealingSteps, s -> s >= 0, "PER betaAnnealingSteps must be non-negative.");
        }
        public PerConfig() { this(DEFAULT_ALPHA, DEFAULT_BETA0, DEFAULT_BETA_ANNEALING_STEPS); }
    }


    public SACAgentConfig {
        Objects.requireNonNull(commonHyperparams, "commonHyperparams must not be null");
        Objects.requireNonNull(policyNetworkConfig, "policyNetworkConfig must not be null");
        Objects.requireNonNull(qNetworkConfig, "qNetworkConfig must not be null");
        Objects.requireNonNull(policyOptimizerConfig, "policyOptimizerConfig must not be null");
        Objects.requireNonNull(qOptimizerConfig, "qOptimizerConfig must not be null");
        // alphaOptimizerConfig is only required if learnAlpha is true
        Objects.requireNonNull(memoryConfig, "memoryConfig must not be null");
        // perConfig is optional

        if (learnAlpha == null) learnAlpha = true; // Default to learning alpha
        if (learnAlpha && alphaOptimizerConfig == null) {
            throw new IllegalArgumentException("alphaOptimizerConfig must be provided if learnAlpha is true.");
        }

        if (tau == null) tau = DEFAULT_TAU;
        if (initialAlpha == null) initialAlpha = DEFAULT_INITIAL_ALPHA;
        // targetEntropy can be null if alpha is fixed. If learning, it should ideally be set.
        // A common heuristic for targetEntropy is -action_dim. This should be set by the user based on actionDim.

        if (policyUpdateFreq == null) policyUpdateFreq = DEFAULT_POLICY_UPDATE_FREQ;
        if (targetUpdateFreq == null) targetUpdateFreq = DEFAULT_TARGET_UPDATE_FREQ;


        Util.validate(tau, t -> t > 0 && t <= 1, "Tau must be between 0 (exclusive) and 1 (inclusive)");
        Util.validate(initialAlpha, alphaVal -> alphaVal > 0, "Initial alpha must be positive"); // Renamed param to avoid conflict
        Util.validate(policyUpdateFreq, f -> f > 0, "Policy update frequency must be positive");
        Util.validate(targetUpdateFreq, f -> f > 0, "Target update frequency must be positive");


        if (memoryConfig.replayBuffer() == null) {
            throw new IllegalArgumentException("MemoryConfig must specify replayBuffer settings for SAC.");
        }
         if (memoryConfig.episodeLength() != null && perConfig == null) {
            System.err.println("Warning: SACAgentConfig: memoryConfig.episodeLength is typically not used for off-policy SAC with standard replay. Ensure replayBuffer.capacity and batchSize are set.");
        }
    }

    /** Minimal constructor with defaults, no PER */
    public SACAgentConfig(HyperparamConfig commonHyperparams, NetworkConfig policyNetworkConfig,
                          NetworkConfig qNetworkConfig, OptimizerConfig policyOptimizerConfig,
                          OptimizerConfig qOptimizerConfig, @Nullable OptimizerConfig alphaOptimizerConfig,
                          MemoryConfig memoryConfig, boolean learnAlpha, @Nullable Float targetEntropy) {
        this(commonHyperparams, policyNetworkConfig, qNetworkConfig,
             policyOptimizerConfig, qOptimizerConfig, alphaOptimizerConfig, memoryConfig,
             null, // No PER by default
             DEFAULT_TAU, DEFAULT_INITIAL_ALPHA, targetEntropy, learnAlpha,
             DEFAULT_POLICY_UPDATE_FREQ, DEFAULT_TARGET_UPDATE_FREQ, null);
    }


    public SACAgentConfig withNotes(String notes) {
        return new SACAgentConfig(commonHyperparams, policyNetworkConfig, qNetworkConfig,
                                  policyOptimizerConfig, qOptimizerConfig, alphaOptimizerConfig, memoryConfig,
                                  perConfig, tau, initialAlpha, targetEntropy, learnAlpha,
                                  policyUpdateFreq, targetUpdateFreq, notes);
    }

    public SACAgentConfig withPerConfig(PerConfig perConfig) {
        return new SACAgentConfig(commonHyperparams, policyNetworkConfig, qNetworkConfig,
                                  policyOptimizerConfig, qOptimizerConfig, alphaOptimizerConfig, memoryConfig,
                                  perConfig, tau, initialAlpha, targetEntropy, learnAlpha,
                                  policyUpdateFreq, targetUpdateFreq, notes);
    }

    // Add other `with...` methods as needed for convenience.

    @Override
    public String toString() {
        return "SACAgentConfig{" +
            "\n  commonHyperparams=" + commonHyperparams +
            ",\n  policyNetworkConfig=" + policyNetworkConfig +
            ",\n  qNetworkConfig=" + qNetworkConfig +
            ",\n  policyOptimizerConfig=" + policyOptimizerConfig +
            ",\n  qOptimizerConfig=" + qOptimizerConfig +
            (alphaOptimizerConfig != null ? ",\n  alphaOptimizerConfig=" + alphaOptimizerConfig : "") +
            ",\n  memoryConfig=" + memoryConfig +
            (perConfig != null ? ",\n  perConfig=" + perConfig : "") +
            ",\n  tau=" + tau +
            ",\n  initialAlpha=" + initialAlpha +
            (targetEntropy != null ? ",\n  targetEntropy=" + targetEntropy : "") +
            ",\n  learnAlpha=" + learnAlpha +
            ",\n  policyUpdateFreq=" + policyUpdateFreq +
            ",\n  targetUpdateFreq=" + targetUpdateFreq +
            (notes != null ? ",\n  notes='" + notes + '\'' : "") +
            "\n}";
    }
}
