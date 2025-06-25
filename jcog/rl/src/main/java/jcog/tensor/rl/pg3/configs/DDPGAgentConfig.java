package jcog.tensor.rl.pg3.configs;

import jcog.Util;
import jcog.math.Range;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;

/**
 * Configuration for the DDPG (Deep Deterministic Policy Gradient) Agent.
 *
 * <p>DDPG is an off-policy algorithm that learns a Q-function (critic) and a deterministic policy (actor)
 * concurrently. It uses experience replay and target networks for stable training.
 *
 * <ul>
 *   <li>Actor Network (Policy): Maps states to specific actions (deterministic).
 *   <li>Critic Network (Q-Value): Estimates the value of state-action pairs.
 *   <li>Target Networks: Time-delayed copies of the actor and critic that are updated slowly,
 *       improving stability.
 *   <li>Replay Buffer: Stores past experiences for off-policy learning.
 *   <li>Noise Process: Adds noise to actions during training for exploration in continuous action spaces.
 * </ul>
 */
public record DDPGAgentConfig(
    HyperparamConfig commonHyperparams,
    NetworkConfig actorNetworkConfig,
    NetworkConfig criticNetworkConfig,
    OptimizerConfig actorOptimizerConfig,
    OptimizerConfig criticOptimizerConfig,
    MemoryConfig memoryConfig, // Used for replay buffer settings
    @Nullable PerConfig perConfig, // Optional: Configuration for Prioritized Experience Replay
    ActionConfig.NoiseConfig noiseConfig, // Configuration for action noise
    Float tau, // Rate for soft updating target networks (e.g., 0.005)
    Integer policyUpdateFreq, // How often to update the policy and target networks (e.g., every 2 steps)
    String notes
) {

    public static final float DEFAULT_TAU = 0.005f;
    public static final int DEFAULT_POLICY_UPDATE_FREQ = 2;

    /**
     * Configuration for Prioritized Experience Replay (PER).
     *
     * @param alpha Alpha parameter for PER, controls prioritization (0=uniform, 1=full).
     * @param beta0 Initial beta parameter for IS weight annealing.
     * @param betaAnnealingSteps Steps over which beta anneals from beta0 to 1.0.
     */
    public record PerConfig(
        double alpha,
        double beta0,
        int betaAnnealingSteps
    ) {
        public static final double DEFAULT_ALPHA = 0.6;
        public static final double DEFAULT_BETA0 = 0.4;
        public static final int DEFAULT_BETA_ANNEALING_STEPS = 1_000_000; // Example value

        public PerConfig {
            Util.validate(alpha, a -> a >= 0, "PER alpha must be non-negative.");
            Util.validate(beta0, b -> b >= 0, "PER beta0 must be non-negative.");
            Util.validate(betaAnnealingSteps, s -> s >= 0, "PER betaAnnealingSteps must be non-negative.");
        }

        public PerConfig() {
            this(DEFAULT_ALPHA, DEFAULT_BETA0, DEFAULT_BETA_ANNEALING_STEPS);
        }
    }


    public DDPGAgentConfig {
        Objects.requireNonNull(commonHyperparams, "commonHyperparams must not be null");
        Objects.requireNonNull(actorNetworkConfig, "actorNetworkConfig must not be null");
        Objects.requireNonNull(criticNetworkConfig, "criticNetworkConfig must not be null");
        Objects.requireNonNull(actorOptimizerConfig, "actorOptimizerConfig must not be null");
        Objects.requireNonNull(criticOptimizerConfig, "criticOptimizerConfig must not be null");
        Objects.requireNonNull(memoryConfig, "memoryConfig must not be null");
        Objects.requireNonNull(noiseConfig, "noiseConfig must not be null");
        // perConfig is optional

        if (tau == null) tau = DEFAULT_TAU;
        if (policyUpdateFreq == null) policyUpdateFreq = DEFAULT_POLICY_UPDATE_FREQ;

        Util.validate(tau, t -> t > 0 && t <= 1, "Tau must be between 0 (exclusive) and 1 (inclusive)");
        Util.validate(policyUpdateFreq, f -> f > 0, "Policy update frequency must be positive");

        if (memoryConfig.replayBuffer() == null) {
            throw new IllegalArgumentException("MemoryConfig must specify replayBuffer settings for DDPG.");
        }
         if (memoryConfig.episodeLength() != null && perConfig == null) { // Only warn if not using PER which might have different step considerations
            System.err.println("Warning: DDPGAgentConfig: memoryConfig.episodeLength is typically not used for off-policy DDPG with standard replay. Ensure replayBuffer.capacity and batchSize are set appropriately.");
        }
    }

    /**
     * Minimal constructor with sensible defaults for some parameters, no PER.
     */
    public DDPGAgentConfig(
        HyperparamConfig commonHyperparams,
        NetworkConfig actorNetworkConfig,
        NetworkConfig criticNetworkConfig,
        OptimizerConfig actorOptimizerConfig,
        OptimizerConfig criticOptimizerConfig,
        MemoryConfig memoryConfig,
        ActionConfig.NoiseConfig noiseConfig
    ) {
        this(commonHyperparams, actorNetworkConfig, criticNetworkConfig,
             actorOptimizerConfig, criticOptimizerConfig, memoryConfig, null, // No PER by default
             noiseConfig, DEFAULT_TAU, DEFAULT_POLICY_UPDATE_FREQ, null);
    }

    public DDPGAgentConfig withNotes(String notes) {
        return new DDPGAgentConfig(commonHyperparams, actorNetworkConfig, criticNetworkConfig,
                                   actorOptimizerConfig, criticOptimizerConfig, memoryConfig, perConfig, noiseConfig,
                                   tau, policyUpdateFreq, notes);
    }

    public DDPGAgentConfig withTau(float tau) {
        return new DDPGAgentConfig(commonHyperparams, actorNetworkConfig, criticNetworkConfig,
                                   actorOptimizerConfig, criticOptimizerConfig, memoryConfig, perConfig, noiseConfig,
                                   tau, policyUpdateFreq, notes);
    }

    public DDPGAgentConfig withPolicyUpdateFreq(int policyUpdateFreq) {
        return new DDPGAgentConfig(commonHyperparams, actorNetworkConfig, criticNetworkConfig,
                                   actorOptimizerConfig, criticOptimizerConfig, memoryConfig, perConfig, noiseConfig,
                                   tau, policyUpdateFreq, notes);
    }

    public DDPGAgentConfig withPerConfig(PerConfig perConfig) {
        return new DDPGAgentConfig(commonHyperparams, actorNetworkConfig, criticNetworkConfig,
                                   actorOptimizerConfig, criticOptimizerConfig, memoryConfig, perConfig, noiseConfig,
                                   tau, policyUpdateFreq, notes);
    }


    @Override
    public String toString() {
        return "DDPGAgentConfig{" +
            "\n  commonHyperparams=" + commonHyperparams +
            ",\n  actorNetworkConfig=" + actorNetworkConfig +
            ",\n  criticNetworkConfig=" + criticNetworkConfig +
            ",\n  actorOptimizerConfig=" + actorOptimizerConfig +
            ",\n  criticOptimizerConfig=" + criticOptimizerConfig +
            ",\n  memoryConfig=" + memoryConfig +
            (perConfig != null ? ",\n  perConfig=" + perConfig : "") +
            ",\n  noiseConfig=" + noiseConfig +
            ",\n  tau=" + tau +
            ",\n  policyUpdateFreq=" + policyUpdateFreq +
            (notes != null ? ",\n  notes='" + notes + '\'' : "") +
            "\n}";
    }
}
