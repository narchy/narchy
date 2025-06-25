package jcog.tensor.rl.pg3.configs;

import jcog.tensor.rl.pg.util.ReplayBuffer2; // Assuming we might use or adapt this

/**
 * Configuration for the DDPG (Deep Deterministic Policy Gradient) agent.
 * <p>
 * This class encapsulates all hyperparameters and structural settings for the DDPG agent,
 * its actor and critic networks, optimizers, and replay buffer.
 */
public record DDPGAgentConfig(
    NetworkConfig actorNetworkConfig,
    NetworkConfig criticNetworkConfig,
    OptimizerConfig actorOptimizerConfig,
    OptimizerConfig criticOptimizerConfig,
    MemoryConfig memoryConfig, // Will likely need specific replay buffer settings
    HyperparamConfig hyperparams, // Common hyperparameters like gamma
    DDPGHyperparams ddpgHyperparams // DDPG specific ones like tau, noise
) {

    /**
     * DDPG-specific hyperparameters.
     *
     * @param tau                Rate for soft target updates (polyak averaging).
     * @param actionNoiseStddev  Standard deviation for action noise (e.g., Ornstein-Uhlenbeck or Gaussian).
     * @param targetPolicyNoise  Standard deviation for noise added to target policy actions (for TD3 extension, optional).
     * @param targetNoiseClip    Clipping range for target policy noise (for TD3 extension, optional).
     * @param policyUpdateFreq   Frequency of policy updates compared to critic updates.
     */
    public record DDPGHyperparams(
        Float tau,
        Float actionNoiseStddev,
        @Deprecated Float targetPolicyNoise, // More relevant for TD3
        @Deprecated Float targetNoiseClip,   // More relevant for TD3
        Integer policyUpdateFreq
    ) {
        public DDPGHyperparams {
            if (tau == null) tau = 0.005f;
            if (actionNoiseStddev == null) actionNoiseStddev = 0.1f;
            if (targetPolicyNoise == null) targetPolicyNoise = 0.2f; // Default for TD3, less critical for DDPG
            if (targetNoiseClip == null) targetNoiseClip = 0.5f;   // Default for TD3
            if (policyUpdateFreq == null) policyUpdateFreq = 1; // Standard DDPG updates actor and critic together usually
        }
    }

    public DDPGAgentConfig {
        if (actorNetworkConfig == null) throw new IllegalArgumentException("Actor network config cannot be null");
        if (criticNetworkConfig == null) throw new IllegalArgumentException("Critic network config cannot be null");
        if (actorOptimizerConfig == null) actorOptimizerConfig = new OptimizerConfig(OptimizerType.ADAM, 1e-4f, null, null);
        if (criticOptimizerConfig == null) criticOptimizerConfig = new OptimizerConfig(OptimizerType.ADAM, 1e-3f, null, null);
        if (memoryConfig == null) {
            // DDPG needs a replay buffer.
            // Let's define a default replay buffer config within MemoryConfig or expect it to be set.
            // For now, assuming MemoryConfig can handle replay buffer specifics or will be adapted.
            // Placeholder, this might need a more specific ReplayBufferConfig record.
             MemoryConfig.ReplayBufferConfig replayDefaults = new MemoryConfig.ReplayBufferConfig(1_000_000, 64);
             memoryConfig = new MemoryConfig(null, replayDefaults, null); // Episode length null for off-policy
        }
        if (hyperparams == null) hyperparams = new HyperparamConfig(
            0.99f, null, null, false, false, 0.0f, 1 // Default gamma, other PG params less relevant
        );
        if (ddpgHyperparams == null) ddpgHyperparams = new DDPGHyperparams(null, null, null, null, null);
    }

    // Convenience constructor with some defaults
    public DDPGAgentConfig(int stateDim, int actionDim) {
        this(
            new NetworkConfig(1e-4f, 256, 256), // actor
            new NetworkConfig(1e-3f, 256, 256), // critic
            new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-4f, null, null),
            new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 1e-3f, null, null),
            new MemoryConfig(null, new MemoryConfig.ReplayBufferConfig(1_000_000, 64), null),
            new HyperparamConfig(0.99f, null, null, false, false, 0.0f,1),
            new DDPGHyperparams(0.005f, 0.1f, 0.2f, 0.5f, 1)
        );
    }
}
