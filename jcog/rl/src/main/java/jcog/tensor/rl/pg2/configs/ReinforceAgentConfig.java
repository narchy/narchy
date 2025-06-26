package jcog.tensor.rl.pg2.configs;

import java.util.Objects;

public record ReinforceAgentConfig(
    HyperparamConfig hyperparams,
    NetworkConfig policyNetworkConfig, // For the policy network
    ActionConfig actionConfig,
    MemoryConfig memoryConfig // For episodeLength / buffer capacity
) {
    public ReinforceAgentConfig {
        Objects.requireNonNull(hyperparams, "hyperparams cannot be null");
        Objects.requireNonNull(policyNetworkConfig, "policyNetworkConfig cannot be null");
        Objects.requireNonNull(actionConfig, "actionConfig cannot be null");
        Objects.requireNonNull(memoryConfig, "memoryConfig cannot be null");
        if (memoryConfig.replayBuffer() != null) {
            // REINFORCE is on-policy, replay buffer is not standard.
            // Allow it but note that episodeLength from memoryConfig is primary.
            System.err.println("Warning: ReplayBufferConfig in MemoryConfig is present for ReinforceAgentConfig. " +
                               "REINFORCE typically uses episode-based on-policy memory via memoryConfig.episodeLength().");
        }
         if (memoryConfig.episodeLength() == null || memoryConfig.episodeLength().intValue() <= 0) {
            throw new IllegalArgumentException("MemoryConfig must define a positive episodeLength for REINFORCE.");
        }
    }

    /**
     * Default constructor that initializes with default configurations.
     * Policy learning rate is taken from the default HyperparamConfig.
     */
    public ReinforceAgentConfig(int episodeLen) {
        this(
            new HyperparamConfig(), // Default hyperparameters
            new NetworkConfig( // Default network config for policy
                OptimizerConfig.of(new HyperparamConfig().policyLR().floatValue()) // Use default policy LR
            ),
            new ActionConfig(),     // Default action config
            new MemoryConfig(       // Default memory config for on-policy
                episodeLen
            )
        );
    }
}
