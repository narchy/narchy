package jcog.tensor.rl.pg2.configs;

import java.util.Objects;

public record VPGAgentConfig(
    HyperparamConfig hyperparams,
    NetworkConfig policyNetworkConfig,
    NetworkConfig valueNetworkConfig, // Specific config for the value network
    ActionConfig actionConfig,
    MemoryConfig memoryConfig
) {
    public VPGAgentConfig {
        Objects.requireNonNull(hyperparams, "hyperparams cannot be null");
        Objects.requireNonNull(policyNetworkConfig, "policyNetworkConfig cannot be null");
        Objects.requireNonNull(valueNetworkConfig, "valueNetworkConfig cannot be null");
        Objects.requireNonNull(actionConfig, "actionConfig cannot be null");
        Objects.requireNonNull(memoryConfig, "memoryConfig cannot be null");

        if (memoryConfig.replayBuffer() != null) {
            // VPG is on-policy, replay buffer is not standard.
            System.err.println("Warning: ReplayBufferConfig in MemoryConfig is present for VPGAgentConfig. " +
                               "VPG typically uses episode-based on-policy memory via memoryConfig.episodeLength().");
        }
        if (memoryConfig.episodeLength() == null || memoryConfig.episodeLength().intValue() <= 0) {
            throw new IllegalArgumentException("MemoryConfig must define a positive episodeLength for VPG.");
        }
    }

    /**
     * Default constructor that initializes with default configurations.
     * Learning rates for policy and value networks are taken from HyperparamConfig defaults.
     */
    public VPGAgentConfig(int[] layers, int episodeLen) {
        this(
            new HyperparamConfig(), // Default hyperparameters
            new NetworkConfig( // Default network config for policy
                OptimizerConfig.of(new HyperparamConfig().policyLR().floatValue()), // Uses default policy LR
                layers
            ),
            new NetworkConfig( // Default network config for value function
                OptimizerConfig.of(new HyperparamConfig().valueLR().floatValue()), // Uses default value LR
                layers
            ),
            new ActionConfig(),     // Default action config
            new MemoryConfig(       // Default memory config for on-policy
                episodeLen
            )
        );
    }
}
