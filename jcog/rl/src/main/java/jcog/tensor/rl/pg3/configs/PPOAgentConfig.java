package jcog.tensor.rl.pg3.configs;

import java.util.Objects;

public record PPOAgentConfig(
    HyperparamConfig hyperparams,
    NetworkConfig policyNetworkConfig,
    NetworkConfig valueNetworkConfig,
    ActionConfig actionConfig,
    MemoryConfig memoryConfig,
    @Nullable ObservationNormalizerConfig obsNormConfig // Added optional observation normalization config
) {
    public PPOAgentConfig {
        Objects.requireNonNull(hyperparams, "hyperparams cannot be null");
        Objects.requireNonNull(policyNetworkConfig, "policyNetworkConfig cannot be null");
        Objects.requireNonNull(valueNetworkConfig, "valueNetworkConfig cannot be null");
        Objects.requireNonNull(actionConfig, "actionConfig cannot be null");
        Objects.requireNonNull(memoryConfig, "memoryConfig cannot be null");
        // obsNormConfig is optional

        if (memoryConfig.replayBuffer() != null) {
            System.err.println("Warning: ReplayBufferConfig in MemoryConfig is present for PPOAgentConfig. " +
                               "PPO typically uses on-policy batch collection via memoryConfig.episodeLength().");
        }
        // PPO is on-policy, requires episodeLength from MemoryConfig.
        if (memoryConfig.episodeLength() == null || memoryConfig.episodeLength().intValue() <= 0) {
            throw new IllegalArgumentException("MemoryConfig must define a positive episodeLength for PPO.");
        }
        // PPO specific hyperparameter checks from HyperparamConfig
        if (hyperparams.ppoClip().floatValue() <= 0) {
            throw new IllegalArgumentException("PPO clipping epsilon (ppoClip from hyperparams) must be positive.");
        }
        if (hyperparams.epochs().intValue() <= 0) {
            throw new IllegalArgumentException("Number of PPO epochs (from hyperparams) must be positive.");
        }
        // Lambda for GAE should be in [0,1] - this is usually validated in HyperparamConfig itself.
    }

    /**
     * Default constructor that initializes with default configurations.
     * Learning rates for policy and value networks are taken from HyperparamConfig defaults.
     * PPO specific parameters like clip range, epochs, lambda are also from HyperparamConfig defaults.
     */
    public PPOAgentConfig() {
        this(
            new HyperparamConfig(),
            new NetworkConfig(
                OptimizerConfig.of(new HyperparamConfig().policyLR().floatValue())
            ),
            new NetworkConfig(
                OptimizerConfig.of(new HyperparamConfig().valueLR().floatValue())
            ),
            new ActionConfig(),
            new MemoryConfig(        // PPO uses on-policy memory
                MemoryConfig.DEFAULT_EPISODE_LENGTH.intValue()
            ),
            null // Default for obsNormConfig
        );
    }
}
